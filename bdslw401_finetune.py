"""
VideoMAE Fine-tuning — BdSLW401 (Original Paper Approach, Kaggle-Ready)
========================================================================
Faithful adaptation of BdSLW401.ipynb for Kaggle T4×2 GPU.

Key differences from the notebook:
  • Works with the flat Kaggle dataset layout (train/*.mp4, no class subfolders)
  • MAX_WORDS = 30  →  only W001–W030 are trained
  • TQDM progress bar per epoch AND per batch via custom TrainerCallback
  • All pip installs at the top — run as-is on a fresh Kaggle session
  • Confusion matrices + CSV metrics saved to /kaggle/working/output/
  • push_to_hub removed (not needed on Kaggle)

Expected dataset layout on Kaggle:
  /kaggle/input/bdslw401/Front/Front/train/W001S01F_01.mp4 ...
  /kaggle/input/bdslw401/Front/Front/val/...
  /kaggle/input/bdslw401/Front/Front/test/...

  File naming: W{NNN}S{NN}{view_char}_{trial}.mp4
"""

# ── 0. Installs ────────────────────────────────────────────────────────────────
import subprocess, sys

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'timm', 'einops', 'av',
     'evaluate', 'scikit-learn', 'seaborn'],
    check=False,
)

# ── 1. Imports ─────────────────────────────────────────────────────────────────
import os, re, logging, pathlib, random
import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms as TF
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    TrainerState,
    TrainerControl,
)
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ── 2. Config ──────────────────────────────────────────────────────────────────
DATA_ROOT    = pathlib.Path('/kaggle/input/datasets/hasanssl/bdslw401/Front/Front')
OUTPUT_DIR   = pathlib.Path('/kaggle/working/output')
MODEL_CKPT   = 'MCG-NJU/videomae-base-finetuned-kinetics'

MAX_WORDS    = 30          # only W001–W030 (set None for all 401)
VIEW_CHAR    = 'F'         # 'F' = Front, 'L' = Lateral
NUM_FRAMES   = 16          # VideoMAE fixed — do not change
NUM_EPOCHS   = 20
BATCH_SIZE   = 2           # per device (2 GPUs → effective 2×2×4 = 16)
GRAD_ACCUM   = 4
LR           = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16         = True
SEED         = 42
NUM_WORKERS  = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)')

# ── 3. Dataset scanning ────────────────────────────────────────────────────────
_PATTERN = re.compile(rf'^W(\d+)S\d+{VIEW_CHAR}_\d+\.mp4$', re.I)

def scan_split(split: str):
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        print(f'  [WARN] {split_dir} not found')
        return []
    files = []
    for f in sorted(split_dir.glob('*.mp4')):
        m = _PATTERN.match(f.name)
        if not m:
            continue
        if MAX_WORDS is not None and int(m.group(1)) > MAX_WORDS:
            continue
        files.append(f)
    return files

train_files = scan_split('train')
val_files   = scan_split('val')
test_files  = scan_split('test')

all_files    = train_files + val_files + test_files
class_labels = sorted({re.match(r'^(W\d+)', f.name, re.I).group(1).upper()
                        for f in all_files})
label2id = {lbl: i for i, lbl in enumerate(class_labels)}
id2label  = {i: lbl for lbl, i in label2id.items()}
NUM_CLASSES  = len(class_labels)

print(f'\nClasses  : {NUM_CLASSES}  {class_labels}')
print(f'Train    : {len(train_files)} videos')
print(f'Val      : {len(val_files)} videos')
print(f'Test     : {len(test_files)} videos')

# ── 4. Load image processor + model ───────────────────────────────────────────
print(f'\nLoading model  {MODEL_CKPT} ...')
image_processor = VideoMAEImageProcessor.from_pretrained(MODEL_CKPT)
model = VideoMAEForVideoClassification.from_pretrained(
    MODEL_CKPT,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model params: {total_params:.1f}M')

# ── 5. Transforms (matching original notebook) ────────────────────────────────
mean = image_processor.image_mean
std  = image_processor.image_std
if 'shortest_edge' in image_processor.size:
    h = w = image_processor.size['shortest_edge']
else:
    h, w = image_processor.size['height'], image_processor.size['width']
resize_to = (h, w)

MEAN_T = torch.tensor(mean).view(3, 1, 1)
STD_T  = torch.tensor(std).view(3, 1, 1)

_train_tf = TF.Compose([
    TF.Resize((int(h * 256 / 224), int(w * 256 / 224)), antialias=True),
    TF.RandomCrop(resize_to),
    TF.RandomHorizontalFlip(p=0.5),
])

_val_tf = TF.Compose([
    TF.Resize(resize_to, antialias=True),
    TF.CenterCrop(resize_to),
])

# ── 6. Dataset class ───────────────────────────────────────────────────────────
def _load_video(path: pathlib.Path) -> torch.Tensor:
    """Decode mp4, uniformly sample NUM_FRAMES, return [T, C, H, W] float."""
    try:
        vframes, _, _ = tvio.read_video(str(path), pts_unit='sec', output_format='TCHW')
    except Exception:
        return torch.zeros(NUM_FRAMES, 3, *resize_to)
    N = vframes.shape[0]
    if N == 0:
        return torch.zeros(NUM_FRAMES, 3, *resize_to)
    idx    = torch.linspace(0, N - 1, NUM_FRAMES).long()
    frames = vframes[idx].float() / 255.0          # [T, C, H, W]
    return frames


class SignVideoDataset(Dataset):
    """
    Reads flat BdSLW401 Kaggle layout: {split}/*.mp4
    Returns video as [C, T, H, W] — same convention as pytorchvideo.Ucf101.
    collate_fn then permutes to [T, C, H, W] before stacking into [B, T, C, H, W].
    """
    def __init__(self, files, is_train: bool):
        self.files    = files
        self.is_train = is_train
        self.tf       = _train_tf if is_train else _val_tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        wc    = re.match(r'^(W\d+)', path.name, re.I).group(1).upper()
        label = label2id[wc]

        frames = _load_video(path)                             # [T, C, H, W]
        frames = torch.stack([self.tf(f) for f in frames])    # [T, C, H, W]
        frames = (frames - MEAN_T) / STD_T                    # normalize
        video  = frames.permute(1, 0, 2, 3)                   # [C, T, H, W]

        return {'video': video, 'label': label}


def collate_fn(examples):
    """Identical to original notebook: permute [C,T,H,W] → [T,C,H,W] per sample."""
    pixel_values = torch.stack(
        [ex['video'].permute(1, 0, 2, 3) for ex in examples]
    )                                                           # [B, T, C, H, W]
    labels = torch.tensor([ex['label'] for ex in examples])
    return {'pixel_values': pixel_values, 'labels': labels}


train_dataset = SignVideoDataset(train_files, is_train=True)
val_dataset   = SignVideoDataset(val_files,   is_train=False)
test_dataset  = SignVideoDataset(test_files,  is_train=False)

# ── 7. Metrics (identical to original notebook) ───────────────────────────────
def compute_metrics(p):
    preds_argmax = np.argmax(p.predictions, axis=1)
    labels       = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_argmax, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, preds_argmax)
    logging.info(
        f'Metrics — Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  '
        f'Recall: {recall:.4f}  F1: {f1:.4f}'
    )
    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1}

# ── 8. TQDM Trainer callback ───────────────────────────────────────────────────
class TqdmProgressCallback(TrainerCallback):
    """Shows an epoch-level bar (outer) and a batch-level bar (inner)."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self._epoch_bar   = None
        self._batch_bar   = None
        self._cur_epoch   = 0
        self._steps_this_epoch = 0

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        self._epoch_bar = tqdm(
            total=self.total_epochs, desc='Training epochs',
            unit='ep', position=0, leave=True,
        )

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        self._cur_epoch        += 1
        self._steps_this_epoch  = 0
        steps_per_epoch = (
            (state.max_steps // self.total_epochs)
            if (state.max_steps and self.total_epochs)
            else None
        )
        self._batch_bar = tqdm(
            total=steps_per_epoch,
            desc=f'  Epoch {self._cur_epoch}/{self.total_epochs}',
            unit='step', position=1, leave=False,
        )

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        self._steps_this_epoch += 1
        if self._batch_bar is not None:
            self._batch_bar.update(1)
            if state.log_history:
                last = state.log_history[-1]
                post = {k: f'{v:.4f}' for k, v in last.items()
                        if k in ('loss', 'learning_rate')}
                if post:
                    self._batch_bar.set_postfix(**post)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        if self._batch_bar is not None:
            self._batch_bar.close()
            self._batch_bar = None
        if self._epoch_bar is not None:
            post = {}
            for entry in reversed(state.log_history):
                for k in ('eval_accuracy', 'eval_loss', 'loss'):
                    if k in entry and k not in post:
                        post[k] = f'{entry[k]:.4f}'
                if len(post) >= 2:
                    break
            self._epoch_bar.set_postfix(**post)
            self._epoch_bar.update(1)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kw):
        if self._batch_bar is not None:
            self._batch_bar.close()
        if self._epoch_bar is not None:
            self._epoch_bar.close()

# ── 9. TrainingArguments (same as original, adapted for Kaggle) ───────────────
_steps_per_epoch = max(1, len(train_files) // (BATCH_SIZE * GRAD_ACCUM))
_warmup_steps    = int(WARMUP_RATIO * _steps_per_epoch * NUM_EPOCHS)

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    remove_unused_columns       = False,
    eval_strategy         = 'epoch',
    save_strategy               = 'epoch',
    learning_rate               = LR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_steps                = _warmup_steps,
    num_train_epochs            = NUM_EPOCHS,
    logging_steps               = 50,
    metric_for_best_model       = 'accuracy',
    greater_is_better           = True,
    load_best_model_at_end      = True,
    report_to                   = 'none',
    fp16                        = FP16,
    weight_decay                = WEIGHT_DECAY,
    dataloader_num_workers      = NUM_WORKERS,
    dataloader_pin_memory       = True,
    seed                        = SEED,
    save_total_limit            = 3,
)

# ── 10. Trainer ────────────────────────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    processing_class = image_processor,
    compute_metrics = compute_metrics,
    data_collator   = collate_fn,
    callbacks       = [
        EarlyStoppingCallback(early_stopping_patience=5),
        TqdmProgressCallback(total_epochs=NUM_EPOCHS),
    ],
)

# ── 11. Train ──────────────────────────────────────────────────────────────────
eff_batch = BATCH_SIZE * max(torch.cuda.device_count(), 1) * GRAD_ACCUM
print(f'\nStarting training')
print(f'  Classes      : {NUM_CLASSES}')
print(f'  Epochs       : {NUM_EPOCHS}')
print(f'  Batch/device : {BATCH_SIZE}   Grad accum: {GRAD_ACCUM}')
print(f'  Eff. batch   : {eff_batch}')
print(f'  Learning rate: {LR}')
print(f'  FP16         : {FP16}')
print(f'  Output dir   : {OUTPUT_DIR}')

trainer.train()
trainer.save_model(str(OUTPUT_DIR / 'final_model'))
print('\nModel saved.')

# ── 12. Confusion matrix helper (identical to original notebook) ───────────────
def plot_confusion_matrix(
    conf_matrix, class_labels,
    accuracy=None, precision=None, recall=None, f1=None,
    dataset_name='Test Data', normalize=False, filename='confusion_matrix.png',
):
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_mat = conf_matrix.astype('float') / np.where(row_sums == 0, 1, row_sums)
    else:
        conf_mat = conf_matrix

    n = len(class_labels)
    fig_size = max(8, n // 2)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        conf_mat, annot=(n <= 40),
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual',    fontsize=12)
    plt.title(
        f'VideoMAE BdSLW401 | {dataset_name} | {NUM_CLASSES} classes\n'
        f'Acc={accuracy:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}',
        fontsize=10,
    )
    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(str(save_path), dpi=100)
    plt.close()
    print(f'  Saved → {save_path}')

# ── 13. Post-training evaluation (all three splits) ───────────────────────────
results = {}
for split_name, ds in [('Train', train_dataset),
                        ('Validation', val_dataset),
                        ('Test', test_dataset)]:
    print(f'\n--- {split_name} evaluation ---')
    preds_out    = trainer.predict(ds)
    preds_argmax = np.argmax(preds_out.predictions, axis=1)
    labels       = preds_out.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds_argmax, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, preds_argmax)
    logging.info(
        f'{split_name} — Acc={accuracy:.4f}  P={precision:.4f}  '
        f'R={recall:.4f}  F1={f1:.4f}'
    )

    results[split_name] = dict(accuracy=accuracy, precision=precision,
                                recall=recall, f1=f1)

    plot_confusion_matrix(
        conf_matrix  = confusion_matrix(labels, preds_argmax),
        class_labels = class_labels,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        dataset_name = f'{split_name} Data',
        filename     = f'confusion_matrix_{split_name.lower()}.png',
    )
    plot_confusion_matrix(
        conf_matrix  = confusion_matrix(labels, preds_argmax),
        class_labels = class_labels,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        dataset_name = f'{split_name} Data (normalised)',
        normalize    = True,
        filename     = f'confusion_matrix_{split_name.lower()}_norm.png',
    )

# ── 14. Save metrics + state ───────────────────────────────────────────────────
trainer.log_metrics('test', results['Test'])
trainer.save_metrics('test', results['Test'])
trainer.save_state()

print('\n' + '='*60)
print('Training complete.')
for split_name, m in results.items():
    print(f"  {split_name:12s}  Acc={m['accuracy']:.4f}  "
          f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
print(f'Outputs saved to: {OUTPUT_DIR}')
print('='*60)
