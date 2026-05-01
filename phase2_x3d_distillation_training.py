"""
Phase 2: X3D-XS Student Training via Offline Knowledge Distillation
=====================================================================
VideoMAE is completely absent from memory.  The teacher's knowledge arrives
as pre-computed logit tensors from phase1_extract_teacher_logits.pt.

Loss (Hinton KD):
    L = α * CE(student_logits, hard_label)
      + (1 - α) * T² * KL(softmax(student/T) ‖ softmax(teacher/T))

Hinton et al. "Distilling the Knowledge in a Neural Network", 2015.

Expected inputs:
  /kaggle/input/datasets/hasanssl/bdslw401/Front/Front/train/*.mp4
  /kaggle/working/teacher_logits.pt      <- produced by phase1_extract_teacher_logits.py

Output:
  /kaggle/working/x3d_kd/best_x3d_kd.pt
  /kaggle/working/x3d_kd/training_curves.png
  /kaggle/working/x3d_kd/metrics.pt
"""

# ── 0. Installs ────────────────────────────────────────────────────────────────
import subprocess, sys

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'av', 'timm', 'einops', 'fvcore', 'decord', 'numexpr'],
    check=False,
)

# ── 1. Imports ─────────────────────────────────────────────────────────────────
import os, re, math, time, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import decord
decord.bridge.set_bridge('torch')
import numexpr
numexpr.set_num_threads(1)
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 2. Config ──────────────────────────────────────────────────────────────────
DATA_ROOT       = pathlib.Path('/kaggle/input/datasets/hasanssl/bdslw401/Front/Front')
TEACHER_LOGITS  = pathlib.Path('/kaggle/input/datasets/rafidadib/teacher-logits/teacher_logits.pt')   # from Phase 1
OUTPUT_DIR      = pathlib.Path('/kaggle/working/x3d_kd')

MAX_WORDS       = 30          # must match bdslw401_finetune.py
VIEW_CHAR       = 'F'

# ── X3D-XS clip settings ──────────────────────────────────────────────────────
CLIP_LEN        = 8           # frames per clip (x3d_xs default=4; 8 works better for signs)
INPUT_SPATIAL   = 182         # spatial crop size (x3d_xs default)

# ── Kinetics-400 normalisation (used by X3D pretrained weights) ───────────────
X3D_MEAN = [0.45, 0.45, 0.45]
X3D_STD  = [0.225, 0.225, 0.225]

# ── Distillation hyperparameters ──────────────────────────────────────────────
TEMPERATURE   = 4.0     # T: controls softness of teacher distribution (try 3–6)
ALPHA         = 0.5     # weight on hard-label CE; (1-ALPHA) on soft KD loss
                        # set ALPHA=0.2 to lean harder on teacher guidance

# ── Training hyperparameters ──────────────────────────────────────────────────
NUM_EPOCHS    = 10
BATCH_SIZE    = 32      # X3D-XS is tiny — can afford large batches
LR            = 1e-3    # OneCycleLR max_lr
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
NUM_WORKERS   = 4
SEED          = 42
FP16          = True
PATIENCE      = 15      # early stopping

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f'Device : {device}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)')

# Normalisation constants on GPU — shape (1,3,1,1,1) broadcasts over [B,C,T,H,W]
_MEAN_GPU = torch.tensor(X3D_MEAN).view(1, 3, 1, 1, 1).to(device)
_STD_GPU  = torch.tensor(X3D_STD).view(1, 3, 1, 1, 1).to(device)

# ── 3. Load teacher logits (no model in memory) ────────────────────────────────
print(f'\nLoading teacher logits from {TEACHER_LOGITS} ...')
ckpt_data     = torch.load(str(TEACHER_LOGITS), map_location='cpu')
logits_dict   = ckpt_data['logits']       # { stem: Tensor(NUM_CLASSES,) }
label2id      = ckpt_data['label2id']
id2label      = ckpt_data['id2label']
class_labels  = ckpt_data['class_labels']
NUM_CLASSES   = ckpt_data['num_classes']

print(f'Teacher logits : {len(logits_dict)} videos   classes={NUM_CLASSES}')
print(f'Temperature T  : {TEMPERATURE}   Alpha α: {ALPHA}')

# ── 4. Dataset scanning ────────────────────────────────────────────────────────
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

print(f'\nTrain: {len(train_files)}   Val: {len(val_files)}   Test: {len(test_files)}')

# Warn about videos without teacher logits (should be 0 if both MAX_WORDS match)
missing_train = [f.stem for f in train_files if f.stem not in logits_dict]
if missing_train:
    print(f'[WARN] {len(missing_train)} train videos have no teacher logits — they will use CE only')

# ── 5. Transforms ──────────────────────────────────────────────────────────────
_train_spatial = int(INPUT_SPATIAL * 256 / 224)   # resize before random crop

_train_tf = v2.Compose([
    v2.Resize((_train_spatial, _train_spatial), antialias=True),
    v2.RandomCrop((INPUT_SPATIAL, INPUT_SPATIAL)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
])

_val_tf = v2.Compose([
    v2.Resize((INPUT_SPATIAL, INPUT_SPATIAL), antialias=True),
    v2.CenterCrop((INPUT_SPATIAL, INPUT_SPATIAL)),
])


def _load_video(path: pathlib.Path, num_frames: int, tf) -> torch.Tensor:
    """Decode only the needed frames via decord, apply spatial transform.
    Returns un-normalised [0, 1] Tensor [C, T, H, W]. Normalisation happens on GPU.
    """
    try:
        vr           = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames == 0:
            return torch.zeros(3, num_frames, INPUT_SPATIAL, INPUT_SPATIAL)
        idx    = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
        frames = vr.get_batch(idx)                             # [T, H, W, C] uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255.0   # [T, C, H, W]
        frames = tf(frames)                                    # [T, C, H, W] — v2 processes whole clip at once
        return frames.permute(1, 0, 2, 3)                      # [C, T, H, W]
    except Exception as e:
        print(f'  [WARN] failed to read {path.name}: {e}')
        return torch.zeros(3, num_frames, INPUT_SPATIAL, INPUT_SPATIAL)


# ── 6. Dataset ─────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    """
    Returns (pixel_values, hard_label, teacher_logits) per video.
      pixel_values    : Tensor [C, T, H, W]  — ready for X3D (after collate adds batch dim)
      hard_label      : int
      teacher_logits  : Tensor [NUM_CLASSES] — raw un-softmaxed teacher output
                        If a video has no pre-computed logit (shouldn't happen),
                        falls back to a zero tensor (CE-only supervision).
    """

    def __init__(self, files, is_train: bool):
        self.files    = files
        self.is_train = is_train
        self.tf       = _train_tf if is_train else _val_tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path  = self.files[idx]
        stem  = path.stem
        wc    = re.match(r'^(W\d+)', path.name, re.I).group(1).upper()
        label = label2id[wc]

        video    = _load_video(path, CLIP_LEN, self.tf)           # [C, T, H, W]
        t_logits = logits_dict.get(stem, torch.zeros(NUM_CLASSES))

        return video, label, t_logits


def collate_fn(batch):
    videos     = torch.stack([b[0] for b in batch])          # [B, C, T, H, W]
    labels     = torch.tensor([b[1] for b in batch])          # [B]
    t_logits   = torch.stack([b[2] for b in batch])           # [B, NUM_CLASSES]
    return videos, labels, t_logits


train_dataset = DistillationDataset(train_files, is_train=True)
val_dataset   = DistillationDataset(val_files,   is_train=False)
test_dataset  = DistillationDataset(test_files,  is_train=False)

# Balanced sampler for training
train_labels   = [label2id[re.match(r'^(W\d+)', f.name, re.I).group(1).upper()] for f in train_files]
class_counts   = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
sample_weights = 1.0 / np.maximum(class_counts[np.array(train_labels)], 1.0)
sampler        = WeightedRandomSampler(
    weights     = torch.tensor(sample_weights, dtype=torch.float64),
    num_samples = len(train_dataset),
    replacement = True,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=collate_fn,
)

# ── 7. Build X3D-XS student model ──────────────────────────────────────────────
print('\nLoading X3D-XS with Kinetics-400 pretrained weights ...')
student = torch.hub.load(
    'facebookresearch/pytorchvideo',
    'x3d_xs',
    pretrained=True,
)

# Locate and replace the final classification head
# PyTorchVideo X3D: model.blocks[-1].proj  is the final nn.Linear(D, 400)
_final_block = student.blocks[-1]
if hasattr(_final_block, 'proj'):
    in_features = _final_block.proj.in_features
    _final_block.proj = nn.Linear(in_features, NUM_CLASSES, bias=True)
    nn.init.trunc_normal_(_final_block.proj.weight, std=0.02)
    nn.init.zeros_(_final_block.proj.bias)
    print(f'Replaced head: Linear({in_features}, {NUM_CLASSES})')
else:
    raise RuntimeError(
        'Could not find `model.blocks[-1].proj` in X3D-XS. '
        'Check the PyTorchVideo version or X3D architecture.'
    )

student.to(device)
total_params = sum(p.numel() for p in student.parameters()) / 1e6
train_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
print(f'Total params   : {total_params:.2f}M')
print(f'Trainable      : {train_params:.2f}M')

# ── 8. Hinton Knowledge Distillation Loss ──────────────────────────────────────
def kd_loss(
    student_logits: torch.Tensor,    # [B, NUM_CLASSES]  raw student output
    teacher_logits: torch.Tensor,    # [B, NUM_CLASSES]  raw teacher output (saved offline)
    hard_labels:    torch.Tensor,    # [B]               integer ground-truth
    T:  float = TEMPERATURE,
    alpha: float = ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, ce_loss, kl_loss).

    L = α  · CE(student, y_hard)
      + (1-α) · T² · KL( softmax(student/T) ‖ softmax(teacher/T) )

    The T² factor compensates for the reduced gradient magnitude when
    using soft targets, restoring it to the same scale as hard-label CE.
    """
    ce_loss = F.cross_entropy(student_logits, hard_labels)

    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits  / T, dim=-1)
    kl_loss      = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

    total_loss = alpha * ce_loss + (1.0 - alpha) * (T ** 2) * kl_loss
    return total_loss, ce_loss, kl_loss


# ── 9. Optimizer & Scheduler ───────────────────────────────────────────────────
STEPS_PER_EPOCH = len(train_loader)
TOTAL_STEPS     = NUM_EPOCHS * STEPS_PER_EPOCH

optimizer = AdamW(student.parameters(), lr=LR / 10, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer, max_lr=LR, total_steps=TOTAL_STEPS,
    pct_start=0.2, anneal_strategy='cos',
    div_factor=10, final_div_factor=1e4,
)
scaler = GradScaler(enabled=FP16)

print(f'\nSteps/epoch: {STEPS_PER_EPOCH}   Total steps: {TOTAL_STEPS}')
print(f'Distillation — T={TEMPERATURE}  α={ALPHA}  (CE weight={ALPHA:.2f}, KD weight={1-ALPHA:.2f})')


# ── 10. Train / Eval helpers ───────────────────────────────────────────────────
def train_one_epoch(epoch: int) -> dict:
    student.train()
    total_loss = total_ce = total_kl = correct = n = 0

    bar = tqdm(train_loader, desc=f'  Ep{epoch+1:03d} train', leave=False)
    for videos, labels, t_logits in bar:
        videos   = videos.to(device, non_blocking=True)       # [B, C, T, H, W]
        videos   = (videos - _MEAN_GPU) / _STD_GPU            # normalise on GPU
        labels   = labels.to(device, non_blocking=True)
        t_logits = t_logits.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=FP16):
            s_logits           = student(videos)               # [B, NUM_CLASSES]
            loss, l_ce, l_kl   = kd_loss(s_logits, t_logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs          = videos.size(0)
        total_loss += loss.item() * bs
        total_ce   += l_ce.item() * bs
        total_kl   += l_kl.item() * bs
        correct    += (s_logits.argmax(1) == labels).sum().item()
        n          += bs

        bar.set_postfix(
            loss=f'{loss.item():.3f}',
            ce  =f'{l_ce.item():.3f}',
            kl  =f'{l_kl.item():.3f}',
            acc =f'{100.*correct/n:.1f}%',
        )

    return {
        'loss': total_loss / n,
        'ce':   total_ce   / n,
        'kl':   total_kl   / n,
        'acc':  correct    / n,
    }


@torch.no_grad()
def evaluate(loader: DataLoader) -> dict:
    student.eval()
    total_loss = correct = correct_top5 = n = 0
    all_preds, all_labels = [], []

    for videos, labels, t_logits in loader:
        videos   = videos.to(device, non_blocking=True)
        videos   = (videos - _MEAN_GPU) / _STD_GPU            # normalise on GPU
        labels   = labels.to(device, non_blocking=True)
        t_logits = t_logits.to(device, non_blocking=True)

        with autocast(enabled=FP16):
            s_logits          = student(videos)
            loss, _, _        = kd_loss(s_logits, t_logits, labels)

        top5 = s_logits.topk(min(5, NUM_CLASSES), dim=1).indices
        total_loss   += loss.item() * videos.size(0)
        correct      += (s_logits.argmax(1) == labels).sum().item()
        correct_top5 += (top5 == labels.unsqueeze(1)).any(1).sum().item()
        n            += videos.size(0)
        all_preds.append(s_logits.argmax(1).detach())
        all_labels.append(labels.detach())

    all_preds  = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    return {
        'loss':     total_loss / n,
        'top1_acc': correct    / n,
        'top5_acc': correct_top5 / n,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
    }


# ── 11. Training loop ──────────────────────────────────────────────────────────
best_val_acc      = 0.0
patience_counter  = 0
best_model_path   = OUTPUT_DIR / 'best_x3d_kd.pt'

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_top1':  [], 'val_top5': [],
}

print(f'\n{"Ep":>4} | {"T-Loss":>8} | {"T-Acc":>7} | {"V-Loss":>8} | '
      f'{"V-Top1":>7} | {"V-Top5":>7} | {"F1":>7} | {"Time":>5}')
print('-' * 75)

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    ep_start = time.time()

    train_m = train_one_epoch(epoch)
    val_m   = evaluate(val_loader)
    ep_time = time.time() - ep_start

    history['train_loss'].append(train_m['loss'])
    history['train_acc'].append(train_m['acc'])
    history['val_loss'].append(val_m['loss'])
    history['val_top1'].append(val_m['top1_acc'])
    history['val_top5'].append(val_m['top5_acc'])

    star = ''
    if val_m['top1_acc'] > best_val_acc:
        best_val_acc = val_m['top1_acc']
        torch.save(
            {
                'epoch':       epoch + 1,
                'model_state': student.state_dict(),
                'val_top1':    best_val_acc,
                'val_top5':    val_m['top5_acc'],
                'label2id':    label2id,
                'id2label':    id2label,
                'num_classes': NUM_CLASSES,
                'clip_len':    CLIP_LEN,
            },
            str(best_model_path),
        )
        patience_counter = 0
        star = '  ★'
    else:
        patience_counter += 1

    print(
        f'{epoch+1:>4} | {train_m["loss"]:>8.4f} | {train_m["acc"]*100:>6.2f}% | '
        f'{val_m["loss"]:>8.4f} | {val_m["top1_acc"]*100:>6.2f}% | '
        f'{val_m["top5_acc"]*100:>6.2f}% | {val_m["f1"]*100:>6.2f}% | '
        f'{ep_time:>4.0f}s{star}'
    )

    if patience_counter >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch + 1}.')
        break

total_min = (time.time() - start_time) / 60
print('=' * 75)
print(f'Training complete in {total_min:.1f} min  |  Best val top-1: {best_val_acc*100:.2f}%')

# ── 12. Final test evaluation (best model) ─────────────────────────────────────
print('\nLoading best model for test evaluation ...')
best_ckpt = torch.load(str(best_model_path), map_location=device)
student.load_state_dict(best_ckpt['model_state'])

test_m = evaluate(test_loader)
print(f'\nTest results (best val ckpt):')
print(f'  Top-1   : {test_m["top1_acc"]*100:.2f}%')
print(f'  Top-5   : {test_m["top5_acc"]*100:.2f}%')
print(f'  F1      : {test_m["f1"]*100:.2f}%')
print(f'  Precision: {test_m["precision"]*100:.2f}%')
print(f'  Recall  : {test_m["recall"]*100:.2f}%')

# ── 13. Save metrics & training curves ────────────────────────────────────────
torch.save(
    {
        'history': history,
        'test_metrics': test_m,
        'best_val_acc': best_val_acc,
        'config': {
            'clip_len': CLIP_LEN, 'spatial': INPUT_SPATIAL,
            'temperature': TEMPERATURE, 'alpha': ALPHA,
            'num_classes': NUM_CLASSES, 'lr': LR,
            'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS,
        },
    },
    str(OUTPUT_DIR / 'metrics.pt'),
)

ep = list(range(1, len(history['train_loss']) + 1))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ep, history['train_loss'], 'b-o', ms=3, label='Train Total Loss')
axes[0].plot(ep, history['val_loss'],   'r-o', ms=3, label='Val Total Loss')
axes[0].set_title('Loss (KD)'); axes[0].legend(); axes[0].set_xlabel('Epoch')

axes[1].plot(ep, [v * 100 for v in history['train_acc']],  'b-o', ms=3, label='Train Top-1')
axes[1].plot(ep, [v * 100 for v in history['val_top1']],   'r-o', ms=3, label='Val Top-1')
axes[1].set_title('Top-1 Accuracy'); axes[1].legend()
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('%')

axes[2].plot(ep, [v * 100 for v in history['val_top5']], 'g-o', ms=3, label='Val Top-5')
axes[2].set_title('Val Top-5 Accuracy'); axes[2].legend()
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('%')

plt.tight_layout()
curve_path = OUTPUT_DIR / 'training_curves.png'
plt.savefig(str(curve_path), dpi=120)
plt.close()
print(f'\nCurves saved → {curve_path}')
print(f'Best model  → {best_model_path}')
print(f'Metrics     → {OUTPUT_DIR / "metrics.pt"}')
print('\n[Phase 2 complete]')
