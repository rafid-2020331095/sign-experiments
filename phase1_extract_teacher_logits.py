"""
Phase 1: Teacher Logit Extraction (Offline, Run Once)
======================================================
Loads the fully fine-tuned VideoMAE from bdslw401_finetune.py's saved output
and extracts raw (un-softmaxed) output logits for every video in train/val/test.

Output:
  /kaggle/working/teacher_logits.pt  — dict { video_stem -> Tensor(NUM_CLASSES,) }

This file takes up virtually no disk space (~401 floats per video).
Run this once and feed teacher_logits.pt into phase2_x3d_distillation_training.py.

Expected input layout (same as bdslw401_finetune.py):
  /kaggle/input/datasets/hasanssl/bdslw401/Front/Front/train/W001S01F_01.mp4 ...
  /kaggle/working/output/final_model/   <- saved by bdslw401_finetune.py
"""

# ── 0. Installs ────────────────────────────────────────────────────────────────
import subprocess, sys

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'av', 'timm', 'einops'],
    check=False,
)

# ── 1. Imports ─────────────────────────────────────────────────────────────────
import os, re, pathlib
import torch
import torchvision.io as tvio
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from tqdm.auto import tqdm

# ── 2. Config ──────────────────────────────────────────────────────────────────
DATA_ROOT   = pathlib.Path('/kaggle/input/datasets/hasanssl/bdslw401/Front/Front')
MODEL_DIR   = pathlib.Path('/kaggle/working/output/final_model')   # saved by bdslw401_finetune.py
LOGITS_OUT  = pathlib.Path('/kaggle/working/teacher_logits.pt')

MAX_WORDS   = 30          # must match bdslw401_finetune.py (None = all 401)
VIEW_CHAR   = 'F'         # 'F' = Front view
NUM_FRAMES  = 16          # VideoMAE fixed input length — do NOT change
BATCH_SIZE  = 16          # inference only — GPU can handle much larger batches
NUM_WORKERS = 8           # more workers to keep GPU fed
SPLITS      = ['train', 'val', 'test']   # extract for all splits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)')

# ── 3. Dataset scanning (identical logic to bdslw401_finetune.py) ──────────────
_PATTERN = re.compile(rf'^W(\d+)S\d+{VIEW_CHAR}_\d+\.mp4$', re.I)


def scan_split(split: str):
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        print(f'  [WARN] {split_dir} not found — skipping')
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


all_files = []
for split in SPLITS:
    split_files = scan_split(split)
    print(f'  {split:6s}: {len(split_files)} videos')
    all_files.extend(split_files)

class_labels = sorted({
    re.match(r'^(W\d+)', f.name, re.I).group(1).upper()
    for f in all_files
})
label2id  = {lbl: i for i, lbl in enumerate(class_labels)}
id2label  = {i: lbl for lbl, i in label2id.items()}
NUM_CLASSES = len(class_labels)
print(f'\nClasses: {NUM_CLASSES}   Total videos: {len(all_files)}')

# ── 4. Load fine-tuned teacher (VideoMAE) ──────────────────────────────────────
print(f'\nLoading fine-tuned VideoMAE from: {MODEL_DIR}')
image_processor = VideoMAEImageProcessor.from_pretrained(str(MODEL_DIR))
teacher = VideoMAEForVideoClassification.from_pretrained(str(MODEL_DIR))
teacher.eval()
teacher.to(device)

total_params = sum(p.numel() for p in teacher.parameters()) / 1e6
print(f'Teacher params: {total_params:.1f}M')
print(f'Teacher head  : {teacher.classifier}')

# ── 5. Transforms — val-style (no augmentation, centre crop) ──────────────────
mean = image_processor.image_mean
std  = image_processor.image_std
if 'shortest_edge' in image_processor.size:
    h = w = image_processor.size['shortest_edge']
else:
    h, w = image_processor.size['height'], image_processor.size['width']
resize_to = (h, w)

MEAN_T = torch.tensor(mean).view(3, 1, 1)
STD_T  = torch.tensor(std).view(3, 1, 1)

_val_tf = TF.Compose([
    TF.Resize(resize_to, antialias=True),
    TF.CenterCrop(resize_to),
])


def _load_video(path: pathlib.Path) -> torch.Tensor:
    """Decode mp4, uniformly sample NUM_FRAMES, return [T, C, H, W] float."""
    try:
        vframes, _, _ = tvio.read_video(str(path), pts_unit='sec', output_format='TCHW')
    except Exception as e:
        print(f'  [WARN] failed to read {path.name}: {e}')
        return torch.zeros(NUM_FRAMES, 3, *resize_to)
    N = vframes.shape[0]
    if N == 0:
        return torch.zeros(NUM_FRAMES, 3, *resize_to)
    idx    = torch.linspace(0, N - 1, NUM_FRAMES).long()
    frames = vframes[idx].float() / 255.0  # [T, C, H, W]
    return frames


# ── 6. Extraction Dataset ──────────────────────────────────────────────────────
class LogitExtractionDataset(Dataset):
    """Returns (video_stem, pixel_values) — no label needed for extraction."""

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path   = self.files[idx]
        stem   = path.stem                                     # e.g. "W001S01F_01"
        frames = _load_video(path)                             # [T, C, H, W]
        frames = _val_tf(frames)                               # [T, C, H, W] — vectorised, no loop
        frames = (frames - MEAN_T) / STD_T                    # normalise
        return stem, frames                                    # [T, C, H, W] — no premature permute


def collate_extraction(batch):
    stems        = [b[0] for b in batch]
    pixel_values = torch.stack([b[1] for b in batch])          # [B, T, C, H, W] — already correct shape
    return stems, pixel_values


# ── 7. Run extraction ──────────────────────────────────────────────────────────
dataset = LogitExtractionDataset(all_files)
loader  = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    prefetch_factor=4,          # workers pre-stage 4 batches ahead while GPU runs
    persistent_workers=True,    # keep worker processes alive between batches
    collate_fn=collate_extraction,
)

logits_dict = {}   # { video_stem (str) : logit_tensor (NUM_CLASSES,) }

print(f'\nExtracting raw logits for {len(all_files)} videos ...')
print('(No gradients, no softmax — raw output of the final linear layer)')

with torch.inference_mode():  # faster than no_grad — disables grad + version counter
    for stems, pixel_values in tqdm(loader, desc='Teacher extraction'):
        pixel_values = pixel_values.to(device)
        outputs      = teacher(pixel_values=pixel_values)
        logits       = outputs.logits.cpu()                    # [B, NUM_CLASSES]
        for stem, lg in zip(stems, logits):
            logits_dict[stem] = lg                             # Tensor(NUM_CLASSES,)

print(f'\nExtracted logits for {len(logits_dict)} videos.')

# ── 8. Sanity check ────────────────────────────────────────────────────────────
missing = [f.stem for f in all_files if f.stem not in logits_dict]
if missing:
    print(f'[WARN] {len(missing)} videos have no logits: {missing[:5]} ...')
else:
    print('Sanity check: all videos have logits.')

sample_key = next(iter(logits_dict))
sample_lg  = logits_dict[sample_key]
print(f'\nSample key    : {sample_key}')
print(f'Logit shape   : {sample_lg.shape}  (expected [{NUM_CLASSES}])')
print(f'Logit range   : {sample_lg.min():.4f}  →  {sample_lg.max():.4f}')
print(f'Argmax class  : {id2label[int(sample_lg.argmax())]}')

# ── 9. Save ────────────────────────────────────────────────────────────────────
torch.save(
    {
        'logits':      logits_dict,
        'label2id':    label2id,
        'id2label':    id2label,
        'num_classes': NUM_CLASSES,
        'class_labels': class_labels,
    },
    str(LOGITS_OUT),
)
print(f'\nSaved → {LOGITS_OUT}')
print(f'File size: {LOGITS_OUT.stat().st_size / 1e6:.2f} MB')
print('\n[Phase 1 complete]  Pass teacher_logits.pt to phase2_x3d_distillation_training.py')
