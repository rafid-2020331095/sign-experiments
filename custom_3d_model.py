"""
Custom 3D Model: MicroX3D_400 Student — Offline Logit Distillation
===================================================================
Replaces X3D-XS with a fully custom MicroX3D_400 architecture trained via
Hinton KD against the same pre-extracted VideoMAE teacher logits from Phase 1.

Model design highlights vs X3D-XS:
  - Depthwise Separable 3D Convolutions → ~1.4 M parameters (ultra-lightweight)
  - 512-D final embedding  → enough capacity to hold 30 (or 401) classes cleanly
  - Delayed Spatial Pooling in Stage 2  → keeps spatial resolution for hand details
  - Optional kd_proj head (768D) for future feature-level distillation (not used here)

Distillation (Option 1 — Logit KD, same as phase2):
    L = α · CE(student, y_hard)
      + (1-α) · T² · KL( softmax(student/T) ‖ softmax(teacher/T) )

Expected inputs:
  /kaggle/input/datasets/hasanssl/bdslw401/Front/Front/{train,val,test}/*.mp4
  /kaggle/working/teacher_logits.pt   ← produced by phase1_extract_teacher_logits.py

Output:
  /kaggle/working/micro_x3d_kd/best_micro_x3d_kd.pt
  /kaggle/working/micro_x3d_kd/training_curves.png
  /kaggle/working/micro_x3d_kd/metrics.pt
"""

# ── 0. Installs ─────────────────────────────────────────────────────────────────
import subprocess, sys

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'decord', 'numexpr'],
    check=False,
)

# ── 1. Imports ──────────────────────────────────────────────────────────────────
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 2. Config ───────────────────────────────────────────────────────────────────
DATA_ROOT       = pathlib.Path('/kaggle/input/datasets/hasanssl/bdslw401/Front/Front')
TEACHER_LOGITS  = pathlib.Path('/kaggle/input/datasets/rafidadib/teacher-logits/teacher_logits.pt')
OUTPUT_DIR      = pathlib.Path('/kaggle/working/micro_x3d_kd')

MAX_WORDS       = 30          # first 30 classes — must match phase1
VIEW_CHAR       = 'F'

# ── Clip settings ───────────────────────────────────────────────────────────────
CLIP_LEN        = 8
INPUT_SPATIAL   = 182

# ── Kinetics-400 normalisation (same pipeline as X3D baseline) ─────────────────
X3D_MEAN = [0.45, 0.45, 0.45]
X3D_STD  = [0.225, 0.225, 0.225]

# ── Distillation hyperparameters ────────────────────────────────────────────────
TEMPERATURE   = 4.0
ALPHA         = 0.5     # CE weight; (1-ALPHA) on soft KD loss

# ── Training hyperparameters ────────────────────────────────────────────────────
NUM_EPOCHS    = 30      # more epochs since model trains from scratch (no pretrained)
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
NUM_WORKERS   = 4
SEED          = 42
FP16          = True
PATIENCE      = 15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f'Device : {device}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)')

# GPU normalisation constants — shape (1,3,1,1,1) broadcasts over [B,C,T,H,W]
_MEAN_GPU = torch.tensor(X3D_MEAN).view(1, 3, 1, 1, 1).to(device)
_STD_GPU  = torch.tensor(X3D_STD).view(1, 3, 1, 1, 1).to(device)


# ── 3. Load teacher logits ──────────────────────────────────────────────────────
print(f'\nLoading teacher logits from {TEACHER_LOGITS} ...')
ckpt_data    = torch.load(str(TEACHER_LOGITS), map_location='cpu')
logits_dict  = ckpt_data['logits']       # { video_stem: Tensor(NUM_CLASSES,) }
label2id     = ckpt_data['label2id']
id2label     = ckpt_data['id2label']
class_labels = ckpt_data['class_labels']
NUM_CLASSES  = ckpt_data['num_classes']  # 30

print(f'Teacher logits : {len(logits_dict)} videos   classes={NUM_CLASSES}')
print(f'Temperature T  : {TEMPERATURE}   Alpha α: {ALPHA}')


# ── 4. MicroX3D_400 Architecture ───────────────────────────────────────────────

class Bottleneck3D(nn.Module):
    """
    Lightweight 3D residual block using Depthwise Separable Convolutions.
    Expand → Depthwise 3D → Project (inverted bottleneck).
    """
    def __init__(self, in_dim, out_dim, stride=(1, 1, 1), expansion=2.0):
        super().__init__()
        mid_dim = int(in_dim * expansion)

        # 1. Pointwise Expansion
        self.conv1 = nn.Conv3d(in_dim, mid_dim, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm3d(mid_dim)

        # 2. Depthwise 3D Convolution
        self.conv2 = nn.Conv3d(mid_dim, mid_dim, kernel_size=3, stride=stride,
                               padding=1, groups=mid_dim, bias=False)
        self.bn2   = nn.BatchNorm3d(mid_dim)

        # 3. Pointwise Projection
        self.conv3 = nn.Conv3d(mid_dim, out_dim, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(out_dim)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection: 1×1 conv whenever shape changes
        self.shortcut = nn.Sequential()
        if stride != (1, 1, 1) or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_dim),
            )

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.relu(self.bn1(self.conv1(x)))
        x   = self.relu(self.bn2(self.conv2(x)))
        x   = self.bn3(self.conv3(x))
        return self.relu(x + res)


class MicroX3D_400(nn.Module):
    """
    Scaled-up MicroX3D student model.

    Key design choices:
      - 512-D final embedding (vs 128 in a naïve tiny model) for class separation
      - Delayed Spatial Pooling: Stage 2 stride=(2,1,1) shrinks time only,
        preserving spatial resolution for fine hand/finger detail
      - Dedicated kd_proj head: maps 512D → videomae_dim (768) for optional
        future feature-level distillation (not used in this script)

    Input  : [B, 3, T, H, W]  — T=8 frames, H=W=182
    Output : logits [B, num_classes]
             (or logits + projected_features if return_features=True)
    """

    def __init__(self, num_classes=401, videomae_dim=768):
        super().__init__()

        # 1. Stem: initial spatial downsample
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # 2. Stage 1: base features
        self.stage1 = Bottleneck3D(32, 64, stride=(1, 2, 2), expansion=2.0)

        # 3. Stage 2: delayed spatial pooling — shrink time only
        self.stage2 = nn.Sequential(
            Bottleneck3D(64,  128, stride=(2, 1, 1), expansion=2.0),
            Bottleneck3D(128, 128, stride=(1, 1, 1), expansion=2.0),
        )

        # 4. Stage 3: shrink both time and space
        self.stage3 = nn.Sequential(
            Bottleneck3D(128, 256, stride=(2, 2, 2), expansion=2.5),
            Bottleneck3D(256, 256, stride=(1, 1, 1), expansion=2.5),
        )

        # 5. Stage 4: final expansion to 512D
        self.stage4 = Bottleneck3D(256, 512, stride=(1, 2, 2), expansion=3.0)

        # 6. Global pooling + dropout
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout     = nn.Dropout(0.4)

        # Classification head
        self.fc = nn.Linear(512, num_classes)

        # Optional projection head for feature-level KD (VideoMAE dim=768)
        self.kd_proj = nn.Linear(512, videomae_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x        = self.global_pool(x)   # [B, 512, 1, 1, 1]
        features = x.flatten(1)          # [B, 512]

        logits = self.fc(self.dropout(features))   # [B, num_classes]

        if return_features:
            projected = self.kd_proj(features)     # [B, 768]  — for feature-level KD
            return logits, projected

        return logits


# ── 5. Dataset scanning ─────────────────────────────────────────────────────────
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

missing_train = [f.stem for f in train_files if f.stem not in logits_dict]
if missing_train:
    print(f'[WARN] {len(missing_train)} train videos have no teacher logits — CE-only supervision')


# ── 6. Transforms ───────────────────────────────────────────────────────────────
_train_spatial = int(INPUT_SPATIAL * 256 / 224)

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
    """Decode only the needed frames, apply spatial transform.
    Returns un-normalised [0,1] Tensor [C, T, H, W]. GPU normalisation done later.
    """
    try:
        vr           = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
        total_frames = len(vr)
        if total_frames == 0:
            return torch.zeros(3, num_frames, INPUT_SPATIAL, INPUT_SPATIAL)
        idx    = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
        frames = vr.get_batch(idx)                              # [T, H, W, C] uint8
        frames = frames.permute(0, 3, 1, 2).float() / 255.0    # [T, C, H, W]
        frames = tf(frames)                                     # v2 processes whole clip
        return frames.permute(1, 0, 2, 3)                       # [C, T, H, W]
    except Exception as e:
        print(f'  [WARN] failed to read {path.name}: {e}')
        return torch.zeros(3, num_frames, INPUT_SPATIAL, INPUT_SPATIAL)


# ── 7. Dataset ──────────────────────────────────────────────────────────────────
class DistillationDataset(Dataset):
    """
    Returns (pixel_values, hard_label, teacher_logits) per video.
      pixel_values   : [C, T, H, W]  un-normalised, ready for collate
      hard_label     : int
      teacher_logits : [NUM_CLASSES] raw teacher logits (CE-only fallback if missing)
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

        video    = _load_video(path, CLIP_LEN, self.tf)
        t_logits = logits_dict.get(stem, torch.zeros(NUM_CLASSES))

        return video, label, t_logits


def collate_fn(batch):
    videos   = torch.stack([b[0] for b in batch])   # [B, C, T, H, W]
    labels   = torch.tensor([b[1] for b in batch])   # [B]
    t_logits = torch.stack([b[2] for b in batch])    # [B, NUM_CLASSES]
    return videos, labels, t_logits


train_dataset = DistillationDataset(train_files, is_train=True)
val_dataset   = DistillationDataset(val_files,   is_train=False)
test_dataset  = DistillationDataset(test_files,  is_train=False)

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

# ── 8. Build MicroX3D_400 student ───────────────────────────────────────────────
print(f'\nBuilding MicroX3D_400 (num_classes={NUM_CLASSES}, trained from scratch) ...')
student = MicroX3D_400(num_classes=NUM_CLASSES, videomae_dim=768).to(device)

total_params = sum(p.numel() for p in student.parameters()) / 1e6
train_params = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6
print(f'Total params   : {total_params:.3f}M')
print(f'Trainable      : {train_params:.3f}M')

# Sanity check forward pass
with torch.no_grad():
    _dummy = torch.randn(2, 3, CLIP_LEN, INPUT_SPATIAL, INPUT_SPATIAL).to(device)
    _out   = student(_dummy)
    print(f'Forward check  : input {list(_dummy.shape)}  →  logits {list(_out.shape)}')
    del _dummy, _out


# ── 9. Hinton KD Loss ───────────────────────────────────────────────────────────
def kd_loss(
    student_logits: torch.Tensor,   # [B, NUM_CLASSES]
    teacher_logits: torch.Tensor,   # [B, NUM_CLASSES]
    hard_labels:    torch.Tensor,   # [B]
    T:     float = TEMPERATURE,
    alpha: float = ALPHA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, ce_loss, kl_loss).

    L = α · CE(student, y_hard)
      + (1-α) · T² · KL( softmax(student/T) ‖ softmax(teacher/T) )
    """
    ce_loss      = F.cross_entropy(student_logits, hard_labels)
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits    / T, dim=-1)
    kl_loss      = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    total_loss   = alpha * ce_loss + (1.0 - alpha) * (T ** 2) * kl_loss
    return total_loss, ce_loss, kl_loss


# ── 10. Optimizer & Scheduler ───────────────────────────────────────────────────
STEPS_PER_EPOCH = len(train_loader)
TOTAL_STEPS     = NUM_EPOCHS * STEPS_PER_EPOCH

optimizer = AdamW(student.parameters(), lr=LR / 10, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer, max_lr=LR, total_steps=TOTAL_STEPS,
    pct_start=0.2, anneal_strategy='cos',
    div_factor=10, final_div_factor=1e4,
)
scaler = GradScaler(enabled=FP16)

print(f'\nSteps/epoch : {STEPS_PER_EPOCH}   Total steps : {TOTAL_STEPS}')
print(f'KD setup    — T={TEMPERATURE}  α={ALPHA}  (CE={ALPHA:.2f}, KD={1-ALPHA:.2f})')


# ── 11. Train / Eval helpers ────────────────────────────────────────────────────
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
            s_logits         = student(videos)                 # [B, NUM_CLASSES]
            loss, l_ce, l_kl = kd_loss(s_logits, t_logits, labels)

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

    return {'loss': total_loss/n, 'ce': total_ce/n, 'kl': total_kl/n, 'acc': correct/n}


@torch.no_grad()
def evaluate(loader: DataLoader) -> dict:
    student.eval()
    total_loss = correct = correct_top5 = n = 0
    all_preds, all_labels = [], []

    for videos, labels, t_logits in loader:
        videos   = videos.to(device, non_blocking=True)
        videos   = (videos - _MEAN_GPU) / _STD_GPU
        labels   = labels.to(device, non_blocking=True)
        t_logits = t_logits.to(device, non_blocking=True)

        with autocast(enabled=FP16):
            s_logits      = student(videos)
            loss, _, _    = kd_loss(s_logits, t_logits, labels)

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
        all_labels, all_preds, average='weighted', zero_division=0,
    )
    return {
        'loss':      total_loss / n,
        'top1_acc':  correct    / n,
        'top5_acc':  correct_top5 / n,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'preds':     all_preds,
        'labels':    all_labels,
    }


# ── 12. Training loop ───────────────────────────────────────────────────────────
best_val_acc     = 0.0
patience_counter = 0
best_model_path  = OUTPUT_DIR / 'best_micro_x3d_kd.pt'

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
                'model_arch':  'MicroX3D_400',
                'embed_dim':   512,
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


# ── 13. Final test evaluation ────────────────────────────────────────────────────
print('\nLoading best model for test evaluation ...')
best_ckpt = torch.load(str(best_model_path), map_location=device)
student.load_state_dict(best_ckpt['model_state'])

test_m = evaluate(test_loader)
macro_f1 = f1_score(test_m['labels'], test_m['preds'], average='macro')

print(f'\nTest results (best val ckpt — epoch {best_ckpt["epoch"]}):')
print('=' * 45)
print(f'  Top-1 Accuracy : {test_m["top1_acc"]*100:.2f}%')
print(f'  Top-5 Accuracy : {test_m["top5_acc"]*100:.2f}%')
print(f'  Macro F1       : {macro_f1*100:.2f}%')
print(f'  Weighted F1    : {test_m["f1"]*100:.2f}%')
print(f'  Precision      : {test_m["precision"]*100:.2f}%')
print(f'  Recall         : {test_m["recall"]*100:.2f}%')
print('=' * 45)

# ── 14. Save metrics & confusion matrix ─────────────────────────────────────────
torch.save(
    {
        'history':    history,
        'test_metrics': {
            'top1_acc':  float(test_m['top1_acc']),
            'top5_acc':  float(test_m['top5_acc']),
            'macro_f1':  float(macro_f1),
            'weighted_f1': float(test_m['f1']),
        },
        'best_val_acc': best_val_acc,
        'config': {
            'model':       'MicroX3D_400',
            'clip_len':    CLIP_LEN,
            'spatial':     INPUT_SPATIAL,
            'temperature': TEMPERATURE,
            'alpha':       ALPHA,
            'num_classes': NUM_CLASSES,
            'lr':          LR,
            'batch_size':  BATCH_SIZE,
            'num_epochs':  NUM_EPOCHS,
            'embed_dim':   512,
        },
    },
    str(OUTPUT_DIR / 'metrics.pt'),
)

# Training curves
ep_list = list(range(1, len(history['train_loss']) + 1))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(ep_list, history['train_loss'], 'b-o', ms=3, label='Train')
axes[0].plot(ep_list, history['val_loss'],   'r-o', ms=3, label='Val')
axes[0].set_title('Loss (KD)'); axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].grid(alpha=0.3)

axes[1].plot(ep_list, [v * 100 for v in history['train_acc']], 'b-o', ms=3, label='Train Top-1')
axes[1].plot(ep_list, [v * 100 for v in history['val_top1']],  'r-o', ms=3, label='Val Top-1')
axes[1].axhline(best_val_acc * 100, color='gray', linestyle='--', label=f'Best {best_val_acc*100:.1f}%')
axes[1].set_title('Top-1 Accuracy (MicroX3D_400)'); axes[1].legend()
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('%'); axes[1].grid(alpha=0.3)

axes[2].plot(ep_list, [v * 100 for v in history['val_top5']], 'g-o', ms=3, label='Val Top-5')
axes[2].set_title('Val Top-5 Accuracy'); axes[2].legend()
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('%'); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'training_curves.png'), dpi=120)
plt.close()
print(f'\nCurves  → {OUTPUT_DIR / "training_curves.png"}')

# Confusion matrix
with open(str(TEACHER_LOGITS).replace('teacher_logits.pt', 'label_map.json'), 'r', errors='ignore') as _f:
    _label_map = json.load(_f) if False else {str(v): k for k, v in label2id.items()}

cm       = confusion_matrix(test_m['labels'], test_m['preds'])
wl       = [_label_map.get(str(i), str(i)) for i in range(NUM_CLASSES)]
fig, ax  = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(wl, rotation=90, fontsize=8)
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(wl, fontsize=8)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'MicroX3D_400 Confusion Matrix  top-1={test_m["top1_acc"]*100:.1f}%', fontsize=13)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'confusion_matrix.png'), dpi=120)
plt.close()
print(f'CM      → {OUTPUT_DIR / "confusion_matrix.png"}')

print(f'Model   → {best_model_path}')
print(f'Metrics → {OUTPUT_DIR / "metrics.pt"}')
print('\n[custom_3d_model.py complete]')
