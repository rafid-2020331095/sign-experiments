"""
BdSL-SPOTER | Keypoint Training Script  (GPU)
==============================================
Trains the BdSL-SPOTER transformer on pre-extracted MediaPipe Holistic
keypoints produced by keypoint_extraction.py.

Input (Kaggle dataset mounted from keypoint_extraction.py output)
-----------------------------------------------------------------
  /kaggle/input/<your-dataset-name>/
      train.npz       X: (N, 200, 150)  float32   y: (N,)  int64
      val.npz         X: (N, 200, 150)  float32   y: (N,)  int64
      test.npz        X: (N, 200, 150)  float32   y: (N,)  int64
      config.json     { num_classes, seq_len, feature_dim, num_landmarks }
      label_map.json  { "0": "W001", "1": "W002", ... }

Feature vector layout per frame (150 values)
---------------------------------------------
  [0   : 66 ]  33 pose landmarks     × (x, y)
  [66  : 108]  21 left-hand landmarks × (x, y)   (zeros if not detected)
  [108 : 150]  21 right-hand landmarks × (x, y)  (zeros if not detected)

Model changes vs. the original notebook (D_MODEL=66, 33 pose landmarks)
------------------------------------------------------------------------
  D_MODEL  : 150  (= feature_dim, up from 66)
  N_HEADS  : 6    (150 / 6 = 25 per head, same design intent as paper)
  flip aug : left-hand block (66:108) ↔ right-hand block (108:150) added
  classes  : 30   (from config.json — set NUM_WORDS=30 in extraction)
  Everything else: identical to notebook (4 layers, d_ff=512, curriculum,
  label smoothing 0.1, dropout 0.15, weight decay 1e-4, OneCycleLR, AMP)

Epoch checkpointing / resume
-----------------------------
  After every epoch saves:
      /kaggle/working/checkpoints/epoch_{N:02d}.pt
  On restart: auto-loads the latest checkpoint and resumes from the next epoch.
  Best model always written to /kaggle/working/best_bdsl_spoter.pt

Usage
-----
    python keypoint_training.py
    # or in a Kaggle notebook cell:  %run keypoint_training.py
"""

import os, json, time, glob, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')   # headless — safe on Kaggle
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Paths — UPDATE KEYPOINTS_DIR to match your Kaggle dataset name ────────────
KEYPOINTS_DIR = '/kaggle/input/datasets/rafidadib/keypoint-30/keypoints'   # <-- set to your dataset name
OUTPUT_DIR    = '/kaggle/working'
CKPT_DIR      = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Load config written by keypoint_extraction.py ─────────────────────────────
with open(os.path.join(KEYPOINTS_DIR, 'config.json')) as f:
    _cfg = json.load(f)

NUM_CLASSES   = _cfg['num_classes']     # 30  (first 30 words)
SEQ_LEN       = _cfg['seq_len']         # 200
FEATURE_DIM   = _cfg['feature_dim']     # 150 (75 landmarks × x, y)
NUM_LANDMARKS = _cfg['num_landmarks']   # 75
NUM_SIGNERS   = _cfg.get('num_signers', 18)  # max signer index (for GRL discriminator)

# ── Model hyperparameters ─────────────────────────────────────────────────────
# FIX: D_MODEL is now INDEPENDENT of FEATURE_DIM.
# A Linear(150 → 128) projection is added at the start of the model so that:
#   (a) attention head width is reasonable (128/8=16 per head)
#   (b) the model learns which landmark combinations matter
#   (c) fewer attention parameters → fits better on 2929 training samples
D_MODEL      = 128           # projection target (was 150 = FEATURE_DIM — caused slow convergence)
N_HEADS      = 8             # 128 / 8 = 16 per head
N_LAYERS     = 4             # exactly as paper
D_FF         = 512           # exactly as paper
DROPOUT      = 0.20          # increased from 0.15 — reduce train→test gap (was 56% train, 39% test)
LABEL_SMOOTH = 0.05          # reduced from 0.1 — too much smoothing suppressed gradient signal
WEIGHT_DECAY = 5e-4          # increased from 1e-4 — stronger L2 regularization for small dataset
MAX_EPOCHS   = 80            # increased from 30 — model was still improving at epoch 30 (not converged)
PATIENCE     = 15            # increased to give model time to push through plateaus
BATCH_SIZE   = 32
MAX_LR       = 3e-4          # FIX: reduced from 1e-3 — previous LR caused overshoot and collapse
SEED         = 42

# Set to False to always train from epoch 1 (ignores any saved checkpoints)
RESUME_FROM_CKPT = False

# ── GRL (Gradient Reversal Layer) — Signer Invariance ────────────────────────────
# When USE_GRL=True the model adds a signer discriminator head with a gradient
# reversal layer (GRL) on top of the pooled feature vector. The GRL negates
# gradients flowing back to the backbone, forcing it to produce features that
# are indistinguishable between signers. This directly targets the val→test gap
# caused by unseen test signers (4 and 8 here).
# Requires signer_id array in the .npz files (re-run keypoint_extraction.py).
USE_GRL     = True    # set False to disable and train without signer invariance
LAMBDA_ADV  = 0.5     # adversarial loss weight (same as experiment E3/E4 in videomae_train.py)
DISC_HIDDEN = 256     # hidden dim of signer discriminator MLP

# FIX: Curriculum learning DISABLED.
# Root cause of epoch-3 collapse: curriculum pads positions [clen:200] with zeros,
# so the learnable positional encoding learns 'these positions = zeros = ignore'.
# At epoch 3, real data suddenly appears in those positions → loss spikes above ln(30)
# → model collapses to predicting only W020 for all inputs.
CURRICULUM_EPOCHS    = 0    # 0 = disabled  (was 3 — caused collapse)
CURRICULUM_START_LEN = 50   # unused when CURRICULUM_EPOCHS=0

assert D_MODEL % N_HEADS == 0, f'D_MODEL ({D_MODEL}) must be divisible by N_HEADS ({N_HEADS})'

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    print(f'VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'\nFeature dim : {FEATURE_DIM}  (150 = 33 pose + 21 L-hand + 21 R-hand) × (x,y)')
print(f'D_MODEL     : {D_MODEL}  (= feature_dim)')
print(f'N_HEADS     : {N_HEADS}  ({D_MODEL}/{N_HEADS} = {D_MODEL//N_HEADS} per head)')
print(f'N_LAYERS    : {N_LAYERS}')
print(f'NUM_CLASSES : {NUM_CLASSES}')
print(f'SEQ_LEN     : {SEQ_LEN}')


# ── Dataset ───────────────────────────────────────────────────────────────────

# BlazePose left↔right swap pairs for horizontal flip (pose section only, 0:66)
BLAZE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


class BdSLDataset(Dataset):
    """
    Loads pre-extracted keypoint .npz files produced by keypoint_extraction.py.
    X shape: (N, 200, 150)  — 200 frames, 150 features (75 landmarks × x, y)

    Data augmentations applied during training:
      1. Temporal dropout   — randomly drop ~10% of frames then resample
      2. Coordinate noise   — Gaussian noise σ=0.002 on all coordinates
      3. Horizontal flip    — negate x, swap pose LR pairs AND left↔right hand blocks
    """

    def __init__(self, npz_path, augment=False, curriculum_len=None):
        data = np.load(npz_path)
        self.X            = data['X'].astype(np.float32)   # (N, 200, 150)
        self.y            = data['y'].astype(np.int64)      # (N,)
        # Load signer IDs if available (written by updated keypoint_extraction.py)
        if 'signer_id' in data:
            self.signer_id = (data['signer_id'].astype(np.int64) - 1)  # convert to 0-based
        else:
            self.signer_id = np.zeros(len(self.y), dtype=np.int64)     # fallback
        self.augment      = augment
        self.curriculum_len = curriculum_len
        has_sid = 'signer_id' in data
        print(f'Loaded {os.path.basename(npz_path)}: '
              f'X={self.X.shape}, classes={len(np.unique(self.y))}'
              f', signer_ids={"found" if has_sid else "NOT FOUND (GRL will be a no-op)"}')

    def __len__(self): return len(self.y)

    # ── Augmentations ─────────────────────────────────────────────────────────

    def temporal_dropout(self, seq, p=0.15):
        T    = len(seq)
        mask = np.random.rand(T) > p
        kept = seq[mask]
        if len(kept) < 2:
            return seq
        old_idx = np.linspace(0, len(kept) - 1, len(kept))
        new_idx = np.linspace(0, len(kept) - 1, T)
        out = np.zeros_like(seq)
        for d in range(seq.shape[1]):
            out[:, d] = np.interp(new_idx, old_idx, kept[:, d])
        return out

    def coordinate_noise(self, seq, sigma=0.004):
        noise = np.zeros_like(seq)
        noise[:, :150] = np.random.normal(0, sigma, (seq.shape[0], 150)).astype(np.float32)
        return seq + noise

    def landmark_dropout(self, seq, p=0.10):
        """
        Randomly zero out entire individual landmarks (both x and y) for all frames.
        Simulates partial occlusion / detection failure for a landmark.
        p = probability each landmark is dropped for the whole sequence.
        Layout: 33 pose (cols 0:66) | 21 L-hand (66:108) | 21 R-hand (108:150)
        Only operates on the first 150 coordinate columns (not geometric features).
        """
        seq  = seq.copy()
        n_lm = 75                          # always 75 landmarks in the coordinate block
        mask = np.random.rand(n_lm) < p   # True = drop this landmark
        for i in np.where(mask)[0]:
            seq[:, i * 2]     = 0.0       # x
            seq[:, i * 2 + 1] = 0.0       # y
        return seq

    def temporal_scale(self, seq):
        """
        Randomly stretch or compress the signing speed by resampling to a
        length in [0.8×, 1.2×] of SEQ_LEN, then resize back to SEQ_LEN.
        Breaks systematic confusions caused by speed-dependent patterns.
        """
        T      = seq.shape[0]
        scale  = np.random.uniform(0.8, 1.2)
        new_T  = max(2, int(T * scale))
        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, new_T)
        scaled  = np.zeros((new_T, seq.shape[1]), dtype=np.float32)
        for d in range(seq.shape[1]):
            scaled[:, d] = np.interp(new_idx, old_idx, seq[:, d])
        final_idx = np.linspace(0, new_T - 1, T)
        out = np.zeros_like(seq)
        for d in range(seq.shape[1]):
            out[:, d] = np.interp(final_idx, np.arange(new_T), scaled[:, d])
        return out

    def horizontal_flip(self, seq):
        """
        Mirror the signing space horizontally.
        For 150-dim features:
          - Negate all x columns (even indices across full 150 dim)
          - Swap pose left↔right pairs  (within cols 0:66)
          - Swap entire left-hand block (cols 66:108) with right-hand block (cols 108:150)
        """
        seq = seq.copy()

        # Step 1: negate x-coordinates only in the 150-dim coordinate block
        seq[:, 0:150:2] = -seq[:, 0:150:2]

        # Step 2: swap pose landmark LR pairs (landmark indices within 0:66)
        for a, b in BLAZE_LR_PAIRS:
            xa, ya = a * 2, a * 2 + 1
            xb, yb = b * 2, b * 2 + 1
            seq[:, [xa, ya]], seq[:, [xb, yb]] = \
                seq[:, [xb, yb]].copy(), seq[:, [xa, ya]].copy()

        # Step 3: swap left-hand block ↔ right-hand block (raw coords)
        left_block          = seq[:, 66:108].copy()
        right_block         = seq[:, 108:150].copy()
        seq[:, 66:108]      = right_block
        seq[:, 108:150]     = left_block

        # Step 4: swap geometric feature blocks for left ↔ right hand
        # Layout of geo features (cols 150+):
        #   [150:170] L-bone-len  [170:190] R-bone-len
        #   [190:200] L-angles    [200:210] R-angles
        #   [210:215] L-tip-dist  [215:220] R-tip-dist
        #   [220]     inter-hand  [221] L-wrist-nose  [222] R-wrist-nose
        if seq.shape[1] > 150:
            for l_s, l_e, r_s, r_e in [
                (150, 170, 170, 190),   # bone lengths
                (190, 200, 200, 210),   # angles
                (210, 215, 215, 220),   # fingertip distances
            ]:
                tmp = seq[:, l_s:l_e].copy()
                seq[:, l_s:l_e] = seq[:, r_s:r_e]
                seq[:, r_s:r_e] = tmp
            # swap wrist-to-nose: col 221 ↔ 222
            if seq.shape[1] > 222:
                seq[:, [221, 222]] = seq[:, [222, 221]]

        return seq

    def apply_curriculum(self, seq):
        if self.curriculum_len is None or self.curriculum_len >= SEQ_LEN:
            return seq
        trimmed = seq[:self.curriculum_len]
        pad     = np.zeros((SEQ_LEN - self.curriculum_len, seq.shape[1]), dtype=np.float32)
        return np.concatenate([trimmed, pad], axis=0)

    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        seq   = self.X[idx].copy()
        label = self.y[idx]
        if self.augment:
            if np.random.rand() < 0.60: seq = self.temporal_dropout(seq)
            if np.random.rand() < 0.60: seq = self.coordinate_noise(seq)
            if np.random.rand() < 0.50: seq = self.horizontal_flip(seq)
            if np.random.rand() < 0.50: seq = self.landmark_dropout(seq)
            if np.random.rand() < 0.40: seq = self.temporal_scale(seq)
        seq = self.apply_curriculum(seq)
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(self.signer_id[idx], dtype=torch.long))


# ── GRL helpers ────────────────────────────────────────────────────────────────────────

class GRLFunction(torch.autograd.Function):
    """Custom autograd function that reverses gradients during backprop."""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GRL(nn.Module):
    """Gradient Reversal Layer — lambda ramps from 0 → 1 over training."""
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def set_lambda(self, lam: float):
        self.lam = lam

    def forward(self, x):
        return GRLFunction.apply(x, self.lam)


class SignerDiscriminator(nn.Module):
    """2-layer MLP that predicts signer identity from backbone features."""
    def __init__(self, in_dim: int, hidden: int, n_signers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, n_signers),
        )

    def forward(self, x):
        return self.net(x)


def grl_lambda_schedule(step: int, total_steps: int) -> float:
    """Ramps lambda from ~0 at step 0 to ~1.0 at step total_steps."""
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ── Model ─────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding — Paper Eq. 2."""
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.encoding, std=0.02)

    def forward(self, x):
        return x + self.encoding


class BdSLSPOTER(nn.Module):
    """
    BdSL-SPOTER adapted for 150-dim MediaPipe Holistic input.

    Input pipeline (FIX vs original):
      1. input_norm  : LayerNorm(150)    — stabilises raw landmark values
      2. input_proj  : Linear(150→128)   — learns which landmark combos matter
      3. pos_enc     : LearnablePos(128) — added AFTER projection (not before)
      4. transformer : 4× encoder layers, D_MODEL=128, 8 heads
      5. global avg pool → 3-layer classifier

    Why input_proj fixes the collapse:
      Raw 150-dim features fed directly into self-attention require the model
      to simultaneously learn landmark weighting AND temporal patterns with only
      2929 training samples. Separating these into a projection layer first
      reduces the effective attention space and greatly speeds up convergence.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        feature_dim=FEATURE_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        n_signers=NUM_SIGNERS,
        disc_hidden=DISC_HIDDEN,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        # Input normalisation + projection
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)

        self.pos_enc = LearnablePositionalEncoding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # 3-layer sign classification head
        h1, h2, h3 = d_model * 2, d_model, num_classes * 2
        self.classifier = nn.Sequential(
            nn.Linear(d_model, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),      nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, h3),      nn.LayerNorm(h3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h3, num_classes),
        )

        # GRL signer discriminator head
        self.grl  = GRL()
        self.disc = SignerDiscriminator(d_model, disc_hidden, n_signers)

        self._init_weights()

    def set_grl_lambda(self, lam: float):
        self.grl.set_lambda(lam)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):                    # x: (B, T, 150)
        x    = self.input_norm(x)            # (B, T, 150)
        x    = self.input_proj(x)            # (B, T, 128)
        x    = self.pos_enc(x)               # (B, T, 128)
        x    = self.transformer(x)           # (B, T, 128)
        feat = x.mean(dim=1)                 # (B, 128) — global avg pool
        sign_out   = self.classifier(feat)   # (B, num_classes)
        # Discriminator only runs during training to save eval compute
        signer_out = self.disc(self.grl(feat)) if (USE_GRL and self.training) else None
        return sign_out, signer_out


# ── DataLoader builder ────────────────────────────────────────────────────────

def get_dataloaders(epoch):
    """
    Build train/val loaders for the current epoch.
    Curriculum learning is disabled (CURRICULUM_EPOCHS=0) — see explanation above.
    Uses WeightedRandomSampler to guarantee balanced class representation per batch,
    which prevents the model from collapsing to predicting only dominant classes.
    """
    clen = None   # curriculum disabled

    train_ds = BdSLDataset(os.path.join(KEYPOINTS_DIR, 'train.npz'),
                           augment=True, curriculum_len=clen)
    val_ds   = BdSLDataset(os.path.join(KEYPOINTS_DIR, 'val.npz'),
                           augment=False, curriculum_len=None)

    # FIX: WeightedRandomSampler — each sample's weight = 1/class_count
    # Even with balanced dataset (90-99/class), this ensures every batch
    # sees each class with equal probability, preventing mode collapse.
    labels        = train_ds.y                                         # (N,)
    class_counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    sample_weights = 1.0 / np.maximum(class_counts[labels], 1)        # (N,)
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, clen


# ── Train / Eval functions ────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    criterion, disc_criterion, epoch, global_step, total_steps):
    model.train()
    total_loss = correct = total = 0

    bar = tqdm(loader, desc=f'  Ep{epoch+1:02d} train', leave=False)
    for X, y, signer_ids in bar:
        X          = X.to(DEVICE, non_blocking=True)
        y          = y.to(DEVICE, non_blocking=True)
        signer_ids = signer_ids.to(DEVICE, non_blocking=True)

        # Ramp GRL lambda: 0 at step 0 → ~1.0 at final step
        lam = grl_lambda_schedule(global_step, total_steps)
        model.set_grl_lambda(lam)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            sign_logits, signer_logits = model(X)
            l_ce = criterion(sign_logits, y)
            if USE_GRL and signer_logits is not None:
                l_adv = disc_criterion(signer_logits, signer_ids)
            else:
                l_adv = torch.tensor(0.0, device=DEVICE)
            loss = l_ce + LAMBDA_ADV * l_adv

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        total_loss += l_ce.item() * X.size(0)
        correct    += (sign_logits.argmax(1) == y).sum().item()
        total      += X.size(0)
        bar.set_postfix(ce=f'{l_ce.item():.3f}', adv=f'{l_adv.item():.3f}',
                        acc=f'{100.*correct/total:.1f}%', lam=f'{lam:.2f}')

    return total_loss / total, correct / total, global_step


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = correct_top5 = total = 0
    all_preds, all_labels = [], []

    for X, y, _signer_ids in loader:   # signer_ids not used during evaluation
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast():
            sign_logits, _ = model(X)   # discriminator skipped in eval mode
            loss = criterion(sign_logits, y)

        top5 = sign_logits.topk(min(5, NUM_CLASSES), dim=1).indices
        total_loss   += loss.item() * X.size(0)
        correct      += (sign_logits.argmax(1) == y).sum().item()
        correct_top5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total        += X.size(0)
        all_preds.extend(sign_logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return {
        'loss':     total_loss / total,
        'top1_acc': correct / total,
        'top5_acc': correct_top5 / total,
        'preds':    np.array(all_preds),
        'labels':   np.array(all_labels),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_epoch_ckpt(model, optimizer, scheduler, scaler, epoch, best_val_acc, patience_counter):
    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state':    scaler.state_dict(),
        'best_val_acc':    best_val_acc,
        'patience_counter': patience_counter,
    }, os.path.join(CKPT_DIR, f'epoch_{epoch:02d}.pt'))


def load_latest_ckpt(model, optimizer, scheduler, scaler):
    """
    Loads the most recent compatible epoch checkpoint.
    Skips gracefully if:
      - RESUME_FROM_CKPT is False (fresh start requested)
      - model architecture has changed (shape mismatch)
      - scheduler total_steps has changed (MAX_EPOCHS was edited)
    Returns (start_epoch, best_val_acc, patience_counter).
    """
    if not RESUME_FROM_CKPT:
        print('  RESUME_FROM_CKPT=False — starting fresh from epoch 1.')
        return 0, 0.0, 0
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'epoch_*.pt')))
    if not files:
        return 0, 0.0, 0
    for fpath in reversed(files):   # try newest first
        ckpt = torch.load(fpath, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt['model_state'])
        except RuntimeError as e:
            print(f'  [WARN] {os.path.basename(fpath)}: architecture mismatch — starting fresh.')
            print(f'  Detail: {str(e)[:120]}...')
            return 0, 0.0, 0
        # Restore optimizer and scaler (non-fatal if mismatched)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception:
            pass
        # Restore scheduler only if total_steps still matches; otherwise rebuild position
        sched_state = ckpt.get('scheduler_state', {})
        if sched_state.get('total_steps') == scheduler.total_steps:
            try:
                scheduler.load_state_dict(sched_state)
            except Exception:
                pass
        else:
            # MAX_EPOCHS changed: fast-forward scheduler to the right step position
            ep          = ckpt['epoch']
            steps_done  = ep * (len(ckpt['model_state']) and (scheduler.total_steps // MAX_EPOCHS))
            steps_done  = ep * (scheduler.total_steps // MAX_EPOCHS)
            for _ in range(steps_done):
                scheduler.step()
            print(f'  [INFO] Scheduler rebuilt for new total_steps={scheduler.total_steps} '
                  f'(fast-forwarded {steps_done} steps to epoch {ep}).')
        ep = ckpt['epoch']
        print(f'Resumed from {os.path.basename(fpath)}  '
              f'(best_val_acc={ckpt["best_val_acc"]*100:.2f}%)')
        return ep, ckpt['best_val_acc'], ckpt['patience_counter']
    return 0, 0.0, 0


# ── Pre-training Diagnostics ────────────────────────────────────────────────────

def run_diagnostics():
    """
    Comprehensive data-quality audit run BEFORE training.
    Prints clear warnings for issues that are known to cause near-random performance.

    The #1 suspected root cause of random-level accuracy (loss stuck at ln(30)≈3.40):
      → Very high zero-frame rate: when MediaPipe fails to detect the body/hands,
        the feature vector is all-zeros for that frame. If >40% of frames are zero,
        all class sequences look nearly identical and the model cannot learn.
    """
    print('\n' + '=' * 70)
    print('  PRE-TRAINING DIAGNOSTICS')
    print('  (Run this before every training run to catch data issues early)')
    print('=' * 70)

    # Load label map written by keypoint_extraction.py
    label_map_path = os.path.join(KEYPOINTS_DIR, 'label_map.json')
    with open(label_map_path) as f:
        label_map = json.load(f)   # {"0": "W001", "1": "W002", ...}

    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(KEYPOINTS_DIR, f'{split}.npz')
        if not os.path.exists(path):
            print(f'  [WARN] {split}.npz NOT FOUND — skipping')
            continue
        d = np.load(path)
        splits[split] = {'X': d['X'].astype(np.float32), 'y': d['y'].astype(np.int64)}

    # ── 1. Basic shapes ──────────────────────────────────────────────────────
    print('\n[1] Shape & Label Range')
    print(f'  {"Split":<8} {"Samples":>8} {"Classes":>8} {"Label min–max":>14} {"Feat dim":>10}')
    print(f'  {"-" * 52}')
    for split, data in splits.items():
        X, y = data['X'], data['y']
        print(f'  {split:<8} {len(y):>8,} {len(np.unique(y)):>8} '
              f'  {y.min():>4}–{y.max():<6} {X.shape[2]:>10}')
        if X.shape[2] != FEATURE_DIM:
            print(f'  *** ERROR: expected feature_dim={FEATURE_DIM}, got {X.shape[2]}. Re-run extraction!')
        if X.shape[1] != SEQ_LEN:
            print(f'  *** ERROR: expected seq_len={SEQ_LEN}, got {X.shape[1]}.')

    # ── 2. Label map spot-check ───────────────────────────────────────────────
    print('\n[2] Label Map Spot-Check  (label_index → word name)')
    for split, data in splits.items():
        unique = np.unique(data['y'])[:6]
        words  = [label_map.get(str(l), '*** MISSING ***') for l in unique]
        print(f'  {split:<8}: {list(zip(unique.tolist(), words))}')
        if any('MISSING' in w for w in words):
            print(f'  *** CRITICAL: label_map.json does not cover all labels in {split}. '
                  'Label mismatch detected!')

    # ── 3. Label alignment across splits ─────────────────────────────────────
    print('\n[3] Label Set Alignment  (same class indices across all splits?)')
    train_set = set(splits['train']['y'].tolist()) if 'train' in splits else set()
    ok = True
    for split, data in splits.items():
        if split == 'train':
            continue
        split_set = set(data['y'].tolist())
        extra = split_set - train_set
        missing = train_set - split_set
        if extra:
            print(f'  *** {split}: labels {sorted(extra)} appear in {split} but NOT in train '
                  '→ model never saw these classes!')
            ok = False
        if missing:
            print(f'  ℹ  {split}: labels {sorted(missing)} in train but absent from {split} '
                  '(model cannot be evaluated on them)')
    if ok:
        print('  ✓  All splits share identical label set — no alignment issues.')

    # ── 4. Class balance ─────────────────────────────────────────────────────
    print('\n[4] Class Balance in Train Split')
    if 'train' in splits:
        y_tr  = splits['train']['y']
        counts = np.bincount(y_tr, minlength=NUM_CLASSES)
        print(f'  Samples/class → min={counts.min()}  max={counts.max()}  '
              f'mean={counts.mean():.1f}  std={counts.std():.1f}')
        scarce = np.where(counts < 20)[0]
        if len(scarce):
            words = [label_map.get(str(i), f'cls{i}') for i in scarce]
            print(f'  ⚠  {len(scarce)} class(es) with <20 train samples: '
                  + ', '.join(f'{w}({counts[i]})' for i, w in zip(scarce, words)))
        else:
            print(f'  ✓  All classes have ≥{counts.min()} train samples.')

        fig, ax = plt.subplots(figsize=(14, 3))
        word_labels = [label_map.get(str(i), str(i)) for i in range(NUM_CLASSES)]
        ax.bar(range(NUM_CLASSES), counts, color='steelblue', alpha=0.8)
        ax.axhline(counts.mean(), color='red', linestyle='--', label=f'Mean={counts.mean():.0f}')
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(word_labels, rotation=90, fontsize=8)
        ax.set_ylabel('# samples'); ax.set_title('Train: samples per class')
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'diag_class_balance.png'), dpi=100, bbox_inches='tight')
        plt.close()
        print('  Saved → diag_class_balance.png')

    # ── 5. Zero-frame rate (PRIMARY SUSPECT for random-level accuracy) ────────
    print('\n[5] Zero-Frame Rate  ← PRIMARY SUSPECT if accuracy ≈ random')
    print('    A frame where ALL 150 features = 0.0 means MediaPipe detected nothing.')
    print('    If this is >30%, the data is dominated by noise and the model cannot learn.')
    zero_warning = False
    for split, data in splits.items():
        X = data['X']                                       # (N, 200, 150)
        frame_is_zero   = (np.abs(X).sum(axis=2) == 0)     # (N, 200) bool
        overall_rate    = frame_is_zero.mean()
        per_sample_rate = frame_is_zero.mean(axis=1)        # (N,)
        bad_samples     = (per_sample_rate > 0.5).sum()
        print(f'  {split:<8}: {overall_rate*100:5.1f}% zero frames overall  |  '
              f'{bad_samples}/{len(X)} samples have >50% zero frames')
        if overall_rate > 0.40:
            print(f'  *** CRITICAL ({split}): {overall_rate*100:.0f}% zero-frame rate is too high. '
                  'Detection failure → model will behave randomly. '
                  'Fix: lower min_detection_confidence in extraction, or improve video quality.')
            zero_warning = True
        elif overall_rate > 0.20:
            print(f'  ⚠  ({split}): {overall_rate*100:.0f}% zero-frame rate is elevated. '
                  'Model may struggle.')

    # ── 6. Body-part detection breakdown ─────────────────────────────────────
    print('\n[6] Body-Part Detection Rate  (% of frames with at least one non-zero keypoint)')
    for split, data in splits.items():
        X = data['X']
        pose_det  = (np.abs(X[:, :,   0: 66]).sum(axis=2) > 0).mean()
        lhand_det = (np.abs(X[:, :,  66:108]).sum(axis=2) > 0).mean()
        rhand_det = (np.abs(X[:, :, 108:150]).sum(axis=2) > 0).mean()
        print(f'  {split:<8}: Pose {pose_det*100:5.1f}%  |  '
              f'L-Hand {lhand_det*100:5.1f}%  |  R-Hand {rhand_det*100:5.1f}%')
        if lhand_det < 0.3 and rhand_det < 0.3:
            print(f'  *** Both hands rarely detected in {split}. '
                  'Hand features (84/150 dims) are mostly zeros — this explains random accuracy. '
                  'Model effectively only has pose (66 dims) with very sparse signal.')

    # ── 7. Feature statistics ─────────────────────────────────────────────────
    print('\n[7] Feature Statistics (non-zero values after BdSL normalization)')
    for split, data in splits.items():
        X  = data['X']
        nz = X[X != 0]
        if len(nz) == 0:
            print(f'  {split:<8}: *** ALL VALUES ARE ZERO — critical data corruption!')
            continue
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        print(f'  {split:<8}: min={nz.min():7.3f}  max={nz.max():7.3f}  '
              f'mean={nz.mean():6.3f}  std={nz.std():6.3f}  '
              f'NaN={nan_count}  Inf={inf_count}')
        if nan_count or inf_count:
            print(f'  *** NaN/Inf detected in {split}! This will cause loss=NaN during training.')
        if nz.max() > 50 or nz.min() < -50:
            print(f'  ⚠  Extreme feature values (expected range roughly -5 to 5 after BdSL norm). '
                  'Check normalization.')

    # ── 8. Class separability ─────────────────────────────────────────────────
    print('\n[8] Class Separability  (cosine similarity of per-class mean feature vectors)')
    print('    Value close to 1.0 → classes look identical → model cannot learn distinctions.')
    if 'train' in splits:
        X, y = splits['train']['X'], splits['train']['y']
        class_means = np.stack([
            X[y == c].mean(axis=(0, 1)) if (y == c).sum() > 0
            else np.zeros(FEATURE_DIM)
            for c in range(NUM_CLASSES)
        ])                                                      # (NUM_CLASSES, FEATURE_DIM)
        norms = np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8
        normed = class_means / norms
        sim = normed @ normed.T                                  # (NUM_CLASSES, NUM_CLASSES)
        np.fill_diagonal(sim, np.nan)
        avg_sim = float(np.nanmean(sim))
        max_sim = float(np.nanmax(sim))
        print(f'  Mean inter-class cosine similarity : {avg_sim:.4f}')
        print(f'  Max  inter-class cosine similarity : {max_sim:.4f}')
        if avg_sim > 0.95:
            print('  *** CRITICAL: Classes are nearly indistinguishable in feature space.')
            print('      This is the direct mathematical reason for random-level accuracy.')
            print('      Root cause: high zero-frame rate makes ALL class means look like zero vectors.')
        elif avg_sim > 0.80:
            print('  ⚠  Classes are highly overlapping. Model will struggle.')
        else:
            print('  ✓  Classes show reasonable separation.')

    # ── 9. Baselines ─────────────────────────────────────────────────────────
    print('\n[9] Baseline Accuracy (to interpret training results)')
    for split, data in splits.items():
        y = data['y']
        random_chance  = 1.0 / NUM_CLASSES
        majority_cls   = int(np.bincount(y).argmax())
        majority_acc   = float((y == majority_cls).mean())
        majority_word  = label_map.get(str(majority_cls), f'cls{majority_cls}')
        print(f'  {split:<8}: random={random_chance*100:.2f}%  '
              f'majority class ({majority_word})={majority_acc*100:.2f}%  '
              f'← model must exceed this to show it learned something')

    print('\n' + '=' * 70)
    print('  DIAGNOSTICS COMPLETE')
    if zero_warning:
        print('  *** HIGH ZERO-FRAME RATE DETECTED — fix keypoint extraction first!')
        print('  Suggested fixes:')
        print('    1. Lower min_detection_confidence to 0.3 in keypoint_extraction.py')
        print('    2. Ensure videos are front-facing with good lighting')
        print('    3. Check that the correct video folder (Front) is used')
        print('    4. Consider using pose-only (66-dim) if hands are rarely detected')
    else:
        print('  No critical issues found — data quality looks acceptable.')
    print('=' * 70 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Run diagnostics BEFORE any training ────────────────────────────────
    run_diagnostics()

    # ── Compute STEPS_PER_EPOCH; also check signer_id availability ───────────
    _tmp = np.load(os.path.join(KEYPOINTS_DIR, 'train.npz'))
    N_TRAIN = len(_tmp['y'])
    if 'signer_id' not in _tmp.files:
        print('[WARN] signer_id not found in train.npz.'
              ' Re-run keypoint_extraction.py to enable GRL.')
    del _tmp
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE   # drop_last=True in DataLoader
    TOTAL_STEPS     = MAX_EPOCHS * STEPS_PER_EPOCH

    # ── Build model, optimizer, scheduler ────────────────────────────────────
    model         = BdSLSPOTER().to(DEVICE)
    optimizer     = AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=WEIGHT_DECAY)
    criterion     = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    disc_criterion = nn.CrossEntropyLoss()   # no smoothing for signer discriminator
    scaler        = GradScaler()
    scheduler = OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=TOTAL_STEPS,
        pct_start=0.3,           # FIX: longer warmup (was 0.1) — don't peak LR too early
        anneal_strategy='cos',
        div_factor=10, final_div_factor=1e4,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nParameters  : {total_params:,}  ({total_params/1e6:.3f} M)')
    print(f'Train size  : {N_TRAIN}  |  steps/epoch: {STEPS_PER_EPOCH}')
    print(f'GRL         : USE_GRL={USE_GRL}  λ_adv={LAMBDA_ADV}  n_signers={NUM_SIGNERS}')

    # ── Resume from checkpoint if available ──────────────────────────────────
    best_model_path = os.path.join(OUTPUT_DIR, 'best_bdsl_spoter.pt')
    start_epoch, best_val_acc, patience_counter = load_latest_ckpt(
        model, optimizer, scheduler, scaler
    )

    history      = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_top5']}
    global_step  = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'  BdSL-SPOTER Training  (150-dim Holistic keypoints, D_MODEL={D_MODEL})')
    print('=' * 70)
    print(f'{"Ep":>3} | {"Tr Loss":>8} | {"Tr Acc":>7} | '
          f'{"Val Loss":>8} | {"Val T1":>7} | {"Val T5":>7} | {"CLen":>5} | {"Time":>6}')
    print('-' * 70)

    start_time = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        t0 = time.time()

        train_loader, val_loader, curr_len = get_dataloaders(epoch)

        tr_loss, tr_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, disc_criterion, epoch, global_step, TOTAL_STEPS)

        vm = evaluate(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vm['loss'])
        history['val_acc'].append(vm['top1_acc'])
        history['val_top5'].append(vm['top5_acc'])

        cl_str = str(curr_len) if curr_len else 'full'
        print(f'{epoch+1:>3} | {tr_loss:>8.4f} | {tr_acc*100:>6.2f}% | '
              f'{vm["loss"]:>8.4f} | {vm["top1_acc"]*100:>6.2f}% | '
              f'{vm["top5_acc"]*100:>6.2f}% | {cl_str:>5} | {time.time()-t0:>5.0f}s')

        # ── Save best model ───────────────────────────────────────────────────
        if vm['top1_acc'] > best_val_acc:
            best_val_acc = vm['top1_acc']
            torch.save({'epoch': epoch + 1, 'model_state': model.state_dict(),
                        'val_top1': vm['top1_acc'], 'val_top5': vm['top5_acc']},
                       best_model_path)
            print(f'  ★ New best → {best_val_acc*100:.2f}%  (best_bdsl_spoter.pt)')
            patience_counter = 0
        else:
            patience_counter += 1

        # ── Save epoch checkpoint ─────────────────────────────────────────────
        save_epoch_ckpt(model, optimizer, scheduler, scaler,
                        epoch + 1, best_val_acc, patience_counter)

        if patience_counter >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch+1}.')
            break

    total_min = (time.time() - start_time) / 60
    print('=' * 70)
    print(f'Training complete in {total_min:.1f} min  |  Best val top-1: {best_val_acc*100:.2f}%')

    # ── Training curves ───────────────────────────────────────────────────────
    ep = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, history['train_loss'], 'b-o', markersize=4, label='Training Loss')
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [v * 100 for v in history['val_acc']], 'r-o', markersize=4, label='Val Top-1')
    axes[1].axhline(best_val_acc * 100, color='gray', linestyle='--',
                    label=f'Best {best_val_acc*100:.2f}%')
    axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Training curves saved → training_curves.png')

    # ── Final test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f'\nLoaded best model from epoch {ckpt["epoch"]}  '
          f'(val top-1={ckpt["val_top1"]*100:.2f}%)')

    test_ds     = BdSLDataset(os.path.join(KEYPOINTS_DIR, 'test.npz'), augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    tm       = evaluate(model, test_loader, criterion)
    macro_f1 = f1_score(tm['labels'], tm['preds'], average='macro')

    print('\n' + '=' * 45)
    print('  TEST SET RESULTS')
    print('=' * 45)
    print(f'  Top-1 Accuracy : {tm["top1_acc"]*100:.2f}%')
    print(f'  Top-5 Accuracy : {tm["top5_acc"]*100:.2f}%')
    print(f'  Macro F1       : {macro_f1:.4f}')
    print('=' * 45)

    results = {
        'test_top1':      float(tm['top1_acc']),
        'test_top5':      float(tm['top5_acc']),
        'macro_f1':       float(macro_f1),
        'best_val_epoch': int(ckpt['epoch']),
        'best_val_top1':  float(ckpt['val_top1']),
        'd_model':        D_MODEL,
        'n_heads':        N_HEADS,
        'n_layers':       N_LAYERS,
        'feature_dim':    FEATURE_DIM,
        'num_landmarks':  NUM_LANDMARKS,
        'num_classes':    NUM_CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved → test_results.json')

    # ── Load label map for human-readable class names ──────────────────────
    with open(os.path.join(KEYPOINTS_DIR, 'label_map.json')) as _f:
        _label_map = json.load(_f)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm            = confusion_matrix(tm['labels'], tm['preds'])
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    word_labels   = [_label_map.get(str(i), str(i)) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(word_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(word_labels, fontsize=7)
    ax.set_xlabel('Predicted Word', fontsize=11)
    ax.set_ylabel('True Word', fontsize=11)
    ax.set_title(
        f'BdSL-SPOTER Confusion Matrix  ({NUM_CLASSES} classes)\n'
        f'Test Top-1: {tm["top1_acc"]*100:.2f}%  |  Macro-F1: {macro_f1:.3f}',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('Confusion matrix saved → confusion_matrix.png')

    # ── Per-class accuracy table ───────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  PER-CLASS ACCURACY  (sorted worst → best)')
    print('=' * 60)
    print(f'  {"#":>3} {"Word":<8} {"Acc":>7}  {"Support":>8}  {"Top Confusion"}')
    print(f'  {"-" * 56}')
    support        = cm.sum(axis=1)
    sorted_idx     = np.argsort(per_class_acc)  # worst first
    for rank, ci in enumerate(sorted_idx):
        word    = word_labels[ci]
        acc_pct = per_class_acc[ci] * 100
        sup     = int(support[ci])
        # top predicted class for samples of this true class (excluding correct)
        row     = cm[ci].copy(); row[ci] = 0
        top_confused_idx  = int(row.argmax()) if row.sum() > 0 else -1
        top_confused_word = word_labels[top_confused_idx] if top_confused_idx >= 0 else 'N/A'
        top_confused_n    = int(row.max()) if row.sum() > 0 else 0
        marker = ' ***' if acc_pct < 5 else (' ⚠' if acc_pct < 50 else '  ✓')
        print(f'  {rank+1:>3} {word:<8} {acc_pct:>6.1f}%  {sup:>8}    '
              f'confused with {top_confused_word} ({top_confused_n}×){marker}')

    perfect  = (per_class_acc == 1.0).sum()
    good     = ((per_class_acc >= 0.95) & (per_class_acc < 1.0)).sum()
    moderate = ((per_class_acc >= 0.50) & (per_class_acc < 0.95)).sum()
    poor     = (per_class_acc < 0.50).sum()
    print(f'\n  Perfect (100%)    : {perfect} classes')
    print(f'  Good    (≥95%)    : {good} classes')
    print(f'  Moderate (50-94%) : {moderate} classes')
    print(f'  Poor     (<50%)   : {poor} classes')

    # ── Top confusion pairs ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  TOP-10 CONFUSION PAIRS  (true → predicted, n times)')
    print('=' * 60)
    cm_nodiag = cm.copy(); np.fill_diagonal(cm_nodiag, 0)
    flat_idx  = np.argsort(cm_nodiag.ravel())[::-1][:10]
    for k, fi in enumerate(flat_idx):
        ti, pi = divmod(fi, NUM_CLASSES)
        n = cm_nodiag[ti, pi]
        if n == 0: break
        print(f'  {k+1:>2}. {word_labels[ti]:>8} → {word_labels[pi]:<8}  ({n}×)')

    # ── Automatic diagnosis of results ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('  AUTOMATIC RESULT DIAGNOSIS')
    print('=' * 60)
    random_chance = 1.0 / NUM_CLASSES
    if tm['top1_acc'] <= random_chance * 1.5:
        print('  *** RESULT: Near-random accuracy. Model learned NOTHING.')
        print('  Possible causes (in order of likelihood):')
        print('   1. High zero-frame rate in keypoints (run diagnostics section above)')
        print('   2. MediaPipe hand detection failure (hands are 84/150 dims)')
        print('   3. Label mismatch between train and val/test splits')
        print('   4. Feature normalization bug (check bdsl_normalize output range)')
        print('   5. Learning rate too high/low (check training loss — stuck at ln(30)≈3.40?)')
        print('  Recommended fix: re-run keypoint_extraction.py with lower')
        print('  min_detection_confidence=0.3 and verify hand detection rate.')
    elif tm['top1_acc'] < 0.50:
        print(f'  ⚠  Low accuracy ({tm["top1_acc"]*100:.1f}%). Partial learning occurred.')
        print('  Check per-class table above for which words are hardest.')
    else:
        print(f'  ✓  Reasonable accuracy ({tm["top1_acc"]*100:.1f}%). Model is learning.')

    if abs(tm['loss'] - np.log(NUM_CLASSES)) < 0.05:
        print(f'  *** Loss ({tm["loss"]:.4f}) ≈ ln({NUM_CLASSES})={np.log(NUM_CLASSES):.4f}: '
              'model is outputting uniform distribution — training completely failed.')
    print('=' * 60)

    # ── Inference speed benchmark ─────────────────────────────────────────────
    model.eval()
    dummy = torch.randn(1, SEQ_LEN, FEATURE_DIM).to(DEVICE)
    for _ in range(20):
        with torch.no_grad(): model(dummy)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(300):
        with torch.no_grad(): model(dummy)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    fps       = 300 / (time.time() - t0)
    model_mb  = os.path.getsize(best_model_path) / 1e6
    total_par = sum(p.numel() for p in model.parameters())

    print('\n' + '=' * 45)
    print('  COMPUTATIONAL EFFICIENCY')
    print('=' * 45)
    print(f'  Parameters  : {total_par/1e6:.3f} M')
    print(f'  Model size  : {model_mb:.1f} MB')
    print(f'  Inference   : {fps:.0f} FPS  ({DEVICE})')
    print(f'  Train time  : {total_min:.1f} min')
    print('=' * 45)
    print(f'\nAll outputs → {OUTPUT_DIR}')
