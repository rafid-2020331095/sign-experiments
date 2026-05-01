"""
KD Signer Variance — Cross-Fold Training with Knowledge Distillation
=====================================================================
Extends signer-variance.py by injecting VideoMAE teacher logits
(from phase1_extract_teacher_logits.py) via KL-Divergence KD loss.

Loss equation per batch:
  loss = (0.5 * l_ce)
       + (ALPHA_KD  * l_kd)    ← KL-div against softened teacher probs
       + (LAMBDA_ADV * l_adv)  ← GRL signer discriminator
       + (ALPHA_SUPCON * l_sc) ← supervised contrastive

Teacher logit alignment:
  phase1 saves  teacher_logits.pt  → { video_stem: Tensor(NUM_CLASSES,) }
  build_master_arrays_kd() loads this dict and aligns logits to the
  keypoint master pool.  If the npz contains a 'video_stem' array the
  alignment is stem-based (exact); otherwise positional (same scan order).
  Per-fold train_teacher_logits.npy is saved alongside the fold npz files
  so BdSLDataset can simply do  self.teacher_logits[idx].

Fold definitions  (13 actual signers: {1,2,3,4,5,6,8,9,10,11,12,13,15}):
  Fold 1  train=[1,2,3,5,6,10,12,13]  val=[4,8,9]    test=[11,15]
  Fold 2  train=[1,2,4,5,8,9,11,15]  val=[3,10,12]  test=[6,13]

Outputs per fold:
  best_kd_fold{N}.pt | training_curves_kd_fold{N}.png
Final:
  kd_fold_results.json | confusion_matrix_kd_fold{N}.png
"""

import os, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
KEYPOINTS_DIR      = '/kaggle/input/datasets/rafidadib/geo-feature-keypoint/keypoints'
TEACHER_LOGITS_PT  = '/kaggle/working/teacher_logits.pt'   # output of phase1_extract_teacher_logits.py
OUTPUT_DIR         = '/kaggle/working'
SPLITS_DIR         = os.path.join(OUTPUT_DIR, 'kd_splits')
os.makedirs(SPLITS_DIR, exist_ok=True)

with open(os.path.join(KEYPOINTS_DIR, 'config.json')) as f:
    _cfg = json.load(f)

NUM_CLASSES   = _cfg['num_classes']
SEQ_LEN       = _cfg['seq_len']
FEATURE_DIM   = _cfg['feature_dim']
NUM_LANDMARKS = _cfg['num_landmarks']
NUM_SIGNERS   = 15   # signer_id max=15, zero-indexed → max idx=14, discriminator head=15

D_MODEL = 128; N_HEADS = 8; N_LAYERS = 4; D_FF = 512; DROPOUT = 0.20
LABEL_SMOOTH = 0.05; WEIGHT_DECAY = 5e-4; MAX_EPOCHS = 80; PATIENCE = 15
BATCH_SIZE = 64; MAX_LR = 3e-4; SEED = 42

USE_GRL = True; LAMBDA_ADV = 0.1; DISC_HIDDEN = 256
USE_SUPCON = True; ALPHA_SUPCON = 0.15; SUPCON_TEMP = 0.10; PROJ_DIM = 64

KD_T      = 3.0    # distillation temperature
ALPHA_KD  = 0.5    # weight for KD loss; CE is scaled by (1 - ALPHA_KD) = 0.5

# ── Fold definitions ───────────────────────────────────────────────────────────
# Actual signer pool: [1,2,3,4,5,6,8,9,10,11,12,13,15]  (13 signers, IDs 7,14 absent)
FOLDS = [
    # Fold 1 — val=[4,8,9]  test=[11,15]  train=remainder
    {'name': 'fold1',
     'train': [1,2,3,5,6,10,12,13],
     'val':   [4,8,9],
     'test':  [11,15]},
    # Fold 2 — val=[3,10,12]  test=[6,13]  train=remainder
    {'name': 'fold2',
     'train': [1,2,4,5,8,9,11,15],
     'val':   [3,10,12],
     'test':  [6,13]},
]

assert D_MODEL % N_HEADS == 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}  | NUM_CLASSES={NUM_CLASSES}  FEATURE_DIM={FEATURE_DIM}')
if torch.cuda.is_available():
    print(f'GPU   : {torch.cuda.get_device_name(0)}  '
          f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')


# ── Dataset scanner / diagnostic ─────────────────────────────────────────────

def scan_dataset_signers():
    """
    Scan EVERY .npz file in KEYPOINTS_DIR, report per-file signer IDs,
    and print the combined unique signer list.
    Returns the sorted list of all signer IDs found.
    """
    import glob
    npz_files = sorted(glob.glob(os.path.join(KEYPOINTS_DIR, '*.npz')))
    print(f'\n{"+"*70}')
    print(f'  DATASET SCAN  →  {KEYPOINTS_DIR}')
    print(f'  Found {len(npz_files)} npz file(s)')
    print(f'{"+"*70}')
    all_signers = set()
    for fpath in npz_files:
        fname = os.path.basename(fpath)
        d     = np.load(fpath, allow_pickle=True)
        n_samples = len(d['y']) if 'y' in d.files else 0
        if 'signer_id' in d.files:
            sids = sorted(set(d['signer_id'].astype(int).tolist()))
            all_signers.update(sids)
            print(f'  {fname:<20}  samples={n_samples:6d}  signers({len(sids):2d})={sids}')
        else:
            print(f'  {fname:<20}  samples={n_samples:6d}  [NO signer_id field]')
    print(f'{"+"*70}')
    print(f'  ALL SIGNERS COMBINED ({len(all_signers)}) : {sorted(all_signers)}')
    print(f'{"+"*70}\n')
    return sorted(all_signers)


# ── Build master pool + align teacher logits ───────────────────────────────────

def build_master_arrays_kd():
    """
    Returns all_X, all_y, all_sid, all_teacher_logits  (all aligned by row index).

    Alignment strategy — stem reconstruction from encoded fields:
      Filename format: W{word:03d}S{signer:02d}F_{instance:02d}
        W001  → word class 1  (y[i] = 0-indexed, so word_1idx = y[i]+1)
        S01   → signer 1      (signer_id is already 1-indexed)
        _01   → instance 1    (tracked per (word, signer) pair in scan order)

      This reconstructs the exact key used in teacher_logits.pt without
      needing a stored video_stem field in the npz.
    """
    import re
    _STEM_RE = re.compile(r'W(\d+)S(\d+)[A-Za-z]_(\d+)')

    print(f'\nLoading teacher logits from: {TEACHER_LOGITS_PT}')
    teacher_pt  = torch.load(TEACHER_LOGITS_PT, map_location='cpu')
    teacher_map = teacher_pt['logits']       # { stem (str): Tensor(NC,) }
    t_nc        = next(iter(teacher_map.values())).shape[0]
    print(f'  Teacher classes : {t_nc}  |  vocab size : {len(teacher_map)}')
    if t_nc != NUM_CLASSES:
        raise ValueError(
            f'Teacher NUM_CLASSES={t_nc} != keypoint NUM_CLASSES={NUM_CLASSES}. '
            'Make sure both models were trained on the same word list.')

    # Pre-parse teacher dict into exact lookup
    # key: (word_1idx, signer_1idx, instance_1idx)  →  numpy float32 array
    teacher_lookup = {}
    unparsed = 0
    for stem, logit in teacher_map.items():
        m = _STEM_RE.search(stem)
        if not m:
            unparsed += 1; continue
        teacher_lookup[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = \
            logit.numpy().astype(np.float32)
    print(f'  Parsed lookup   : {len(teacher_lookup)} entries'
          f'{f"  ({unparsed} unparseable stems skipped)" if unparsed else ""}')

    all_X, all_y, all_sid, all_tl = [], [], [], []

    for subset in ['train', 'val', 'test']:
        path = os.path.join(KEYPOINTS_DIR, f'{subset}.npz')
        if not os.path.exists(path):
            print(f'  [WARN] {path} not found'); continue
        d   = np.load(path, allow_pickle=True)
        X   = d['X'].astype(np.float32)
        y   = d['y'].astype(np.int64)
        sid = d['signer_id'].astype(np.int64) if 'signer_id' in d.files \
              else np.zeros(len(y), dtype=np.int64)

        # ── Exact stem-based alignment ────────────────────────────────────────
        # Track per-(word, signer) instance counter to reconstruct _01, _02 ...
        instance_ctr = {}
        tl      = np.zeros((len(X), t_nc), dtype=np.float32)
        missing = 0
        for i in range(len(X)):
            word_1   = int(y[i]) + 1      # 0-indexed label → 1-indexed W number
            signer_1 = int(sid[i])         # already 1-indexed
            pair     = (word_1, signer_1)
            instance_ctr[pair] = instance_ctr.get(pair, 0) + 1
            key = (word_1, signer_1, instance_ctr[pair])
            if key in teacher_lookup:
                tl[i] = teacher_lookup[key]
            else:
                missing += 1

        status = f'✓ all {len(X)} matched' if missing == 0 \
                 else f'[WARN] {missing}/{len(X)} unmatched (zero-filled)'
        print(f'  {subset:5s}  samples={len(X):5d}  teacher align: {status}')

        all_X.append(X); all_y.append(y); all_sid.append(sid); all_tl.append(tl)

    all_X   = np.concatenate(all_X)
    all_y   = np.concatenate(all_y)
    all_sid = np.concatenate(all_sid)
    all_tl  = np.concatenate(all_tl)
    print(f'Master pool: X={all_X.shape}  teacher_logits={all_tl.shape}  '
          f'signers={sorted(set(all_sid.tolist()))}')
    return all_X, all_y, all_sid, all_tl


def save_fold_npz_kd(fold: dict, all_X, all_y, all_sid, all_tl):
    """Write train/val/test.npz + train_teacher_logits.npy for one fold."""
    fold_dir = os.path.join(SPLITS_DIR, fold['name'])
    os.makedirs(fold_dir, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        mask = np.isin(all_sid, fold[subset])
        np.savez(os.path.join(fold_dir, f'{subset}.npz'),
                 X=all_X[mask], y=all_y[mask], signer_id=all_sid[mask])
        print(f'  {subset:5s} → {mask.sum():5d} samples  '
              f'signers={sorted(set(all_sid[mask].tolist()))}')
        if subset == 'train':
            tl_path = os.path.join(fold_dir, 'train_teacher_logits.npy')
            np.save(tl_path, all_tl[mask].astype(np.float32))
            print(f'         teacher logits → {os.path.basename(tl_path)}  '
                  f'shape={all_tl[mask].shape}')


# ── BlazePose LR pairs ─────────────────────────────────────────────────────────
BLAZE_LR_PAIRS = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
                   (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]


# ── Dataset ───────────────────────────────────────────────────────────────────

class BdSLDataset(Dataset):
    """
    Returns (X, y, signer_id)               when teacher_logits_path is None
    Returns (X, y, signer_id, t_logit)      when teacher_logits_path is given
    """
    def __init__(self, npz_path, augment=False, teacher_logits_path=None):
        d = np.load(npz_path)
        self.X         = d['X'].astype(np.float32)
        self.y         = d['y'].astype(np.int64)
        self.signer_id = (d['signer_id'].astype(np.int64) - 1) \
                         if 'signer_id' in d.files \
                         else np.zeros(len(d['y']), dtype=np.int64)
        self.augment   = augment

        if teacher_logits_path is not None:
            self.teacher_logits = np.load(teacher_logits_path).astype(np.float32)
            assert len(self.teacher_logits) == len(self.y), \
                f'teacher_logits length {len(self.teacher_logits)} != dataset length {len(self.y)}'
        else:
            self.teacher_logits = None

        print(f'  {os.path.basename(npz_path)}: X={self.X.shape}  '
              f'classes={len(np.unique(self.y))}  '
              f'signers={sorted(set((self.signer_id+1).tolist()))}'
              f'{"  [+teacher logits]" if self.teacher_logits is not None else ""}')

    def __len__(self): return len(self.y)

    def temporal_dropout(self, seq, p=0.15):
        T = len(seq); mask = np.random.rand(T) > p; kept = seq[mask]
        if len(kept) < 2: return seq
        oi = np.linspace(0, len(kept)-1, len(kept)); ni = np.linspace(0, len(kept)-1, T)
        out = np.zeros_like(seq)
        for d in range(seq.shape[1]): out[:, d] = np.interp(ni, oi, kept[:, d])
        return out

    def coordinate_noise(self, seq, sigma=0.004):
        noise = np.zeros_like(seq)
        noise[:, :150] = np.random.normal(0, sigma, (len(seq), 150)).astype(np.float32)
        return seq + noise

    def landmark_dropout(self, seq, p=0.10):
        seq = seq.copy()
        for i in np.where(np.random.rand(75) < p)[0]:
            seq[:, i*2] = 0.0; seq[:, i*2+1] = 0.0
        return seq

    def temporal_scale(self, seq):
        T = seq.shape[0]; new_T = max(2, int(T * np.random.uniform(0.8, 1.2)))
        oi = np.linspace(0, T-1, T); ni = np.linspace(0, T-1, new_T)
        sc = np.zeros((new_T, seq.shape[1]), dtype=np.float32)
        for d in range(seq.shape[1]): sc[:, d] = np.interp(ni, oi, seq[:, d])
        fi = np.linspace(0, new_T-1, T); out = np.zeros_like(seq)
        for d in range(seq.shape[1]): out[:, d] = np.interp(fi, np.arange(new_T), sc[:, d])
        return out

    def horizontal_flip(self, seq):
        seq = seq.copy(); seq[:, 0:150:2] = -seq[:, 0:150:2]
        for a, b in BLAZE_LR_PAIRS:
            seq[:, [a*2, a*2+1]], seq[:, [b*2, b*2+1]] = \
                seq[:, [b*2, b*2+1]].copy(), seq[:, [a*2, a*2+1]].copy()
        lb = seq[:, 66:108].copy(); rb = seq[:, 108:150].copy()
        seq[:, 66:108] = rb; seq[:, 108:150] = lb
        if seq.shape[1] > 150:
            for ls, le, rs, re in [(150,170,170,190),(190,200,200,210),(210,215,215,220)]:
                t = seq[:, ls:le].copy(); seq[:, ls:le] = seq[:, rs:re]; seq[:, rs:re] = t
            if seq.shape[1] > 222: seq[:, [221, 222]] = seq[:, [222, 221]]
        return seq

    def augment_seq(self, seq):
        if np.random.rand() < 0.60: seq = self.temporal_dropout(seq)
        if np.random.rand() < 0.60: seq = self.coordinate_noise(seq)
        if np.random.rand() < 0.50: seq = self.horizontal_flip(seq)
        if np.random.rand() < 0.50: seq = self.landmark_dropout(seq)
        if np.random.rand() < 0.40: seq = self.temporal_scale(seq)
        return seq

    def __getitem__(self, idx):
        seq = self.X[idx].copy()
        if self.augment: seq = self.augment_seq(seq)
        x   = torch.tensor(seq,                   dtype=torch.float32)
        y   = torch.tensor(self.y[idx],            dtype=torch.long)
        sid = torch.tensor(self.signer_id[idx],    dtype=torch.long)
        if self.teacher_logits is not None:
            t   = torch.tensor(self.teacher_logits[idx], dtype=torch.float32)
            return x, y, sid, t
        return x, y, sid


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

def supervised_contrastive_loss(emb, labels, temp=SUPCON_TEMP):
    N = emb.shape[0]
    sim = torch.mm(emb, emb.T) / temp
    self_mask = torch.eye(N, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(self_mask, float('-inf'))
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~self_mask
    n_pos = pos_mask.sum(1).float(); valid = n_pos > 0
    if not valid.any(): return emb.sum() * 0.0
    log_prob = F.log_softmax(sim, dim=1)
    loss = -(log_prob * pos_mask.float()).sum(1)
    return (loss[valid] / n_pos[valid]).mean()


# ── GRL ───────────────────────────────────────────────────────────────────────

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.clone()
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None

class GRL(nn.Module):
    def __init__(self): super().__init__(); self.lam = 0.0
    def set_lambda(self, lam): self.lam = lam
    def forward(self, x): return GRLFunction.apply(x, self.lam)

class SignerDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden, n_signers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, n_signers))
    def forward(self, x): return self.net(x)

def grl_lambda_schedule(step, total_steps):
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ── Model ─────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=D_MODEL, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, proj_dim))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.encoding, std=0.02)
    def forward(self, x): return x + self.encoding

class BdSLSPOTER(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, seq_len=SEQ_LEN,
                 feature_dim=FEATURE_DIM, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
                 n_signers=NUM_SIGNERS, disc_hidden=DISC_HIDDEN):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc    = LearnablePositionalEncoding(seq_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        h1, h2, h3 = d_model*2, d_model, num_classes*2
        self.classifier = nn.Sequential(
            nn.Linear(d_model, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),      nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, h3),      nn.LayerNorm(h3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h3, num_classes))
        self.grl       = GRL()
        self.disc      = SignerDiscriminator(d_model, disc_hidden, n_signers)
        self.proj_head = ProjectionHead(d_model, PROJ_DIM) if USE_SUPCON else None
        self._init_weights()

    def set_grl_lambda(self, lam): self.grl.set_lambda(lam)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x); x = self.input_proj(x)
        x = self.pos_enc(x);    x = self.transformer(x)
        feat = x.mean(dim=1)
        sign_out = self.classifier(feat)
        if self.training:
            signer_out = self.disc(self.grl(feat)) if USE_GRL    else None
            proj_out   = self.proj_head(feat)       if USE_SUPCON else None
        else:
            signer_out = proj_out = None
        return sign_out, signer_out, proj_out


# ── Training helpers ──────────────────────────────────────────────────────────

def augment_batch(X_np, ds):
    return torch.from_numpy(np.stack([ds.augment_seq(X_np[i]) for i in range(len(X_np))]))

_kl_loss = nn.KLDivLoss(reduction='batchmean')

def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    criterion, disc_criterion, epoch, global_step, total_steps, ds):
    model.train()
    total_loss = correct = total = 0
    bar = tqdm(loader, desc=f'  Ep{epoch+1:02d}', leave=False)

    # loader yields 4-tuples (X, y, sids, t_logits) from training BdSLDataset
    for X, y, sids, t_logits in bar:
        X_np = X.numpy().copy()
        X        = X.to(DEVICE, non_blocking=True)
        y        = y.to(DEVICE, non_blocking=True)
        sids     = sids.to(DEVICE, non_blocking=True)
        t_logits = t_logits.to(DEVICE, non_blocking=True)

        lam = grl_lambda_schedule(global_step, total_steps)
        model.set_grl_lambda(lam)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            sl, dl, pl = model(X)

            # ── CE loss (scaled down to balance with KD) ───────────────────
            l_ce = criterion(sl, y)

            # ── KD loss: only on samples with a valid teacher logit ────────
            # Unmatched samples have all-zero logits (stored as fallback).
            # Applying softmax to all-zeros gives uniform dist → wrong signal.
            # We detect them and exclude from KD entirely.
            matched = t_logits.abs().sum(dim=1) > 0   # [B] bool mask
            if matched.any():
                student_log_probs = F.log_softmax(sl[matched] / KD_T, dim=1)
                teacher_probs     = F.softmax(t_logits[matched] / KD_T, dim=1)
                l_kd = _kl_loss(student_log_probs, teacher_probs) * (KD_T * KD_T)
            else:
                l_kd = torch.tensor(0.0, device=DEVICE)

            # ── GRL adversarial loss ───────────────────────────────────────
            l_adv = disc_criterion(dl, sids) if (USE_GRL and dl is not None) \
                    else torch.tensor(0.0, device=DEVICE)

            # ── Supervised Contrastive loss ────────────────────────────────
            l_sc = torch.tensor(0.0, device=DEVICE)
            if USE_SUPCON and pl is not None:
                Xa = augment_batch(X_np, ds).to(DEVICE, non_blocking=True)
                _, _, pla = model(Xa)
                if pla is not None:
                    l_sc = supervised_contrastive_loss(
                        torch.cat([pl, pla]), torch.cat([y, y]))

            # ── Combined loss ──────────────────────────────────────────────
            # When no matched samples in batch, ALPHA_KD weight falls back
            # fully to CE (equivalent to vanilla training for that batch).
            kd_weight = ALPHA_KD * matched.float().mean()   # scales by match rate
            loss = ((1.0 - kd_weight) * l_ce
                    + kd_weight        * l_kd
                    + LAMBDA_ADV       * l_adv
                    + ALPHA_SUPCON     * l_sc)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        global_step += 1

        total_loss += l_ce.item() * X.size(0)
        correct    += (sl.argmax(1) == y).sum().item()
        total      += X.size(0)
        bar.set_postfix(ce=f'{l_ce.item():.3f}', kd=f'{l_kd.item():.3f}',
                        sc=f'{l_sc.item():.3f}', acc=f'{100.*correct/total:.1f}%')
    return total_loss / total, correct / total, global_step


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Val/test loaders return 3-tuples (X, y, signer_id) — no teacher logits."""
    model.eval()
    total_loss = correct = correct5 = total = 0
    preds_all, labels_all = [], []
    for X, y, _ in loader:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast():
            sl, _, _ = model(X); loss = criterion(sl, y)
        top5 = sl.topk(min(5, NUM_CLASSES), dim=1).indices
        total_loss += loss.item() * X.size(0)
        correct    += (sl.argmax(1) == y).sum().item()
        correct5   += (top5 == y.unsqueeze(1)).any(1).sum().item()
        total      += X.size(0)
        preds_all.append(sl.argmax(1).detach()); labels_all.append(y.detach())
    if total == 0:
        return {'loss': float('inf'), 'top1_acc': 0.0, 'top5_acc': 0.0,
                'preds': np.array([], dtype=np.int64),
                'labels': np.array([], dtype=np.int64)}
    return {'loss': total_loss/total, 'top1_acc': correct/total,
            'top5_acc': correct5/total,
            'preds': torch.cat(preds_all).cpu().numpy(),
            'labels': torch.cat(labels_all).cpu().numpy()}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(metrics, title, word_labels):
    print(f'\n{"="*55}')
    print(f'  {title}')
    print(f'{"="*55}')
    print(f'  Top-1   : {metrics["top1_acc"]*100:.2f}%')
    print(f'  Top-5   : {metrics["top5_acc"]*100:.2f}%')
    print(f'  Macro F1: {metrics["macro_f1"]*100:.2f}%')
    cm_mat        = confusion_matrix(metrics['labels'], metrics['preds'])
    per_class_acc = cm_mat.diagonal() / (cm_mat.sum(axis=1) + 1e-8)
    print(f'  Perfect (100%): {(per_class_acc==1.0).sum()}  |  '
          f'Poor (<50%): {(per_class_acc<0.50).sum()}')
    print(f'{"="*55}')
    return cm_mat

def save_cm(cm_mat, word_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_mat, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(len(word_labels))); ax.set_xticklabels(word_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(word_labels))); ax.set_yticklabels(word_labels, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title, fontsize=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()
    print(f'  CM → {os.path.basename(out_path)}')


# ── Main: OUTER FOLD LOOP ─────────────────────────────────────────────────────

if __name__ == '__main__':

    with open(os.path.join(KEYPOINTS_DIR, 'label_map.json')) as f:
        _label_map = json.load(f)
    word_labels = [_label_map.get(str(i), str(i)) for i in range(NUM_CLASSES)]

    # ── Scan dataset: list all signers in every npz file ─────────────────────
    actual_signers = scan_dataset_signers()

    # ── Build master pool + align teacher logits ─────────────────────────────
    print('\n' + '='*70)
    print('  Building master pool  (train.npz + val.npz + test.npz merged)')
    print('='*70)
    all_X, all_y, all_sid, all_tl = build_master_arrays_kd()

    fold_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # OUTER FOLD LOOP
    # ══════════════════════════════════════════════════════════════════════════
    for fold_idx, fold in enumerate(FOLDS):
        fold_name = fold['name']
        print(f'\n{"="*70}')
        print(f'  FOLD {fold_idx+1}/{len(FOLDS)}  [{fold_name.upper()}]  (KD variant)')
        print(f'  train signers : {fold["train"]}')
        print(f'  val   signers : {fold["val"]}')
        print(f'  test  signers : {fold["test"]}')
        print(f'{"="*70}')

        # ── Save fold npz + teacher logits .npy ───────────────────────────────
        print(f'\nCreating fold data for {fold_name}:')
        save_fold_npz_kd(fold, all_X, all_y, all_sid, all_tl)

        fold_dir  = os.path.join(SPLITS_DIR, fold_name)
        best_path = os.path.join(OUTPUT_DIR, f'best_kd_{fold_name}.pt')
        tl_path   = os.path.join(fold_dir, 'train_teacher_logits.npy')

        # ── FATAL MISTAKE PREVENTION: fresh model + optimizer each fold ───────
        model     = BdSLSPOTER().to(DEVICE)
        n_train   = len(np.load(os.path.join(fold_dir, 'train.npz'))['y'])
        spe       = max(1, n_train // BATCH_SIZE)
        tot_steps = MAX_EPOCHS * spe

        optimizer  = AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=WEIGHT_DECAY)
        criterion  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        disc_crit  = nn.CrossEntropyLoss()
        scaler     = GradScaler()
        scheduler  = OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=tot_steps,
                                 pct_start=0.3, anneal_strategy='cos',
                                 div_factor=10, final_div_factor=1e4)

        # ── Create DataLoaders for THIS fold ──────────────────────────────────
        print(f'\nDatasets for {fold_name}:')
        train_ds = BdSLDataset(os.path.join(fold_dir, 'train.npz'),
                                augment=True, teacher_logits_path=tl_path)
        val_ds   = BdSLDataset(os.path.join(fold_dir, 'val.npz'),   augment=False)
        test_ds  = BdSLDataset(os.path.join(fold_dir, 'test.npz'),  augment=False)

        labels = train_ds.y
        cc     = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
        sw     = 1.0 / np.maximum(cc[labels], 1)
        train_ldr = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            sampler=WeightedRandomSampler(torch.tensor(sw, dtype=torch.float64),
                                          len(train_ds), replacement=True),
            num_workers=2, pin_memory=True, drop_last=True)
        val_ldr  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)
        test_ldr = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)

        print(f'\nParams: {sum(p.numel() for p in model.parameters()):,}  '
              f'train={n_train}  steps/ep={spe}')
        print(f'KD: T={KD_T}  ALPHA_KD={ALPHA_KD}  '
              f'CE_weight={1-ALPHA_KD}  KD_weight={ALPHA_KD}')

        # ══════════════════════════════════════════════════════════════════════
        # INNER TRAINING LOOP
        # ══════════════════════════════════════════════════════════════════════
        best_val_acc = 0.0
        patience_ctr = 0
        history      = {'tl': [], 'va': [], 'v5': []}
        gs           = 0
        t0           = time.time()

        print(f'\n{"Ep":>4} | {"T-Loss":>9} | {"T-Acc":>8} | {"V-Loss":>8} | '
              f'{"V-Top1":>8} | {"V-Top5":>8} | {"Time":>5}')
        print('-' * 70)

        val_empty = len(val_ds) == 0
        if val_empty:
            print('  [WARN] val set is empty — early stopping will use train accuracy')

        for epoch in range(MAX_EPOCHS):
            es = time.time()
            tl, ta, gs = train_one_epoch(
                model, train_ldr, optimizer, scheduler, scaler,
                criterion, disc_crit, epoch, gs, tot_steps, train_ds)
            vm = evaluate(model, val_ldr, criterion)
            et = time.time() - es

            track_acc = ta if val_empty else vm['top1_acc']

            history['tl'].append(tl)
            history['va'].append(track_acc)
            history['v5'].append(vm['top5_acc'] if not val_empty else ta)

            v_loss_str = '     N/A' if val_empty else f'{vm["loss"]:>8.4f}'
            v_top1_str = f'{ta*100:>7.2f}%*' if val_empty else f'{vm["top1_acc"]*100:>7.2f}%'
            v_top5_str = '      N/A' if val_empty else f'{vm["top5_acc"]*100:>7.2f}%'
            print(f'{epoch+1:>4} | {tl:>9.4f} | {ta*100:>7.2f}% | '
                  f'{v_loss_str} | {v_top1_str} | {v_top5_str} | {et:>4.0f}s')
            if val_empty: print('         (* train acc used — no val signers in pool)')

            if track_acc > best_val_acc:
                best_val_acc = track_acc
                torch.save({'model_state': model.state_dict(),
                            'epoch':       epoch + 1,
                            'val_top1':    best_val_acc,
                            'val_top5':    vm['top5_acc'] if not val_empty else ta,
                            'fold':        fold_name,
                            'feature_dim': FEATURE_DIM,
                            'num_classes': NUM_CLASSES,
                            'kd_T':        KD_T,
                            'alpha_kd':    ALPHA_KD}, best_path)
                flag = '(train acc)' if val_empty else ''
                print(f'  ★ New best → {best_val_acc*100:.2f}%  (best_kd_{fold_name}.pt) {flag}')
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= PATIENCE:
                print(f'\nEarly stopping at epoch {epoch+1}.')
                break

        fold_min = (time.time() - t0) / 60
        print(f'{"="*70}')
        print(f'{fold_name} done in {fold_min:.1f} min  |  Best val: {best_val_acc*100:.2f}%')

        eps = list(range(1, len(history['tl'])+1))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(eps, history['tl'], 'b-o', ms=3)
        axes[0].set_title(f'{fold_name} KD Loss'); axes[0].set_xlabel('Epoch'); axes[0].grid(alpha=0.3)
        axes[1].plot(eps, [v*100 for v in history['va']], 'r-o', ms=3, label='Val Top-1')
        axes[1].axhline(best_val_acc*100, color='gray', ls='--',
                        label=f'Best {best_val_acc*100:.1f}%')
        axes[1].set_title(f'{fold_name} Val Acc'); axes[1].legend()
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('%'); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'training_curves_kd_{fold_name}.png'), dpi=120)
        plt.close()

        # ── FOLD CONCLUSION ────────────────────────────────────────────────────
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        test_m = evaluate(model, test_ldr, criterion)
        test_m['macro_f1'] = float(f1_score(test_m['labels'], test_m['preds'], average='macro'))

        print(f'\nFold {fold_idx+1} [{fold_name}] Test : '
              f'Top-1={test_m["top1_acc"]*100:.2f}%  '
              f'Top-5={test_m["top5_acc"]*100:.2f}%  '
              f'F1={test_m["macro_f1"]*100:.2f}%')

        cm_f = confusion_matrix(test_m['labels'], test_m['preds'])
        save_cm(cm_f, word_labels,
                f'{fold_name} KD Test  top-1={test_m["top1_acc"]*100:.1f}%  '
                f'signers={fold["test"]}',
                os.path.join(OUTPUT_DIR, f'confusion_matrix_kd_{fold_name}.png'))

        fold_results.append({
            'fold':          fold_name,
            'test_signers':  fold['test'],
            'best_val_epoch': int(ckpt['epoch']),
            'best_val_top1':  float(best_val_acc),
            'test_top1':      float(test_m['top1_acc']),
            'test_top5':      float(test_m['top5_acc']),
            'test_macro_f1':  float(test_m['macro_f1']),
        })

    # ══════════════════════════════════════════════════════════════════════════
    # CROSS-FOLD SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    accs = [r['test_top1'] for r in fold_results]
    cross_subject_acc = float(np.mean(accs))
    cross_subject_std = float(np.std(accs))

    print('\n' + '='*70)
    print('  CROSS-FOLD KD RESULTS  (signer-independent evaluation)')
    print('='*70)
    print(f'  {"Fold":<8} {"Test Signers":<18} {"Val Top-1":>9} {"Test Top-1":>10} {"F1":>8}')
    print(f'  {"-"*57}')
    for r in fold_results:
        print(f'  {r["fold"]:<8} {str(r["test_signers"]):<18} '
              f'{r["best_val_top1"]*100:>8.2f}%  '
              f'{r["test_top1"]*100:>9.2f}%  '
              f'{r["test_macro_f1"]*100:>7.2f}%')
    print(f'  {"-"*57}')
    print(f'  {"MEAN":<8} {"":<18} '
          f'{"":>9}  {cross_subject_acc*100:>9.2f}%  '
          f'{np.mean([r["test_macro_f1"] for r in fold_results])*100:>7.2f}%')
    print(f'  {"STD":<8} {"":<18} '
          f'{"":>9}  {cross_subject_std*100:>9.2f}%')
    print('='*70)
    print(f'\n  Final Cross-Subject Accuracy (KD) : '
          f'{cross_subject_acc*100:.2f}% ± {cross_subject_std*100:.2f}%')

    results = {
        'folds':             fold_results,
        'cross_subject_acc': cross_subject_acc,
        'cross_subject_std': cross_subject_std,
        'config': {
            'model': 'BdSLSPOTER-v2+KD', 'D_MODEL': D_MODEL, 'N_LAYERS': N_LAYERS,
            'KD_T': KD_T, 'ALPHA_KD': ALPHA_KD,
            'LAMBDA_ADV': LAMBDA_ADV, 'ALPHA_SUPCON': ALPHA_SUPCON,
            'SUPCON_TEMP': SUPCON_TEMP, 'BATCH_SIZE': BATCH_SIZE,
            'NUM_SIGNERS': NUM_SIGNERS, 'NUM_CLASSES': NUM_CLASSES,
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'kd_fold_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nAll outputs → {OUTPUT_DIR}')
    print('[kd-signer-variance.py complete]')
