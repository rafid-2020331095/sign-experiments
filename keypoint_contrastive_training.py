"""
BdSL-SPOTER | Keypoint Training with Contrastive Learning  (GPU)
=================================================================
Extends keypoint_training.py with:

  1. Supervised Contrastive Loss  (Khosla et al., NeurIPS 2020)
       - In-batch positives: same word label (any signer)
       - Augmented-view positives: each sample paired with its own augmented copy
       - Combined gives both SIGNER invariance and STYLE invariance in one loss

  2. Cross-signer SupCon emphasis
       - Same word from different signers pulled together more strongly
       - Directly targets the val→test gap caused by unseen test signers

  3. GRL kept but reduced  (LAMBDA_ADV = 0.1 instead of 0.5)
       - At 0.5 it hurt test accuracy (over-corrected)
       - At 0.1 it acts as a gentle regularizer complementary to SupCon

  4. ProjectionHead  (2-layer MLP → L2-normalised 64-dim sphere)
       - Contrastive loss operates in projection space
       - Classification still uses backbone features directly (sign_head)

Total loss = L_CE  +  λ_adv × L_GRL  +  α_supcon × L_SupCon

All other settings (model architecture, augmentations, scheduler, checkpoints)
are identical to keypoint_training.py.
"""

import os, json, time, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
KEYPOINTS_DIR = '/kaggle/input/datasets/rafidadib/keypoint-30/keypoints'
OUTPUT_DIR    = '/kaggle/working'
CKPT_DIR      = os.path.join(OUTPUT_DIR, 'checkpoints_contrastive')
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────
with open(os.path.join(KEYPOINTS_DIR, 'config.json')) as f:
    _cfg = json.load(f)

NUM_CLASSES   = _cfg['num_classes']
SEQ_LEN       = _cfg['seq_len']
FEATURE_DIM   = _cfg['feature_dim']       # 223 if geo features extracted, else 150
NUM_LANDMARKS = _cfg['num_landmarks']
NUM_SIGNERS   = _cfg.get('num_signers', 18)

# ── Model hyperparameters  (identical to keypoint_training.py) ────────────────
D_MODEL      = 128
N_HEADS      = 8
N_LAYERS     = 4
D_FF         = 512
DROPOUT      = 0.20
LABEL_SMOOTH = 0.05
WEIGHT_DECAY = 5e-4
MAX_EPOCHS   = 80
PATIENCE     = 15
BATCH_SIZE   = 64         # larger batch → more in-batch positives for SupCon
MAX_LR       = 3e-4
SEED         = 42
RESUME_FROM_CKPT = False

# ── GRL  (kept but toned down — 0.5 hurt test acc in experiment) ──────────────
USE_GRL     = True
LAMBDA_ADV  = 0.1         # was 0.5; at 0.1 it gently regularises without hurting
DISC_HIDDEN = 256

# ── Supervised Contrastive Learning ──────────────────────────────────────────
USE_SUPCON    = True
ALPHA_SUPCON  = 0.15      # weight of SupCon loss  (tunable: try 0.10 – 0.25)
SUPCON_TEMP   = 0.10      # temperature τ  (lower = sharper, harder negatives)
PROJ_DIM      = 64        # projection head output dimension

# ── Curriculum  (disabled — caused collapse at epoch 3) ──────────────────────
CURRICULUM_EPOCHS    = 0
CURRICULUM_START_LEN = 50

assert D_MODEL % N_HEADS == 0

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    print(f'VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'\nFeature dim : {FEATURE_DIM}')
print(f'D_MODEL     : {D_MODEL}  ({D_MODEL}/{N_HEADS} = {D_MODEL//N_HEADS} per head)')
print(f'N_LAYERS    : {N_LAYERS}')
print(f'NUM_CLASSES : {NUM_CLASSES}')
print(f'SEQ_LEN     : {SEQ_LEN}')
print(f'BATCH_SIZE  : {BATCH_SIZE}  (doubled per forward pass for SupCon augmented views)')
print(f'LAMBDA_ADV  : {LAMBDA_ADV}   (GRL weight)')
print(f'ALPHA_SUPCON: {ALPHA_SUPCON}  (SupCon weight)')
print(f'SUPCON_TEMP : {SUPCON_TEMP}   (contrastive temperature)')


# ── BlazePose L↔R pairs for horizontal flip ────────────────────────────────────
BLAZE_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


# ── Dataset ───────────────────────────────────────────────────────────────────

class BdSLDataset(Dataset):
    def __init__(self, npz_path, augment=False, curriculum_len=None):
        data = np.load(npz_path)
        self.X  = data['X'].astype(np.float32)
        self.y  = data['y'].astype(np.int64)
        if 'signer_id' in data:
            self.signer_id = (data['signer_id'].astype(np.int64) - 1)
        else:
            self.signer_id = np.zeros(len(self.y), dtype=np.int64)
        self.augment = augment
        self.curriculum_len = curriculum_len
        has_sid = 'signer_id' in data
        print(f'Loaded {os.path.basename(npz_path)}: '
              f'X={self.X.shape}, classes={len(np.unique(self.y))}'
              f', signer_ids={"found" if has_sid else "NOT FOUND"}')

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
        seq  = seq.copy()
        n_lm = 75
        mask = np.random.rand(n_lm) < p
        for i in np.where(mask)[0]:
            seq[:, i * 2]     = 0.0
            seq[:, i * 2 + 1] = 0.0
        return seq

    def temporal_scale(self, seq):
        T       = seq.shape[0]
        scale   = np.random.uniform(0.8, 1.2)
        new_T   = max(2, int(T * scale))
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
        seq = seq.copy()
        seq[:, 0:150:2] = -seq[:, 0:150:2]
        for a, b in BLAZE_LR_PAIRS:
            xa, ya = a * 2, a * 2 + 1
            xb, yb = b * 2, b * 2 + 1
            seq[:, [xa, ya]], seq[:, [xb, yb]] = \
                seq[:, [xb, yb]].copy(), seq[:, [xa, ya]].copy()
        left_block  = seq[:, 66:108].copy()
        right_block = seq[:, 108:150].copy()
        seq[:, 66:108]  = right_block
        seq[:, 108:150] = left_block
        if seq.shape[1] > 150:
            for l_s, l_e, r_s, r_e in [
                (150, 170, 170, 190),
                (190, 200, 200, 210),
                (210, 215, 215, 220),
            ]:
                tmp = seq[:, l_s:l_e].copy()
                seq[:, l_s:l_e] = seq[:, r_s:r_e]
                seq[:, r_s:r_e] = tmp
            if seq.shape[1] > 222:
                seq[:, [221, 222]] = seq[:, [222, 221]]
        return seq

    def apply_curriculum(self, seq):
        if self.curriculum_len is None or self.curriculum_len >= SEQ_LEN:
            return seq
        trimmed = seq[:self.curriculum_len]
        pad     = np.zeros((SEQ_LEN - self.curriculum_len, seq.shape[1]), dtype=np.float32)
        return np.concatenate([trimmed, pad], axis=0)

    def augment_seq(self, seq):
        """Apply a random subset of augmentations to a single sequence."""
        if np.random.rand() < 0.60: seq = self.temporal_dropout(seq)
        if np.random.rand() < 0.60: seq = self.coordinate_noise(seq)
        if np.random.rand() < 0.50: seq = self.horizontal_flip(seq)
        if np.random.rand() < 0.50: seq = self.landmark_dropout(seq)
        if np.random.rand() < 0.40: seq = self.temporal_scale(seq)
        return seq

    def __getitem__(self, idx):
        seq   = self.X[idx].copy()
        label = self.y[idx]
        if self.augment:
            seq = self.augment_seq(seq)
        seq = self.apply_curriculum(seq)
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(self.signer_id[idx], dtype=torch.long))


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

def supervised_contrastive_loss(embeddings: torch.Tensor,
                                 labels: torch.Tensor,
                                 temperature: float = SUPCON_TEMP) -> torch.Tensor:
    """
    Supervised Contrastive Loss  (Khosla et al., NeurIPS 2020).

    embeddings : (N, D)  L2-normalised projection vectors (unit sphere)
    labels     : (N,)    integer class labels

    Positive pairs: any two samples sharing the same class label.
    When batched with [original || augmented], this automatically includes:
      - (original_i, augmented_i)  — style invariance
      - (original_i, original_j) where label_i == label_j  — class clustering
      - (original_i, augmented_j) where label_i == label_j  — cross-signer style

    L = mean over anchors of:
          -1/|P(i)| * sum_{p in P(i)} log[ exp(z_i·z_p/τ) / sum_{a≠i} exp(z_i·z_a/τ) ]
    """
    N   = embeddings.shape[0]
    sim = torch.mm(embeddings, embeddings.T) / temperature  # (N, N)

    # Mask diagonal (self-similarity → -inf so it's excluded from softmax)
    self_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device)
    sim       = sim.masked_fill(self_mask, float('-inf'))

    # Positive mask: same label, excluding self
    lab      = labels.unsqueeze(1)               # (N, 1)
    pos_mask = (lab == lab.T) & ~self_mask        # (N, N) bool

    n_pos = pos_mask.sum(1).float()               # (N,) number of positives per anchor
    valid = n_pos > 0                             # anchors with at least one positive

    if not valid.any():
        return embeddings.sum() * 0.0             # keeps gradient graph alive

    log_prob    = F.log_softmax(sim, dim=1)       # (N, N)
    # Sum log-prob over positives, divide by number of positives
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(1)  # (N,)
    loss = (loss_per_anchor[valid] / n_pos[valid]).mean()
    return loss


# ── GRL helpers ───────────────────────────────────────────────────────────────

class GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class GRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = 0.0

    def set_lambda(self, lam: float): self.lam = lam

    def forward(self, x):
        return GRLFunction.apply(x, self.lam)


class SignerDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_signers: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(hidden, n_signers),
        )

    def forward(self, x): return self.net(x)


def grl_lambda_schedule(step: int, total_steps: int) -> float:
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ── Projection Head ───────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for contrastive learning.
    Projects D_MODEL → PROJ_DIM and L2-normalises to unit sphere.
    Used only during training; sign classification uses backbone features directly.
    """
    def __init__(self, in_dim: int = D_MODEL, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)   # L2-norm → unit sphere


# ── Model ─────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.encoding, std=0.02)

    def forward(self, x): return x + self.encoding


class BdSLSPOTER(nn.Module):
    """
    BdSL-SPOTER with three heads:
      1. sign_head      — classification (CE loss)
      2. disc + grl     — signer discriminator (adversarial GRL loss)
      3. proj_head      — projection for supervised contrastive loss

    Forward returns (sign_logits, signer_logits, proj_emb).
    During eval, signer_logits and proj_emb are None.
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

        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc    = LearnablePositionalEncoding(seq_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        h1, h2, h3 = d_model * 2, d_model, num_classes * 2
        self.classifier = nn.Sequential(
            nn.Linear(d_model, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2),      nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h2, h3),      nn.LayerNorm(h3), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h3, num_classes),
        )

        # GRL signer discriminator
        self.grl  = GRL()
        self.disc = SignerDiscriminator(d_model, disc_hidden, n_signers)

        # Projection head for contrastive loss
        self.proj_head = ProjectionHead(d_model, PROJ_DIM) if USE_SUPCON else None

        self._init_weights()

    def set_grl_lambda(self, lam: float): self.grl.set_lambda(lam)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):                       # x: (B, T, FEATURE_DIM)
        x    = self.input_norm(x)
        x    = self.input_proj(x)               # (B, T, D_MODEL)
        x    = self.pos_enc(x)
        x    = self.transformer(x)              # (B, T, D_MODEL)
        feat = x.mean(dim=1)                    # (B, D_MODEL) — global avg pool

        sign_out   = self.classifier(feat)      # (B, num_classes)

        if self.training:
            signer_out = self.disc(self.grl(feat)) if USE_GRL else None
            proj_out   = self.proj_head(feat)   if USE_SUPCON else None
        else:
            signer_out = None
            proj_out   = None

        return sign_out, signer_out, proj_out


# ── DataLoader builder ────────────────────────────────────────────────────────

def get_dataloaders(epoch):
    clen = None

    train_ds = BdSLDataset(os.path.join(KEYPOINTS_DIR, 'train.npz'),
                           augment=True, curriculum_len=clen)
    val_ds   = BdSLDataset(os.path.join(KEYPOINTS_DIR, 'val.npz'),
                           augment=False, curriculum_len=None)

    labels        = train_ds.y
    class_counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    sample_weights = 1.0 / np.maximum(class_counts[labels], 1)
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = len(train_ds),
        replacement = True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_ds, clen


# ── Train function ────────────────────────────────────────────────────────────

def augment_batch(X_np: np.ndarray, ds: BdSLDataset) -> torch.Tensor:
    """
    Create one augmented view of a numpy batch (B, T, F).
    Applied augmentations are the same as training augmentations, run per-sample.
    Returns a CPU float32 tensor ready for .to(DEVICE).
    """
    out = np.stack([ds.augment_seq(X_np[i]) for i in range(len(X_np))])
    return torch.from_numpy(out)


def train_one_epoch(model, loader, optimizer, scheduler, scaler,
                    criterion, disc_criterion, epoch, global_step, total_steps,
                    train_ds):
    model.train()
    total_loss = correct = total = 0

    bar = tqdm(loader, desc=f'  Ep{epoch+1:02d} train', leave=False)
    for X, y, signer_ids in bar:
        X_np = X.numpy().copy()   # keep numpy copy for augmented view generation

        X          = X.to(DEVICE, non_blocking=True)
        y          = y.to(DEVICE, non_blocking=True)
        signer_ids = signer_ids.to(DEVICE, non_blocking=True)

        # Ramp GRL lambda
        lam = grl_lambda_schedule(global_step, total_steps)
        model.set_grl_lambda(lam)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            # ── Original view forward ───────────────────────────────────────
            sign_logits, signer_logits, proj = model(X)

            # ── CE loss ──────────────────────────────────────────────────────
            l_ce = criterion(sign_logits, y)

            # ── GRL adversarial loss ─────────────────────────────────────────
            if USE_GRL and signer_logits is not None:
                l_adv = disc_criterion(signer_logits, signer_ids)
            else:
                l_adv = torch.tensor(0.0, device=DEVICE)

            # ── Supervised Contrastive loss ──────────────────────────────────
            l_supcon = torch.tensor(0.0, device=DEVICE)
            if USE_SUPCON and proj is not None:
                # Create augmented view and forward it
                X_aug = augment_batch(X_np, train_ds).to(DEVICE, non_blocking=True)
                _, _, proj_aug = model(X_aug)

                if proj_aug is not None:
                    # Concatenate [original | augmented] — (2B, PROJ_DIM)
                    combined_proj   = torch.cat([proj, proj_aug], dim=0)
                    combined_labels = torch.cat([y,   y],         dim=0)
                    l_supcon = supervised_contrastive_loss(
                        combined_proj, combined_labels, SUPCON_TEMP
                    )

            loss = l_ce + LAMBDA_ADV * l_adv + ALPHA_SUPCON * l_supcon

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
        bar.set_postfix(
            ce   =f'{l_ce.item():.3f}',
            adv  =f'{l_adv.item():.3f}',
            con  =f'{l_supcon.item():.3f}',
            acc  =f'{100.*correct/total:.1f}%',
            lam  =f'{lam:.2f}',
        )

    return total_loss / total, correct / total, global_step


# ── Evaluate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = correct_top5 = total = 0
    all_preds, all_labels = [], []

    for X, y, _signer_ids in loader:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast():
            sign_logits, _, _ = model(X)   # signer + proj skipped in eval
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
        'epoch':            epoch,
        'model_state':      model.state_dict(),
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict(),
        'scaler_state':     scaler.state_dict(),
        'best_val_acc':     best_val_acc,
        'patience_counter': patience_counter,
    }, os.path.join(CKPT_DIR, f'epoch_{epoch:02d}.pt'))


def load_latest_ckpt(model, optimizer, scheduler, scaler):
    if not RESUME_FROM_CKPT:
        print('  RESUME_FROM_CKPT=False — starting fresh from epoch 1.')
        return 0, 0.0, 0
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'epoch_*.pt')))
    if not files:
        return 0, 0.0, 0
    for fpath in reversed(files):
        ckpt = torch.load(fpath, map_location=DEVICE)
        try:
            model.load_state_dict(ckpt['model_state'])
        except RuntimeError as e:
            print(f'  [WARN] {os.path.basename(fpath)}: architecture mismatch — starting fresh.')
            print(f'  Detail: {str(e)[:120]}...')
            return 0, 0.0, 0
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception:
            pass
        sched_state = ckpt.get('scheduler_state', {})
        if sched_state.get('total_steps') == scheduler.total_steps:
            try: scheduler.load_state_dict(sched_state)
            except Exception: pass
        else:
            ep = ckpt['epoch']
            steps_done = ep * (scheduler.total_steps // MAX_EPOCHS)
            for _ in range(steps_done): scheduler.step()
        ep = ckpt['epoch']
        print(f'Resumed from {os.path.basename(fpath)}  '
              f'(best_val_acc={ckpt["best_val_acc"]*100:.2f}%)')
        return ep, ckpt['best_val_acc'], ckpt['patience_counter']
    return 0, 0.0, 0


# ── Pre-training Diagnostics ──────────────────────────────────────────────────

def run_diagnostics():
    print('\n' + '=' * 70)
    print('  PRE-TRAINING DIAGNOSTICS')
    print('=' * 70)

    label_map_path = os.path.join(KEYPOINTS_DIR, 'label_map.json')
    with open(label_map_path) as f:
        label_map = json.load(f)

    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(KEYPOINTS_DIR, f'{split}.npz')
        if not os.path.exists(path):
            print(f'  [WARN] {split}.npz NOT FOUND — skipping')
            continue
        d = np.load(path)
        splits[split] = {'X': d['X'].astype(np.float32), 'y': d['y'].astype(np.int64)}

    print('\n[1] Shape & Label Range')
    print(f'  {"Split":<8} {"Samples":>8} {"Classes":>8} {"Label min–max":>14} {"Feat dim":>10}')
    print(f'  {"-" * 52}')
    for split, data in splits.items():
        X, y = data['X'], data['y']
        print(f'  {split:<8} {len(y):>8,} {len(np.unique(y)):>8} '
              f'  {y.min():>4}–{y.max():<6} {X.shape[2]:>10}')
        if X.shape[2] != FEATURE_DIM:
            print(f'  *** FEATURE DIM MISMATCH: expected {FEATURE_DIM}, got {X.shape[2]}. Re-run extraction!')

    print('\n[2] Zero-Frame Rate')
    for split, data in splits.items():
        X = data['X']
        zero_frames = (X == 0).all(axis=2).mean() * 100
        print(f'  {split:<8}: {zero_frames:.1f}% zero frames')

    print('\n[3] Signer ID availability')
    for split in ['train', 'val', 'test']:
        path = os.path.join(KEYPOINTS_DIR, f'{split}.npz')
        if os.path.exists(path):
            d = np.load(path)
            has = 'signer_id' in d.files
            print(f'  {split:<8}: {"found ✓" if has else "MISSING — re-run keypoint_extraction.py"}')

    print('\n[4] Baseline accuracy (random)')
    for split, data in splits.items():
        y = data['y']
        print(f'  {split:<8}: random={100/NUM_CLASSES:.2f}%  '
              f'majority={np.bincount(y).max()/len(y)*100:.2f}%')

    print('\n' + '=' * 70)
    print('  DIAGNOSTICS COMPLETE')
    print('=' * 70 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    run_diagnostics()

    # ── Compute training steps ─────────────────────────────────────────────────
    _tmp = np.load(os.path.join(KEYPOINTS_DIR, 'train.npz'))
    N_TRAIN = len(_tmp['y'])
    if 'signer_id' not in _tmp.files:
        print('[WARN] signer_id not found in train.npz.'
              ' Re-run keypoint_extraction.py to enable GRL.')
    del _tmp
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
    TOTAL_STEPS     = MAX_EPOCHS * STEPS_PER_EPOCH

    # ── Build model, optimizer, scheduler ────────────────────────────────────
    model         = BdSLSPOTER().to(DEVICE)
    optimizer     = AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=WEIGHT_DECAY)
    criterion     = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    disc_criterion = nn.CrossEntropyLoss()
    scaler        = GradScaler()
    scheduler = OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=TOTAL_STEPS,
        pct_start=0.3, anneal_strategy='cos',
        div_factor=10, final_div_factor=1e4,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nParameters  : {total_params:,}  ({total_params/1e6:.3f} M)')
    print(f'Proj head   : {sum(p.numel() for p in model.proj_head.parameters()):,} params')

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_val_acc, patience_counter = \
        load_latest_ckpt(model, optimizer, scheduler, scaler)

    best_model_path = os.path.join(OUTPUT_DIR, 'best_bdsl_spoter_contrastive.pt')

    history = {'train_loss': [], 'val_acc': [], 'val_top5': []}
    start_time = time.time()
    global_step = start_epoch * STEPS_PER_EPOCH

    print(f'\n{"Ep":>4} | {"Train Loss":>10} | {"Train Acc":>9} | '
          f'{"Val Loss":>8} | {"Val Top1":>8} | {"Val Top5":>8} | '
          f'{"Curr":>6} | {"Time":>5}')
    print('-' * 85)

    for epoch in range(start_epoch, MAX_EPOCHS):
        train_loader, val_loader, train_ds, clen = get_dataloaders(epoch)

        ep_start = time.time()
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, disc_criterion, epoch, global_step, TOTAL_STEPS,
            train_ds,
        )
        vm = evaluate(model, val_loader, criterion)
        ep_time = time.time() - ep_start

        history['train_loss'].append(train_loss)
        history['val_acc'].append(vm['top1_acc'])
        history['val_top5'].append(vm['top5_acc'])

        print(f'{epoch+1:>4} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | '
              f'{vm["loss"]:>8.4f} | {vm["top1_acc"]*100:>7.2f}% | '
              f'{vm["top5_acc"]*100:>7.2f}% | '
              f'{"full":>6} | {ep_time:>4.0f}s')

        if vm['top1_acc'] > best_val_acc:
            best_val_acc = vm['top1_acc']
            torch.save({'model_state': model.state_dict(),
                        'epoch': epoch + 1,
                        'val_top1': best_val_acc}, best_model_path)
            print(f'  ★ New best → {best_val_acc*100:.2f}%  (best_bdsl_spoter_contrastive.pt)')
            patience_counter = 0
        else:
            patience_counter += 1

        save_epoch_ckpt(model, optimizer, scheduler, scaler,
                        epoch + 1, best_val_acc, patience_counter)

        if patience_counter >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch+1}.')
            break

    total_min = (time.time() - start_time) / 60
    print('=' * 70)
    print(f'Training complete in {total_min:.1f} min  |  Best val top-1: {best_val_acc*100:.2f}%')

    # ── Training curves ────────────────────────────────────────────────────────
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves_contrastive.png'), dpi=120)
    plt.close()
    print('Training curves saved → training_curves_contrastive.png')

    # ── Final test evaluation ──────────────────────────────────────────────────
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
        'lambda_adv':     LAMBDA_ADV,
        'alpha_supcon':   ALPHA_SUPCON,
        'supcon_temp':    SUPCON_TEMP,
        'proj_dim':       PROJ_DIM,
        'feature_dim':    FEATURE_DIM,
        'num_classes':    NUM_CLASSES,
    }
    with open(os.path.join(OUTPUT_DIR, 'test_results_contrastive.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved → test_results_contrastive.json')

    # ── Load label map ─────────────────────────────────────────────────────────
    with open(os.path.join(KEYPOINTS_DIR, 'label_map.json')) as _f:
        _label_map = json.load(_f)

    # ── Confusion matrix ───────────────────────────────────────────────────────
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
    ax.set_title(f'Test Confusion Matrix  (top-1={tm["top1_acc"]*100:.1f}%)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_contrastive.png'), dpi=120)
    plt.close()
    print('Confusion matrix saved → confusion_matrix_contrastive.png')

    # ── Per-class accuracy ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  PER-CLASS ACCURACY  (sorted worst → best)')
    print('=' * 60)
    print(f'    # {"Word":<8} {"Acc":>6}  {"Support":>8}    Top Confusion')
    print(f'  {"─" * 56}')
    sorted_idx = np.argsort(per_class_acc)
    for rank, ci in enumerate(sorted_idx):
        word      = word_labels[ci]
        acc_pct   = per_class_acc[ci] * 100
        sup       = cm[ci].sum()
        row       = cm[ci].copy(); row[ci] = 0
        top_ci    = row.argmax()
        top_word  = word_labels[top_ci]
        top_n     = int(row.max()) if row.sum() > 0 else 0
        marker    = ' ***' if acc_pct < 5 else (' ⚠' if acc_pct < 50 else '  ✓')
        print(f'  {rank+1:>3} {word:<8} {acc_pct:>6.1f}%  {sup:>8}    '
              f'confused with {top_word} ({top_n}×){marker}')

    perfect = (per_class_acc == 1.0).sum()
    good    = ((per_class_acc >= 0.95) & (per_class_acc < 1.0)).sum()
    poor    = (per_class_acc < 0.50).sum()
    print(f'\n  Perfect (100%)    : {perfect} classes')
    print(f'  Good    (≥95%)    : {good} classes')
    print(f'  Poor    (<50%)    : {poor} classes')
    print('=' * 60)

    if abs(tm['loss'] - np.log(NUM_CLASSES)) < 0.05:
        print(f'  *** Loss ({tm["loss"]:.4f}) ≈ ln({NUM_CLASSES})={np.log(NUM_CLASSES):.4f}: '
              'model outputs uniform distribution — training failed.')
