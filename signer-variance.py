"""
Signer Variance — Cross-Fold Training  (signer-variance.py)
============================================================
Implements proper signer-independent cross-validation using BdSL-SPOTER
(plain CE transformer — signer-invariance modules removed for baseline testing).

Architecture: OUTER FOLD LOOP  ⟶  INNER TRAINING LOOP
  For each fold:
    1.  Fresh model + fresh optimizer (FATAL MISTAKE PREVENTION — no weight leak)
    2.  Create DataLoaders filtered by signer_id for train / val / test
    3.  Run standard PyTorch training inner loop with early stopping
    4.  Load best val checkpoint, evaluate on fold's test signers
  After all folds:
    5.  Report per-fold test accuracy
    6.  Average across folds  →  Final Cross-Subject Accuracy
    7.  Ensemble all fold models on the ORIGINAL test.npz for boosted accuracy

Fold definitions  (18-signer pool, signer IDs 1-indexed):
  Fold 1  train=[1,2,3,5,6,7,10,12,13,14,16,17,18]  val=[4,8,9]    test=[11,15]
  Fold 2  train=[1,2,3,5,6,8,9,10,11,13,15,17,18]   val=[7,14,16]  test=[4,12]

Output per fold:
  best_fold{N}.pt  |  training_curves_fold{N}.png
Final output:
  fold_results.json  |  confusion_matrix_ensemble.png
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
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
KEYPOINTS_DIR = '/kaggle/input/datasets/rafidadib/geo-feature-keypoint/keypoints'
OUTPUT_DIR    = '/kaggle/working'
SPLITS_DIR    = os.path.join(OUTPUT_DIR, 'splits')
os.makedirs(SPLITS_DIR, exist_ok=True)

with open(os.path.join(KEYPOINTS_DIR, 'config.json')) as f:
    _cfg = json.load(f)

NUM_CLASSES   = _cfg['num_classes']
SEQ_LEN       = _cfg['seq_len']
FEATURE_DIM   = _cfg['feature_dim']
NUM_LANDMARKS = _cfg['num_landmarks']
D_MODEL = 128; N_HEADS = 8; N_LAYERS = 4; D_FF = 512; DROPOUT = 0.20
LABEL_SMOOTH = 0.05; WEIGHT_DECAY = 5e-4; MAX_EPOCHS = 80; PATIENCE = 15
BATCH_SIZE = 64; MAX_LR = 3e-4; SEED = 42

# ── Fold definitions (outer loop iterates over these) ─────────────────────────
# FATAL MISTAKE PREVENTION: model is completely re-initialised at each fold.
# Test signers are NEVER seen during training or validation within that fold.
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
    Returns the set of all signer IDs found.
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
            print(f'  {fname:<20}  samples={n_samples:6d}  '
                  f'signers({len(sids):2d})={sids}')
        else:
            print(f'  {fname:<20}  samples={n_samples:6d}  [NO signer_id field]')

    print(f'{"+"*70}')
    print(f'  ALL SIGNERS COMBINED ({len(all_signers)}) : {sorted(all_signers)}')
    print(f'{"+"*70}\n')
    return sorted(all_signers)


# ── Split creation ─────────────────────────────────────────────────────────────

def build_master_arrays():
    all_X, all_y, all_sid = [], [], []
    for subset in ['train', 'val', 'test']:
        path = os.path.join(KEYPOINTS_DIR, f'{subset}.npz')
        if not os.path.exists(path):
            print(f'  [WARN] {path} not found'); continue
        d = np.load(path)
        all_X.append(d['X'].astype(np.float32))
        all_y.append(d['y'].astype(np.int64))
        if 'signer_id' not in d.files:
            raise RuntimeError(f'{subset}.npz missing signer_id — re-run extraction')
        all_sid.append(d['signer_id'].astype(np.int64))
    all_X   = np.concatenate(all_X)
    all_y   = np.concatenate(all_y)
    all_sid = np.concatenate(all_sid)
    print(f'Master pool: X={all_X.shape}  signers={sorted(set(all_sid.tolist()))}')
    return all_X, all_y, all_sid


def save_fold_npz(fold: dict, all_X, all_y, all_sid):
    """Write train/val/test.npz for one fold under SPLITS_DIR/<fold_name>/."""
    fold_dir = os.path.join(SPLITS_DIR, fold['name'])
    os.makedirs(fold_dir, exist_ok=True)
    for subset in ['train', 'val', 'test']:
        mask = np.isin(all_sid, fold[subset])
        np.savez(os.path.join(fold_dir, f'{subset}.npz'),
                 X=all_X[mask], y=all_y[mask], signer_id=all_sid[mask])
        print(f'  {subset:5s} → {mask.sum():5d} samples  '
              f'signers={sorted(set(all_sid[mask].tolist()))}')


# ── BlazePose LR pairs ─────────────────────────────────────────────────────────
BLAZE_LR_PAIRS = [(1,4),(2,5),(3,6),(7,8),(9,10),(11,12),(13,14),(15,16),
                   (17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]


# ── Dataset ───────────────────────────────────────────────────────────────────

class BdSLDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        d = np.load(npz_path)
        self.X         = d['X'].astype(np.float32)
        self.y         = d['y'].astype(np.int64)
        self.signer_id = (d['signer_id'].astype(np.int64) - 1) \
                         if 'signer_id' in d.files \
                         else np.zeros(len(d['y']), dtype=np.int64)
        self.augment   = augment
        print(f'  {os.path.basename(npz_path)}: X={self.X.shape}  '
              f'classes={len(np.unique(self.y))}  '
              f'signers={sorted(set((self.signer_id+1).tolist()))}')

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
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.long),
                torch.tensor(self.signer_id[idx], dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.encoding, std=0.02)
    def forward(self, x): return x + self.encoding

class BdSLSPOTER(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, seq_len=SEQ_LEN,
                 feature_dim=FEATURE_DIM, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
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
        self._init_weights()

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
        return self.classifier(x.mean(dim=1))


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, epoch):
    model.train()
    total_loss = correct = total = 0
    bar = tqdm(loader, desc=f'  Ep{epoch+1:02d}', leave=False)
    for X, y, _ in bar:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            sl   = model(X)
            loss = criterion(sl, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item() * X.size(0)
        correct    += (sl.argmax(1) == y).sum().item()
        total      += X.size(0)
        bar.set_postfix(ce=f'{loss.item():.3f}', acc=f'{100.*correct/total:.1f}%')
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = correct5 = total = 0
    preds_all, labels_all = [], []
    for X, y, _ in loader:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with autocast():
            sl = model(X); loss = criterion(sl, y)
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



# (train_split removed — training is now done inside the outer fold loop in main)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(metrics, title, word_labels):
    print(f'\n{"="*55}')
    print(f'  {title}')
    print(f'{"="*55}')
    print(f'  Top-1  : {metrics["top1_acc"]*100:.2f}%')
    print(f'  Top-5  : {metrics["top5_acc"]*100:.2f}%')
    print(f'  Macro F1: {metrics["macro_f1"]*100:.2f}%')
    cm_mat        = confusion_matrix(metrics['labels'], metrics['preds'])
    per_class_acc = cm_mat.diagonal() / (cm_mat.sum(axis=1) + 1e-8)
    perfect = (per_class_acc == 1.0).sum()
    poor    = (per_class_acc < 0.50).sum()
    print(f'  Perfect (100%): {perfect}  |  Poor (<50%): {poor}')
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


# ── Main: OUTER FOLD LOOP ────────────────────────────────────────────────────

if __name__ == '__main__':

    with open(os.path.join(KEYPOINTS_DIR, 'label_map.json')) as f:
        _label_map = json.load(f)
    word_labels = [_label_map.get(str(i), str(i)) for i in range(NUM_CLASSES)]

    # ── Scan dataset: list all signers in every npz file ─────────────────────
    actual_signers = scan_dataset_signers()

    # ── Build master pool (merges train.npz + val.npz + test.npz) ─────────────
    print('\n' + '='*70)
    print('  Building master pool  (train.npz + val.npz + test.npz merged)')
    print('='*70)
    all_X, all_y, all_sid = build_master_arrays()

    # Storage for cross-fold summary
    fold_results = []   # one entry per fold

    # ══════════════════════════════════════════════════════════════════════════
    # OUTER FOLD LOOP
    # ══════════════════════════════════════════════════════════════════════════
    for fold_idx, fold in enumerate(FOLDS):
        fold_name = fold['name']
        print(f'\n{"="*70}')
        print(f'  FOLD {fold_idx+1}/{len(FOLDS)}  [{fold_name.upper()}]')
        print(f'  train signers : {fold["train"]}')
        print(f'  val   signers : {fold["val"]}')
        print(f'  test  signers : {fold["test"]}')
        print(f'{"="*70}')

        # ── Save fold npz files ───────────────────────────────────────────────
        save_fold_npz(fold, all_X, all_y, all_sid)

        fold_dir  = os.path.join(SPLITS_DIR, fold_name)
        best_path = os.path.join(OUTPUT_DIR, f'best_{fold_name}.pt')

        # ── FATAL MISTAKE PREVENTION: fresh model + optimizer each fold ───────
        # Do NOT reuse weights from the previous fold.
        model     = BdSLSPOTER().to(DEVICE)
        n_train   = len(np.load(os.path.join(fold_dir, 'train.npz'))['y'])
        spe       = max(1, n_train // BATCH_SIZE)   # steps per epoch
        tot_steps = MAX_EPOCHS * spe

        optimizer  = AdamW(model.parameters(), lr=MAX_LR/10, weight_decay=WEIGHT_DECAY)
        criterion  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        scaler     = GradScaler()
        scheduler  = OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=tot_steps,
                                 pct_start=0.3, anneal_strategy='cos',
                                 div_factor=10, final_div_factor=1e4)

        # ── Create DataLoaders for THIS fold ──────────────────────────────────
        print(f'\nDatasets for {fold_name}:')
        train_ds = BdSLDataset(os.path.join(fold_dir, 'train.npz'), augment=True)
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

        total_params = sum(p.numel() for p in model.parameters())
        print(f'\nParams: {total_params:,}  train={n_train}  steps/ep={spe}')

        # ══════════════════════════════════════════════════════════════════════
        # INNER TRAINING LOOP
        # ══════════════════════════════════════════════════════════════════════
        best_val_acc = 0.0
        patience_ctr = 0
        history      = {'tl': [], 'va': [], 'v5': []}
        t0           = time.time()

        print(f'\n{"Ep":>4} | {"T-Loss":>9} | {"T-Acc":>8} | {"V-Loss":>8} | '
              f'{"V-Top1":>8} | {"V-Top5":>8} | {"Time":>5}')
        print('-' * 70)

        val_empty = len(val_ds) == 0
        if val_empty:
            print('  [WARN] val set is empty — early stopping will use train accuracy')

        for epoch in range(MAX_EPOCHS):
            es = time.time()
            tl, ta = train_one_epoch(
                model, train_ldr, optimizer, scheduler, scaler,
                criterion, epoch)
            vm = evaluate(model, val_ldr, criterion)
            et = time.time() - es

            # When val is empty, substitute train accuracy as the tracking metric
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

            # ── Early stopping / model checkpointing ──────────────────────────
            if track_acc > best_val_acc:
                best_val_acc = track_acc
                torch.save({'model_state': model.state_dict(),
                            'epoch':       epoch + 1,
                            'val_top1':    best_val_acc,
                            'val_top5':    vm['top5_acc'] if not val_empty else ta,
                            'fold':        fold_name,
                            'feature_dim': FEATURE_DIM,
                            'num_classes': NUM_CLASSES}, best_path)
                flag = '(train acc)' if val_empty else ''
                print(f'  ★ New best → {best_val_acc*100:.2f}%  (best_{fold_name}.pt) {flag}')
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= PATIENCE:
                print(f'\nEarly stopping at epoch {epoch+1}.')
                break

        fold_min = (time.time() - t0) / 60
        print(f'{"="*70}')
        print(f'{fold_name} training done in {fold_min:.1f} min  |  '
              f'Best val: {best_val_acc*100:.2f}%')

        # ── Save training curves ──────────────────────────────────────────────
        eps = list(range(1, len(history['tl'])+1))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(eps, history['tl'], 'b-o', ms=3)
        axes[0].set_title(f'{fold_name} Loss'); axes[0].set_xlabel('Epoch'); axes[0].grid(alpha=0.3)
        axes[1].plot(eps, [v*100 for v in history['va']], 'r-o', ms=3, label='Val Top-1')
        axes[1].axhline(best_val_acc*100, color='gray', ls='--',
                        label=f'Best {best_val_acc*100:.1f}%')
        axes[1].set_title(f'{fold_name} Val Acc'); axes[1].legend()
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('%'); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'training_curves_{fold_name}.png'), dpi=120)
        plt.close()

        # ── FOLD CONCLUSION: load best weights, test on UNSEEN test signers ────
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        test_m = evaluate(model, test_ldr, criterion)
        test_m['macro_f1'] = float(f1_score(test_m['labels'], test_m['preds'], average='macro'))

        print(f'\nFold {fold_idx+1} [{fold_name}] Test Accuracy : '
              f'{test_m["top1_acc"]*100:.2f}%  '
              f'Top-5={test_m["top5_acc"]*100:.2f}%  '
              f'F1={test_m["macro_f1"]*100:.2f}%')

        # Save fold confusion matrix
        cm_f = confusion_matrix(test_m['labels'], test_m['preds'])
        save_cm(cm_f, word_labels,
                f'{fold_name} Test  top-1={test_m["top1_acc"]*100:.1f}%  '
                f'signers={fold["test"]}',
                os.path.join(OUTPUT_DIR, f'confusion_matrix_{fold_name}.png'))

        fold_results.append({
            'fold': fold_name,
            'test_signers': fold['test'],
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
    print('  CROSS-FOLD RESULTS  (signer-independent evaluation)')
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
    print(f'\n  Final Cross-Subject Accuracy : {cross_subject_acc*100:.2f}% ± {cross_subject_std*100:.2f}%')

    # ── Save results JSON ──────────────────────────────────────────────────────
    results = {
        'folds':              fold_results,
        'cross_subject_acc':  cross_subject_acc,
        'cross_subject_std':  cross_subject_std,
        'config': {
            'model': 'BdSLSPOTER-v2', 'D_MODEL': D_MODEL, 'N_LAYERS': N_LAYERS,
            'BATCH_SIZE': BATCH_SIZE, 'NUM_CLASSES': NUM_CLASSES,
        },
    }
    with open(os.path.join(OUTPUT_DIR, 'fold_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nAll outputs → {OUTPUT_DIR}')
    print('[signer-variance.py complete]')
