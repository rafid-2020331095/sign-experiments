"""
VideoMAE Frame Preprocessing Script  (CPU-only — no GPU needed)
===============================================================
Run this in a CPU-only Kaggle notebook FIRST to extract and save frame tensors
from all BdSLW401 videos to disk.  The GPU training script (videomae_train.py)
then loads these .pt files directly, eliminating all video-decode overhead.

What this does per video
------------------------
  1. Decode the full MP4 with torchvision
  2. Uniformly sample 16 frames
  3. Resize + centre-crop to 224×224
  4. ImageNet-normalise
  5. Save as float16 .pt  →  ~4.8 MB per video

Output structure
----------------
  /kaggle/working/frames/
      train/  W001S01F_01.pt  W001S02F_01.pt  ...
      val/    ...
      test/   ...

Resume / checkpointing
----------------------
  The script skips any video whose .pt file already exists.
  Just re-run after a Kaggle session cut-off — it picks up from where it stopped.

Storage cap
-----------
  Set MAX_STORAGE_GB to your Kaggle working-dir budget (default 15 GB).
  The script halts cleanly when the cap is hit; re-run next session to continue.

  Storage estimates (float16):
    Full train  (38,876 videos) ≈ 186 GB  — requires multiple sessions
    Full val    ( 4,389 videos) ≈  21 GB
    Full test   ( 7,833 videos) ≈  38 GB
    First 30 words (~2,900 videos) ≈  14 GB  ← fits in one session

After all sessions complete
---------------------------
  Download /kaggle/working/frames/ from the Output tab, upload as a Kaggle
  Dataset, then mount it in videomae_train.py.
"""

import os, re, sys, time, subprocess
from pathlib import Path

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'transformers==4.40.0', 'timm', 'einops', 'av'],
    check=False
)

import torch
import torchvision.io as tvio
import torchvision.transforms as TF
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = Path('/kaggle/input/datasets/hasanssl/bdslw401/Front/Front')
FRAMES_DIR = Path('/kaggle/working/frames')

# ── Settings ──────────────────────────────────────────────────────────────────
VIEW            = 'Front'          # 'Front' | 'Lateral'
NUM_FRAMES      = 16               # VideoMAE fixed input — do not change
IMG_SIZE        = 224              # VideoMAE fixed spatial size — do not change
MAX_STORAGE_GB  = 15.0             # stop when this many GB used in FRAMES_DIR
MAX_WORDS       = 30               # only process W001–W030  (set None for all 401)
SPLITS          = ['val', 'test', 'train']   # val/test first (reused every epoch)

SAVE_DTYPE = torch.float16         # ~4.8 MB/video  (float32 = ~9.6 MB/video)
VIEW_CHAR  = {'Front': 'F', 'Lateral': 'L'}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for s in SPLITS:
    (FRAMES_DIR / s).mkdir(parents=True, exist_ok=True)

print('VideoMAE Frame Preprocessing')
print(f'  Data root    : {DATA_ROOT}')
print(f'  Frames dir   : {FRAMES_DIR}')
print(f'  View         : {VIEW}')
print(f'  Frames/video : {NUM_FRAMES} × {IMG_SIZE}×{IMG_SIZE}')
print(f'  Save dtype   : {SAVE_DTYPE}')
print(f'  Storage cap  : {MAX_STORAGE_GB} GB')
print()


# ── Frame extractor ───────────────────────────────────────────────────────────

_transform = TF.Compose([
    TF.Resize(int(IMG_SIZE * 256 / 224), antialias=True),
    TF.CenterCrop(IMG_SIZE),
])


def extract_frames(video_path: str) -> torch.Tensor:
    """
    Extract NUM_FRAMES uniformly sampled, resized, normalised frames.
    Returns float16 tensor [16, 3, 224, 224], or None on failure.
    """
    try:
        vframes, _, _ = tvio.read_video(str(video_path),
                                        pts_unit='sec', output_format='TCHW')
    except Exception as e:
        print(f'  [WARN] Cannot read {Path(video_path).name}: {e}')
        return None

    N = vframes.shape[0]
    if N == 0:
        return None

    idx    = torch.linspace(0, N - 1, NUM_FRAMES).long()
    frames = vframes[idx].float() / 255.0                       # [16, 3, H, W]
    frames = torch.stack([_transform(f) for f in frames])      # [16, 3, 224, 224]
    frames = (frames - MEAN) / STD
    return frames.to(SAVE_DTYPE)


# ── Disk-usage helper ─────────────────────────────────────────────────────────

def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1e9


# ── Video scanner ─────────────────────────────────────────────────────────────

def collect_videos(split: str):
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        print(f'  [WARN] {split_dir} not found')
        return []
    char    = VIEW_CHAR.get(VIEW, 'F')
    pattern = re.compile(rf'^W(\d+)S\d+{char}_\d+\.mp4$', re.I)
    videos  = []
    for f in split_dir.glob('*.mp4'):
        m = pattern.match(f.name)
        if not m:
            continue
        if MAX_WORDS is not None and int(m.group(1)) > MAX_WORDS:
            continue
        videos.append(f)
    return sorted(videos)


# ── Per-split processing ──────────────────────────────────────────────────────

def preprocess_split(split: str) -> bool:
    """
    Extract frames for one split.
    Returns False if the storage cap was hit (caller should stop all splits).
    """
    videos  = collect_videos(split)
    out_dir = FRAMES_DIR / split
    total   = len(videos)
    saved   = skipped = failed = 0

    # Count already-done files to show accurate resumption info
    already_done = sum(1 for v in videos if (out_dir / (v.stem + '.pt')).exists())
    print(f'[{split}]  {total} videos  |  already done: {already_done}  |  '
          f'remaining: {total - already_done}')

    pbar = tqdm(videos, desc=f'  {split}', unit='vid')
    for video_path in pbar:
        pt_path = out_dir / (video_path.stem + '.pt')

        # ── Resume: skip already-extracted ────────────────────────────────────
        if pt_path.exists():
            skipped += 1
            continue

        # ── Storage cap guard ─────────────────────────────────────────────────
        used_gb = dir_size_gb(FRAMES_DIR)
        if used_gb >= MAX_STORAGE_GB:
            pbar.close()
            print(f'\n  [STOP] Cap reached: {used_gb:.2f} / {MAX_STORAGE_GB} GB used.')
            print(f'  Saved this run: {saved}  |  Failed: {failed}')
            print(f'  Re-run to continue from this point (already-done files skipped).')
            return False   # signal caller to stop

        frames = extract_frames(str(video_path))
        if frames is None:
            failed += 1
            continue

        torch.save(frames, pt_path)
        saved += 1
        pbar.set_postfix(saved=saved, disk_gb=f'{dir_size_gb(FRAMES_DIR):.1f}')

    used_gb = dir_size_gb(FRAMES_DIR)
    print(f'  Saved: {saved}  |  Skipped (already done): {skipped}  |  Failed: {failed}')
    print(f'  Disk usage: {used_gb:.2f} GB')
    return True   # completed without hitting cap


# ── Run ───────────────────────────────────────────────────────────────────────

print('=' * 60)
print('Starting preprocessing  (CPU — no GPU needed)')
print('Processes val → test → train (val/test reused every epoch,')
print('so they give the most training speedup per GB of storage).')
print('=' * 60)

t0 = time.time()
for split in SPLITS:
    ok = preprocess_split(split)
    if not ok:
        print('\nStorage cap hit. Download /kaggle/working/frames/ now,')
        print('upload as a Kaggle Dataset, then re-run to continue.')
        break
else:
    elapsed = time.time() - t0
    print(f'\nAll splits done in {elapsed/60:.1f} min')
    print(f'Total disk used: {dir_size_gb(FRAMES_DIR):.2f} GB')

print()
print('Next steps:')
print('  1. Output tab → download the frames/ folder (or save as Kaggle Dataset)')
print('  2. In videomae_train.py, set FRAMES_DIR to where frames/ is mounted')
print('  3. Run videomae_train.py on a GPU session')
