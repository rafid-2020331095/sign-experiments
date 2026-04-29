"""
BdSL-SPOTER | Keypoint Extraction Script
========================================
Extracts MediaPipe Holistic keypoints (x, y) from BdSLW401.
Includes body pose AND both hand landmarks — essential for sign language.

Settings
--------
Solution      : mediapipe Holistic (Pose + Left Hand + Right Hand)
Landmarks     : 33 pose + 21 left hand + 21 right hand = 75 total
Feature dim   : 75 × (x, y) = 150 features / frame
Sequence len  : T = 200 frames (uniform resample)
Words         : First 60 (W001–W060)
Splits        : taken directly from dataset train / val / test folders
Normalisation : BdSL-specific (Paper Eq. 1, α = 0.85)

Feature vector layout per frame (150 values):
  [0:66]    — 33 pose landmarks  × (x, y)
  [66:108]  — 21 left-hand landmarks × (x, y)   (zeros if hand not detected)
  [108:150] — 21 right-hand landmarks × (x, y)  (zeros if hand not detected)

FIX: mp.solutions removed in mediapipe >= 0.10.14.
     Script pins to 0.10.13 (oldest available for Python 3.12) and uses
     direct submodule import so it works even if pinning fails.
"""

import subprocess, sys

# ── Pin mediapipe to oldest version available for Python 3.12 ───────────────
# check=False: if pinning fails (e.g. already installed at newer version),
# we fall through to the direct submodule import which works on all versions.
_install = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q',
     'mediapipe==0.10.13', 'opencv-python-headless', 'tqdm'],
    check=False
)
if _install.returncode != 0:
    print('[install] mediapipe==0.10.13 unavailable — using already-installed version.')
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'opencv-python-headless', 'tqdm'],
        check=True
    )

# ── Imports ───────────────────────────────────────────────────────────────────
import os, re, json, glob, warnings
import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── MediaPipe import — works on both old and new installs ────────────────────
# Direct submodule import avoids the 'mp.solutions' AttributeError entirely.
try:
    from mediapipe.python.solutions import holistic as _mp_holistic_mod
    MP_Holistic = _mp_holistic_mod.Holistic
    print('[mediapipe] Using mediapipe.python.solutions.holistic (direct import)')
except (ImportError, AttributeError):
    try:
        import mediapipe as mp
        _mp_holistic_mod = mp.solutions.holistic
        MP_Holistic = _mp_holistic_mod.Holistic
        print('[mediapipe] Using mp.solutions.holistic (legacy import)')
    except AttributeError:
        raise RuntimeError(
            'MediaPipe solutions API not found.\n'
            'Run: pip install mediapipe==0.10.13\n'
            'Available for Python 3.12: 0.10.13, 0.10.14, 0.10.15 ...'
        )

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = '/kaggle/input/datasets/hasanssl/bdslw401/Front/Front'
OUTPUT_DIR   = '/kaggle/working/keypoints'

# ── Settings ──────────────────────────────────────────────────────────────────
NUM_WORDS     = 30     # first 30 words (W001–W030)
TARGET_FRAMES = 200    # fixed sequence length T
NUM_POSE_LM   = 33     # BlazePose body landmarks
NUM_HAND_LM   = 21     # MediaPipe hand landmarks per hand
NUM_LANDMARKS = NUM_POSE_LM + NUM_HAND_LM * 2   # 33 + 21 + 21 = 75
BASE_DIM      = NUM_LANDMARKS * 2               # 75 × (x, y) = 150  — raw coordinates
GEO_DIM       = 73                              # signer-invariant geometric features
FEATURE_DIM   = BASE_DIM + GEO_DIM             # 150 + 73 = 223 total per frame
ALPHA             = 0.85   # BdSL signing-space normalisation factor (Paper Eq. 1)
CHECKPOINT_EVERY  = 200    # save a checkpoint .npz every N successfully extracted videos

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Dataset root  : {DATASET_ROOT}')
print(f'Output dir    : {OUTPUT_DIR}')
print(f'Landmarks     : {NUM_LANDMARKS}  (33 pose + 21 left hand + 21 right hand)')
print(f'Feature dim   : {FEATURE_DIM}  (150 raw coords + 73 geometric = 223)')
print(f'Target frames : {TARGET_FRAMES}')
print(f'Norm alpha    : {ALPHA}')
print(f'Normalisation : BdSL (α={ALPHA})  +  shoulder/torso (body-centric)')


# ── Section 1: Dataset Scanner ───────────────────────────────────────────────

def parse_word_id(fname: str):
    """Return 1-based word ID from filename like W001S01F_01.mp4, or None."""
    m = re.match(r'W(\d{3})', os.path.basename(fname))
    return int(m.group(1)) if m else None


def parse_signer_id(fname: str) -> int:
    """Return 1-based signer ID from filename like W001S03F_01.mp4, or 0 if not found."""
    m = re.match(r'W\d+S(\d+)', os.path.basename(fname), re.I)
    return int(m.group(1)) if m else 0


def collect_split(split_name: str, max_word: int = 60):
    """
    Walk DATASET_ROOT/<split_name>/ and collect videos whose word ID ≤ max_word.
    Returns list of (video_path, label_index, signer_id) — label = word_id - 1 (0-based),
    signer_id = 1-based signer number parsed from filename S\d+ field.
    """
    split_dir = os.path.join(DATASET_ROOT, split_name)
    entries   = []
    if not os.path.isdir(split_dir):
        print(f'[WARN] Directory not found: {split_dir}')
        return entries
    for fname in sorted(os.listdir(split_dir)):
        if not fname.lower().endswith('.mp4'):
            continue
        wid = parse_word_id(fname)
        if wid is None or wid > max_word:
            continue
        sid = parse_signer_id(fname)
        entries.append((os.path.join(split_dir, fname), wid - 1, sid))
    return entries


train_entries = collect_split('train', NUM_WORDS)
val_entries   = collect_split('val',   NUM_WORDS)
test_entries  = collect_split('test',  NUM_WORDS)

if len(val_entries) == 0:
    print('[INFO] No "val" folder found — using "test" folder as validation.')
    val_entries = test_entries

print(f'Train : {len(train_entries)} videos | {len(set(e[1] for e in train_entries))} classes')
print(f'Val   : {len(val_entries)}  videos | {len(set(e[1] for e in val_entries))} classes')
print(f'Test  : {len(test_entries)}  videos | {len(set(e[1] for e in test_entries))} classes')


# ── Section 2: Frame-level Keypoint Extraction ───────────────────────────────

def extract_frame_keypoints(results) -> np.ndarray:
    """
    Pull landmarks from one MediaPipe Holistic result.
    Layout: [pose×66] + [left_hand×42] + [right_hand×42] = 150 values.
    Any undetected component is filled with zeros.
    Returns np.ndarray (150,).
    """
    kps = []

    # ── Pose: 33 landmarks (66 values) ──────────────────────────────────────
    if results.pose_landmarks is None:
        kps.extend([0.0] * (NUM_POSE_LM * 2))
    else:
        for lm in results.pose_landmarks.landmark:
            kps.extend([lm.x, lm.y])

    # ── Left hand: 21 landmarks (42 values) ─────────────────────────────────
    if results.left_hand_landmarks is None:
        kps.extend([0.0] * (NUM_HAND_LM * 2))
    else:
        for lm in results.left_hand_landmarks.landmark:
            kps.extend([lm.x, lm.y])

    # ── Right hand: 21 landmarks (42 values) ────────────────────────────────
    if results.right_hand_landmarks is None:
        kps.extend([0.0] * (NUM_HAND_LM * 2))
    else:
        for lm in results.right_hand_landmarks.landmark:
            kps.extend([lm.x, lm.y])

    return np.array(kps, dtype=np.float32)   # (150,)


def resample_sequence(seq: np.ndarray, target_len: int = TARGET_FRAMES) -> np.ndarray:
    """
    Uniformly resample (T_orig, 150) → (target_len, 150) via linear interpolation.
    Paper: videos resampled to fixed T = 200.
    """
    T = len(seq)
    if T == target_len:
        return seq
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(new_idx, old_idx, seq[:, d])
    return out


def extract_video_keypoints(video_path: str):
    """
    Run MediaPipe Holistic on stride-2 sampled frames of a video.
    Frame pattern: take frame 0, skip 1, take 2, skip 3, ...
    This halves the number of frames processed while preserving temporal coverage,
    then resamples to TARGET_FRAMES via linear interpolation.
    Returns np.ndarray (TARGET_FRAMES, 150) or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Read all raw frames first, then apply stride-2 pattern
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return None

    # Alternating pattern: indices 0, 2, 4, 6, ... (take one, skip one)
    sampled_frames = raw_frames[::2]

    frames_kps = []
    with MP_Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for frame in sampled_frames:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            frames_kps.append(extract_frame_keypoints(results))

    if not frames_kps:
        return None

    seq = np.stack(frames_kps, axis=0)           # (T_orig//2, 150)
    seq = resample_sequence(seq, TARGET_FRAMES)   # (200, 150)
    return seq


# ── Section 3: BdSL Signing-Space Normalisation (Paper Eq. 1) ────────────────
# x'_t = (x_t - x_c) / (α · w)
# y'_t = (y_t - y_c) / (α · h)
# α = 0.85 — empirically optimised for BdSL's compact signing space

def bdsl_normalize(seq: np.ndarray) -> np.ndarray:
    """
    Apply BdSL-specific signing-space normalisation (Paper Eq. 1).
    seq : np.ndarray (T, 150) — even cols = x, odd cols = y.
    Returns normalised array of same shape.
    """
    seq      = seq.copy()
    x_coords = seq[:, 0::2]   # (T, 75)
    y_coords = seq[:, 1::2]   # (T, 75)

    x_nz = x_coords[x_coords != 0]
    y_nz = y_coords[y_coords != 0]

    if len(x_nz) == 0 or len(y_nz) == 0:
        return seq

    x_c = (x_nz.min() + x_nz.max()) / 2.0
    y_c = (y_nz.min() + y_nz.max()) / 2.0
    w   = max(x_nz.max() - x_nz.min(), 1e-6)
    h   = max(y_nz.max() - y_nz.min(), 1e-6)

    seq[:, 0::2] = (x_coords - x_c) / (ALPHA * w)
    seq[:, 1::2] = (y_coords - y_c) / (ALPHA * h)
    return seq


print(f'BdSL normalisation ready  (α = {ALPHA}).')


# ── Section 3a-2: Shoulder / Torso Normalisation ─────────────────────────────
# Applied AFTER bdsl_normalize().  Re-expresses every landmark relative to the
# shoulder midpoint and scales by the torso length so that:
#   • Signer POSITION is removed  (shoulder midpoint → origin)
#   • Signer HEIGHT / camera DISTANCE is removed  (torso length → 1.0 unit)
#
# MediaPipe BlazePose landmark → column mapping  (col = lm_idx × 2):
#   lm 11 = left  shoulder → cols 22, 23
#   lm 12 = right shoulder → cols 24, 25
#   lm 23 = left  hip      → cols 46, 47
#   lm 24 = right hip      → cols 48, 49

def shoulder_normalize(seq: np.ndarray) -> np.ndarray:
    """
    Re-centre all landmarks on the shoulder midpoint and scale by torso length.
    seq : (T, 150)  — output of bdsl_normalize().
    Returns same shape; no-op if shoulders / hips are undetected (all-zero).
    """
    seq = seq.copy()

    lsh_x, lsh_y = seq[:, 22], seq[:, 23]   # left  shoulder
    rsh_x, rsh_y = seq[:, 24], seq[:, 25]   # right shoulder
    lhp_x, lhp_y = seq[:, 46], seq[:, 47]   # left  hip
    rhp_x, rhp_y = seq[:, 48], seq[:, 49]   # right hip

    smid_x = (lsh_x + rsh_x) / 2.0   # (T,) shoulder midpoint x
    smid_y = (lsh_y + rsh_y) / 2.0   # (T,) shoulder midpoint y
    hmid_x = (lhp_x + rhp_x) / 2.0   # (T,) hip midpoint x
    hmid_y = (lhp_y + rhp_y) / 2.0   # (T,) hip midpoint y

    torso_per_frame = np.sqrt((smid_x - hmid_x) ** 2 + (smid_y - hmid_y) ** 2)
    valid = torso_per_frame[torso_per_frame > 1e-6]
    torso_scale = float(np.median(valid)) if len(valid) > 0 else 0.0
    if torso_scale < 1e-6:
        return seq   # pose not detected — leave sequence unchanged

    # Broadcast: (T, 75) − (T, 1)  then divide by scalar
    seq[:, 0::2] = (seq[:, 0::2] - smid_x[:, None]) / torso_scale
    seq[:, 1::2] = (seq[:, 1::2] - smid_y[:, None]) / torso_scale
    return seq


print('Shoulder / torso normalisation ready.')


# ── Section 3b: Geometric Feature Computation ────────────────────────────────
# These 73 signer-invariant features are appended to the 150 raw coordinates,
# giving a final feature vector of 223 values per frame.
#
# MediaPipe hand landmark indices (0-based):
#   0: Wrist
#   1: Thumb_CMC  2: Thumb_MCP  3: Thumb_IP   4: Thumb_TIP
#   5: Index_MCP  6: Index_PIP  7: Index_DIP  8: Index_TIP
#   9: Mid_MCP   10: Mid_PIP   11: Mid_DIP   12: Mid_TIP
#  13: Ring_MCP  14: Ring_PIP  15: Ring_DIP  16: Ring_TIP
#  17: Pink_MCP  18: Pink_PIP  19: Pink_DIP  20: Pink_TIP

# 20 bone connections per hand (5 palm + 3×5 finger bones)
HAND_BONES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),   # palm fan
    (1, 2), (2, 3), (3, 4),                       # thumb
    (5, 6), (6, 7), (7, 8),                       # index
    (9, 10), (10, 11), (11, 12),                  # middle
    (13, 14), (14, 15), (15, 16),                 # ring
    (17, 18), (18, 19), (19, 20),                 # pinky
]

# 10 joint angle triples per hand (2 per finger — at PIP and DIP equivalent)
HAND_ANGLES = [
    (1, 2, 3), (2, 3, 4),     # thumb:  MCP-IP-TIP angles
    (5, 6, 7), (6, 7, 8),     # index:  PIP and DIP angles
    (9, 10, 11), (10, 11, 12),# middle: PIP and DIP angles
    (13, 14, 15), (14, 15, 16),# ring:  PIP and DIP angles
    (17, 18, 19), (18, 19, 20),# pinky: PIP and DIP angles
]

# Fingertip landmark indices (for fingertip-to-wrist distances)
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


def _hand_pt(hand_kps: np.ndarray, lm_idx: int) -> np.ndarray:
    """Return (x, y) for landmark lm_idx from a 42-dim hand block."""
    return hand_kps[lm_idx * 2: lm_idx * 2 + 2]


def _angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex b formed by rays b→a and b→c (radians, 0–π)."""
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def compute_geometric_features(frame: np.ndarray) -> np.ndarray:
    """
    Compute 73 signer-invariant geometric features from a normalised (150,) frame.

    Input layout:  frame[0:66]   = 33 pose landmarks × (x,y)
                   frame[66:108] = 21 left-hand landmarks × (x,y)
                   frame[108:150]= 21 right-hand landmarks × (x,y)

    Output (73 values):
      [0:20]  Left  hand bone lengths      (20 bones: 5 palm + 3×5 fingers)
      [20:40] Right hand bone lengths      (20 bones)
      [40:50] Left  hand joint angles      (10 angles: 2 per finger, in radians)
      [50:60] Right hand joint angles      (10 angles)
      [60:65] Left  fingertip-to-wrist     (5 distances: thumb→pinky)
      [65:70] Right fingertip-to-wrist     (5 distances)
      [70]    Inter-hand wrist distance    (1 — how far apart the two hands are)
      [71]    Left  wrist-to-nose distance (1 — hand height relative to face)
      [72]    Right wrist-to-nose distance (1)
    """
    geo = []
    lh = frame[66:108]     # left hand block  (42,)
    rh = frame[108:150]    # right hand block (42,)
    lh_ok = bool(lh.any())
    rh_ok = bool(rh.any())

    # ── Bone lengths (20 per hand) ────────────────────────────────────────────
    for hand_kps, ok in [(lh, lh_ok), (rh, rh_ok)]:
        for a_i, b_i in HAND_BONES:
            if ok:
                geo.append(float(np.linalg.norm(_hand_pt(hand_kps, a_i) - _hand_pt(hand_kps, b_i))))
            else:
                geo.append(0.0)

    # ── Joint angles (10 per hand) ────────────────────────────────────────────
    for hand_kps, ok in [(lh, lh_ok), (rh, rh_ok)]:
        for a_i, b_i, c_i in HAND_ANGLES:
            if ok:
                geo.append(_angle_2d(_hand_pt(hand_kps, a_i),
                                     _hand_pt(hand_kps, b_i),
                                     _hand_pt(hand_kps, c_i)))
            else:
                geo.append(0.0)

    # ── Fingertip-to-wrist distances (5 per hand) ─────────────────────────────
    for hand_kps, ok in [(lh, lh_ok), (rh, rh_ok)]:
        wrist = _hand_pt(hand_kps, 0)
        for tip_i in FINGERTIP_IDS:
            geo.append(float(np.linalg.norm(_hand_pt(hand_kps, tip_i) - wrist)) if ok else 0.0)

    # ── Inter-hand wrist-to-wrist distance (1) ────────────────────────────────
    if lh_ok and rh_ok:
        geo.append(float(np.linalg.norm(_hand_pt(lh, 0) - _hand_pt(rh, 0))))
    else:
        geo.append(0.0)

    # ── Wrist-to-nose distances (2) using pose landmarks ─────────────────────
    # Pose lm 0: nose (cols 0,1)   lm 15: left wrist (cols 30,31)   lm 16: right wrist (cols 32,33)
    nose    = frame[0:2]
    lw_pose = frame[30:32]
    rw_pose = frame[32:34]
    geo.append(float(np.linalg.norm(lw_pose - nose)) if (lw_pose != 0).any() else 0.0)
    geo.append(float(np.linalg.norm(rw_pose - nose)) if (rw_pose != 0).any() else 0.0)

    return np.array(geo, dtype=np.float32)  # (73,)


def add_geometric_features(seq: np.ndarray) -> np.ndarray:
    """
    Append geometric features to every frame of a (T, 150) normalised sequence.
    Returns (T, 223) — original coordinates kept as-is, 73 features appended.
    """
    geo = np.stack([compute_geometric_features(seq[t]) for t in range(len(seq))])
    return np.concatenate([seq, geo], axis=1)  # (T, 223)


# ── Section 4: Process All Splits and Save ───────────────────────────────────

def process_and_save(entries, split_name: str):
    """
    Extract keypoints for all videos in a split with incremental checkpoint saving.

    Checkpoints are written every CHECKPOINT_EVERY successfully extracted videos to:
        OUTPUT_DIR/<split_name>_ckpts/ckpt_NNNN.npz

    On restart the function detects existing checkpoints, skips already-done videos,
    and resumes from where it left off. The final merged .npz is saved at the end.
    """
    save_path = os.path.join(OUTPUT_DIR, f'{split_name}.npz')
    ckpt_dir  = os.path.join(OUTPUT_DIR, f'{split_name}_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume: load already-saved checkpoints ───────────────────────────────
    sequences, labels, signer_ids = [], [], []
    done_paths = set()
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_*.npz')))
    for cf in ckpt_files:
        d = np.load(cf, allow_pickle=True)
        sequences.extend(d['X'].tolist())
        labels.extend(d['y'].tolist())
        done_paths.update(d['paths'].tolist())
        if 'signer_id' in d:
            signer_ids.extend(d['signer_id'].tolist())
        else:
            # backward-compatible: derive from stored paths
            signer_ids.extend([parse_signer_id(p) for p in d['paths'].tolist()])
    if done_paths:
        print(f'  Resumed  : {len(done_paths)} videos from {len(ckpt_files)} checkpoints')

    # ── Filter to remaining (unprocessed) entries ─────────────────────────────
    remaining = [(p, l, s) for p, l, s in entries if p not in done_paths]
    print(f'  Total    : {len(entries)} | Done: {len(done_paths)} | Remaining: {len(remaining)}')

    # ── Process remaining videos ─────────────────────────────────────────────
    failed = []
    buf_seqs, buf_labels, buf_sids, buf_paths = [], [], [], []
    ckpt_counter = len(ckpt_files)

    for video_path, label, signer_id in tqdm(remaining, desc=f'[{split_name}]'):
        seq = extract_video_keypoints(video_path)
        if seq is None:
            failed.append(video_path)
            continue
        seq = bdsl_normalize(seq)
        seq = shoulder_normalize(seq)          # body-centric reference frame
        seq = add_geometric_features(seq)  # (200, 150) → (200, 223)
        buf_seqs.append(seq)
        buf_labels.append(label)
        buf_sids.append(signer_id)
        buf_paths.append(video_path)

        # ── Save checkpoint every CHECKPOINT_EVERY videos ─────────────────
        if len(buf_seqs) >= CHECKPOINT_EVERY:
            ckpt_path = os.path.join(ckpt_dir, f'ckpt_{ckpt_counter:04d}.npz')
            np.savez_compressed(
                ckpt_path,
                X=np.stack(buf_seqs).astype(np.float32),
                y=np.array(buf_labels, dtype=np.int64),
                signer_id=np.array(buf_sids, dtype=np.int64),
                paths=np.array(buf_paths),
            )
            total_so_far = len(sequences) + len(buf_seqs)
            print(f'\n  Checkpoint → {os.path.basename(ckpt_path)}  '
                  f'({len(buf_seqs)} samples | {total_so_far} total)')
            sequences.extend(buf_seqs)
            labels.extend(buf_labels)
            signer_ids.extend(buf_sids)
            buf_seqs, buf_labels, buf_sids, buf_paths = [], [], [], []
            ckpt_counter += 1

    # ── Flush remaining buffer as final checkpoint ────────────────────────────
    if buf_seqs:
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_{ckpt_counter:04d}.npz')
        np.savez_compressed(
            ckpt_path,
            X=np.stack(buf_seqs).astype(np.float32),
            y=np.array(buf_labels, dtype=np.int64),
            signer_id=np.array(buf_sids, dtype=np.int64),
            paths=np.array(buf_paths),
        )
        print(f'\n  Checkpoint → {os.path.basename(ckpt_path)}  ({len(buf_seqs)} samples)')
        sequences.extend(buf_seqs)
        labels.extend(buf_labels)
        signer_ids.extend(buf_sids)

    if not sequences:
        print(f'[ERROR] No sequences extracted for split: {split_name}')
        return None, None

    # ── Merge all checkpoints into final .npz ─────────────────────────────────
    X  = np.stack(sequences).astype(np.float32)     # (N, 200, 150)
    y  = np.array(labels, dtype=np.int64)            # (N,)
    s  = np.array(signer_ids, dtype=np.int64)        # (N,)  1-based signer IDs
    np.savez_compressed(save_path, X=X, y=y, signer_id=s)
    print(f'  Merged   → {save_path}')
    print(f'  Shape    : X={X.shape}, y={y.shape}, signer_id={s.shape}')
    print(f'  Classes  : {len(np.unique(y))}  |  Signers: {sorted(np.unique(s).tolist())}  |  Failed: {len(failed)} videos')
    if failed:
        for f in failed[:3]:
            print(f'    [fail] {os.path.basename(f)}')
    return X, y


print('=' * 60)
print('Starting keypoint extraction  (CPU — no GPU needed)')
print('Expected: ~1–3 min per 100 videos depending on resolution.')
print('=' * 60)

X_train, y_train = process_and_save(train_entries, 'train')
X_val,   y_val   = process_and_save(val_entries,   'val')
X_test,  y_test  = process_and_save(test_entries,  'test')


# ── Section 5: Save Config & Label Map (for Notebook 2) ─────────────────────

label_map = {str(i): f'W{(i + 1):03d}' for i in range(NUM_WORDS)}
with open(os.path.join(OUTPUT_DIR, 'label_map.json'), 'w') as f:
    json.dump(label_map, f, indent=2)

# Collect all signer IDs across splits to determine num_signers for GRL
_all_sids = []
for _split in ['train', 'val', 'test']:
    _p = os.path.join(OUTPUT_DIR, f'{_split}.npz')
    if os.path.exists(_p):
        _d = np.load(_p)
        if 'signer_id' in _d:
            _all_sids.extend(_d['signer_id'].tolist())
_num_signers = int(max(_all_sids)) if _all_sids else 18
print(f'Signer IDs found across all splits: {sorted(set(_all_sids))}  (num_signers={_num_signers})')

config = {
    'num_classes'  : NUM_WORDS,
    'seq_len'      : TARGET_FRAMES,
    'feature_dim'  : FEATURE_DIM,
    'num_landmarks': NUM_LANDMARKS,
    'num_signers'  : _num_signers,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print('label_map.json and config.json saved.')


# ── Section 6: Quick Verification ────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')   # headless — safe on Kaggle / servers
import matplotlib.pyplot as plt

print('\nSplit verification:')
for split in ['train', 'val', 'test']:
    path = os.path.join(OUTPUT_DIR, f'{split}.npz')
    if not os.path.exists(path):
        print(f'  {split:5s} | NOT FOUND — skipping')
        continue
    d    = np.load(path)
    X, y = d['X'], d['y']
    print(f'  {split:5s} | X: {X.shape} | y: {y.shape} | '
          f'classes: {len(np.unique(y))} | range: [{X.min():.3f}, {X.max():.3f}]')

# Trajectory plot for one training sample
train_npz = os.path.join(OUTPUT_DIR, 'train.npz')
if os.path.exists(train_npz):
    d      = np.load(train_npz)
    sample = d['X'][0]   # (200, 150)

    # (name, absolute_col_index_for_x)  — y = col+1
    # Pose landmarks: col = lm_idx * 2
    # Left hand starts at col 66:  col = 66 + lm_idx * 2
    # Right hand starts at col 108: col = 108 + lm_idx * 2
    lm_cols = [
        ('Nose (pose)',          0),          # pose lm 0
        ('Left wrist (pose)',    30),         # pose lm 15
        ('Right wrist (pose)',   32),         # pose lm 16
        ('L index tip (hand)',   66 + 8*2),   # left hand lm 8
        ('R index tip (hand)',   108 + 8*2),  # right hand lm 8
        ('L thumb tip (hand)',   66 + 4*2),   # left hand lm 4
        ('R thumb tip (hand)',   108 + 4*2),  # right hand lm 4
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, col in lm_cols:
        axes[0].plot(sample[:, col],     label=name)
        axes[1].plot(sample[:, col + 1], label=name)

    axes[0].set_title('X trajectories (normalised)')
    axes[0].set_xlabel('Frame'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_title('Y trajectories (normalised)')
    axes[1].set_xlabel('Frame'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle(f'Train sample — label: W{(d["y"][0] + 1):03d}', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'sample_trajectory.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Trajectory plot saved → {fig_path}')

print('\nAll done!')
print(f'Files saved to: {OUTPUT_DIR}')
print('  train.npz, val.npz, test.npz, label_map.json, config.json')
print('\nNext step: upload the keypoints/ folder as a Kaggle dataset, then run Notebook 2.')
