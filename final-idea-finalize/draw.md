# Figma Diagram Specifications — BdSL-SPOTER-v2 Paper
## Target: Q1 Journal (IEEE TNNLS / Pattern Recognition / CVIU level)

---

## Diagrams Needed (3 Total)

| # | Title | Type | Purpose |
|---|---|---|---|
| **D1** | End-to-End Feature Extraction Pipeline | Flow diagram | Show the full preprocessing chain from raw video to 223-D feature tensor |
| **D2** | BdSL-SPOTER-v2 Model Architecture | Block/layer diagram | Show the full model with all three output heads (CE, GRL, SupCon) |
| **D3** | Signer-Independent Cross-Validation Protocol | Protocol diagram | Show the 2-fold signer-disjoint training protocol and evaluation strategy |

---

---

## D1 — End-to-End Feature Extraction Pipeline

### Purpose
Show how a raw `.mp4` video becomes a (200 × 223) feature tensor ready for the model.

### Canvas size suggestion
1800 × 500 px, left-to-right flow

### Stages (7 boxes in sequence, connected by arrows)

---

**Stage 1 — Raw Video Input**
- Box label: `Raw Video (.mp4)`
- Sub-label: `BdSLW401 Dataset`
- Content to show inside:
  - Filename format: `W{word}S{signer}F_{instance}.mp4`
  - Example: `W001S03F_01.mp4`
  - Splits: `train / val / test` folders
  - Total: 30 classes × ~13 signers × multiple instances
- Visual: small video frame icon, gray background box

---

**Stage 2 — Frame Sampling (Stride-2)**
- Box label: `Frame Sampling`
- Sub-label: `Stride-2 subsampling`
- Content:
  - Read all frames
  - Keep frames at indices 0, 2, 4, 6, … (every other frame)
  - Purpose: halves processing time, preserves temporal coverage
- Visual: row of frames, alternate ones greyed out/crossed

---

**Stage 3 — MediaPipe Holistic Keypoint Extraction**
- Box label: `MediaPipe Holistic`
- Sub-label: `Per-frame landmark detection`
- Content:
  - `model_complexity = 1`
  - `min_detection_confidence = 0.5`
  - `min_tracking_confidence = 0.5`
  - Outputs per frame: 3 groups of landmarks
    - **Pose:** 33 landmarks → 66 values (cols 0–65)
    - **Left hand:** 21 landmarks → 42 values (cols 66–107)
    - **Right hand:** 21 landmarks → 42 values (cols 108–149)
  - Undetected body part → zero-filled
  - Per-frame vector: **150-D** (x, y for each of 75 landmarks)
- Visual: skeleton figure with pose + both hands highlighted in different colours

---

**Stage 4 — Temporal Resampling**
- Box label: `Temporal Resampling`
- Sub-label: `Linear interpolation → T = 200`
- Content:
  - Input: variable-length `(T_orig/2, 150)` sequence
  - Output: fixed `(200, 150)` sequence
  - Method: `np.interp` per feature dimension along time axis
  - Formula label: `new_idx = linspace(0, T_orig-1, 200)`
- Visual: two timelines — jagged (variable T) → uniform (200 ticks)

---

**Stage 5 — Two-Stage Normalisation**
- Box label: `Normalisation`
- Sub-label: `BdSL-specific + Shoulder-centric`
- Content (two sub-boxes stacked vertically):

  **Sub-box A — BdSL Signing-Space Normalisation (Paper Eq. 1):**
  - `x'_t = (x_t − x_c) / (α · w)`,  `α = 0.85`
  - `y'_t = (y_t − y_c) / (α · h)`
  - `x_c, y_c` = bounding-box centroid over all non-zero landmarks
  - `w, h` = bounding box width/height
  - Removes: camera position, signing-space scale

  **Sub-box B — Shoulder/Torso Normalisation:**
  - Origin → shoulder midpoint (avg of lm 11 + lm 12)
  - Scale → torso length (shoulder midpoint to hip midpoint, lm 23+24)
  - Removes: signer position, height, camera distance
  - Applied after Sub-box A

- Visual: two skeleton figures — before (offset, different scales) → after (aligned, same scale)

---

**Stage 6 — Geometric Feature Computation (73-D)**
- Box label: `Geometric Features`
- Sub-label: `73 signer-invariant descriptors appended`
- Content (table of feature groups):

  | Feature Group | Count | Description |
  |---|:---:|---|
  | Left hand bone lengths | 20 | Euclidean distance for each of 20 hand bones (5 palm + 3×5 fingers) |
  | Right hand bone lengths | 20 | Same 20 bones for right hand |
  | Left hand joint angles | 10 | Interior angle at PIP/DIP joints, 2 per finger (radians) |
  | Right hand joint angles | 10 | Same 10 angles for right hand |
  | Left fingertip-to-wrist | 5 | Distance from wrist to each fingertip (thumb→pinky) |
  | Right fingertip-to-wrist | 5 | Same 5 distances for right hand |
  | Inter-hand wrist distance | 1 | Left wrist to right wrist distance |
  | Left wrist-to-nose | 1 | Hand height relative to face |
  | Right wrist-to-nose | 1 | Hand height relative to face |
  | **Total** | **73** | |

- Visual: hand skeleton with bone connections and angle arcs highlighted

---

**Stage 7 — Final Feature Tensor**
- Box label: `Feature Tensor`
- Sub-label: `Input to BdSL-SPOTER`
- Content:
  - Shape: `(200, 223)`
  - Breakdown bar (horizontal):
    - Blue block: `cols 0–149` → Raw xy coordinates (150-D)
      - `0–65`: Pose landmarks
      - `66–107`: Left hand
      - `108–149`: Right hand
    - Orange block: `cols 150–222` → Geometric features (73-D)
      - `150–169`: Left bone lengths
      - `170–189`: Right bone lengths
      - `190–199`: Left joint angles
      - `200–209`: Right joint angles
      - `210–214`: Left fingertip-wrist
      - `215–219`: Right fingertip-wrist
      - `220`: Inter-hand wrist dist
      - `221`: Left wrist-nose
      - `222`: Right wrist-nose
  - Saved as `.npz` with fields: `X`, `y`, `signer_id`

---

### Connecting Arrows for D1
- Stage 1 → 2: label `cv2.VideoCapture + raw_frames[::2]`
- Stage 2 → 3: label `RGB frame → holistic.process()`
- Stage 3 → 4: label `(T//2, 150)`
- Stage 4 → 5: label `(200, 150)`
- Stage 5 → 6: label `normalised (200, 150)`
- Stage 6 → 7: label `concat([coords, geo], axis=1)` → `(200, 223)`

---

---

## D2 — BdSL-SPOTER-v2 Model Architecture

### Purpose
Show the full transformer model with input projection, positional encoding, transformer encoder, and the three output branches (classification, GRL adversarial, SupCon projection).

### Canvas size suggestion
1400 × 900 px, top-to-bottom flow (input at top, losses at bottom)

### Input (top)
- Box: `Input Sequence`
- Shape label: `(B, 200, 223)`  — Batch × Frames × Features

---

### Block 1 — Input Normalisation
- Box: `LayerNorm(223)`
- Output shape: `(B, 200, 223)`
- Note: applied along feature dim, stabilises varied scale geometric + raw features

---

### Block 2 — Input Projection
- Box: `Linear(223 → 128)`
- Output shape: `(B, 200, 128)`
- Projects heterogeneous 223-D input into uniform d_model space

---

### Block 3 — Learnable Positional Encoding
- Box: `Learnable Pos. Encoding`
- Parameter: `nn.Parameter(zeros(1, 200, 128))` — init: trunc_normal(std=0.02)
- Operation: element-wise add to token sequence
- Output shape: `(B, 200, 128)`
- Note: learned (not sinusoidal) — adapts to BdSL temporal patterns

---

### Block 4 — Transformer Encoder (repeated ×4)
- Outer box label: `TransformerEncoder (×4 layers)`
- Each layer contains (in order, norm_first=True):
  1. `Pre-LN: LayerNorm(128)`
  2. `Multi-Head Self-Attention` — 8 heads, head_dim=16, GELU
  3. `Pre-LN: LayerNorm(128)` (before FFN)
  4. `FFN: Linear(128→512) → GELU → Linear(512→128)`
  5. `Dropout(0.20)` after each sub-layer
- Output shape: `(B, 200, 128)` — same as input

---

### Block 5 — Mean Pooling
- Box: `Mean Pooling (dim=1)`
- Operation: average over 200 time steps
- Output shape: `(B, 128)` — one vector per sample
- Label: `Global sign representation`

---

### (From Block 5, THREE branches split out)

---

### Branch A — Classification Head (always active)
- Label: `Classification Head`
- Architecture (sequential):
  ```
  Linear(128 → 256) → LayerNorm(256) → GELU → Dropout(0.20)
  Linear(256 → 128) → LayerNorm(128) → GELU → Dropout(0.20)
  Linear(128 →  60) → LayerNorm(60)  → GELU → Dropout(0.20)
  Linear( 60 →  30)
  ```
- Output: `(B, 30)` logits
- Loss: `CrossEntropy(label_smoothing=0.05)`
- Loss label: `L_CE`
- Colour: **Blue**

---

### Branch B — GRL Signer Discriminator (training only, USE_GRL=True)
- Label: `GRL + Signer Discriminator`
- First operation: `Gradient Reversal Layer (GRL)`
  - Forward: identity pass `x`
  - Backward: gradient multiplied by `-λ`
  - Lambda schedule: `λ = 2 / (1 + exp(-10p)) - 1`,  `p = step / total_steps`
  - Range: λ starts near 0, grows to 1.0
  - λ_adv coefficient: 0.1
- Discriminator architecture:
  ```
  Linear(128 → 256) → ReLU → Dropout(0.30)
  Linear(256 →  15)            ← 15 signer classes
  ```
- Output: `(B, 15)` signer logits
- Loss: `CrossEntropy` on signer ID
- Total loss contribution: `0.1 × L_GRL`
- Loss label: `L_GRL`
- Colour: **Red/Orange**
- Note on diagram: dashed arrow for backward pass reversal

---

### Branch C — SupCon Projection Head (training only, USE_SUPCON=True)
- Label: `Projection Head (SupCon)`
- Architecture:
  ```
  Linear(128 → 128) → ReLU
  Linear(128 →  64)
  L2-Normalise (unit sphere)
  ```
- Output: `(B, 64)` — unit-norm embeddings
- Loss: `Supervised Contrastive Loss` (temperature τ = 0.10)
  - Requires augmented views: input batch passed through `augment_seq()` → second projection `z_aug`
  - Concat: `[z, z_aug]` shape `(2B, 64)`
  - Positives = same class label across views and signers
  - Formula: `L = -1/|P| Σ log( exp(z·z+/τ) / Σ exp(z·z_j/τ) )`
- Total loss contribution: `0.15 × L_SupCon`
- Loss label: `L_SupCon`
- Colour: **Green**

---

### Total Loss Box (bottom centre)
```
L_total = L_CE + 0.1 × L_GRL + 0.15 × L_SupCon
```
- Three arrows feeding in from branches A, B, C with their respective weights

---

### Optimiser / Scheduler block (attach to the side)
- `AdamW` — lr_start = 3e-5, weight_decay = 5e-4
- `OneCycleLR` — max_lr = 3e-4, warmup = 30%, cosine anneal
- `GradScaler` — Mixed Precision (AMP)
- `Gradient clipping` — max norm 1.0

---

### Model Stats box (corner annotation)
- Total params: **985,831** (Exp C)  /  **914,694** (Exp A, 150-D)
- Input: `(B=64, T=200, D=223)`

---

---

## D3 — Signer-Independent Cross-Validation Protocol

### Purpose
Show how the 13-signer dataset is partitioned into 2 folds with strict signer disjointness, and how the outer/inner training loop operates.

### Canvas size suggestion
1600 × 900 px, structured in 3 rows

---

### Row 1 — Dataset Overview (top, full width)

**Master Pool box:**
- Label: `Master Pool`
- Sub: `Merged from train.npz + val.npz + test.npz`
- Stats: `3,843 samples × (200, 223)  |  30 classes  |  13 signers`
- Show a horizontal strip divided into 13 colour-coded blocks, one per signer:
  - Signers: `1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15`
  - Block width proportional to sample count per signer
- Note: `Signer IDs 7 and 14 absent from dataset`

---

### Row 2 — Two Folds Side by Side

**Fold 1 (left half):**
- Header: `Fold 1`
- Three sub-boxes:
  - **Train** (largest, blue): signers `[1, 2, 3, 5, 6, 10, 12, 13]` → **2,363 samples**
  - **Val** (medium, yellow): signers `[4, 8, 9]` → **883 samples**  
  - **Test** (red, locked icon): signers `[11, 15]` → **597 samples**  ← NEVER seen during train/val
- Key annotation: `★ Fold 1 Test Top-1: 84.59% | F1: 83.86%`

**Fold 2 (right half):**
- Header: `Fold 2`
- Three sub-boxes:
  - **Train** (largest, blue): signers `[1, 2, 4, 5, 8, 9, 11, 15]` → **2,356 samples**
  - **Val** (medium, yellow): signers `[3, 10, 12]` → **888 samples**
  - **Test** (red, locked icon): signers `[6, 13]` → **599 samples** ← NEVER seen during train/val
- Key annotation: `★ Fold 2 Test Top-1: 75.63% | F1: 75.19%`

**Between folds:** dashed vertical divider line

**Signer overlap matrix** (small grid, 13×2):
- Rows = 13 signers, Cols = Fold 1, Fold 2
- Colour: Blue=Train, Yellow=Val, Red=Test for each signer per fold
- Show that NO signer appears in BOTH test sets

---

### Row 3 — Inner Training Loop (bottom, full width)

**Left: Initialisation box**
```
Fresh BdSLSPOTER() ← random init (no weight reuse from prev fold)
AdamW + OneCycleLR + GradScaler
```

**Center: Training loop flowchart (linear left-to-right)**

```
[Epoch 1..80]
    ↓
[train_one_epoch]
  - WeightedRandomSampler (class-balanced)
  - 5 augmentations (temporal dropout, coord noise, flip, landmark dropout, temporal scale)
  - Forward pass → L_CE + λ·L_GRL + α·L_SupCon
  - AMP backward, grad clip 1.0
    ↓
[evaluate on Val set]
  - Compute Val Top-1
    ↓
[Early Stopping check]
  - If Val Top-1 > best → save checkpoint (best_fold{N}.pt)  ★
  - Patience = 15 epochs
  - If patience exhausted → STOP
    ↓
[repeat until stop or epoch 80]
```

**Right: Evaluation box**
```
Load best_fold{N}.pt
↓
evaluate(test_ldr, criterion)
→ Test Top-1, Top-5, Macro F1
```

---

### Row 4 — Cross-Fold Summary (bottom strip, full width)

**Results table (actual numbers):**

| | Fold 1 | Fold 2 | Mean | Std |
|---|:---:|:---:|:---:|:---:|
| Test Top-1 | 84.59% | 75.63% | **80.11%** | ±4.48% |
| Test Top-5 | 96.65% | 95.66% | **96.16%** | |
| Macro F1 | 83.86% | 75.19% | **79.53%** | |

**Final metric box (highlighted):**
```
Final Cross-Subject Accuracy: 80.11% ± 4.48%
```

**Key annotation boxes (3 warning notes):**
1. `⚠ No weight reuse between folds`
2. `⚠ Test signers disjoint from train AND val`
3. `⚠ No ensemble on original test.npz (data leakage prevention)`

---

---

## General Figma Design Notes

### Colour Palette
| Element | Hex |
|---|---|
| Input/data boxes | `#E8F4FD` (light blue) |
| Model layer boxes | `#1A73E8` (Google blue) |
| Classification branch | `#1565C0` (dark blue) |
| GRL/adversarial branch | `#E53935` (red) |
| SupCon branch | `#2E7D32` (green) |
| Loss boxes | `#FF6F00` (amber) |
| Train split | `#1976D2` |
| Val split | `#F9A825` |
| Test split | `#C62828` |
| Annotation/warning | `#6A1B9A` (purple) |
| Background | `#FAFAFA` |
| Arrow/connector | `#424242` |

### Typography
- Section headers: **Bold 18–20px**
- Box labels: **Bold 14px**
- Sub-labels / formulas: Regular 11–12px, monospace for code/shapes
- Shape annotations: Italic 11px

### Arrow conventions
- Solid arrow → data flow
- Dashed arrow → gradient flow (backward pass)
- Double-headed arrow → "never overlap" constraint between train/test
- Lock icon → test set is sealed/held-out

### Annotation badges
- `(B, 200, 223)` style shape labels on every arrow between model blocks
- Small `★` star next to best checkpoint save
- `⚠` warning badges for protocol constraints

---

## Summary of Key Numbers to Embed in Diagrams

| Fact | Value |
|---|---|
| Dataset | BdSLW401 |
| Total samples | 3,843 |
| Classes | 30 |
| Signers | 13 (IDs: 1,2,3,4,5,6,8,9,10,11,12,13,15) |
| Sequence length | 200 frames |
| Raw feature dim | 150 (75 landmarks × xy) |
| Geometric features | 73 |
| Total feature dim | 223 |
| Model params (full) | 985,831 |
| Model params (raw-only) | 914,694 |
| d_model | 128 |
| Attention heads | 8 |
| Transformer layers | 4 |
| d_ff | 512 |
| Batch size | 64 |
| Max epochs | 80 |
| Patience | 15 |
| λ_adv (GRL weight) | 0.1 |
| α_SupCon (SC weight) | 0.15 |
| SupCon temperature τ | 0.10 |
| Projection head dim | 64 |
| Signer discriminator classes | 15 |
| Label smoothing | 0.05 |
| Dropout | 0.20 |
| Weight decay | 5e-4 |
| Max LR | 3e-4 |
| Normalisation α | 0.85 |
| **Fold 1 Test Top-1** | **84.59%** |
| **Fold 2 Test Top-1** | **75.63%** |
| **Final Cross-Subject Acc** | **80.11% ± 4.48%** |
| Baseline (raw, CE only) | 48.41% ± 1.50% |
| Geometric CE only | 73.00% ± 2.38% |


