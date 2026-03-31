# Adversarial Patch V2 Design

**Date:** 2026-03-31
**Output:** `src/patch_V2.ipynb` — a self-contained Colab notebook

---

## Goal

Train a universal adversarial patch that maximally disrupts YOLOv8-seg's person detection and silhouette segmentation, regardless of which person or background is in the frame. The patch is placed on the shirt/torso region and physically realizable (printable).

## Root Causes Fixed from V1

1. **Dead gradient problem**: V1 targeted shirt-region anchors (~0.02 confidence → gradient ≈ 0). V2 uses Top-K loss to target wherever YOLO actually fires (head/shoulder anchors, ~0.85 confidence).
2. **Wrong training data**: V1 used stock images (`bus.jpg`, `zidane.jpg`). V2 extracts frames from actual walking videos in `data/`.
3. **Single loss axis**: V1 only attacked detection confidence. V2 also attacks the assembled segmentation mask probability directly in the patch region.

---

## Architecture

Three components inside one notebook:

### 1. Frame Extractor
- Input: video files uploaded to Colab (`test_1.mp4`, `test_2.mp4`, etc.)
- Samples ~300 frames uniformly across all videos
- Resizes to 640×640 (YOLOv8 native resolution)
- Pre-filters: skips frames where no person is detected (avoids wasted gradient steps)
- Output: list of numpy arrays

### 2. Patch Trainer
- **Patch tensor**: `[1, 3, PATCH_H, PATCH_W]`, initialized with random noise in `[0.4, 0.6]`
- **Frozen model**: YOLOv8-seg weights never updated; only patch pixels are trained
- **Optimizer**: Adam, `lr=0.01`
- **Training loop**: 200 steps × 30 sampled frames × 8 EoT transforms = 48,000 forward passes total (~15–20 min on Colab T4)

### 3. Evaluator
- Runs clean video frames and patched frames through YOLOv8
- Reports confidence suppression, detection success rate, and mask probability over the patch region
- Saves patch PNG and side-by-side visualization frames

---

## Hyperparameters (all in one config cell)

```python
NUM_FRAMES   = 300       # frames extracted from videos
PATCH_H      = 200       # patch height (pixels)
PATCH_W      = 150       # patch width (pixels)
PATCH_SCALE  = 0.3       # fraction of detected person width
SHIRT_TOP    = 0.30      # shirt region top boundary (fraction of person bbox height)
SHIRT_BOT    = 0.70      # shirt region bottom boundary
EPSILON      = 0.15      # max pixel perturbation (0–1 range)
LR           = 0.01      # Adam learning rate
NUM_STEPS    = 200       # training steps
EOT_N        = 8         # EoT transforms per frame per step
TOP_K        = 50        # number of anchors targeted by conf loss
CONF_LOSS_W  = 0.6       # weight for Top-K confidence loss
SEG_LOSS_W   = 0.4       # weight for mask suppression loss
```

---

## Raw Output Contract

YOLOv8-seg's torch model returns a nested tuple, not a dict. `yolo_raw_forward` packages it explicitly:

```python
def yolo_raw_forward(torch_model, img_tensor):
    """
    img_tensor: [1, 3, 640, 640] float, values in [0, 1]
    Returns dict:
      'preds': [1, 116, 8400]  — 4 box coords + 80 class logits + 32 mask coeffs per anchor
      'proto': [1, 32, 160, 160]  — mask prototype basis
    """
    out = torch_model(img_tensor)   # (preds_tensor, (proto_tensor, ...))
    return {'preds': out[0], 'proto': out[1][0]}
```

Both loss functions consume this dict. No other raw output access patterns used.

---

## Loss Functions

### `topk_conf_loss(raw, k=50)`
Targets where YOLO actually fires — fixes the dead gradient problem.

```python
def topk_conf_loss(raw, k=50):
    preds = raw['preds']                      # [1, 116, 8400]
    class_logits = preds[0, 4:84, :]          # [80, 8400] — class scores
    max_logits = class_logits.max(dim=0)[0]   # [8400] — best class per anchor
    conf = torch.sigmoid(max_logits)          # [8400] — confidence per anchor
    top_k = torch.topk(conf, k=k).values      # [k] — strongest anchors
    return top_k.mean()                       # minimize → suppress detection
```

Gradient signal: targets ~0.85 confidence anchors (head/shoulders) instead of ~0.02 (shirt region). The patch on the shirt affects head-region anchors through YOLOv8's large convolutional receptive fields.

### `mask_suppression_loss(raw, patch_box_mask_coords)`
Directly suppresses the assembled segmentation mask over the patch region.

```python
def mask_suppression_loss(raw, patch_box_mask_coords):
    """
    patch_box_mask_coords: (x1, y1, x2, y2) in 160x160 mask space
    """
    preds = raw['preds']                            # [1, 116, 8400]
    proto = raw['proto']                            # [1, 32, 160, 160]

    # Find highest-confidence anchor (person detection)
    class_logits = preds[0, 4:84, :]               # [80, 8400]
    conf = torch.sigmoid(class_logits.max(0)[0])   # [8400]
    best = conf.argmax()
    if conf[best] < 0.1:
        return torch.tensor(0.0, requires_grad=True)  # no detection, skip

    # Assemble mask for that detection
    mask_coeff = preds[0, 84:, best]               # [32]
    proto_flat = proto[0].reshape(32, -1)          # [32, 160*160]
    mask_logits = (mask_coeff @ proto_flat).reshape(160, 160)   # [160, 160]
    mask_prob = torch.sigmoid(mask_logits)         # [160, 160] in [0,1]

    # Crop to patch region and minimize mask probability there
    x1, y1, x2, y2 = patch_box_mask_coords
    patch_region = mask_prob[y1:y2, x1:x2]
    return patch_region.mean()                     # minimize → erase mask in patch area
```

### Combined
```python
loss = CONF_LOSS_W * topk_conf_loss(raw) + SEG_LOSS_W * mask_suppression_loss(raw, patch_box_mask)
```

---

## EoT Transforms

Applied to the patch tensor *before* pasting. Must be **differentiable** — use `kornia` for rotation and perspective warp so gradients flow back to the patch pixels. Non-differentiable transforms (PIL, numpy) break the autograd graph.

| Transform | Range | Library | Simulates |
|---|---|---|---|
| Brightness | 0.6 – 1.4× | `kornia.enhance` | lighting variation |
| Hue/saturation | ±10% | `kornia.enhance` | color shift |
| Rotation | ±15° | `kornia.geometry` | patch tilt on fabric |
| Perspective warp | slight | `kornia.geometry` | fabric curvature |
| Motion blur | kernel 3–7px | `kornia.filters` | video motion |
| Gaussian noise | σ 0–10 | manual `torch` | sensor noise |

Each applied independently with 50% probability.

---

## Training Loop

```python
patch = torch.empty(1, 3, PATCH_H, PATCH_W).uniform_(0.4, 0.6).requires_grad_(True)
patch_init = patch.data.clone()
optimizer = torch.optim.Adam([patch], lr=LR)

for step in range(NUM_STEPS):
    frames_batch = random.sample(training_frames, 30)
    optimizer.zero_grad()
    step_loss = 0.0
    n_valid = 0

    for frame in frames_batch:
        with torch.no_grad():                        # no graph for shirt detection
            shirt_box, shirt_box_mask = detect_shirt_region(yolo, frame)
        if shirt_box is None:
            continue

        for _ in range(EOT_N):
            aug_patch = apply_eot_transforms(patch)  # differentiable via kornia
            patched_img = paste_patch(frame, aug_patch, shirt_box)
            raw = yolo_raw_forward(torch_model, patched_img)
            loss = combined_loss(raw, shirt_box_mask)
            loss.backward()                          # accumulates into patch.grad
            step_loss += loss.item()
            n_valid += 1

    if n_valid > 0:
        optimizer.step()
        # Project to epsilon ball around initialization
        with torch.no_grad():
            patch.data = (patch_init + (patch.data - patch_init)
                          .clamp(-EPSILON, EPSILON)).clamp(0, 1)

    if step % 10 == 0:
        mean_loss = step_loss / max(n_valid, 1)
        print(f"Step {step:3d} | loss {mean_loss:.4f}")
```

Key details:
- `detect_shirt_region` called under `torch.no_grad()` to avoid building a graph for the pre-detection pass
- Loss accumulated across all frames × EoT samples before `optimizer.step()` (true gradient average)
- `step_loss / n_valid` gives the actual mean loss for logging

---

## Notebook Cell Structure

| Cell | Content |
|---|---|
| 1 | `!pip install ultralytics kornia` |
| 2 | Imports |
| 3 | Config block (all hyperparameters) |
| 4 | Load YOLOv8-seg, extract `torch_model`, freeze all weights |
| 5 | Frame extractor (video → filtered frames list) |
| 6 | Baseline inference (log per-frame confidence before patch) |
| 7 | `yolo_raw_forward`, `detect_shirt_region`, `paste_patch` helpers |
| 8 | EoT transform functions (kornia-based) |
| 9 | `topk_conf_loss`, `mask_suppression_loss`, `combined_loss` |
| 10 | Initialize patch tensor |
| 11 | Training loop |
| 12 | Post-training evaluation |
| 13 | Loss curve plot |
| 14 | Side-by-side visualizations (clean | patched | silhouette) |
| 15 | Save `patch_v2.png`, `patch_v2_print.png` |

---

## Evaluation Metrics

| Metric | How measured | Target |
|---|---|---|
| Confidence drop | mean(baseline_conf) − mean(patched_conf) | > 30% reduction |
| Detection suppression rate | % frames where top anchor conf < 0.5 | > 50% (vs ~5% baseline) |
| Mask probability in patch region | mean mask prob over patch box, clean vs patched | > 40% reduction |

Note: "detection suppression rate" and "mask probability" are measured only on frames where baseline detected a person, to avoid trivially good numbers on frames YOLO never fired on.
