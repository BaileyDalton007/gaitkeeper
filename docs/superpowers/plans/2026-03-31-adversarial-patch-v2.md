# Adversarial Patch V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `src/patch_V2.ipynb`, a self-contained Google Colab notebook that trains a universal adversarial patch against YOLOv8-seg using a single input image, with a dual loss (Top-K confidence suppression + mask probability suppression) and differentiable EoT transforms via kornia.

**Architecture:** A single Jupyter notebook with 15 cells. The patch tensor is the only learnable parameter; YOLOv8-seg weights are frozen throughout. Training uses Adam with gradient accumulation across EoT transforms before each optimizer step. A `yolo_raw_forward` helper explicitly packages the nested tuple output into a dict so all loss functions consume a stable contract.

> **Scope note:** This plan intentionally uses a single training image (debug/verification phase). The spec describes a multi-frame video pipeline as the eventual goal — that is a follow-up step once this single-image loop is confirmed to work. EoT transforms (8 per step) compensate for the lack of frame diversity during this phase.

**Tech Stack:** Python 3.10, PyTorch, Ultralytics YOLOv8-seg (`yolov8n-seg`), kornia (differentiable vision transforms), OpenCV, matplotlib

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/patch_V2.ipynb` | Create | Full training notebook |

No other files are created or modified.

---

### Task 1: Notebook scaffold — installs, imports, config

**Files:**
- Create: `src/patch_V2.ipynb`

- [ ] **Step 1: Create the notebook file with Cell 1 (installs)**

```python
# Cell 1 — Installs
!pip install ultralytics kornia --quiet
```

- [ ] **Step 2: Add Cell 2 (imports)**

```python
# Cell 2 — Imports
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import kornia.augmentation as K
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
```

- [ ] **Step 3: Add Cell 3 (config block — all hyperparameters in one place)**

```python
# Cell 3 — Config
# ── Training ──────────────────────────────────────────
PATCH_H       = 200      # patch height in pixels
PATCH_W       = 150      # patch width in pixels
PATCH_SCALE   = 0.3      # fraction of detected person width used for patch
SHIRT_TOP     = 0.30     # top of shirt region as fraction of person bbox height
SHIRT_BOT     = 0.70     # bottom of shirt region as fraction of person bbox height
EPSILON       = 0.15     # max pixel delta from initialization (0–1 range)
LR            = 0.01     # Adam learning rate
NUM_STEPS     = 200      # total training steps (use 30 for debug run)
EOT_N         = 8        # EoT transforms per step
TOP_K         = 50       # top anchors to target in confidence loss
CONF_LOSS_W   = 0.6      # weight for Top-K confidence loss
SEG_LOSS_W    = 0.4      # weight for mask suppression loss

# ── Input ──────────────────────────────────────────────
# Upload one image to Colab and set this path.
# Any image with a clearly visible person works (e.g., someone walking).
IMAGE_PATH    = "person.jpg"

# ── Model ──────────────────────────────────────────────
YOLO_MODEL    = "yolov8n-seg"   # nano for speed; swap yolov8x-seg for eval
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
```

- [ ] **Step 4: Verify cells render without error by running Cell 1–3 in Colab**

Expected: no import errors, prints `Using device: cuda`

- [ ] **Step 5: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: patch_V2 scaffold — installs, imports, config"
```

---

### Task 2: Load YOLOv8-seg and define `yolo_raw_forward`

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 4 (load model and freeze weights)**

```python
# Cell 4 — Load YOLOv8-seg, freeze weights
yolo = YOLO(YOLO_MODEL)
torch_model = yolo.model.to(DEVICE)
torch_model.eval()

# Freeze all weights — only patch pixels will be updated
for p in torch_model.parameters():
    p.requires_grad_(False)

print(f"Model loaded: {YOLO_MODEL} | Parameters frozen: {sum(p.numel() for p in torch_model.parameters()):,}")
```

- [ ] **Step 2: Add Cell 5 (`yolo_raw_forward` — stable output contract)**

```python
# Cell 5 — Raw forward pass helper
def yolo_raw_forward(img_tensor):
    """
    img_tensor: [1, 3, 640, 640] float32 on DEVICE, values in [0, 1]
    Returns dict:
        'preds': [1, 116, 8400]  — 4 box + 80 class logits + 32 mask coeffs per anchor
        'proto': [1, 32, 160, 160] — mask prototype basis
    """
    out = torch_model(img_tensor)
    return {'preds': out[0], 'proto': out[1][0]}


def img_to_tensor(img_np):
    """uint8 HxWx3 numpy → [1, 3, 640, 640] float32 tensor on DEVICE"""
    img = cv2.resize(img_np, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(DEVICE)
```

- [ ] **Step 3: Verify by running a test forward pass**

```python
# Quick sanity check — run in its own cell temporarily, delete after
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
raw = yolo_raw_forward(img_to_tensor(test_img))
print("preds shape:", raw['preds'].shape)   # expect [1, 116, 8400]
print("proto shape:", raw['proto'].shape)   # expect [1, 32, 160, 160]
```

Expected output:
```
preds shape: torch.Size([1, 116, 8400])
proto shape: torch.Size([1, 32, 160, 160])
```

- [ ] **Step 4: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: load YOLOv8-seg and raw forward helper"
```

---

### Task 3: Shirt detection helper and `paste_patch`

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 6 (shirt region detection and patch pasting)**

```python
# Cell 6 — Shirt detection and patch placement
def detect_shirt_region(img_np):
    """
    Detects the highest-confidence person and returns:
        shirt_box_img: (x1, y1, x2, y2) in 640x640 image coordinates
        shirt_box_mask: (x1, y1, x2, y2) in 160x160 mask coordinates
    Returns (None, None) if no person detected.
    """
    with torch.no_grad():
        results = yolo(cv2.resize(img_np, (640, 640)), verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    # Find highest-confidence person detection (class 0)
    boxes = results[0].boxes
    person_mask = (boxes.cls == 0)
    if not person_mask.any():
        return None, None

    best_idx = boxes.conf[person_mask].argmax()
    box = boxes.xyxy[person_mask][best_idx].cpu().numpy()  # [x1, y1, x2, y2]

    x1, y1, x2, y2 = box
    h = y2 - y1
    # Shirt region: middle vertical band of the person bbox
    sx1 = int(x1)
    sy1 = int(y1 + h * SHIRT_TOP)
    sx2 = int(x2)
    sy2 = int(y1 + h * SHIRT_BOT)

    # Scale to mask resolution (160x160 = 640/4)
    scale = 160 / 640
    mx1, my1 = int(sx1 * scale), int(sy1 * scale)
    mx2, my2 = int(sx2 * scale), int(sy2 * scale)

    # Clamp to valid range
    mx1, my1 = max(0, mx1), max(0, my1)
    mx2, my2 = min(159, mx2), min(159, my2)

    if mx2 <= mx1 or my2 <= my1:
        return None, None

    return (sx1, sy1, sx2, sy2), (mx1, my1, mx2, my2)


def paste_patch(img_tensor, patch, shirt_box):
    """
    img_tensor: [1, 3, 640, 640] float32
    patch: [1, 3, PATCH_H, PATCH_W] float32, differentiable
    shirt_box: (x1, y1, x2, y2) in image coords
    Returns: patched image tensor [1, 3, 640, 640]
    """
    x1, y1, x2, y2 = shirt_box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return img_tensor

    # Resize patch to shirt region dimensions
    patch_resized = F.interpolate(patch, size=(h, w), mode='bilinear', align_corners=False)

    # Clone and paste (preserve gradient graph through patch_resized)
    patched = img_tensor.clone()
    patched[:, :, y1:y2, x1:x2] = patch_resized
    return patched
```

- [ ] **Step 2: Test on a real image — upload `person.jpg` to Colab and verify**

```python
img = cv2.imread(IMAGE_PATH)
box_img, box_mask = detect_shirt_region(img)
print("Image shirt box:", box_img)
print("Mask shirt box:", box_mask)
```

Expected: two tuples of integers (not None). If None, try a clearer image.

- [ ] **Step 3: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: shirt detection and patch paste helpers"
```

---

### Task 4: Loss functions

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 7 (`topk_conf_loss`)**

```python
# Cell 7 — Loss functions
def topk_conf_loss(raw, k=TOP_K):
    """
    Minimize the mean confidence of the k highest-confidence anchors.
    Targets wherever YOLO actually fires (head/shoulders ~0.85 conf),
    not the shirt region which has near-zero confidence and dead gradients.
    """
    preds = raw['preds']                         # [1, 116, 8400]
    class_logits = preds[0, 4:84, :]             # [80, 8400]
    max_logits = class_logits.max(dim=0)[0]      # [8400]
    conf = torch.sigmoid(max_logits)             # [8400]
    top_k_vals = torch.topk(conf, k=k).values    # [k]
    return top_k_vals.mean()
```

- [ ] **Step 2: Add Cell 8 (`mask_suppression_loss`)**

```python
def mask_suppression_loss(raw, patch_box_mask):
    """
    Assemble the segmentation mask for the highest-confidence detection,
    then minimize its probability in the patch region.
    Directly degrades silhouette quality where the patch sits.
    """
    preds = raw['preds']                              # [1, 116, 8400]
    proto = raw['proto']                              # [1, 32, 160, 160]

    class_logits = preds[0, 4:84, :]
    conf = torch.sigmoid(class_logits.max(0)[0])      # [8400]
    best = conf.argmax()

    # Skip if no confident detection — return zero loss with grad
    if conf[best].item() < 0.1:
        return torch.zeros(1, device=DEVICE, requires_grad=True).squeeze()

    mask_coeff = preds[0, 84:, best]                  # [32]
    proto_flat = proto[0].reshape(32, -1)             # [32, 160*160]
    mask_logits = (mask_coeff @ proto_flat).reshape(160, 160)   # [160, 160]
    mask_prob = torch.sigmoid(mask_logits)            # [160, 160]

    x1, y1, x2, y2 = patch_box_mask
    patch_region = mask_prob[y1:y2, x1:x2]

    if patch_region.numel() == 0:
        return torch.zeros(1, device=DEVICE, requires_grad=True).squeeze()

    return patch_region.mean()


def combined_loss(raw, patch_box_mask):
    conf_loss = topk_conf_loss(raw)
    seg_loss  = mask_suppression_loss(raw, patch_box_mask)
    return CONF_LOSS_W * conf_loss + SEG_LOSS_W * seg_loss
```

- [ ] **Step 3: Verify losses run and return scalars**

```python
# Verify loss functions work on real input (run in a temp cell, delete after)
img = cv2.imread(IMAGE_PATH)
t = img_to_tensor(img)
box_img, box_mask = detect_shirt_region(img)

raw = yolo_raw_forward(t)
c = topk_conf_loss(raw)
s = mask_suppression_loss(raw, box_mask)
print(f"conf_loss={c.item():.4f}  seg_loss={s.item():.4f}")
```

Expected: both are floats between 0 and 1. If seg_loss is 0.0, the person confidence is below 0.1 — use a clearer image.

- [ ] **Step 4: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: topk_conf_loss and mask_suppression_loss"
```

---

### Task 5: EoT transforms (differentiable via kornia)

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 9 (EoT transform pipeline)**

```python
# Cell 9 — Differentiable EoT transforms via kornia
def apply_eot_transforms(patch):
    """
    patch: [1, 3, PATCH_H, PATCH_W] float32, grad-enabled
    Returns: transformed patch, same shape, gradient preserved
    All transforms use kornia or raw torch — no PIL/numpy (which break autograd).
    """
    p = patch  # keep reference for conditional applications

    # Brightness: multiply by scalar in [0.6, 1.4]
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)
        p = kornia.enhance.adjust_brightness(p, factor)

    # Hue/Saturation jitter ±10%
    if random.random() < 0.5:
        p = K.ColorJitter(hue=0.1, saturation=0.1)(p)

    # Rotation ±15°
    if random.random() < 0.5:
        angle = torch.tensor([random.uniform(-15, 15)], device=DEVICE)
        p = kornia.geometry.rotate(p, angle)

    # Perspective warp (slight fabric curvature)
    if random.random() < 0.5:
        p = K.RandomPerspective(distortion_scale=0.2, p=1.0)(p)

    # Motion blur
    if random.random() < 0.5:
        ksize = random.choice([3, 5, 7])
        angle = random.uniform(0, 360)
        p = kornia.filters.motion_blur(p, kernel_size=ksize, angle=angle, direction=0.0)

    # Gaussian noise (pure torch, stays differentiable)
    if random.random() < 0.5:
        sigma = random.uniform(0, 10) / 255.0
        p = p + torch.randn_like(p) * sigma

    return p.clamp(0, 1)
```

- [ ] **Step 2: Verify gradient flows through EoT transform**

```python
# Verify autograd is intact through transforms (temp cell, delete after)
dummy = torch.rand(1, 3, PATCH_H, PATCH_W, device=DEVICE, requires_grad=True)
out = apply_eot_transforms(dummy)
out.mean().backward()
print("EoT grad check:", dummy.grad is not None)   # must print True
```

Expected: `EoT grad check: True`

- [ ] **Step 3: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: differentiable EoT transforms via kornia"
```

---

### Task 6: Baseline inference (before-patch benchmark)

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 10 (baseline inference)**

```python
# Cell 10 — Baseline: measure confidence before training
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH} — check Colab upload")

img_tensor = img_to_tensor(img)

with torch.no_grad():
    raw_baseline = yolo_raw_forward(img_tensor)

baseline_conf = topk_conf_loss(raw_baseline).item()
print(f"Baseline Top-{TOP_K} mean confidence: {baseline_conf:.4f}")
# Expected: ~0.6–0.9 for a clear person image
```

- [ ] **Step 2: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: baseline inference cell"
```

---

### Task 7: Patch initialization and training loop

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 11 (patch initialization)**

```python
# Cell 11 — Initialize patch
torch.manual_seed(42)
patch = torch.empty(1, 3, PATCH_H, PATCH_W, device=DEVICE).uniform_(0.4, 0.6)
patch.requires_grad_(True)
patch_init = patch.data.clone()  # store for epsilon projection

optimizer = torch.optim.Adam([patch], lr=LR)
print(f"Patch initialized: shape {list(patch.shape)}, range [{patch.min():.2f}, {patch.max():.2f}]")
```

- [ ] **Step 2: Add Cell 12 (training loop)**

```python
# Cell 12 — Training loop
img = cv2.imread(IMAGE_PATH)
box_img, box_mask = detect_shirt_region(img)
if box_img is None:
    raise RuntimeError("No person detected in training image — use a clearer image")

img_tensor = img_to_tensor(img)  # [1, 3, 640, 640], no grad needed
loss_history = []

for step in range(NUM_STEPS):
    optimizer.zero_grad()
    step_loss = 0.0

    for _ in range(EOT_N):
        aug_patch = apply_eot_transforms(patch)
        patched_img = paste_patch(img_tensor, aug_patch, box_img)
        raw = yolo_raw_forward(patched_img)
        loss = combined_loss(raw, box_mask)
        loss.backward()
        step_loss += loss.item()

    optimizer.step()

    # Project patch to epsilon ball around initialization
    with torch.no_grad():
        delta = (patch.data - patch_init).clamp(-EPSILON, EPSILON)
        patch.data = (patch_init + delta).clamp(0, 1)

    mean_loss = step_loss / EOT_N
    loss_history.append(mean_loss)

    if step % 10 == 0:
        print(f"Step {step:3d}/{NUM_STEPS} | loss {mean_loss:.4f}")

print("Training complete.")
```

- [ ] **Step 3: Run a debug pass (NUM_STEPS=5) to confirm loss decreases**

Temporarily set `NUM_STEPS = 5` in the config cell and run. Expected output:
```
Step   0/5 | loss 0.XXXX
Step   0/5 already passed, Step 5 should show lower loss
```
Loss should decrease across steps. If it increases, check that `patch.requires_grad_(True)` is set and gradients are not being zeroed by a numpy conversion.

- [ ] **Step 4: Restore `NUM_STEPS = 200` in config and commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: patch init and training loop"
```

---

### Task 8: Evaluation, visualization, and save outputs

**Files:**
- Modify: `src/patch_V2.ipynb`

- [ ] **Step 1: Add Cell 13 (post-training evaluation)**

```python
# Cell 13 — Evaluation
img = cv2.imread(IMAGE_PATH)
img_tensor = img_to_tensor(img)
box_img, box_mask = detect_shirt_region(img)

with torch.no_grad():
    # Clean image metrics
    raw_clean = yolo_raw_forward(img_tensor)
    clean_conf = topk_conf_loss(raw_clean).item()
    clean_mask_prob = mask_suppression_loss(raw_clean, box_mask).item()

    # Patched image metrics
    patch_applied = paste_patch(img_tensor, patch, box_img)
    raw_patched = yolo_raw_forward(patch_applied)
    patched_conf = topk_conf_loss(raw_patched).item()
    patched_mask_prob = mask_suppression_loss(raw_patched, box_mask).item()

print("─" * 45)
print(f"{'Metric':<30} {'Clean':>6}  {'Patched':>7}")
print("─" * 45)
print(f"{'Top-K conf (lower=better)':<30} {clean_conf:>6.4f}  {patched_conf:>7.4f}")
print(f"{'Mask prob in patch region':<30} {clean_mask_prob:>6.4f}  {patched_mask_prob:>7.4f}")
conf_drop_pct = (clean_conf - patched_conf) / clean_conf * 100
print("─" * 45)
print(f"Confidence drop: {conf_drop_pct:.1f}%")
```

- [ ] **Step 2: Add Cell 14 (loss curve and side-by-side visualization)**

```python
# Cell 14 — Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loss curve
axes[0].plot(loss_history)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

# Clean image with YOLO overlay
results_clean = yolo(cv2.resize(cv2.imread(IMAGE_PATH), (640, 640)), verbose=False)
axes[1].imshow(cv2.cvtColor(results_clean[0].plot(), cv2.COLOR_BGR2RGB))
axes[1].set_title("Clean — YOLO detections")
axes[1].axis('off')

# Patched image with YOLO overlay
patch_np = patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
patch_np = (patch_np * 255).astype(np.uint8)
img_patched_np = cv2.resize(cv2.imread(IMAGE_PATH), (640, 640)).copy()
x1, y1, x2, y2 = box_img
patch_resized = cv2.resize(patch_np, (x2 - x1, y2 - y1))
img_patched_np[y1:y2, x1:x2] = cv2.cvtColor(patch_resized, cv2.COLOR_RGB2BGR)
results_patched = yolo(img_patched_np, verbose=False)
axes[2].imshow(cv2.cvtColor(results_patched[0].plot(), cv2.COLOR_BGR2RGB))
axes[2].set_title("Patched — YOLO detections")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("eval_v2.png", dpi=150)
plt.show()
print("Saved eval_v2.png")
```

- [ ] **Step 3: Add Cell 15 (save patch)**

```python
# Cell 15 — Save outputs
patch_np = patch.detach().cpu().squeeze().permute(1, 2, 0).numpy()
patch_np = (patch_np * 255).astype(np.uint8)

# Native resolution
cv2.imwrite("patch_v2.png", cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))

# Print-ready upscaled
patch_print = cv2.resize(patch_np, (300, 300), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("patch_v2_print.png", cv2.cvtColor(patch_print, cv2.COLOR_RGB2BGR))

print("Saved: patch_v2.png, patch_v2_print.png")
print(f"Patch pixel range: [{patch_np.min()}, {patch_np.max()}]")
```

- [ ] **Step 4: Run full notebook end-to-end and verify**

Expected end state:
- Loss decreases over 200 steps
- Confidence drop ≥ 20% (30%+ is the target)
- `patch_v2.png` and `patch_v2_print.png` saved
- Side-by-side shows fewer/weaker detections on patched image

- [ ] **Step 5: Commit**

```bash
git add src/patch_V2.ipynb
git commit -m "feat: evaluation, visualization, and save cells — patch_V2 complete"
```

---

## Success Criteria

| Check | Pass condition |
|---|---|
| Loss decreasing | `loss_history[-1] < loss_history[0]` |
| Confidence suppression | ≥ 20% drop (clean vs. patched Top-K conf) |
| Mask suppression | ≥ 20% drop in mask prob over patch region |
| Outputs exist | `patch_v2.png` and `patch_v2_print.png` saved |
| Notebook runs clean | All cells execute top-to-bottom without errors on Colab T4 |
