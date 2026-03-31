# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gaitkeeper is an adversarial ML research project that defeats gait-based biometric identification by applying visual patches to clothing regions in video. The pipeline: segment clothing → generate adversarial patch → apply to video → test against gait recognition model.

## Environment Setup

```bash
conda env create -f conda_env.yml
conda activate gaitkeeper

# Verify GPU is available
python gpu_test.py
```

Requires CUDA 11.8 and Python 3.10. See `conda_guide.md` for exporting updated environments.

## Architecture

Development is notebook-driven. The core reusable logic lives in `video_segmentation.py`, which all notebooks import.

### Pipeline Stages

1. **Segmentation** (`preprocessing.ipynb`, `video_segmentation.ipynb`)
   Uses HuggingFace SegFormer B2 (`mattmdjaga/segformer_b2_clothes`) to extract per-frame binary masks for clothing regions. Target label `4` = upper-clothes/shirt. Output is a boolean tensor of shape `[num_frames, 1024, 1024]`.

2. **Patch Application** (`src/patch_V1.ipynb`)
   Overlays adversarial textures/patterns onto masked clothing regions. `write_mask_overlay_video2()` supports jitter transforms (lighting, rotation, translation, Gaussian noise) to make patches robust. Outputs go to `output/` or `demos/`.

3. **Feature Analysis** (`feature_heatmaps.ipynb`)
   GradCAM visualization on the gait recognition model to identify which clothing features matter. Frames saved to `external/GaitRecognitionSystem/gradcam_frames/`.

### Key Files

- **`video_segmentation.py`** — core module with all segmentation and overlay functions
  - `load_segformer_model()` — loads model to GPU
  - `get_single_mask_from_video(video_path, target_label=4, target_resolution=1024)` — main segmentation entry point
  - `write_mask_overlay_video2()` — applies patches with jitter
  - `apply_jitter()` — lighting/rotation/translation/noise transforms

- **`external/GaitRecognitionSystem/`** — git submodule providing the target gait recognition model used for testing and GradCAM extraction

- **`data/`** — raw input videos and images; `data_processed/` — preprocessed outputs

### Segmentation Label Reference

SegFormer B2 clothes uses 18 classes; label `4` is upper-clothes (shirt/jacket). Masks are processed at 1024×1024 then resized to the source video resolution before overlay.
