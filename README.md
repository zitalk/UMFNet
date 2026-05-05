# Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection

**CVPR 2026**

> Official PyTorch implementation of **UMFNet**, a novel framework for RGB-T salient object detection that explicitly models per-modality uncertainty to handle spatial misalignment between visible and thermal image pairs.

---

## Overview

RGB-T salient object detection (SOD) typically assumes well-aligned image pairs. In practice, however, visible and thermal cameras have different fields of view and optics, resulting in **unaligned modality pairs** that degrade fusion quality. UMFNet addresses this by treating each modality's feature representation as a **Gaussian random variable**: uncertain (misaligned) regions produce high variance, which is then used to suppress unreliable cross-modal influence during fusion.

### Key Components

| Module | Role |
|--------|------|
| **UAM** — Uncertainty-Aware Module | Models each modality as a Gaussian distribution via a reparameterization head; predicts per-spatial-location mean and log-variance; applies KL regularization toward a standard normal prior |
| **CGM** — Confidence-Guided Modulation | Converts uncertainty estimates into confidence weights; uses channel-wise affine modulation (γ, β) and spatial gating to selectively inject thermal cues into the RGB stream |
| **Dual Swin-B Backbone** | Independent Swin Transformer encoders for RGB and thermal streams; features extracted at 4 scales (1/4 → 1/32) |
| **Progressive Decoder** | Skip-connection upsampling decoder with deep supervision at all 4 scales; each scale predicts both a saliency map and a boundary map |

### Architecture

```
RGB  ──► Swin-B Encoder ──► Fv_1..4 ─┐
                                       ├─ UAM + CGM (×4 scales) ──► Fused_1..4 ──► Decoder ──► Sal / Boundary
T    ──► Swin-B Encoder ──► Ft_1..4 ─┘
```

**UAM** encodes each scale's RGB/thermal features into stochastic latent codes, estimates a cross-modal joint distribution, and returns a calibrated thermal signal `F_t_tilde` alongside KL divergence losses.  
**CGM** computes a scalar confidence map from both modalities' log-variances and applies it to spatial-channel fusion, ensuring that uncertain thermal regions contribute less to the final representation.

---

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `timm`, `numpy`, `opencv-python`, `Pillow`, `tqdm`, `py-sod-metrics`.

---

## Datasets

| Split | Dataset | Description |
|-------|---------|-------------|
| Unaligned test | **UVT20K** | Large-scale unaligned RGB-T pairs |
| Unaligned test | **UVT2000** | Unaligned RGB-T benchmark |
| Weakly aligned | **U-VT5000 / U-VT1000 / U-VT821** | Weakly aligned variants of classic benchmarks |
| Aligned (reference) | **VT5000 / VT1000 / VT821** | Standard aligned RGB-T SOD benchmarks |

Set the following environment variables before running:

```bash
export UMFNET_SOD_ROOT=/path/to/unaligned_datasets   # UVT20K, UVT2000, WeaklyAligned/
export UMFNET_RGBTSOD_ROOT=/path/to/aligned_datasets  # VT5000, VT1000, VT821
```

---

## Training

**Using the provided script:**

```bash
export UMFNET_PRETRAIN=/path/to/swin_base_patch4_window12_384_22k.pth
export UMFNET_TEST_RGB_ROOT=/path/to/test/RGB/
export UMFNET_TEST_DEPTH_ROOT=/path/to/test/T/
export UMFNET_TEST_GT_ROOT=/path/to/test/GT/

bash run_umfnet_train.sh
```

**Key hyperparameters** (configurable via `options.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--epoch` | 100 | Total training epochs |
| `--lr` | 5e-5 | Peak learning rate |
| `--batchsize` | 8 | Batch size |
| `--trainsize` | 384 | Input resolution |
| `--lr_sched` | cosine | LR schedule (`cosine` / `step`) |
| `--warmup_epochs` | 5 | Linear LR warm-up epochs |
| `--load_pre` | — | Path to pretrained Swin-B checkpoint |

Checkpoints and logs are saved to `./Results/Result_UMFNet/` by default.

---

## Evaluation

```bash
python UMFNet_test.py \
    --pth_path ./Results/Result_UMFNet/UMFNet_best.pth \
    --datasets UVT20K UVT2000 U-VT5000 U-VT1000 U-VT821 \
    --save_predictions \
    --save_root ./test_maps
```

Metrics reported: **S-measure (Sm)**, **E-measure (Em)**, **Weighted F-measure (Fw)**. Results are written to `metrics_summary.json` under the output directory.

---

## Model Zoo

| Checkpoint | Backbone | UVT20K Sm | UVT2000 Sm |
|------------|----------|-----------|------------|
| Coming soon | Swin-B | — | — |

Pretrained Swin-B weights: [Swin Transformer Model Zoo](https://github.com/microsoft/Swin-Transformer)

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{umfnet2026,
  title     = {Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

---

## License

This repository is released for research purposes only.
