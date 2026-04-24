# Scientific Image Forgery Detection with DINOv2

A segmentation pipeline for detecting copy-move forgeries in biomedical research images, built for the [Recod.ai/LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) Kaggle competition.

# **Result:** 65th place, silver medal 🥈

![img](github.com/muhammadibrahim313/scientific-img-forgery-dinov2-65th-place-silver/blob/main/img/LUC%20-%20Scientific%20Image%20Forgery%20Detection.jpeg?raw=true)
## Overview

DINOv2 (base, 768-dim features) is used as a frozen feature extractor. A small convolutional decoder (384 → 192 → 96 → 1 channels) upsamples the 37×37 token grid back to the full 518×518 input resolution and produces pixel-level forgery masks.

Training runs in two stages:

1. **Decoder warmup.** DINOv2 is fully frozen, only the decoder is trained.
2. **Joint fine-tuning.** The last 12 DINOv2 transformer blocks are unfrozen and trained jointly with the decoder at a much smaller learning rate.

Inference uses horizontal + vertical flip TTA, gradient-enhanced adaptive thresholding, and morphological cleanup. Area and probability thresholds are grid-searched on the validation split before test-time inference.

## Kaggle artifacts

| Resource | Link |
|---|---|
| Trained weights (dataset) | https://www.kaggle.com/datasets/ibrahimqasimi/dinov2-forgery-seg-weights |
| Training notebook | https://www.kaggle.com/code/ibrahimqasimi/train-scientific-img-forgery-dinov2-65th-place |
| Inference notebook | https://www.kaggle.com/code/ibrahimqasimi/infer-scientific-img-forgery-dinov2-65th-place |
| Competition | https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection |

The fastest way to reproduce the silver-medal submission is to open the inference notebook and click "Copy and Edit" on Kaggle. All data, weights, and the DINOv2 base model are already wired up there.

## Repo structure

```
src/
  config.py      # paths, hyperparameters
  model.py       # SegDecoder + ForgerySegmenter
  data.py        # Dataset, split, dataloader factories
  inference.py   # TTA, adaptive masking, classify, RLE
  train.py       # two-stage training entry point
  infer.py       # test-set inference + submission.csv
requirements.txt
```

## Setup

```bash
git clone https://github.com/<you>/scientific-forensics-dinov2.git
cd scientific-forensics-dinov2
pip install -r requirements.txt
```

The code paths default to Kaggle mount points. If running locally, edit `src/config.py` to point `DINO_PATH`, `COMP_DIR`, and `WEIGHTS_PATH` at your local directories.

## Usage

Train from scratch:

```bash
cd src
python train.py
```

Generates `model_seg_final.pt` in the current directory.

Run inference and write `submission.csv`:

```bash
cd src
python infer.py
```

## Method notes

**Why DINOv2.** DINOv2 features are self-supervised, domain-agnostic, and carry strong locality. Copy-move forgeries produce regions that are textually identical to other parts of the same image, and DINOv2 patch features pick up this redundancy without any forgery-specific pretraining.

**Why a tiny decoder.** The 37×37 DINOv2 grid already encodes most of the spatial semantics. A heavy decoder adds parameters without adding useful signal. Three conv blocks plus bilinear upsampling is enough to recover mask boundaries.

**Why two-stage training.** Starting with a randomly initialized decoder against frozen features lets the head converge without disturbing the backbone. Unfreezing only the last 12 blocks in stage 2 gives the model room to adapt the upper representations to biomedical imagery while keeping the lower-level features (edges, textures) intact.

**Inference tricks.**

- Flip TTA (horizontal + vertical) averaged into the probability map.
- Gradient-enhanced thresholding: the Sobel magnitude of the probability map is blended in at `alpha=0.45` before thresholding, which sharpens mask boundaries around high-confidence regions.
- Morphological close then open removes speckle noise and fills in small gaps inside detected regions.
- Area and mean-probability thresholds are grid-searched over the full validation set to decide whether to output a forged mask or `"authentic"`.
## Credit

All public work on this competition, and especially [@pankajiitr](https://www.kaggle.com/pankajiitr).

## License

Code: MIT. Competition data: see the competition page. DINOv2 weights: see Meta's DINOv2 license.
