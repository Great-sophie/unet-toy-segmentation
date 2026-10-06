# U-Net Toy Segmentation — End-to-End Binary Image Segmentation

A compact, reproducible PyTorch project that trains a U-Net to segment simple geometric objects from noisy synthetic images. The project is intentionally small enough to run on CPU while still demonstrating the complete segmentation workflow used in biomedical image analysis.

## Why this project

The goal is to demonstrate an end-to-end segmentation pipeline:

**data generation → U-Net → training → Dice/IoU evaluation → inference → visualisation**

This repository is a learning baseline before moving to histology / biomedical image segmentation.

## Key concepts demonstrated

- Binary semantic segmentation
- U-Net encoder-decoder architecture with skip connections
- `BCEWithLogitsLoss + soft Dice loss`
- Dice coefficient and Intersection-over-Union (IoU)
- Reproducible procedural dataset generation
- CPU/GPU compatible PyTorch training
- Basic unit tests
- Prediction overlays for qualitative evaluation

## Repository structure

```text
unet-toy-segmentation/
├── src/
│   ├── dataset.py       # Procedural toy segmentation dataset
│   ├── metrics.py       # Dice, IoU, soft Dice loss
│   └── model.py         # Compact U-Net
├── tests/
│   └── test_project.py
├── train.py
├── evaluate.py
├── infer.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train:

```bash
python train.py
```

Evaluate on an independent synthetic test set:

```bash
python evaluate.py
```

Generate qualitative predictions:

```bash
python infer.py
```

Run tests:

```bash
pytest -q
```

## Metrics

This project reports:

- **Dice coefficient**: overlap between predicted and ground-truth masks
- **IoU / Jaccard index**: intersection divided by union

For binary segmentation:

```text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

## Reproduced result

A reference CPU-friendly run included with this repository achieved:

- **Test Dice:** 0.9660
- **Test IoU:** 0.9345
- **Independent test samples:** 40

Results may vary slightly by hardware and PyTorch version.


## Example output

After running `infer.py`, see:

```text
outputs/sample_predictions.png
```

The figure shows the input image, ground-truth mask, and prediction overlay.

## What I learned

1. How an encoder-decoder architecture converts image-level features into pixel-level predictions.
2. Why U-Net skip connections preserve spatial detail.
3. The difference between classification and semantic segmentation.
4. How Dice loss addresses overlap directly rather than treating pixels independently.
5. How to evaluate segmentation quantitatively and qualitatively.

## Next step: biomedical / histology segmentation

The same pipeline can be adapted to histology by replacing the synthetic dataset with H&E image-mask pairs and extending it with:

- stain normalisation / augmentation
- multi-class tissue segmentation
- patch sampling from whole-slide images
- external validation
- class-wise Dice / IoU
- uncertainty estimation

## CV-ready description

> Implemented an end-to-end semantic segmentation pipeline in PyTorch using a U-Net architecture, including reproducible data generation, combined BCE/Dice optimisation, Dice/IoU evaluation, inference, visualisation, and automated testing.

## License

MIT
