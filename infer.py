from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.dataset import SyntheticShapesDataset
from src.model import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("outputs/sample_predictions.png"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=8).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = SyntheticShapesDataset(args.num_samples, args.image_size, seed=30_000)
    fig, axes = plt.subplots(args.num_samples, 3, figsize=(8, 2.5 * args.num_samples))
    if args.num_samples == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for i in range(args.num_samples):
            image, mask = ds[i]
            logits = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0].cpu() >= 0.5).numpy()
            axes[i, 0].imshow(image[0], cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title("Input")
            axes[i, 1].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Ground truth")
            axes[i, 2].imshow(image[0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(pred, alpha=0.45, cmap="viridis")
            axes[i, 2].set_title("Prediction overlay")
            for ax in axes[i]:
                ax.axis("off")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
