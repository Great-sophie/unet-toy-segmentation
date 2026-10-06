from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticShapesDataset
from src.metrics import dice_score, iou_score
from src.model import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.pt"))
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(base=8).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ds = SyntheticShapesDataset(args.samples, args.image_size, seed=args.seed)
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    d, j, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            bs = images.size(0)
            d += dice_score(logits, masks).item() * bs
            j += iou_score(logits, masks).item() * bs
            n += bs

    metrics = {"test_dice": d / n, "test_iou": j / n, "n_samples": n}
    print(json.dumps(metrics, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
