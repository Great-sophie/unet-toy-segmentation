from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SyntheticShapesDataset
from src.metrics import dice_score, iou_score, soft_dice_loss
from src.model import UNet


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    bce = nn.BCEWithLogitsLoss()
    total_loss = total_dice = total_iou = n = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = bce(logits, masks) + soft_dice_loss(logits, masks)
            if train:
                loss.backward()
                optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_dice += dice_score(logits.detach(), masks).item() * bs
        total_iou += iou_score(logits.detach(), masks).item() * bs
        n += bs

    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=20)
    parser.add_argument("--val-samples", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SyntheticShapesDataset(args.train_samples, args.image_size, seed=args.seed)
    val_ds = SyntheticShapesDataset(args.val_samples, args.image_size, seed=args.seed + 10_000)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(base=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f}"
        )
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump({"best_val_dice": best_dice, "device": str(device)}, f, indent=2)

    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Saved model to {args.output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
