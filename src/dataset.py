from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticShapesDataset(Dataset):
    """Procedurally generated binary segmentation dataset.

    Each image contains 1-3 simple foreground objects (circles or rectangles)
    on a noisy grayscale background. A deterministic seed makes each sample
    reproducible across runs.
    """

    def __init__(self, n_samples: int = 400, image_size: int = 96, seed: int = 42):
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        h = w = self.image_size

        image = rng.normal(0.06, 0.015, size=(h, w)).astype(np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.mgrid[:h, :w]

        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                radius = int(rng.integers(max(5, h // 12), max(7, h // 5)))
                cy = int(rng.integers(radius, h - radius))
                cx = int(rng.integers(radius, w - radius))
                obj = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            else:
                rh = int(rng.integers(max(8, h // 10), max(12, h // 3)))
                rw = int(rng.integers(max(8, w // 10), max(12, w // 3)))
                y0 = int(rng.integers(0, h - rh))
                x0 = int(rng.integers(0, w - rw))
                obj = np.zeros((h, w), dtype=bool)
                obj[y0 : y0 + rh, x0 : x0 + rw] = True

            mask[obj] = 1.0
            intensity = float(rng.uniform(0.82, 0.98))
            image[obj] = intensity + rng.normal(0, 0.01, size=int(obj.sum()))

        image = np.clip(image, 0.0, 1.0)

        image_t = torch.from_numpy(image[None, ...])
        mask_t = torch.from_numpy(mask[None, ...])
        return image_t, mask_t
