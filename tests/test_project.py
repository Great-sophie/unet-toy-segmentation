import torch

from src.dataset import SyntheticShapesDataset
from src.metrics import dice_score, iou_score
from src.model import UNet


def test_dataset_shapes():
    image, mask = SyntheticShapesDataset(n_samples=1, image_size=64)[0]
    assert image.shape == (1, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_unet_output_shape():
    model = UNet(base=8)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_perfect_prediction_metrics():
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = torch.where(targets > 0, torch.tensor(20.0), torch.tensor(-20.0))
    assert torch.isclose(dice_score(logits, targets), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(iou_score(logits, targets), torch.tensor(1.0), atol=1e-5)
