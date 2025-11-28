import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

# Ensure local src/ is importable when running tests without installation
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import FaceDataset
from src.model import FaceModel


def _make_dummy_dataset(root: str, img_size: int = 112) -> None:
    for cls in ["alice", "bob"]:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(2):
            arr = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(cls_dir, f"{i}.jpg"))


def test_one_batch_forward():
    tmp = tempfile.mkdtemp()
    try:
        _make_dummy_dataset(tmp, img_size=128)
        ds = FaceDataset(
            root=tmp,
            img_size=112,
            use_heatmaps=True,
            extractor="mediapipe",
            cache_heatmaps=False,
            cache_landmarks=False,
            add_mask_channel=True,
            hflip_p=0.0,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
        images, labels, _ = next(iter(loader))
        k = len(ds.group_order)
        expected_ch = 3 + k + 1
        assert images.shape[1] == expected_ch
        model = FaceModel(
            num_classes=len(ds.dataset.classes),
            in_ch=expected_ch,
            backbone="resnet50",
            pretrained=False,
            use_cosface=False,
        )
        out = model(images, labels)
        assert out.shape[0] == images.size(0)
    finally:
        shutil.rmtree(tmp)
