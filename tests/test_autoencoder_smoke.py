import os
import tempfile
import numpy as np
import torch

from autoencoder_data import AutoencoderDataset
from autoencoder import LandmarkAutoencoder


def _make_fake_chunk(path: str, classes=2, samples_per_class=3):
    rows = []
    idx = 0
    for c in range(classes):
        cid = f"person_{c}"
        for s in range(samples_per_class):
            # minimal fields: idx, class_id, h, w, raw(128), norm(128), yaw, pitch, roll
            raw = np.zeros(128, dtype=np.float32)
            norm = np.random.randn(128).astype(np.float32)
            yaw = np.random.uniform(-10, 10)
            pitch = np.random.uniform(-10, 10)
            roll = np.random.uniform(-10, 10)
            row = [idx, cid, 112.0, 112.0]
            row.extend(raw.tolist())
            row.extend(norm.tolist())
            row.extend([yaw, pitch, roll])
            rows.append(row)
            idx += 1
    np.save(path, np.asarray(rows, dtype=object))


def test_autoencoder_dataset_and_forward():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_dir = os.path.join(tmp, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_path = os.path.join(chunk_dir, "chunk_00000.npy")
        _make_fake_chunk(chunk_path)

        ds = AutoencoderDataset(chunk_dir)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape[0] == 131  # 128 landmarks + 3 pose
        assert y.shape[0] == 128

        model = LandmarkAutoencoder(landmark_dim=128, pose_dim=3, hidden=64)
        xb = torch.stack([x, x])
        out = model(xb)
        assert out.shape == (2, 128)
