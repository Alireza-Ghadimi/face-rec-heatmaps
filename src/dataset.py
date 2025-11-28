"""Dataset that stacks RGB with landmark heatmap channels."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .heatmaps import default_sigma, make_group_centers, rasterize_heatmaps
from .indices_dlib68 import GROUP_ORDER as DLIB_GROUP_ORDER, get_groups as dlib_groups
from .indices_mediapipe import GROUP_ORDER as MP_GROUP_ORDER, get_groups as mp_groups
from .landmarks import LandmarkExtractor
from .transforms import JointTransform


class FilteredImageFolder(ImageFolder):
    """ImageFolder that ignores hidden/cache directories."""

    def find_classes(self, directory: str):
        classes, class_to_idx = super().find_classes(directory)
        keep = [c for c in classes if not c.startswith(".cache") and not c.startswith(".")]
        class_to_idx = {c: i for i, c in enumerate(keep)}
        return keep, class_to_idx


class FaceDataset(Dataset):
    """ImageFolder-style dataset that augments RGB with landmark heatmaps."""

    def __init__(
        self,
        root: str,
        img_size: int = 112,
        use_heatmaps: bool = True,
        extractor: str = "mediapipe",
        cache_heatmaps: bool = False,
        cache_landmarks: bool = False,
        add_mask_channel: bool = True,
        hflip_p: float = 0.5,
        group_all_landmarks: bool = False,
    ) -> None:
        self.root = root
        self.img_size = img_size
        self.use_heatmaps = use_heatmaps
        self.cache_heatmaps = cache_heatmaps
        self.cache_landmarks = cache_landmarks
        self.add_mask_channel = add_mask_channel
        self.group_all_landmarks = group_all_landmarks
        self.dataset = FilteredImageFolder(root)
        self.extractor = LandmarkExtractor(extractor)
        self.transform = JointTransform(img_size=img_size, hflip_p=hflip_p)
        if extractor == "dlib":
            if group_all_landmarks:
                self.group_mapping = {f"pt_{i}": [i] for i in range(68)}
                self.group_order = list(self.group_mapping.keys())
            else:
                self.group_mapping = dlib_groups()
                self.group_order = DLIB_GROUP_ORDER
        else:
            if group_all_landmarks:
                self.group_mapping = {f"pt_{i}": [i] for i in range(468)}
                self.group_order = list(self.group_mapping.keys())
            else:
                self.group_mapping = mp_groups()
                self.group_order = MP_GROUP_ORDER
        # Cache inside dataset root, but directories are ignored by FilteredImageFolder.
        self.hm_cache_dir = os.path.join(root, ".cache_heatmaps")
        self.lm_cache_dir = os.path.join(root, ".cache_landmarks")
        os.makedirs(self.hm_cache_dir, exist_ok=True)
        os.makedirs(self.lm_cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        path, label = self.dataset.samples[idx]
        image = Image.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        landmarks, ok = self._load_landmarks(path, bgr)
        tensor, landmarks_t, groups = self.transform.apply(image, landmarks, self.group_mapping)

        heatmap_tensor = torch.empty(0)
        if self.use_heatmaps:
            heatmap_tensor = self._build_heatmaps(path, landmarks_t, groups, ok)
        stacked = tensor if not self.use_heatmaps else torch.cat([tensor, heatmap_tensor], dim=0)

        meta = {"path": path, "ok": ok}
        return stacked, label, meta

    def _build_heatmaps(
        self, path: str, landmarks: np.ndarray, groups: Dict[str, list], ok: bool
    ) -> torch.Tensor:
        key = self._cache_key(path)
        cached = os.path.join(self.hm_cache_dir, f"{key}.npy")
        if self.cache_heatmaps:
            if os.path.exists(cached):
                arr = np.load(cached)
                return torch.from_numpy(arr)

        centers = make_group_centers(landmarks, groups)
        sigma = default_sigma(self.img_size, self.img_size)
        arr = rasterize_heatmaps(centers, self.img_size, self.img_size, sigma, self.group_order, self.add_mask_channel, ok)
        if self.cache_heatmaps:
            np.save(cached, arr)
        return torch.from_numpy(arr)

    def _load_landmarks(self, path: str, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        key = self._cache_key(path)
        cached = os.path.join(self.lm_cache_dir, f"{key}.json")
        if self.cache_landmarks and os.path.exists(cached):
            with open(cached, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = np.asarray(data.get("points", []), dtype=np.float32)
            return pts, bool(data.get("ok", False))

        landmarks, ok = self.extractor(image_bgr)
        if self.cache_landmarks:
            with open(cached, "w", encoding="utf-8") as f:
                json.dump({"points": landmarks.tolist(), "ok": ok}, f)
        return landmarks, ok

    def _cache_key(self, path: str) -> str:
        rel = os.path.relpath(path, self.root)
        h = hashlib.sha1(rel.encode("utf-8")).hexdigest()
        flags = f"{self.img_size}_{int(self.use_heatmaps)}_{int(self.add_mask_channel)}"
        return f"{flags}_{h}"
