"""Joint image + landmark transforms."""

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


class JointTransform:
    """Resize, optional center crop, random flip, and normalization."""

    def __init__(self, img_size: int = 112, hflip_p: float = 0.5) -> None:
        self.img_size = img_size
        self.hflip_p = hflip_p
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

    def apply(
        self, image: Image.Image, landmarks: np.ndarray, groups: Dict[str, list]
    ) -> Tuple[torch.Tensor, np.ndarray, Dict[str, list]]:
        """Apply transforms to image and landmark coordinates."""
        orig_w, orig_h = image.size
        scale = self.img_size / float(min(orig_w, orig_h))
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        landmarks_scaled = landmarks.copy()
        if landmarks_scaled.size > 0:
            landmarks_scaled[:, 0] *= scale
            landmarks_scaled[:, 1] *= scale

        # Center crop to square
        left = max(0, (new_w - self.img_size) // 2)
        top = max(0, (new_h - self.img_size) // 2)
        image = image.crop((left, top, left + self.img_size, top + self.img_size))
        if landmarks_scaled.size > 0:
            landmarks_scaled[:, 0] -= left
            landmarks_scaled[:, 1] -= top
            landmarks_scaled[:, 0] = np.clip(landmarks_scaled[:, 0], 0, self.img_size - 1)
            landmarks_scaled[:, 1] = np.clip(landmarks_scaled[:, 1], 0, self.img_size - 1)

        if random.random() < self.hflip_p:
            image = F.hflip(image)
            if landmarks_scaled.size > 0:
                landmarks_scaled[:, 0] = self.img_size - 1 - landmarks_scaled[:, 0]
            groups = self._swap_groups(groups)

        tensor = F.to_tensor(image)
        tensor = F.normalize(tensor, self.mean, self.std)
        return tensor, landmarks_scaled, groups

    def _swap_groups(self, groups: Dict[str, list]) -> Dict[str, list]:
        swapped = dict(groups)
        if "eye_L" in swapped and "eye_R" in swapped:
            swapped["eye_L"], swapped["eye_R"] = swapped["eye_R"], swapped["eye_L"]
        if "mouth_L" in swapped and "mouth_R" in swapped:
            swapped["mouth_L"], swapped["mouth_R"] = swapped["mouth_R"], swapped["mouth_L"]
        return swapped
