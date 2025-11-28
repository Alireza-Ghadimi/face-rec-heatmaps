"""Gaussian heatmap generation for grouped facial landmarks."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def make_group_centers(
    landmarks: np.ndarray, mapping: Dict[str, Iterable[int]]
) -> Dict[str, Tuple[float, float]]:
    """Compute group centroids from full landmark set."""
    centers = {}
    if landmarks.size == 0:
        return centers
    for name, idxs in mapping.items():
        valid = [i for i in idxs if 0 <= i < len(landmarks)]
        if not valid:
            continue
        pts = landmarks[valid]
        centers[name] = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    return centers


def rasterize_heatmaps(
    centers: Dict[str, Tuple[float, float]],
    height: int,
    width: int,
    sigma_px: float,
    groups: Iterable[str],
    add_mask: bool = False,
    ok: bool = True,
) -> np.ndarray:
    """Vectorized Gaussian heatmaps in channel-first layout."""
    g_list = list(groups)
    c = len(g_list) + (1 if add_mask else 0)
    heatmaps = np.zeros((c, height, width), dtype=np.float32)
    if not centers:
        return heatmaps

    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    denom = 2 * (sigma_px ** 2)

    for i, g in enumerate(g_list):
        if g not in centers:
            continue
        cx, cy = centers[g]
        heatmaps[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / denom)
    if add_mask:
        heatmaps[-1] = 1.0 if ok else 0.0
    heatmaps = np.clip(heatmaps, 0.0, 1.0)
    return heatmaps


def default_sigma(height: int, width: int) -> float:
    """Compute sigma in pixels with clipping to [2, 4]."""
    base = 0.02 * min(height, width)
    return float(np.clip(base, 2.0, 4.0))
