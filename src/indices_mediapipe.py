"""
Stable MediaPipe FaceMesh landmark subsets grouped for heatmaps.

These points are chosen because they are consistently localized and less
prone to drift: eye centers from rims, nose tip, mouth corners, and chin tip.
"""

from __future__ import annotations

from typing import Dict, List

# Grouped indices for MediaPipe FaceMesh (468 points).
MEDIAPIPE_GROUPS: Dict[str, List[int]] = {
    "eye_L": [33, 133, 159, 145],
    "eye_R": [362, 263, 386, 374],
    "nose": [1],  # fallback index 4 handled downstream
    "mouth_L": [78],
    "mouth_R": [308],
    "chin": [152],
}

GROUP_ORDER = ["eye_L", "eye_R", "nose", "mouth_L", "mouth_R", "chin"]


def get_groups() -> Dict[str, List[int]]:
    """Return mapping of group name to landmark indices."""
    return MEDIAPIPE_GROUPS
