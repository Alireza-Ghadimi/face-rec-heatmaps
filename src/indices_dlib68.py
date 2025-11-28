"""
Group mapping for the standard 68-point dlib landmark model.
"""

from __future__ import annotations

from typing import Dict, List

DLib_GROUPS: Dict[str, List[int]] = {
    "eye_L": list(range(36, 42)),  # mean over 36..41
    "eye_R": list(range(42, 48)),  # mean over 42..47
    "nose": [30],
    "mouth_L": [48],
    "mouth_R": [54],
    "chin": [8],
}

GROUP_ORDER = ["eye_L", "eye_R", "nose", "mouth_L", "mouth_R", "chin"]


def get_groups() -> Dict[str, List[int]]:
    """Return mapping of group name to landmark indices."""
    return DLib_GROUPS
