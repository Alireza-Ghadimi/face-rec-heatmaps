"""Simple head pose estimation using 2D landmarks and a sparse 3D face model."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


# 3D model points (in mm) for key facial landmarks, approximate generic model.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -110.0, -5.0),    # Chin
        (-70.0, 0.0, -50.0),    # Left eye outer corner
        (70.0, 0.0, -50.0),     # Right eye outer corner
        (-60.0, 50.0, -50.0),   # Left mouth corner
        (60.0, 50.0, -50.0),    # Right mouth corner
    ],
    dtype=np.float32,
)

# Landmark indices for mediapipe and dlib-68 that correspond to the above model order.
MEDIAPIPE_IDXS = [1, 152, 33, 263, 78, 308]
DLIB68_IDXS = [30, 8, 36, 45, 48, 54]

# Subset ordering for reduced landmark exports (matches scripts/build_vggface2.py LANDMARKS_MP)
LANDMARK_SUBSET_NAMES = [
    "left_eye_outer",
    "left_eye_inner",
    "right_eye_inner",
    "right_eye_outer",
    "nose_tip",
    "mouth_left_corner",
    "mouth_right_corner",
    "upper_lip_center",
    "lower_lip_center",
    "chin",
    "left_ear_tragus",
    "right_ear_tragus",
    "left_cheek",
    "right_cheek",
    "left_eyebrow_outer",
    "left_eyebrow_inner",
    "right_eyebrow_inner",
    "right_eyebrow_outer",
]
SUBSET_ORDER = ["nose_tip", "chin", "left_eye_outer", "right_eye_outer", "mouth_left_corner", "mouth_right_corner"]
SUBSET_IDX_MAP = {name: idx for idx, name in enumerate(LANDMARK_SUBSET_NAMES)}


def _select_points(landmarks: np.ndarray, idxs: List[int]) -> np.ndarray:
    pts = []
    for i in idxs:
        if i < 0 or i >= len(landmarks):
            pts.append([np.nan, np.nan])
        else:
            pts.append(landmarks[i])
    return np.asarray(pts, dtype=np.float32)


def estimate_head_pose(
    landmarks: np.ndarray,
    image_size: Tuple[int, int],
    mode: str = "mediapipe",
) -> Tuple[float, float, float]:
    """
    Estimate yaw/pitch/roll in degrees given 2D landmarks and image size.

    Returns (yaw, pitch, roll). If estimation fails, returns (nan, nan, nan).
    """
    h, w = image_size
    # Choose selection strategy: full landmarks vs subset export.
    if len(landmarks) >= max(MEDIAPIPE_IDXS) + 1 and mode == "mediapipe":
        image_points = _select_points(landmarks, MEDIAPIPE_IDXS)
    elif len(landmarks) >= max(DLIB68_IDXS) + 1 and mode == "dlib":
        image_points = _select_points(landmarks, DLIB68_IDXS)
    else:
        # Assume subset ordering as in LANDMARK_SUBSET_NAMES
        pts = []
        try:
            for name in SUBSET_ORDER:
                pts.append(landmarks[SUBSET_IDX_MAP[name]])
        except Exception:
            return float("nan"), float("nan"), float("nan")
        image_points = np.asarray(pts, dtype=np.float32)
    if np.isnan(image_points).any() or (image_points < 0).any():
        return float("nan"), float("nan"), float("nan")

    focal_length = max(h, w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    success, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return float("nan"), float("nan"), float("nan")
    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        yaw = np.arctan2(-rot_mat[2, 0], sy)
        roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        yaw = np.arctan2(-rot_mat[2, 0], sy)
        roll = 0
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def frontalize_landmarks(
    landmarks: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    origin_idx: int | None = None,
    origin_point: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply inverse head rotation to approximate frontalized 2D landmarks.

    Assumptions: z=0 for all points. If origin_idx is provided and valid, use that
    landmark as the pivot (nose tip recommended). Otherwise, use origin_point if
    provided; else fall back to the centroid of all points.
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim == 1 and pts.size % 2 == 0:
        pts = pts.reshape(-1, 2)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return pts
    if not np.isfinite([yaw_deg, pitch_deg, roll_deg]).all():
        return pts

    if origin_point is not None:
        origin = np.asarray(origin_point, dtype=np.float32)
    elif origin_idx is not None and 0 <= origin_idx < len(pts):
        origin = pts[origin_idx].copy()
    else:
        origin = np.nanmean(pts, axis=0)
    if not np.isfinite(origin).all():
        origin = np.zeros(2, dtype=np.float32)

    pts3d = np.concatenate([pts, np.zeros((len(pts), 1), dtype=np.float32)], axis=1)
    pts3d = pts3d - np.concatenate([origin, [0.0]]).reshape(1, 3)

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32
    )
    Rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]], dtype=np.float32
    )
    R = Rz @ Ry @ Rx  # roll * yaw * pitch
    pts_frontal = (R.T @ pts3d.T).T
    pts_frontal = pts_frontal[:, :2] + origin
    return pts_frontal.astype(np.float32)
