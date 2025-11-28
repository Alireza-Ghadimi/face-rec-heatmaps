"""
Unified landmark extraction for MediaPipe FaceMesh and dlib-68.
Falls back to a deterministic dummy pattern when detectors are unavailable.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency
    mp = None

try:
    import dlib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    dlib = None


class LandmarkExtractor:
    """
    Simple wrapper that returns (landmarks, ok).

    Landmarks are pixel coordinates with shape (N_points, 2). The extractor
    gracefully falls back to a fixed template when the requested backend is
    unavailable so downstream code can continue to run.
    """

    def __init__(self, mode: str = "mediapipe") -> None:
        self.mode = mode.lower()
        self._mp_face_mesh = None
        self._dlib_detector = None
        self._dlib_predictor = None

        if self.mode == "mediapipe" and mp is not None:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=True,
                max_num_faces=1,
            )
        if self.mode == "dlib" and dlib is not None:
            self._dlib_detector = dlib.get_frontal_face_detector()
            predictor_path = os.environ.get("DLIB_LANDMARK_MODEL", "shape_predictor_68_face_landmarks.dat")
            if predictor_path and os.path.exists(predictor_path):
                self._dlib_predictor = dlib.shape_predictor(predictor_path)
            else:
                print('could not load the predictor')

    def __call__(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Return landmarks and success flag."""
        if self.mode == "mediapipe":
            return self._extract_mediapipe(image_bgr)
        if self.mode == "dlib":
            return self._extract_dlib(image_bgr)
        return self._dummy_landmarks(image_bgr.shape[:2])

    def _extract_mediapipe(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self._mp_face_mesh is None:
            return self._dummy_landmarks(image_bgr.shape[:2])
        image_rgb = image_bgr[:, :, ::-1]
        result = self._mp_face_mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return self._dummy_landmarks(image_bgr.shape[:2])
        points = []
        h, w = image_bgr.shape[:2]
        for lm in result.multi_face_landmarks[0].landmark:
            points.append([lm.x * w, lm.y * h])
        return np.asarray(points, dtype=np.float32), True

    def _extract_dlib(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if self._dlib_detector is None or self._dlib_predictor is None:
            return self._dummy_landmarks(image_bgr.shape[:2])
        gray = image_bgr[:, :, 0] if image_bgr.ndim == 3 else image_bgr
        dets = self._dlib_detector(gray, 1)
        if len(dets) == 0:
            return self._dummy_landmarks(image_bgr.shape[:2])
        shape = self._dlib_predictor(gray, dets[0])
        points = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        return points, True

    def _dummy_landmarks(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
        """Return a stable pattern so downstream code can continue running."""
        h, w = shape
        xs = np.array([0.35, 0.65, 0.5, 0.38, 0.62, 0.5], dtype=np.float32) * w
        ys = np.array([0.35, 0.35, 0.45, 0.55, 0.55, 0.7], dtype=np.float32) * h
        points = np.stack([xs, ys], axis=1)
        return points, False


def landmarks_to_numpy(landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """Convenience helper to convert a list of (x, y) to float32 array."""
    return np.asarray(landmarks, dtype=np.float32)
