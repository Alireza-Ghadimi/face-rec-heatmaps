"""
Download VGGFace2 from Hugging Face, optionally stream/subsample, and write
an ImageFolder layout plus a CSV of selected MediaPipe landmarks.
"""

from __future__ import annotations

import argparse
import csv
import random
from itertools import islice
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import sys

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

_MP_MESH = None
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.headpose import estimate_head_pose  # noqa: E402

LANDMARKS_MP: Sequence[Tuple[str, int]] = [
    ("left_eye_outer", 33),
    ("left_eye_inner", 133),
    ("right_eye_inner", 362),
    ("right_eye_outer", 263),
    ("nose_tip", 1),
    ("mouth_left_corner", 78),
    ("mouth_right_corner", 308),
    ("upper_lip_center", 13),
    ("lower_lip_center", 14),
    ("chin", 152),
    ("left_ear_tragus", 234),
    ("right_ear_tragus", 454),
    ("left_cheek", 187),
    ("right_cheek", 411),
    # Note: full 468 landmarks exported below; these 18 remain for head-pose defaults.
    ("left_eyebrow_outer", 70),
    ("left_eyebrow_inner", 105),
    ("right_eyebrow_inner", 334),
    ("right_eyebrow_outer", 296),
]
MP_LANDMARK_COUNT = 468
IMPORTANT_LANDMARKS = list(range(MP_LANDMARK_COUNT))  # export all landmarks


def save_image(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        img = Image.fromarray(np.array(img)).convert("RGB")
    img.save(path)


def extract_landmarks_mediapipe(image: Image.Image) -> Tuple[np.ndarray, bool]:
    """Return (points, ok) where points shape (468,2) in pixel coords."""
    global _MP_MESH
    if mp is None:
        return np.full((MP_LANDMARK_COUNT, 2), -1.0, dtype=np.float32), False
    if _MP_MESH is None:
        _MP_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
        )
    rgb = np.array(image)
    result = _MP_MESH.process(rgb)
    if not result.multi_face_landmarks:
        return np.full((MP_LANDMARK_COUNT, 2), -1.0, dtype=np.float32), False
    h, w = rgb.shape[:2]
    pts = []
    lms = result.multi_face_landmarks[0].landmark
    for idx in range(MP_LANDMARK_COUNT):
        lm = lms[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.asarray(pts, dtype=np.float32), True


def build_vggface2(
    dataset_name: str,
    split: str,
    out_dir: Path,
    max_samples: Optional[int] = None,
    limit_classes: Optional[int] = None,
    streaming: bool = False,
    sample_prob: float = 1.0,
    seed: int = 0,
    csv_path: Optional[Path] = None,
    npy_path: Optional[Path] = None,
    chunk_size: int = 1000,
    chunk_dir: Optional[Path] = None,
    save_images: bool = True,
) -> None:
    rng = random.Random(seed)
    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    total = None
    if not streaming:
        total = len(ds) if max_samples is None else min(len(ds), max_samples)
        iterator = enumerate(ds)
    else:
        iterator = enumerate(ds) if max_samples is None else enumerate(islice(ds, max_samples))

    rows = []
    writer = None
    csv_file = None
    chunk_dir = chunk_dir or (out_dir / "chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    current_chunk = 0
    current_chunk_file = chunk_dir / f"chunk_{current_chunk:05d}.npy"
    skip_chunk = current_chunk_file.exists()
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        header = ["idx", "class_id", "height", "width"]
        for idx in IMPORTANT_LANDMARKS:
            header.extend([f"lm_{idx}_x", f"lm_{idx}_y"])
        for idx in IMPORTANT_LANDMARKS:
            header.extend([f"lm_{idx}_x_norm", f"lm_{idx}_y_norm"])
        header.extend(["yaw_deg", "pitch_deg", "roll_deg"])
        writer = csv.writer(csv_file)
        writer.writerow(header)

    for i, sample in tqdm(iterator, total=total, desc=f"Writing {split}"):
        if sample_prob < 1.0 and rng.random() > sample_prob:
            continue
        chunk_idx = i // chunk_size
        if chunk_idx != current_chunk:
            if rows and not skip_chunk:
                np.save(current_chunk_file, np.asarray(rows, dtype=object))
            rows = []
            current_chunk = chunk_idx
            current_chunk_file = chunk_dir / f"chunk_{current_chunk:05d}.npy"
            skip_chunk = current_chunk_file.exists()
        label = sample.get("label") or sample.get("identity") or sample.get("class_id")
        if label is None:
            raise ValueError("Sample missing 'label'/'identity' field")
        if limit_classes is not None and len(counts) >= limit_classes and label not in counts:
            continue
        counts[label] = counts.get(label, 0) + 1
        img = sample["image"]
        if skip_chunk:
            continue
        if save_images:
            save_image(img, out_dir / str(label) / f"{counts[label]:06d}.jpg")
        if writer is not None or npy_path is not None:
            h, w = img.size[1], img.size[0]  # PIL size is (w, h)
            pts, _ = extract_landmarks_mediapipe(img)
            row = [i, label, h, w]
            pts_sel = pts[IMPORTANT_LANDMARKS]
            row.extend(pts_sel.flatten().tolist())
            nose = pts[1] if len(pts) > 1 else np.array([0.0, 0.0], dtype=np.float32)
            top = pts[10] if len(pts) > 10 else np.array([0.0, 0.0], dtype=np.float32)
            chin = pts[152] if len(pts) > 152 else np.array([1.0, 1.0], dtype=np.float32)
            scale = top - chin
            scale[scale == 0] = np.finfo(np.float32).eps
            pts_norm = (pts_sel - nose) / scale
            row.extend(pts_norm.flatten().tolist())
            yaw, pitch, roll = estimate_head_pose(pts, (h, w), mode="mediapipe")
            row.extend([yaw, pitch, roll])
            rows.append(row)
            if writer is not None and len(rows) == 1:
                writer.writerow(row)

    if csv_file is not None:
        csv_file.close()
    if _MP_MESH is not None:
        try:
            _MP_MESH.close()
        except Exception:
            pass
    if rows and not skip_chunk:
        np.save(current_chunk_file, np.asarray(rows, dtype=object))
    if npy_path is not None and rows:
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, np.asarray(rows, dtype=object))

    print(f"Wrote images to {out_dir} with {len(counts)} identities and {sum(counts.values())} files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VGGFace2 ImageFolder from Hugging Face")
    parser.add_argument("--dataset", type=str, default="anhnct/vggface2", help="HF dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")
    parser.add_argument("--out_dir", type=str, default="data_vggface2", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on total samples")
    parser.add_argument("--limit_classes", type=int, default=None, help="Limit number of identities")
    parser.add_argument("--streaming", action="store_true", help="Stream without full download")
    parser.add_argument("--sample_prob", type=float, default=1.0, help="Subsample probability when streaming")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    parser.add_argument("--csv_path", type=str, default=None, help="Optional CSV path for landmark export")
    parser.add_argument("--npy_path", type=str, default=None, help="Optional NPY path to store all rows")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Rows per chunk npy")
    parser.add_argument("--chunk_dir", type=str, default=None, help="Directory to store chunk npys")
    parser.add_argument("--save_images", type=str, default="true", help="Save extracted JPEGs (true/false)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args.dataset, "------------------------------------------------------")
    build_vggface2(
        dataset_name=args.dataset,
        split=args.split,
        out_dir=Path(args.out_dir),
        max_samples=args.max_samples,
        limit_classes=args.limit_classes,
        streaming=args.streaming,
        sample_prob=args.sample_prob,
        seed=args.seed,
        csv_path=Path(args.csv_path) if args.csv_path else None,
        npy_path=Path(args.npy_path) if args.npy_path else None,
        chunk_size=args.chunk_size,
        chunk_dir=Path(args.chunk_dir) if args.chunk_dir else None,
        save_images=args.save_images.lower() in ("1", "true", "yes", "y"),
    )


if __name__ == "__main__":
    main()
