"""
Dataset utilities for training a landmark-to-canonical autoencoder.

Loads chunked NPY files produced by scripts/build_vggface2.py. Each row layout:
idx, class_id, height, width, raw_landmarks(64*2), norm_landmarks(64*2), yaw, pitch, roll.

We select one canonical sample per class (minimum |yaw|+|pitch|+|roll|) as the target.
Input features = normalized landmarks concat yaw/pitch/roll. Target = canonical normalized landmarks.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # optional acceleration
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except Exception:  # pragma: no cover
    SparkSession = None

FEATURE_COUNT = 64 * 2  # normalized landmarks count


def _load_chunk(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    return arr


def _extract_fields(row: Sequence) -> Tuple[str, np.ndarray, float, float, float]:
    class_id = str(row[1])
    norm_start = 4 + FEATURE_COUNT  # skip raw coords
    norm_end = norm_start + FEATURE_COUNT
    norm = np.asarray(row[norm_start:norm_end], dtype=np.float32)
    yaw, pitch, roll = map(float, row[-3:])
    return class_id, norm, yaw, pitch, roll


def _canonical_map_numpy(rows: np.ndarray) -> Dict[str, np.ndarray]:
    best: Dict[str, Tuple[float, np.ndarray]] = {}
    for r in rows:
        cid, norm, yaw, pitch, roll = _extract_fields(r)
        score = abs(yaw) + abs(pitch) + abs(roll)
        if cid not in best or score < best[cid][0]:
            best[cid] = (score, norm)
    return {k: v[1] for k, v in best.items()}


def build_canonical_map(chunk_paths: List[str]) -> Dict[str, np.ndarray]:
    if SparkSession is None:
        all_rows = [row for p in chunk_paths for row in _load_chunk(p)]
        return _canonical_map_numpy(np.asarray(all_rows, dtype=object))
    spark = SparkSession.builder.appName("canon_map").getOrCreate()
    # Flatten chunks into a DataFrame
    dfs = []
    for p in chunk_paths:
        arr = _load_chunk(p)
        # explode into rows: (class_id, score, norm_vector)
        data = []
        for r in arr:
            cid, norm, yaw, pitch, roll = _extract_fields(r)
            score = abs(yaw) + abs(pitch) + abs(roll)
            data.append((cid, float(score), norm.tolist()))
        df = spark.createDataFrame(data, ["class_id", "score", "norm"])
        dfs.append(df)
    full = dfs[0]
    for df in dfs[1:]:
        full = full.union(df)
    w = (
        full.groupBy("class_id")
        .agg(F.min("score").alias("best_score"))
        .join(full, on=["class_id", F.col("score") == F.col("best_score")], how="inner")
        .select("class_id", "norm")
    )
    canon = {row["class_id"]: np.asarray(row["norm"], dtype=np.float32) for row in w.collect()}
    spark.stop()
    return canon


@dataclass
class AutoencoderSample:
    features: np.ndarray
    target: np.ndarray


class AutoencoderDataset(Dataset):
    """Torch dataset yielding (features, target) tensors."""

    def __init__(self, chunk_dir: str) -> None:
        chunk_paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npy")))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunks found in {chunk_dir}")
        # Build canonical map once
        self.canonical = build_canonical_map(chunk_paths)
        # Flatten all samples
        samples: List[AutoencoderSample] = []
        for p in chunk_paths:
            for r in _load_chunk(p):
                cid, norm, yaw, pitch, roll = _extract_fields(r)
                if cid not in self.canonical:
                    continue
                feats = np.concatenate([norm, np.array([yaw, pitch, roll], dtype=np.float32)], axis=0)
                samples.append(AutoencoderSample(feats, self.canonical[cid]))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.from_numpy(s.features)
        y = torch.from_numpy(s.target)
        return x, y
