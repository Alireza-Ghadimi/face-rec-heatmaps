"""
Dataset utilities for training a landmark-to-canonical autoencoder.

Loads chunked NPY files produced by scripts/build_vggface2.py. Each row layout:
idx, class_id, height, width, raw_landmarks(64*2), norm_landmarks(64*2), yaw, pitch, roll.

We select one canonical sample per class (minimum |yaw|+|pitch|+|roll|) as the target.
Input features = raw landmarks concat yaw/pitch/roll. Target = canonical raw landmarks.
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


def _load_chunk(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def _infer_feature_count(row: Sequence) -> int:
    # row length = 4 + raw_len + norm_len + 3, with raw_len == norm_len
    total = len(row)
    raw_len = (total - 7) // 2  # since raw_len + norm_len = total - 7
    return raw_len


def _extract_fields(row: Sequence, feature_count: int) -> Tuple[str, np.ndarray, float, float, float]:
    class_id = str(row[1])
    raw_start = 4
    raw_end = raw_start + feature_count
    raw = np.asarray(row[raw_start:raw_end], dtype=np.float32)
    yaw, pitch, roll = map(float, row[-3:])
    return class_id, raw, yaw, pitch, roll


def _canonical_map_numpy(rows: np.ndarray, feature_count: int) -> Dict[str, np.ndarray]:
    best: Dict[str, Tuple[float, np.ndarray]] = {}
    for r in rows:
        cid, raw, yaw, pitch, roll = _extract_fields(r, feature_count)
        score = abs(yaw) + abs(pitch) + abs(roll)
        if cid not in best or score < best[cid][0]:
            best[cid] = (score, raw)
    return {k: v[1] for k, v in best.items()}


def build_canonical_map(chunk_paths: List[str]) -> Dict[str, np.ndarray]:
    # If pyspark unavailable or initialization fails, fallback to numpy path.
    first_chunk = _load_chunk(chunk_paths[0])
    feature_count = _infer_feature_count(first_chunk[0])
    if SparkSession is None:
        all_rows = [row for p in chunk_paths for row in _load_chunk(p)]
        return _canonical_map_numpy(np.asarray(all_rows, dtype=object), feature_count)
    try:
        spark = SparkSession.builder.master("local[*]").appName("canon_map").getOrCreate()
        dfs = []
        for p in chunk_paths:
            arr = _load_chunk(p)
            data = []
            for r in arr:
                cid, raw, yaw, pitch, roll = _extract_fields(r, feature_count)
                score = abs(yaw) + abs(pitch) + abs(roll)
                data.append((cid, float(score), raw.tolist()))
            df = spark.createDataFrame(data, ["class_id", "score", "raw"])
            dfs.append(df)
        full = dfs[0]
        for df in dfs[1:]:
            full = full.union(df)
        w = (
            full.groupBy("class_id")
            .agg(F.min("score").alias("best_score"))
            .join(full, on=["class_id", F.col("score") == F.col("best_score")], how="inner")
            .select("class_id", "raw")
        )
        canon = {row["class_id"]: np.asarray(row["raw"], dtype=np.float32) for row in w.collect()}
        spark.stop()
        return canon
    except Exception:
        all_rows = [row for p in chunk_paths for row in _load_chunk(p)]
        return _canonical_map_numpy(np.asarray(all_rows, dtype=object), feature_count)


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
        first_chunk = _load_chunk(chunk_paths[0])
        self.feature_count = _infer_feature_count(first_chunk[0])
        self.canonical = build_canonical_map(chunk_paths)
        samples: List[AutoencoderSample] = []
        for p in chunk_paths:
            for r in _load_chunk(p):
                cid, raw, yaw, pitch, roll = _extract_fields(r, self.feature_count)
                if cid not in self.canonical:
                    continue
                feats = np.concatenate([raw, np.array([yaw, pitch, roll], dtype=np.float32)], axis=0)
                target = self.canonical[cid]
                if not np.isfinite(feats).all() or not np.isfinite(target).all():
                    continue
                samples.append(AutoencoderSample(feats, target))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return torch.from_numpy(s.features), torch.from_numpy(s.target)
