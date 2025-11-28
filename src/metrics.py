"""Verification metrics and yaw-bin reporting."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def build_verification_pairs(
    labels: Sequence[int], paths: Sequence[str], num_pairs: int = 200
) -> List[Tuple[int, int, int]]:
    """Return list of (i, j, same_flag)."""
    by_class: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        by_class.setdefault(lbl, []).append(idx)
    if len(by_class) < 2:
        return []
    pairs: List[Tuple[int, int, int]] = []
    rng = random.Random(0)
    for _ in range(num_pairs // 2):
        cls = rng.choice(list(by_class.keys()))
        if len(by_class[cls]) >= 2:
            i, j = rng.sample(by_class[cls], 2)
            pairs.append((i, j, 1))
        cls_a, cls_b = rng.sample(list(by_class.keys()), 2)
        i = rng.choice(by_class[cls_a])
        j = rng.choice(by_class[cls_b])
        pairs.append((i, j, 0))
    return pairs


def verification_scores(
    embeddings: np.ndarray, pairs: Iterable[Tuple[int, int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cosine similarities and labels for pairs."""
    sims, labs = [], []
    for i, j, same in pairs:
        v1 = embeddings[i]
        v2 = embeddings[j]
        sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        sims.append(sim)
        labs.append(same)
    return np.asarray(sims), np.asarray(labs)


def compute_auc_accuracy(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return ROC AUC and thresholded accuracy."""
    if scores.size == 0:
        return 0.0, 0.0
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.0
    preds = (scores > 0.5).astype(np.int64)
    acc = float((preds == labels).mean())
    return auc, acc


def yaw_binned_accuracy(
    pairs: Iterable[Tuple[int, int, int]],
    scores: np.ndarray,
    labels: np.ndarray,
    yaw_lookup: Dict[int, float] | None = None,
    path_map: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Compute accuracy per yaw bin when a yaw lookup is provided."""
    if yaw_lookup is None or path_map is None:
        return {}
    bins = {"0-15": (0.0, 15.0), "15-45": (15.0, 45.0), "45-90": (45.0, 90.0)}
    accs: Dict[str, List[int]] = {k: [] for k in bins}
    for idx, (i, j, _) in enumerate(pairs):
        if i >= len(path_map) or j >= len(path_map):
            continue
        yaw_i = yaw_lookup.get(path_map[i])
        yaw_j = yaw_lookup.get(path_map[j])
        if yaw_i is None or yaw_j is None:
            continue
        yaw = max(abs(yaw_i), abs(yaw_j))
        for k, (lo, hi) in bins.items():
            if lo <= yaw < hi:
                accs[k].append(int((scores[idx] > 0.5) == labels[idx]))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in accs.items()}
