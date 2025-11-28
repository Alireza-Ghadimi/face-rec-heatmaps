"""
Download VGGFace2 from Hugging Face and convert to an ImageFolder layout.

Defaults target the public `anhnct/vggface2` dataset. You can override the dataset
name or split. Hugging Face auth may be required if the dataset is gated.
"""

from __future__ import annotations

import argparse
import random
from itertools import islice
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_image(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    else:
        img = Image.fromarray(np.array(img)).convert("RGB")
    img.save(path)


def build_vggface2(
    dataset_name: str,
    split: str,
    out_dir: Path,
    max_samples: Optional[int] = None,
    limit_classes: Optional[int] = None,
    streaming: bool = False,
    sample_prob: float = 1.0,
    seed: int = 0,
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

    for i, sample in tqdm(iterator, total=total, desc=f"Writing {split}"):
        if sample_prob < 1.0 and rng.random() > sample_prob:
            continue
        label = sample.get("label") or sample.get("identity")
        if label is None:
            raise ValueError("Sample missing 'label'/'identity' field")
        if limit_classes is not None and len(counts) >= limit_classes and label not in counts:
            continue
        counts[label] = counts.get(label, 0) + 1
        img = sample["image"]
        save_image(img, out_dir / str(label) / f"{counts[label]:06d}.jpg")

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
    )


if __name__ == "__main__":
    main()
