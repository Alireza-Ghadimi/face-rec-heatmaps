"""Training script with argparse config."""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml

from .dataset import FaceDataset
from .metrics import build_verification_pairs, compute_auc_accuracy, verification_scores, yaw_binned_accuracy
from .model import FaceModel
from .utils import save_checkpoint, set_seed


def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_loaders(cfg: Dict[str, Any], pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, FaceDataset]:
    print("root= ", cfg["data_root"])
    dataset = FaceDataset(
        root=cfg["data_root"],
        img_size=cfg["img_size"],
        use_heatmaps=cfg["use_heatmaps"],
        extractor=cfg["extractor"],
        cache_heatmaps=cfg["cache_heatmaps"],
        cache_landmarks=cfg.get("cache_landmarks", False),
        add_mask_channel=cfg["add_mask_channel"],
    )
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, dataset


def train_one_epoch(
    model: FaceModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    global_step: int,
) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(
            device_type=device.type if device.type in ("cuda", "mps") else "cpu",
            enabled=device.type in ("cuda", "mps"),
        ):
            logits = model(images, labels)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
            global_step += 1
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(1, total), total_correct / max(1, total), global_step


def evaluate(model: FaceModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    feats_list, labels_list, paths = [], [], []
    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, labels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            feats_list.append(model.encode(images).cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            if isinstance(meta, dict) and "path" in meta:
                paths_val = meta["path"]
                if isinstance(paths_val, (list, tuple)):
                    paths.extend(paths_val)
                else:
                    paths.append(str(paths_val))
    feats_np = np.concatenate(feats_list, axis=0) if feats_list else np.zeros((0, model.head.weight.shape[1]))
    labels_np = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0,))
    pairs = build_verification_pairs(labels_np.tolist(), paths, num_pairs=100)
    scores, pair_labels = verification_scores(feats_np, pairs)
    auc, ver_acc = compute_auc_accuracy(scores, pair_labels)
    yaw_lookup = None  # placeholder for user-provided yaw metadata
    yaw_acc = yaw_binned_accuracy(pairs, scores, pair_labels, yaw_lookup, paths)
    if not yaw_acc:
        print("Yaw metadata not provided; skipping yaw-binned metrics.")
    return {
        "loss": total_loss / max(1, total),
        "cls_acc": total_correct / max(1, total),
        "ver_auc": auc,
        "ver_acc": ver_acc,
        **{f"yaw_{k}": v for k, v in yaw_acc.items()},
    }


def build_scheduler(optimizer: optim.Optimizer, steps: int, warmup_steps: int):
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face recognition training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--data_root", type=str, required=True, help="ImageFolder root")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None, help="Force device")
    parser.add_argument("--use_heatmaps", type=str, default=None, help="true|false")
    parser.add_argument("--add_mask_channel", type=str, default=None, help="true|false")
    parser.add_argument("--extractor", type=str, choices=["mediapipe", "dlib"], default=None)
    parser.add_argument("--cache_heatmaps", type=str, default=None, help="true|false")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--use_cosface", type=str, default=None, help="true|false")
    return parser.parse_args()


def merge_config(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    for key in ["data_root", "img_size", "extractor"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.device is not None:
        cfg["device"] = args.device
    for key in ["use_heatmaps", "add_mask_channel", "cache_heatmaps", "use_cosface"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = str2bool(val)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = merge_config(load_config(args.config), args)
    set_seed(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False))
    device_arg = cfg.get("device")
    if device_arg:
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pin_mem = device.type == "cuda"
    train_loader, val_loader, dataset = build_loaders(cfg, pin_memory=pin_mem)
    in_ch = 3
    if cfg["use_heatmaps"]:
        k = len(dataset.group_order)
        in_ch += k + (1 if cfg["add_mask_channel"] else 0)
    num_classes = len(dataset.dataset.classes)
    model = FaceModel(
        num_classes=num_classes,
        in_ch=in_ch,
        backbone=cfg["backbone"],
        pretrained=True,
        use_cosface=cfg.get("use_cosface", False),
        arc_margin=cfg["arcface"]["margin"],
        arc_scale=cfg["arcface"]["scale"],
        cos_margin=cfg["cosface"]["margin"],
        cos_scale=cfg["cosface"]["scale"],
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    total_steps = cfg["epochs"] * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, total_steps, cfg["scheduler"]["warmup_steps"])
    scaler = GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    global_step = 0
    for epoch in range(cfg["epochs"]):
        print(f"Epoch {epoch + 1}/{cfg['epochs']}")
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device, scheduler, global_step
        )
        val_metrics = evaluate(model, val_loader, device)
        print(f"Train loss {train_loss:.4f} acc {train_acc:.4f}")
        print(
            "Val loss {loss:.4f} cls_acc {cls_acc:.4f} ver_acc {ver_acc:.4f} auc {ver_auc:.4f}".format(
                **val_metrics
            )
        )
        if val_metrics["cls_acc"] > best_acc:
            best_acc = val_metrics["cls_acc"]
            save_path = os.path.join(cfg.get("save_dir", "checkpoints"), "best.pt")
            save_checkpoint({"model": model.state_dict(), "cfg": cfg}, save_path)
    print("Training finished. Best cls_acc {:.4f}".format(best_acc))


if __name__ == "__main__":
    main()
