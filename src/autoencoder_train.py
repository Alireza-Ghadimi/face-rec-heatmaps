"""Training script for the landmark autoencoder."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm
import csv

from src.autoencoder_data import AutoencoderDataset
from src.autoencoder import LandmarkAutoencoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk_dir", type=str, required=True, help="Directory containing chunk_*.npy")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Select device")
    p.add_argument("--save_path", type=str, default="autoencoder.pt")
    p.add_argument("--loss_log", type=str, default="autoencoder_losses.csv", help="CSV to log train/val loss per epoch")
    return p.parse_args()


def build_loaders(dataset: AutoencoderDataset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    crit = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_loss += crit(out, y).item() * x.size(0)
            steps += 1
    return total_loss / max(1, steps)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    dataset = AutoencoderDataset(args.chunk_dir)
    train_loader, val_loader = build_loaders(dataset, args.batch_size)
    model = LandmarkAutoencoder(landmark_dim=128, pose_dim=3, hidden=args.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    best_val = float("inf")
    losses = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            steps += 1
        train_loss = total / max(1, steps)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        losses.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, args.save_path)
            print(f"Saved best model to {args.save_path}")
    # save loss log
    if args.loss_log:
        with open(args.loss_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(losses)
        print(f"Wrote loss log to {args.loss_log}")


if __name__ == "__main__":
    main()
