#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --- your modules (must contain the classes we added earlier) ---
from dataset import FinalCornersHandDataset
from model import FinalCornersAuxModel


# -------------------------------
# Util: seeding for reproducibility
# -------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------------
# Metrics (ROC-AUC, PR-AUC) w/ pure NumPy fallback
# -------------------------------
def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    # Sort by score desc
    order = np.argsort(-y_score)
    y_true = y_true[order].astype(np.int32)
    # Cum TP/FP
    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        # degenerate case: return trivial curve
        tpr = np.array([0.0, 1.0], dtype=np.float64)
        fpr = np.array([0.0, 1.0], dtype=np.float64)
        return fpr, tpr
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    tpr = tp / (P + 1e-12)
    fpr = fp / (N + 1e-12)
    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr = _roc_curve(y_true, y_score)
    # trapezoidal rule; ensure monotonic fpr
    return float(np.trapz(tpr, fpr))


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true = y_true[order].astype(np.int32)
    P = y_true.sum()
    if P == 0:
        # no positives -> define a trivial PR curve
        return np.array([1.0, 0.0]), np.array([0.0, 0.0])
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (P + 1e-12)
    # prepend (recall=0, precision=1) for standard AP integration
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall = _precision_recall_curve(y_true, y_score)
    # integrate precision(recall) with trapezoidal rule on recall axis
    return float(np.trapz(precision, recall))


# -------------------------------
# Training / Validation epoch
# -------------------------------
def run_epoch(model, loader, optimizer, scaler, criterion, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_probs = []
    all_labels = []

    if train:
        iterator = tqdm(loader, desc="[Train]", leave=False)
    else:
        iterator = tqdm(loader, desc="[Val]", leave=False)

    for batch in iterator:
        corners, aux, label = batch               # corners: [B,24], aux: [B,12], label: [B]
        corners = corners.to(device, non_blocking=True)
        aux     = aux.to(device, non_blocking=True)
        label   = label.to(device, non_blocking=True).view(-1, 1)   # [B,1] float 0/1

        if train:
            optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=torch.cuda.is_available()):
            logits = model(corners, aux)          # [B,1]
            loss = criterion(logits, label)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            probs = torch.sigmoid(logits)         # [B,1]
            preds01 = (probs > 0.5).float()
            total_correct += (preds01.eq(label).sum().item())
            total_count   += label.size(0)
            total_loss    += loss.item() * label.size(0)

            all_probs.append(probs.detach().view(-1).cpu().numpy())
            all_labels.append(label.detach().view(-1).cpu().numpy())

    # Aggregate
    all_probs  = np.concatenate(all_probs)  if all_probs else np.array([], dtype=np.float32)
    all_labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float32)

    acc = (total_correct / max(1, total_count))
    loss_mean = total_loss / max(1, total_count)
    auc_roc = roc_auc(all_labels, all_probs) if all_probs.size else 0.0
    auc_pr  = pr_auc(all_labels, all_probs)  if all_probs.size else 0.0

    return {
        "loss": loss_mean,
        "acc":  acc,
        "roc_auc": auc_roc,
        "pr_auc":  auc_pr,
        "n": total_count,
    }, all_probs, all_labels


# -------------------------------
# Main
# -------------------------------
def main(dir_path: str,
         emb_path=None,              # ignored, kept for CLI compatibility
         corners_only: bool = True   # ignored (we always use corners+aux model)
         ):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- paths ----
    data_dir = Path(dir_path)
    comb_dir = data_dir / "combined_data"
    train_path = str(comb_dir / "train.json")
    val_path   = str(comb_dir / "val.json")

    out_root = data_dir / "training"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    logs_dir = out_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- args / hyperparams (simple) ----
    batch_size = 4096
    epochs     = 150
    lr         = 5e-4
    weight_decay = 1e-4
    use_balanced_sampler = True
    patience  = 20  # early stop on val ROC-AUC

    print("Loading datasets with feature normalization...")
    train_dataset = FinalCornersHandDataset(
        train_path,
        normalization_stats=None,   # compute from train
        is_training=True,
    )
    val_dataset = FinalCornersHandDataset(
        val_path,
        normalization_stats=train_dataset.normalization_stats,  # reuse train stats
        is_training=False,
    )
    print(f"  → {len(train_dataset)} train samples")
    print(f"  → {len(val_dataset)} val samples")

    # ---- sampler or shuffle ----
    if use_balanced_sampler:
        labels_mm = train_dataset.mm_label[:, 1]  # numpy memmap 0/1 for collision-at-placement
        n_pos = int((labels_mm == 1).sum())
        n_neg = int((labels_mm == 0).sum())
        w_pos = 1.0 / max(1, n_pos)
        w_neg = 1.0 / max(1, n_neg)
        sample_weights = np.where(labels_mm == 1, w_pos, w_neg).astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(labels_mm),
            replacement=True
        )
        shuffle_flag = False
        print(f"Using WeightedRandomSampler (neg={n_neg}, pos={n_pos})")
    else:
        sampler = None
        shuffle_flag = True
        print("Using plain shuffling")

    # ---- loaders ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    # ---- model / opt / loss ----
    model = FinalCornersAuxModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # BCE; sampler handles class balance. (If you disable sampler, consider pos_weight.)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # ---- schedulers (simple cosine) ----
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)

    # ---- training ----
    best_val_auc = -math.inf
    best_epoch = -1
    early_stop_counter = 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"training_{run_id}.log"
    with open(log_path, "w") as logf:
        logf.write(f"run: {run_id}\n")
        logf.write(f"data: {dir_path}\n")
        logf.write(f"batch_size={batch_size} epochs={epochs} lr={lr} wd={weight_decay}\n")
        logf.write(f"use_balanced_sampler={use_balanced_sampler}\n")
        logf.flush()

        for epoch in range(epochs):
            # TRAIN
            train_stats, _, _ = run_epoch(model, train_loader, optimizer, scaler, criterion, device, train=True)
            # VAL
            with torch.no_grad():
                val_stats, val_probs, val_labels = run_epoch(model, val_loader, optimizer, scaler, criterion, device, train=False)

            scheduler.step()

            msg = (f"Epoch {epoch+1:03d}/{epochs} | "
                   f"Train: loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} "
                   f"| Val: loss={val_stats['loss']:.4f} acc={val_stats['acc']:.4f} "
                   f"ROC-AUC={val_stats['roc_auc']:.4f} PR-AUC={val_stats['pr_auc']:.4f} "
                   f"| LR={scheduler.get_last_lr()[0]:.2e}")
            print(msg)
            logf.write(msg + "\n")
            logf.flush()

            # early stopping on ROC-AUC
            cur_auc = val_stats["roc_auc"]
            if cur_auc > best_val_auc + 1e-5:
                best_val_auc = cur_auc
                best_epoch = epoch
                early_stop_counter = 0

                # save checkpoint
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "normalization_stats": train_dataset.normalization_stats,
                    "val_roc_auc": best_val_auc,
                    "val_loss": val_stats["loss"],
                    "config": {
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "use_balanced_sampler": use_balanced_sampler,
                    },
                    "model_type": "FinalCornersAuxModel",
                }
                torch.save(ckpt, ckpt_dir / "best.pt")
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best at {best_epoch+1} with ROC-AUC={best_val_auc:.4f})")
                break

    print(f"✅ Training complete. Best ROC-AUC={best_val_auc:.4f} @ epoch {best_epoch+1}")
    print(f"📝 Log:   {str(log_path)}")
    print(f"💾 Checkpoint saved to: {str(ckpt_dir / 'best.pt')}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str,
                    help="Path to data_collection directory containing combined_data/train.json and val.json")
    # Kept for compatibility; ignored internally
    ap.add_argument("--embeddings", type=str, default=None)
    ap.add_argument("--corners-only", action="store_true")
    args = ap.parse_args()
    args.corners_only = True
    args.data = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection"

    main(args.data, args.embeddings, args.corners_only)
