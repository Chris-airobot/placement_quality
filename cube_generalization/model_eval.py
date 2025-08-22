#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import glob
from dataset import FinalCornersHandDataset
from model import FinalCornersAuxModel
import json
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report




def predict_and_analyze(trials: list[dict], ckpt_path: str, device: str | torch.device = "auto", threshold: float | None = None) -> dict:
    """
    Minimal evaluation:
      - Assume each trial dict is valid and contains keys: 'dims', 'grasp_pose', 'init_pose', 'final_pose', and 'had_collision'.
      - Build features exactly as in training (final corners in world; t_loc, R6 from init/grasp; dims).
      - Normalize with saved stats; run FinalCornersAuxModel; record predictions.
      - Compute a single analysis: accuracy and confusion counts at threshold.

    Returns a dict with per-trial predictions and a simple summary.
    """

    import numpy as np
    import torch

    # Small helpers (kept local for simplicity)
    def quat_wxyz_to_R(q):
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        n = (w*w + x*x + y*y + z*z) ** 0.5 or 1.0
        w, x, y, z = w/n, x/n, y/n, z/n
        xx, yy, zz = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        return np.array([
            [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
            [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
        ], dtype=np.float32)

    def rot6d_from_R(R):
        a1 = R[:, 0]; a2 = R[:, 1]
        return np.concatenate([a1, a2], axis=0).astype(np.float32)

    def zscore(x, mean, std, eps=1e-8):
        return (x - mean) / (std + eps)

    # Device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Load model and stats
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = FinalCornersAuxModel(aux_in=12, use_film=True, two_head=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stats = ckpt.get("normalization_stats", None)
    final_corners_mean = np.asarray(stats["corners_mean"], dtype=np.float32)
    final_corners_std  = np.asarray(stats["corners_std"],  dtype=np.float32)
    tloc_mean          = np.asarray(stats["tloc_mean"],          dtype=np.float32)
    tloc_std           = np.asarray(stats["tloc_std"],           dtype=np.float32)
    dims_mean          = np.asarray(stats["dims_mean"],          dtype=np.float32)
    dims_std           = np.asarray(stats["dims_std"],           dtype=np.float32)

    SIGNS_8x3 = np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], dtype=np.float32)

    predictions: list[dict] = []
    y_true_list: list[int] = []
    p_coll_list: list[float] = []
    group_list: list[str | None] = []

    with torch.no_grad():
        for trial in trials:
            dx, dy, dz = trial["dims"]
            init_pose  = np.asarray(trial["init_pose"],  dtype=np.float32)
            final_pose = np.asarray(trial["final_pose"], dtype=np.float32)
            grasp_pose = np.asarray(trial["grasp_pose"], dtype=np.float32)

            # Final corners in world → (24,)
            T = final_pose[:3].astype(np.float32)
            q = final_pose[3:7].astype(np.float32)
            R = quat_wxyz_to_R(q)
            half = 0.5 * np.array([dx, dy, dz], dtype=np.float32)
            corners_local = SIGNS_8x3 * half
            final_world = (R @ corners_local.T).T + T[None, :]
            corners_24 = final_world.reshape(-1)

            # Aux from init(object) and grasp(hand)
            t_o, q_o = init_pose[:3], init_pose[3:7]
            t_h, q_h = grasp_pose[:3], grasp_pose[3:7]
            R_o = quat_wxyz_to_R(q_o)
            R_h = quat_wxyz_to_R(q_h)
            R_loc = R_o.T @ R_h
            t_loc = R_o.T @ (t_h - t_o)
            R6    = rot6d_from_R(R_loc)
            dims  = np.array([dx, dy, dz], dtype=np.float32)

            # Normalize (R6 left as raw)
            corners_24_n = zscore(corners_24, final_corners_mean, final_corners_std).astype(np.float32)
            tloc_n       = zscore(t_loc,       tloc_mean,          tloc_std).astype(np.float32)
            dims_n       = zscore(dims,        dims_mean,          dims_std).astype(np.float32)
            aux_12_n     = np.concatenate([tloc_n, R6, dims_n], axis=0).astype(np.float32)

            # Tensors
            c_t = torch.from_numpy(corners_24_n).unsqueeze(0).to(device)
            a_t = torch.from_numpy(aux_12_n).unsqueeze(0).to(device)

            logits = model(c_t, a_t)
            p_collision = torch.sigmoid(logits).item()
            p_no_collision = 1.0 - p_collision
            y_true_coll = int(trial.get("had_collision", 0))
            y_true_list.append(y_true_coll)
            p_coll_list.append(float(p_collision))
            group_list.append(trial.get("bucket"))

            predictions.append({
                "index": trial.get("index"),
                "bucket": trial.get("bucket"),
                "dims": [float(dx), float(dy), float(dz)],
                "had_collision": bool(y_true_coll),
                "p_collision": float(p_collision),
                "p_no_collision": float(p_no_collision),
                # will fill pred after threshold selection
            })

    # Convert to arrays for metrics
    import numpy as _np
    y = _np.asarray(y_true_list, dtype=_np.int32)
    p = _np.asarray(p_coll_list, dtype=_np.float32)

    # AUC metrics
    auc_roc = float(roc_auc_score(y, p)) if len(_np.unique(y)) > 1 else float("nan")
    auc_pr  = float(average_precision_score(y, p)) if len(_np.unique(y)) > 1 else float("nan")

    # Threshold selection: maximize F1 over unique probabilities
    def _f1_at_thresh(t: float) -> tuple[float, int, int, int, int]:
        pred = (p >= t).astype(_np.int32)
        tp = int(_np.sum((pred == 1) & (y == 1)))
        fp = int(_np.sum((pred == 1) & (y == 0)))
        fn = int(_np.sum((pred == 0) & (y == 1)))
        tn = int(_np.sum((pred == 0) & (y == 0)))
        denom = (2*tp + fp + fn)
        f1 = (2*tp / denom) if denom > 0 else 0.0
        return f1, tp, tn, fp, fn

    if threshold is None:
        cands = _np.unique(p)
        # add endpoints for stability
        cands = _np.concatenate(([0.0], cands, [1.0]))
        best = (0.0, 0.0, 0, 0, 0, 0)  # f1, tau, tp, tn, fp, fn
        for t in cands:
            f1, tp, tn, fp, fn = _f1_at_thresh(float(t))
            if f1 > best[0]:
                best = (f1, float(t), tp, tn, fp, fn)
        best_tau = best[1]
        tp, tn, fp, fn = best[2], best[3], best[4], best[5]
        threshold_used = best_tau
    else:
        threshold_used = float(threshold)
        f1, tp, tn, fp, fn = _f1_at_thresh(threshold_used)

    # Fill predictions' discrete label with selected threshold
    pred_bin = (p >= threshold_used).astype(_np.int32)
    for j in range(len(predictions)):
        predictions[j]["pred_collision"] = int(pred_bin[j])

    total = int(len(trials))
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # Per-group error analysis by 'bucket' if available
    group_metrics: dict[str, dict] = {}
    if any(g is not None for g in group_list):
        unique_groups = sorted({g for g in group_list if g is not None})
        for g in unique_groups:
            idx = _np.array([i for i, gg in enumerate(group_list) if gg == g], dtype=_np.int32)
            yy = y[idx]
            pp = pred_bin[idx]
            tp_g = int(_np.sum((pp == 1) & (yy == 1)))
            tn_g = int(_np.sum((pp == 0) & (yy == 0)))
            fp_g = int(_np.sum((pp == 1) & (yy == 0)))
            fn_g = int(_np.sum((pp == 0) & (yy == 1)))
            denom_f1 = (2*tp_g + fp_g + fn_g)
            f1_g = (2*tp_g / denom_f1) if denom_f1 > 0 else 0.0
            acc_g = float((tp_g + tn_g) / max(1, len(idx)))
            prec_g = float(tp_g / max(1, (tp_g + fp_g)))
            rec_g  = float(tp_g / max(1, (tp_g + fn_g)))
            base_rate = float(_np.mean(yy)) if len(idx) > 0 else 0.0
            group_metrics[str(g)] = {
                "count": int(len(idx)),
                "base_rate": base_rate,
                "accuracy": acc_g,
                "precision": prec_g,
                "recall": rec_g,
                "f1": f1_g,
                "tp": tp_g, "tn": tn_g, "fp": fp_g, "fn": fn_g,
            }

    # Hardest examples (top FP/FN)
    # FP: y=0, pred=1, sort by p desc; FN: y=1, pred=0, sort by p asc
    fp_idx = [i for i in range(total) if y[i] == 0 and pred_bin[i] == 1]
    fn_idx = [i for i in range(total) if y[i] == 1 and pred_bin[i] == 0]
    fp_idx = sorted(fp_idx, key=lambda i: -p[i])[:10]
    fn_idx = sorted(fn_idx, key=lambda i: p[i])[:10]

    top_fp = [{
        "index": predictions[i].get("index"),
        "bucket": predictions[i].get("bucket"),
        "dims": predictions[i].get("dims"),
        "p_collision": float(p[i]),
    } for i in fp_idx]
    top_fn = [{
        "index": predictions[i].get("index"),
        "bucket": predictions[i].get("bucket"),
        "dims": predictions[i].get("dims"),
        "p_collision": float(p[i]),
    } for i in fn_idx]

    # Data distribution (labels)
    num_collisions = int(_np.sum(y))
    num_non = int(total - num_collisions)
    base_rate_overall = float(num_collisions / total) if total > 0 else 0.0

    # Additional field distributions (if present)
    field_distributions: dict[str, dict] = {}
    candidate_keys = [
        "bucket",
        "surface_id",
        "surface",
        "target_surface",
        "placement_surface",
        "contact_surface",
    ]
    for key in candidate_keys:
        counts_dict: dict = {}
        present = False
        for tr in trials:
            if key in tr:
                present = True
                v = tr.get(key)
                try:
                    counts_dict[v] = counts_dict.get(v, 0) + 1
                except TypeError:
                    # skip unhashable values
                    continue
        if present and len(counts_dict) > 0:
            # sort by count desc
            sorted_items = sorted(counts_dict.items(), key=lambda kv: -kv[1])
            field_distributions[key] = {str(k): int(c) for k, c in sorted_items}

    summary = {
        "threshold": float(threshold_used),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": accuracy,
        "counts": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "total": int(total)},
        "label_distribution": {
            "collision": num_collisions,
            "no_collision": num_non,
            "base_rate": base_rate_overall,
        },
        "field_distributions": field_distributions,
        "per_group": group_metrics,
        "top_fp": top_fp,
        "top_fn": top_fn,
    }

    return {"predictions": predictions, "summary": summary}


if __name__ == '__main__':

    experiment_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/experiments/experiment_results.jsonl"

    model_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/training/bigger_data/best_roc.pt"

    # Load trials (assumed JSONL with one dict per line)
    trials = []
    with open(experiment_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))

    # Run predictions and analysis (auto-select threshold)
    result = predict_and_analyze(trials, ckpt_path=model_path, device="auto", threshold=0.7)

    # Pretty, readable output
    summary = result["summary"]
    counts = summary.get("counts", {})
    per_group = summary.get("per_group", {})
    top_fp = summary.get("top_fp", [])
    top_fn = summary.get("top_fn", [])
    ld = summary.get("label_distribution", {})
    fd = summary.get("field_distributions", {})

    print("\n======================== Model Evaluation Summary ========================")
    print(f"Threshold (tau): {summary.get('threshold'):.4f}")
    print(f"ROC-AUC        : {summary.get('auc_roc')}  |  PR-AUC: {summary.get('auc_pr')}")
    print(f"Accuracy       : {summary.get('accuracy'):.4f}")
    print("Counts         : TP={tp}  TN={tn}  FP={fp}  FN={fn}  Total={tot}".format(
        tp=counts.get('tp',0), tn=counts.get('tn',0), fp=counts.get('fp',0), fn=counts.get('fn',0), tot=counts.get('total',0)))

    # Label distribution
    if ld:
        print("\n------------------------------ Label Distribution ------------------------------")
        print(f"collision: {ld.get('collision',0)}  |  no_collision: {ld.get('no_collision',0)}  |  base_rate: {ld.get('base_rate',0.0):.3f}")

    # Field distributions
    if fd:
        print("\n----------------------------- Field Distributions ------------------------------")
        for key, counts_map in fd.items():
            print(f"{key}:")
            # show top up to 10 entries
            shown = 0
            for val, c in counts_map.items():
                print(f"  - {val}: {c}")
                shown += 1
                if shown >= 10:
                    break

    # Per-group breakdown
    if per_group:
        print("\n--------------------------- Per-Group (bucket) ---------------------------")
        header = f"{'group':<12}{'n':>6}{'base':>8}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'tp':>6}{'tn':>6}{'fp':>6}{'fn':>6}"
        print(header)
        print("-" * len(header))
        for g, m in per_group.items():
            print(f"{g:<12}{m.get('count',0):>6}{m.get('base_rate',0.0):>8.3f}{m.get('accuracy',0.0):>8.3f}{m.get('precision',0.0):>8.3f}{m.get('recall',0.0):>8.3f}{m.get('f1',0.0):>8.3f}{m.get('tp',0):>6}{m.get('tn',0):>6}{m.get('fp',0):>6}{m.get('fn',0):>6}")

    # Hardest errors
    if top_fp:
        print("\n------------------------------ Top False Positives ------------------------------")
        for r in top_fp:
            print(f"idx={r.get('index')}  bucket={r.get('bucket')}  dims={r.get('dims')}  p_coll={r.get('p_collision'):.4f}")
    if top_fn:
        print("\n------------------------------ Top False Negatives ------------------------------")
        for r in top_fn:
            print(f"idx={r.get('index')}  bucket={r.get('bucket')}  dims={r.get('dims')}  p_coll={r.get('p_collision'):.4f}")
    print("========================================================================\n")

    # Save per-trial predictions next to the input file
    pred_out = f"{experiment_file}.predictions.jsonl"
    with open(pred_out, "w") as f:
        for r in result["predictions"]:
            f.write(json.dumps(r) + "\n")
    print("Wrote predictions to:", pred_out)