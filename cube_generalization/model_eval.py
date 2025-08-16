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




def generate_experiment_predictions(corners_only: bool = True):
    """
    Generate model predictions for experiments and write predictions JSON.

    New model signature: FinalCornersAuxModel(corners_24, aux_12)
      - corners_24: final corners in world frame, flattened (24,)
      - aux_12: [t_loc(3), R_loc6D(6), dims(3)]
        t_loc, R_loc computed from (grasp_pose, initial_object_pose)
    """
    import json
    import numpy as np
    import torch
    from tqdm import tqdm

    # --- tiny helpers (self-contained) ---
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
        # Zhou et al. CVPR'19: take first two columns
        a1 = R[:, 0]; a2 = R[:, 1]
        return np.concatenate([a1, a2], axis=0).astype(np.float32)

    def zscore(x, mean, std, eps=1e-8):
        if isinstance(mean, torch.Tensor): mean = mean.detach().cpu().numpy()
        if isinstance(std, torch.Tensor):  std  = std.detach().cpu().numpy()
        return (x - mean) / (std + eps)

    # --- hardcoded paths (kept) ---
    experiment_file  = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiments.json"
    predictions_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions_corners_only.json"

    # --- load experiments ---
    with open(experiment_file, "r") as f:
        experiments = json.load(f)
    print(f"✅ Loaded {len(experiments)} experiments")

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load model + stats ---
    from model import FinalCornersAuxModel
    ckpt = torch.load("/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/training/checkpoints/best.pt",
                      map_location="cpu")
    model = FinalCornersAuxModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stats = ckpt.get("normalization_stats", None)
    assert stats is not None, "Checkpoint missing normalization_stats"

    # expected keys: final_corners_mean/std, tloc_mean/std, dims_mean/std
    for k in ("final_corners_mean", "final_corners_std", "tloc_mean", "tloc_std", "dims_mean", "dims_std"):
        assert k in stats, f"Missing {k} in normalization_stats"

    # --- use your dataset helpers for corners ---
    from dataset import cuboid_corners_local_ordered, transform_points_to_world

    results = []
    print(f"\n🔮 Generating predictions for {len(experiments)} experiments...")
    for i, data in enumerate(tqdm(experiments, desc="Predicting")):
        dx, dy, dz = data["object_dimensions"]
        init_pose  = np.asarray(data["initial_object_pose"], dtype=np.float32)  # [tx,ty,tz, qw,qx,qy,qz]
        final_pose = np.asarray(data["final_object_pose"],   dtype=np.float32)
        grasp_pose = np.asarray(data["grasp_pose"],          dtype=np.float32)

        # --- final corners in world (flatten to 24) ---
        corners_local = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)  # [8,3]
        final_world   = transform_points_to_world(corners_local, final_pose)         # [8,3]
        corners_24    = final_world.reshape(-1)                                      # (24,)

        # --- hand-relative block computed wrt INIT object pose (constant per grasp) ---
        t_o  = init_pose[:3];      q_o  = init_pose[3:7]
        t_h  = grasp_pose[:3];     q_h  = grasp_pose[3:7]
        R_o  = quat_wxyz_to_R(q_o)                 # 3x3
        R_h  = quat_wxyz_to_R(q_h)                 # 3x3
        R_loc = R_o.T @ R_h                        # 3x3
        t_loc = R_o.T @ (t_h - t_o)                # (3,)
        R6    = rot6d_from_R(R_loc)                # (6,)
        dims  = np.array([dx, dy, dz], np.float32) # (3,)

        aux_12 = np.concatenate([t_loc.astype(np.float32), R6, dims], axis=0)  # (12,)

        # --- z-score normalize with training stats ---
        corners_24_n = zscore(corners_24, stats["final_corners_mean"], stats["final_corners_std"]).astype(np.float32)
        tloc_n       = zscore(t_loc,       stats["tloc_mean"],          stats["tloc_std"]).astype(np.float32)
        dims_n       = zscore(dims,        stats["dims_mean"],          stats["dims_std"]).astype(np.float32)

        aux_12_n = np.concatenate([tloc_n, R6, dims_n], axis=0).astype(np.float32)

        # --- tensors & forward ---
        c_t  = torch.from_numpy(corners_24_n).unsqueeze(0).to(device)  # [1,24]
        a_t  = torch.from_numpy(aux_12_n).unsqueeze(0).to(device)      # [1,12]
        with torch.no_grad():
            logits = model(c_t, a_t)                                   # [1,1]
            p_collision = torch.sigmoid(logits).item()
            p_nocoll    = 1.0 - p_collision

        # --- write both directions (avoid the earlier confusion) ---
        results.append({
            "object_dimensions": data["object_dimensions"],
            "pred_collision": float(p_collision),
            "pred_no_collision": float(p_nocoll),
            "pred_no_collision_label": int(p_nocoll > 0.9)
        })

        if (i+1) % 200 == 0:
            print(f"Processed {i+1}/{len(experiments)}")

    # --- save line-delimited JSON ---
    with open(predictions_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"✅ Wrote {len(results)} predictions to {predictions_file}")

    preds = [r["pred_no_collision"] for r in results]
    print("\n📊 PREDICTION STATS")
    print(f"  mean={np.mean(preds):.4f}  min={np.min(preds):.4f}  max={np.max(preds):.4f}")
    print("🔍 first 5:")
    for r in results[:5]:
        print(f"  dims={r['object_dimensions']}  p_no_coll={r['pred_no_collision']:.4f}  p_coll={r['pred_collision']:.4f}")



def replace_scores_with_indices(
    experiment_results_path: str,
    predictions_path: str,
    output_path: str | None = None,
    one_based: bool = True,
):
    """
    Replace each record's 'prediction_score' in experiment results JSONL with the
    value looked up by index from the predictions file.

    Index semantics:
    - If an experiment record has field 'index' == k, we use the k-th value from
      the predictions file (1-based if one_based=True) and set it as prediction_score.
    - If 'index' is missing, we fall back to the record's line order.

    - experiment_results_path: JSONL with one JSON object per line, containing 'prediction_score'.
    - predictions_path: JSON or JSONL with one JSON object per line, each holding a score (key: 'pred_no_collision' preferred).
    - output_path: path to write the modified JSONL. If None, writes alongside input with suffix '.indexed.jsonl'.
    - one_based: if True, indices start at 1; else 0.
    """

    if output_path is None:
        root, ext = os.path.splitext(experiment_results_path)
        output_path = f"{root}.indexed.jsonl"

    # Load prediction scores into a list for O(1) index lookup
    pred_scores: list[float] = []
    with open(predictions_path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Stop if predictions file is a single large JSON array; attempt to load once
                f.seek(0)
                try:
                    arr = json.load(f)
                    for it in arr:
                        v = (
                            it.get("pred_no_collision")
                            if isinstance(it, dict)
                            else None
                        )
                        if v is not None:
                            pred_scores.append(float(v))
                except Exception:
                    pass
                break
            else:
                # Typical JSONL line
                v = obj.get("pred_no_collision")
                if v is None:
                    v = obj.get("prediction_score")
                if v is None:
                    v = obj.get("score")
                if v is not None:
                    pred_scores.append(float(v))

    total = len(pred_scores)
    print(f"Loaded {total} prediction scores from {predictions_path}")

    # Process experiment results line by line
    replaced = 0
    with open(experiment_results_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(tqdm(fin, desc="Replacing scores")):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Write through unmodified if malformed
                fout.write(line)
                continue

            # Determine index for lookup: prefer explicit 'index' field
            if "index" in rec and isinstance(rec["index"], int):
                k = rec["index"]
                idx = (k - 1) if one_based else k
            else:
                # Fallback to file order
                idx = (i if one_based else i - 1)

            # Replace if within range
            if 0 <= idx < total and "prediction_score" in rec:
                rec["prediction_score"] = pred_scores[idx]
                replaced += 1

            # Write updated record
            fout.write(json.dumps(rec) + "\n")

    print(f"Updated 'prediction_score' in {replaced} records → {output_path}")

if __name__ == '__main__':


    # generate_experiment_predictions()
     # Defaults to the paths you mentioned; adjust as needed
    experiment_results_path = \
        "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/experiment_results_test_data.jsonl"
    predictions_path = \
        "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions_corners_only.json"
    output_path = \
        "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/experiments/new_corners_only.jsonl"

    replace_scores_with_indices(
        experiment_results_path=experiment_results_path,
        predictions_path=predictions_path,
        output_path=output_path,
        one_based=True,
    )
