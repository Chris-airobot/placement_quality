#!/usr/bin/env python3
"""
Preflight for palm/ground clearance scalars on your VAL split.
Computs std and single-feature ROC-AUC for:
 - PRC (Palm–Rim Clearance): R_ped - ||palm_xy - ped_center|| - r_palm
 - PHM (Palm Height Margin): (palm_bottom_z - z_ped)
 - GM  (Ground Margin):      (palm_bottom_z - z_ground)

We also report the fraction of samples with negative margin (risk zone).

USAGE (edit paths & constants as needed):
  python preflight_palm_clearances.py \
      --val_json /path/to/v6/data_collection/combined_data/val.json \
      --ped_center_x 0.0 --ped_center_y 0.0 \
      --ped_radius 0.10 --z_ped 0.90 --z_ground 0.0 \
      --hand_down_axis z --hand_down_sign -1 \
      --r_palm 0.015 --h_palm_down 0.015 \
      --sample 200000 --seed 0

IMPORTANT:
- Edit the IMPORT line below so this script can import your FinalCornersHandDataset.
- This script does NOT train anything; it's a read-only scan over VAL.
"""

import argparse, sys, os, json, numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.spatial.transform import Rotation as R

# ---------- EDIT THIS IMPORT TO MATCH YOUR PROJECT -------------
# Example: from mypkg.dataset import FinalCornersHandDataset
try:
    from dataset import FinalCornersHandDataset  # <-- CHANGE THIS
except Exception as e:
    print("❌ Import error. Edit the import near the top of this file to point at your FinalCornersHandDataset.")
    print("   For example: 'from mypkg.dataset import FinalCornersHandDataset'")
    print("   Error was:", e)
    sys.exit(1)
# ---------------------------------------------------------------

@dataclass
class EnvCfg:
    ped_center_xy: np.ndarray  # (2,)
    ped_radius: float
    z_ped: float
    z_ground: float
    hand_down_axis: str  # 'x'|'y'|'z'
    hand_down_sign: int  # +1 or -1
    r_palm: float
    h_palm_down: float

def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney U-based ROC-AUC. Larger score => more likely collision (label=1)."""
    order = np.argsort(scores)
    y = labels[order]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n0 == 0 or n1 == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)
    sr_pos = ranks[y == 1].sum()
    auc = (sr_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)

def reconstruct_hand_place(grasp: np.ndarray, init: np.ndarray, final: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (R_place, t_place) for the hand at placement.
    Input quaternions are in Isaac order [tx,ty,tz,qw,qx,qy,qz].
    SciPy expects [x,y,z,w].
    """
    # Hand & object at grasp (init frame available)
    R_h = R.from_quat([grasp[4], grasp[5], grasp[6], grasp[3]]).as_matrix()  # [qx,qy,qz,qw]
    R_o = R.from_quat([init[4],  init[5],  init[6],  init[3]]).as_matrix()
    t_h = np.asarray(grasp[:3], dtype=np.float32)
    t_o = np.asarray(init[:3],  dtype=np.float32)

    # Final object pose
    R_f = R.from_quat([final[4], final[5], final[6], final[3]]).as_matrix()
    t_f = np.asarray(final[:3], dtype=np.float32)

    R_loc = (R_o.T @ R_h).astype(np.float32)
    t_loc = (R_o.T @ (t_h - t_o)).astype(np.float32)

    R_place = (R_f @ R_loc).astype(np.float32)
    t_place = (t_f + (R_f @ t_loc)).astype(np.float32)
    return R_place, t_place

def compute_scalars(ds: FinalCornersHandDataset, idxs: np.ndarray, cfg: EnvCfg):
    labels = np.asarray(ds.mm_label[idxs, 1], dtype=np.int32)
    prc = np.empty(len(idxs), dtype=np.float32)
    phm = np.empty(len(idxs), dtype=np.float32)
    gm  = np.empty(len(idxs), dtype=np.float32)

    axis_map = {"x":0, "y":1, "z":2}
    ai = axis_map[cfg.hand_down_axis.lower()]
    sign = float(cfg.hand_down_sign)
    e = np.zeros(3, dtype=np.float32); e[ai] = sign

    for k, i in enumerate(idxs):
        g = np.asarray(ds.mm_grasp[i], dtype=np.float32)  # [tx,ty,tz,qw,qx,qy,qz]
        o = np.asarray(ds.mm_init[i],  dtype=np.float32)
        f = np.asarray(ds.mm_final[i], dtype=np.float32)

        Rp, tp = reconstruct_hand_place(g, o, f)  # palm center proxy at place
        palm_center = tp  # (x,y,z)

        # PRC: radial clearance from pedestal rim (negative => must collide)
        rho = np.linalg.norm(palm_center[:2] - cfg.ped_center_xy)
        prc[k] = cfg.ped_radius - rho - cfg.r_palm

        # PHM/GM: palm bottom along hand "down" axis
        a_world = Rp @ e
        a_world = a_world / (np.linalg.norm(a_world) + 1e-9)
        palm_bottom = palm_center + a_world * cfg.h_palm_down
        phm[k] = palm_bottom[2] - cfg.z_ped
        gm[k]  = palm_bottom[2] - cfg.z_ground

    return labels, prc, phm, gm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_json", default="/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/combined_data/val.json", help="Path to val.json (v6)")
    ap.add_argument("--ped_center_x", type=float, default=0.2)
    ap.add_argument("--ped_center_y", type=float, default=-0.3)
    ap.add_argument("--ped_radius",   type=float, default=0.08)
    ap.add_argument("--z_ped",        type=float, default=0.10)
    ap.add_argument("--z_ground",     type=float, default=0.0)
    ap.add_argument("--hand_down_axis", type=str, default="z", choices=["x","y","z"])
    ap.add_argument("--hand_down_sign", type=int, default=+1, choices=[-1, +1])
    ap.add_argument("--r_palm",         type=float, default=0.018)
    ap.add_argument("--h_palm_down",    type=float, default=0.018)
    ap.add_argument("--sample",         type=int, default=400_000)
    ap.add_argument("--seed",           type=int, default=0)
    ap.add_argument("--cache_dir",      type=str, default=None)
    ap.add_argument("--report_out",     type=str, default=None, help="Optional path to save a JSON report")
    args = ap.parse_args()

    # Build dataset ONLY to access memmaps; pass empty stats to avoid recomputation
    ds = FinalCornersHandDataset(
        data_path=args.val_json,
        normalization_stats={},  # not used here
        is_training=False,
        cache_dir=args.cache_dir
    )
    N = ds.N
    take = min(args.sample, N)
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(N, size=take, replace=False)

    cfg = EnvCfg(
        ped_center_xy=np.array([args.ped_center_x, args.ped_center_y], dtype=np.float32),
        ped_radius=float(args.ped_radius),
        z_ped=float(args.z_ped),
        z_ground=float(args.z_ground),
        hand_down_axis=args.hand_down_axis,
        hand_down_sign=int(args.hand_down_sign),
        r_palm=float(args.r_palm),
        h_palm_down=float(args.h_palm_down),
    )

    labels, prc, phm, gm = compute_scalars(ds, idxs, cfg)

    # Risk-aligned scores: larger => more collision
    scores = {
        "PRC": -prc,
        "PHM": -phm,
        "GM":  -gm,
    }
    stds = {
        "PRC": float(np.std(prc)),
        "PHM": float(np.std(phm)),
        "GM":  float(np.std(gm)),
    }
    aucs = {k: auc_from_scores(v, labels) for k, v in scores.items()}
    neg_fracs = {
        "PRC": float((prc < 0).mean()),
        "PHM": float((phm < 0).mean()),
        "GM":  float((gm  < 0).mean()),
    }

    print("=== Preflight: PRC / PHM / GM on VAL ===")
    print(f"Samples: {take} / {N}")
    for k in ["PRC","PHM","GM"]:
        print(f"{k:>3} | std={stds[k]:.6f} | AUC={aucs[k]:.4f} | frac( {k} < 0 )={neg_fracs[k]*100:.2f}%")
    print("\nKeep ONLY scalars with std clearly > 0 and AUC >= 0.58 (rule of thumb).")

    if args.report_out:
        out = {
            "args": vars(args),
            "stds": stds,
            "aucs": aucs,
            "neg_frac": neg_fracs,
            "n_total": int(N),
            "n_sampled": int(take),
        }
        with open(args.report_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n📄 Saved report to: {args.report_out}")

if __name__ == "__main__":
    main()