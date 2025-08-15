import json, math, os
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# ---------- quaternion helpers (WXYZ order) ----------
import numpy as np, math

def quat_wxyz_to_R(q):
    w,x,y,z = [float(v) for v in q]
    n = math.sqrt(w*w+x*x+y*y+z*z) or 1.0
    w,x,y,z = w/n, x/n, y/n, z/n
    xx,yy,zz = x*x, y*y, z*z
    wx,wy,wz = w*x, w*y, w*z
    xy,xz,yz = x*y, x*z, y*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)],
    ], float)

def local_signature(item, nd=5):
    """item = [hand_pose7, obj_pose7, ...] with WXYZ quats"""
    hand, obj = item[0], item[1]
    t_h = np.array(hand[:3], float);  R_h = quat_wxyz_to_R(hand[3:7])
    t_o = np.array(obj[:3], float);   R_o = quat_wxyz_to_R(obj[3:7])
    R_loc = R_o.T @ R_h
    t_loc = R_o.T @ (t_h - t_o)
    # Quantize to be robust to float noise
    t_sig = tuple(np.round(t_loc, nd))
    R_sig = tuple(np.round(R_loc.flatten(), nd))
    return (t_sig, R_sig)

# ---------- audit determinism ----------
def audit_raw_v5(path):
    data = json.load(open(path))
    keys = sorted([k for k in data if k.startswith("grasp_")], key=lambda k: int(k.split("_")[1]))
    lens = [len(data[k]) for k in keys]
    all_same_len = len(set(lens)) == 1

    # signatures per pose
    sig_lists = []
    for k in keys:
        sigs = []
        for itm in data[k]:
            ok, sig = local_signature(itm, nd=5)
            if ok:
                sigs.append(sig)
        sig_lists.append(sigs)

    # identical order?
    identical = all(sig_lists[i] == sig_lists[0] for i in range(1, len(sig_lists))) if sig_lists else True

    return {
        "num_pose_buckets": len(keys),
        "lengths": lens,
        "all_same_len": all_same_len,
        "identical_order_by_signature": identical,
        "first_pose_unique_sigs": len(set(sig_lists[0])) if sig_lists else 0,
    }

# ---------- regroup to grasp-first (intersection across all poses) ----------
def regroup_raw_v5_to_grasp_first(raw_path, out_path, nd=5):
    data = json.load(open(raw_path))
    keys = sorted([k for k in data if k.startswith("grasp_")], key=lambda k: int(k.split("_")[1]))

    # build per-pose maps: sig -> item
    per_pose_maps = []
    for k in keys:
        m = {}
        for itm in data[k]:
            ok, sig = local_signature(itm, nd=nd)
            if ok:
                m[sig] = itm  # last one wins if duplicates
        per_pose_maps.append(m)

    # intersection of signatures present in *every* pose
    common = set(per_pose_maps[0].keys())
    for m in per_pose_maps[1:]:
        common &= set(m.keys())

    # deterministic order of grasp IDs
    common = sorted(list(common))

    # build rectangular "grasp-first" structure
    out = {}
    for g, sig in enumerate(common):
        out[f"grasp_{g}"] = [m[sig] for m in per_pose_maps]

    with open(out_path, "w") as f:
        json.dump(out, f)
    return {"num_grasps": len(common), "num_poses": len(keys), "out": out_path}



import numpy as np, math, json

def R_to_quat_wxyz(R):
    # minimal, numerically stable enough for verification
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / s; x = 0.25*s
            y = (R[0,1] + R[1,0]) / s; z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / s; y = 0.25*s
            x = (R[0,1] + R[1,0]) / s; z = (R[1,2] + R[2,1]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / s; z = 0.25*s
            x = (R[0,2] + R[2,0]) / s; y = (R[1,2] + R[2,1]) / s
    return np.array([w,x,y,z], float)

def quat_to_R(q):  # same as quat_wxyz_to_R
    return quat_wxyz_to_R(q)

def geodesic_deg(RA, RB):
    R = RA.T @ RB
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    return math.degrees(math.acos(c))

def verify_repacked(path):
    data = json.load(open(path))  # grasp-first: grasp_g -> [samples over poses]
    # grab object poses from each sample’s second element
    # (assumes all samples across grasps share the same 216 object poses order)
    poses = [s[1] for s in next(iter(data.values()))]
    R_objs = [quat_to_R(p[3:7]) for p in poses]
    t_objs = [np.array(p[:3], float) for p in poses]

    max_pos_err = 0.0
    max_rot_err = 0.0
    for g, lst in data.items():
        # Use pose 0 to compute canonical local transform
        t_h0 = np.array(lst[0][0][:3], float)
        R_h0 = quat_to_R(lst[0][0][3:7])
        R_o0 = R_objs[0]; t_o0 = t_objs[0]
        R_loc = R_o0.T @ R_h0
        t_loc = R_o0.T @ (t_h0 - t_o0)

        for i, sample in enumerate(lst):
            # predict world from local + object pose i
            R_pred = R_objs[i] @ R_loc
            t_pred = t_objs[i] + R_objs[i] @ t_loc

            R_true = quat_to_R(sample[0][3:7])
            t_true = np.array(sample[0][:3], float)

            pos_err = float(np.linalg.norm(t_pred - t_true))
            rot_err = geodesic_deg(R_pred, R_true)

            max_pos_err = max(max_pos_err, pos_err)
            max_rot_err = max(max_rot_err, rot_err)

    return {"max_position_error_m": max_pos_err, "max_rotation_error_deg": max_rot_err}

data_folder = "/media/chris/LABPassport/Chris/data/box_simulation/v5/data_collection/raw_data"
output_dir = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/fixed_data/"
files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
for file in tqdm(files, desc="Regrouping raw v5 -> grasp-first", unit="file"):
	out_path = os.path.join(output_dir, f"reordered_{file}")
	original_path = os.path.join(data_folder, file)
	res = regroup_raw_v5_to_grasp_first(original_path, out_path, nd=5)
