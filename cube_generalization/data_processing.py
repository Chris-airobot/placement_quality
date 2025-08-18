# Minimal data pipeline: regroup -> rich pairs -> combine -> split
# Lean & deterministic. Uses your new extras if present.

import os, json, glob, math, re, hashlib, random
from collections import Counter
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------- CONFIG (edit these) ----------------------------
DATA_DIR          = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
RAW_SUBDIR        = "raw_data"          # your raw files
FIXED_SUBDIR      = "fixed_data"        # regrouped (grasp-first)
PROCESSED_SUBDIR  = "processed_data"    # paired samples
COMBINED_SUBDIR   = "combined_data"     # all_data.json + splits
RAW_GLOB          = "object_*.json"

# Robust regroup rounding (5 is safe)
REGROUP_ROUND_DECIMALS = 5

# Bucket quotas & prior target (robot ≈ 0.371 success / 0.629 collision)
QUOTAS = {'SAME':0, 'SMALL':2, 'MEDIUM':3, 'ADJACENT':3, 'OPPOSITE':2}
TARGET_SUCCESS_RATE = 0.37   # fraction of non-collision placements per bucket
GROUP_CAP = 300              # max pairs per grasp group (per grasp key)
SEED = 42
# ----------------------------------------------------------------------------


# ========================== 1) REGROUP (robust) ==============================
def _quat_wxyz_to_R(q):
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

def _local_signature_from_extras(item, nd=5):
    """Use recorded extras when present; fall back to quat math otherwise."""
    # item layout: [hand_pose7, obj_pose7, success, collision, ..., extras_dict?]
    extras = item[7] if (len(item) > 7 and isinstance(item[7], dict)) else None
    if extras and ('t_loc' in extras) and ('R_loc6' in extras):
        t = tuple(np.round(np.asarray(extras['t_loc'], float), nd))
        r = tuple(np.round(np.asarray(extras['R_loc6'], float).reshape(-1), nd))
        return (t, r)

    # fallback: derive R_loc/t_loc from poses
    hand, obj = item[0], item[1]
    t_h = np.array(hand[:3], float)
    R_h = _quat_wxyz_to_R(hand[3:7])
    t_o = np.array(obj[:3], float)
    R_o = _quat_wxyz_to_R(obj[3:7])
    R_loc = R_o.T @ R_h
    t_loc = R_o.T @ (t_h - t_o)
    t_sig = tuple(np.round(t_loc, nd))
    R_sig = tuple(np.round(R_loc.flatten(), nd))
    return (t_sig, R_sig)

def regroup_raw_to_grasp_first(raw_path, out_path, nd=REGROUP_ROUND_DECIMALS):
    """INPUT (raw): pose-buckets. OUTPUT: grasp-first via pose-invariant signatures."""
    with open(raw_path, "r") as f:
        data = json.load(f)

    pose_keys = sorted([k for k in data if k.startswith("grasp_")],
                       key=lambda k: int(k.split("_")[1]))
    if not pose_keys:
        with open(out_path, "w") as f:
            json.dump({}, f)
        return {"num_grasps": 0, "num_poses": 0, "out": out_path}

    per_pose_maps = []
    for k in pose_keys:
        m = {}
        for itm in data[k]:
            sig = _local_signature_from_extras(itm, nd=nd)
            m[sig] = itm  # last wins if duplicates
        per_pose_maps.append(m)

    common = set(per_pose_maps[0].keys())
    for m in per_pose_maps[1:]:
        common &= set(m.keys())
    common = sorted(list(common))  # deterministic order

    out = {f"grasp_{g}": [m[sig] for m in per_pose_maps] for g, sig in enumerate(common)}
    with open(out_path, "w") as f:
        json.dump(out, f)
    return {"num_grasps": len(common), "num_poses": len(pose_keys), "out": out_path}

def regroup_all_raw(data_dir=DATA_DIR, nd=REGROUP_ROUND_DECIMALS):
    raw_dir = os.path.join(data_dir, RAW_SUBDIR)
    fix_dir = os.path.join(data_dir, FIXED_SUBDIR)
    os.makedirs(fix_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, RAW_GLOB)))
    print(f"[regroup] {len(files)} raw files in {raw_dir}")
    for p in tqdm(files, desc="regrouping"):
        out = os.path.join(fix_dir, f"repacked_{os.path.basename(p)}")
        info = regroup_raw_to_grasp_first(p, out, nd=nd)
        print(f"  {os.path.basename(out)}: grasps={info['num_grasps']} poses={info['num_poses']}")


# ====================== 2) BUILD RICH (paired) ===============================
_ADJ = {1:{2,4,5,6}, 2:{1,3,5,6}, 3:{2,4,5,6}, 4:{1,3,5,6}, 5:{1,2,3,4}, 6:{1,2,3,4}}

def _face_id_from_extras_or_q(item):
    extras = item[7] if (len(item) > 7 and isinstance(item[7], dict)) else None
    if extras and ('final_face_id' in extras):
        return int(extras['final_face_id'])
    # fallback from quaternion
    w,x,y,z = item[1][3:7]
    rot = R.from_quat([x,y,z,w])  # [x,y,z,w]
    up = np.array([0,0,1], float)
    faces = {1:[0,0,1], 2:[1,0,0], 3:[0,0,-1], 4:[-1,0,0], 5:[0,-1,0], 6:[0,1,0]}
    return max(faces, key=lambda k: np.dot(rot.apply(faces[k]), up))

def _bucket(meta_i, meta_j):
    fi, fj = meta_i['face'], meta_j['face']
    if fi == fj:
        dq = R.from_quat(meta_j['quat']) * R.from_quat(meta_i['quat']).inv()
        ang = dq.magnitude() * 180.0 / math.pi
        if ang < 1:   return 'SAME'
        if ang <= 30: return 'SMALL'
        if ang <= 90: return 'MEDIUM'
        return 'MEDIUM'
    return 'ADJACENT' if fj in _ADJ[fi] else 'OPPOSITE'

def _sample_bucket(indices, k, cases, rng, target_success_rate=TARGET_SUCCESS_RATE):
    if not indices or k <= 0:
        return []
    pos = [j for j in indices if not cases[j][3]]  # no collision at placement
    neg = [j for j in indices if cases[j][3]]
    k_pos = min(len(pos), int(round(k * target_success_rate)))
    k_neg = min(len(neg), max(0, k - k_pos))
    chosen = []
    if pos:
        chosen.extend(pos if len(pos) <= k_pos else rng.sample(pos, k_pos))
    if neg:
        chosen.extend(neg if len(neg) <= k_neg else rng.sample(neg, k_neg))
    rem = k - len(chosen)
    if rem > 0:
        pool = [j for j in indices if j not in chosen]
        if pool:
            chosen.extend(pool if len(pool) <= rem else rng.sample(pool, rem))
    return chosen

def build_rich_pairs(data_dir=DATA_DIR,
                     source_subdir=FIXED_SUBDIR,
                     source_glob="repacked_object_*.json",
                     out_subdir=PROCESSED_SUBDIR,
                     quotas=QUOTAS,
                     group_cap=GROUP_CAP,
                     seed=SEED):
    rng = random.Random(seed)
    src_dir = os.path.join(data_dir, source_subdir)
    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    global_counts = Counter()
    total_pairs = 0
    succ_pairs  = 0

    files = sorted(glob.glob(os.path.join(src_dir, source_glob)))
    print(f"[pairs] {len(files)} fixed files in {src_dir}")
    for path in tqdm(files, desc="building pairs"):
        with open(path, "r") as f:
            grasp_first = json.load(f)

        out = {k: [] for k in ("grasp_poses","initial_object_poses",
                               "final_object_poses","success_labels",
                               "collision_labels")}

        for _, cases in grasp_first.items():
            # meta per pose index for this grasp
            meta = []
            for c in cases:
                qw,qx,qy,qz = c[1][3:7]
                meta.append({'quat':[qx,qy,qz,qw], 'face': _face_id_from_extras_or_q(c)})

            # clean pick (success & no collision & no extra flags if present)
            clean_idx = []
            for i,c in enumerate(cases):
                ok = (bool(c[2]) and not bool(c[3]))
                if len(c) > 6:
                    ok = ok and (not bool(c[4])) and (not bool(c[5])) and (not bool(c[6]))
                if ok:
                    clean_idx.append(i)

            produced = 0
            for i in clean_idx:
                if group_cap is not None and produced >= group_cap:
                    break
                # buckets
                buckets = {b: [] for b in quotas.keys()}
                for j in range(len(cases)):
                    b = _bucket(meta[i], meta[j])
                    if b in buckets:
                        buckets[b].append(j)
                # choose placements
                chosen = []
                leftovers = 0
                for b, k in quotas.items():
                    take = _sample_bucket(buckets.get(b, []), k, cases, rng, TARGET_SUCCESS_RATE)
                    chosen.extend(take)
                    leftovers += max(0, k - len(take))
                if leftovers > 0:
                    pool = []
                    for b, lst in buckets.items():
                        pool.extend([j for j in lst if j not in chosen])
                    if pool:
                        chosen.extend(pool if len(pool) <= leftovers else rng.sample(pool, leftovers))
                # ensure at least one ADJACENT & OPPOSITE if available
                for must in ("ADJACENT","OPPOSITE"):
                    lst = buckets.get(must, [])
                    if lst and not any(j in lst for j in chosen):
                        pool = [j for j in lst if j not in chosen]
                        if pool:
                            chosen.append(rng.sample(pool, 1)[0])
                # cap
                if group_cap is not None and len(chosen) > (group_cap - produced):
                    chosen = rng.sample(chosen, group_cap - produced)

                pick = cases[i]
                for j in chosen:
                    place = cases[j]
                    out['grasp_poses'].append(pick[0])
                    out['initial_object_poses'].append(pick[1])
                    out['final_object_poses'].append(place[1])
                    out['success_labels'].append(float(place[2]))
                    out['collision_labels'].append(float(place[3]))
                    # stats
                    b = _bucket(meta[i], meta[j])
                    global_counts[b] += 1
                    total_pairs += 1
                    if not place[3]:
                        succ_pairs += 1
                produced += len(chosen)

        # save this file
        fname = f"processed_{os.path.basename(path)}"
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(out, f)
        col = sum(1 for x in out['collision_labels'] if x > 0.5)
        n = len(out['collision_labels'])
        print(f"  {fname}: {n} pairs | collision rate {col/n:.3f}")

    if total_pairs > 0:
        succ_rate = succ_pairs / total_pairs
        print(f"[pairs] GLOBAL: pairs={total_pairs}  success_rate={succ_rate:.3f}  (target={TARGET_SUCCESS_RATE:.3f})")
        for k in ('SAME','SMALL','MEDIUM','ADJACENT','OPPOSITE'):
            n = global_counts[k]
            print(f"        {k:<8} {n:>8}  {n/total_pairs:6.3f}")


# ==================== 3) COMBINE (streaming writer) =========================
def combine_processed_data(data_dir=DATA_DIR,
                           processed_subdir=PROCESSED_SUBDIR,
                           out_subdir=COMBINED_SUBDIR):
    proc_dir = os.path.join(data_dir, processed_subdir)
    out_dir  = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(proc_dir, "processed_*.json")))
    out_path = os.path.join(out_dir, "all_data.json")
    total_written = 0
    succ_total = 0
    dims_fail = 0

    print(f"[combine] writing {out_path} from {len(files)} files")
    with open(out_path, "w") as out_f:
        out_f.write("[\n")
        first = True
        for fp in tqdm(files, desc="combining"):
            m = re.search(r'object_([0-9.]+)_([0-9.]+)_([0-9.]+)\.json', fp)
            if m is None:
                dims_fail += 1
            dims = [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else [None,None,None]
            d = json.load(open(fp, "r"))
            n = len(d["grasp_poses"])
            for i in range(n):
                obj = {
                    "grasp_pose": d["grasp_poses"][i],
                    "initial_object_pose": d["initial_object_poses"][i],
                    "final_object_pose": d["final_object_poses"][i],
                    "success_label": d["success_labels"][i],
                    "collision_label": d["collision_labels"][i],
                    "object_dimensions": dims
                }
                if not first: out_f.write(",\n")
                first = False
                out_f.write(json.dumps(obj))
                total_written += 1
                if obj["success_label"] and not obj["collision_label"]:
                    succ_total += 1
        out_f.write("\n]")
    print("[combine] done")
    if total_written > 0:
        print(f"[combine] total={total_written}  success_rate={succ_total/total_written:.3f}  dims_parse_fail={dims_fail}")


# ====================== 4) SPLIT (no leakage) ===============================
def _dims_key(dims, nd=6):
    if not dims: return None
    try:
        return tuple(round(float(x), nd) for x in dims)
    except Exception:
        return None

def _stream_array(path):
    """Stream a JSON array of objects from disk without loading into memory."""
    with open(path, "r") as f:
        in_array = False
        buf = []
        depth = 0
        in_str = False
        esc = False

        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            for c in chunk:
                if not in_array:
                    if c == '[':
                        in_array = True
                    continue
                if c == ']' and not in_str and depth == 0:
                    return
                if c == '"' and not esc:
                    in_str = not in_str
                esc = (c == '\\' and in_str and not esc)
                if in_str:
                    if depth > 0:
                        buf.append(c)
                    continue
                if c == '{':
                    if depth == 0:
                        buf = ['{']; depth = 1
                    else:
                        buf.append('{'); depth += 1
                elif c == '}':
                    if depth > 0:
                        buf.append('}'); depth -= 1
                        if depth == 0:
                            yield json.loads(''.join(buf)); buf = []
                else:
                    if depth > 0:
                        buf.append(c)

def _count_split_file(path):
    n = succ = coll = 0
    dims_set = set()
    for obj in _stream_array(path):
        n += 1
        if obj.get("collision_label"):
            coll += 1
        if obj.get("success_label") and not obj.get("collision_label"):
            succ += 1
        dims_set.add(_dims_key(obj.get("object_dimensions")))
    succ_rate = (succ / n) if n else 0.0
    coll_rate = (coll / n) if n else 0.0
    return {"n": n, "succ_rate": succ_rate, "coll_rate": coll_rate, "dims": dims_set}

def split_all_data(data_dir=DATA_DIR,
                   out_subdir=COMBINED_SUBDIR,
                   train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=SEED,
                   expected_total: int | None = None,
                   show_progress: bool = True):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    out_dir = os.path.join(data_dir, out_subdir)
    in_path = os.path.join(out_dir, "all_data.json")
    train_p = os.path.join(out_dir, "train.json")
    val_p   = os.path.join(out_dir, "val.json")
    test_p  = os.path.join(out_dir, "test.json")

    outs = {"train.json": open(train_p, "w"),
            "val.json":   open(val_p, "w"),
            "test.json":  open(test_p, "w")}
    for f in outs.values():
        f.write("[\n")
    first = {k: True for k in outs}

    def pick_split_by_key(key_tuple):
        if key_tuple is None:
            r = random.Random(seed).random()
            return "train.json" if r < train_frac else ("val.json" if r < train_frac + val_frac else "test.json")
        h = hashlib.md5((str(key_tuple) + f"|{seed}").encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        return "train.json" if u < train_frac else ("val.json" if u < train_frac + val_frac else "test.json")

    print(f"[split] streaming {in_path}")
    iterator = _stream_array(in_path)
    if show_progress and expected_total is not None:
        iterator = tqdm(iterator, total=expected_total, unit="obj", desc="Splitting", smoothing=0.05)

    for obj in iterator:
        key = _dims_key(obj.get("object_dimensions"))
        tgt = pick_split_by_key(key)
        if not first[tgt]:
            outs[tgt].write(",\n")
        else:
            first[tgt] = False
        outs[tgt].write(json.dumps(obj))

    for k, f in outs.items():
        f.write("\n]"); f.close()
    print("[split] wrote train/val/test")

    train_stats = _count_split_file(train_p)
    val_stats   = _count_split_file(val_p)
    test_stats  = _count_split_file(test_p)

    print(f"train.json: n={train_stats['n']}  success_rate={train_stats['succ_rate']:.3f}  collision_rate={train_stats['coll_rate']:.3f}")
    print(f"val.json:   n={val_stats['n']}    success_rate={val_stats['succ_rate']:.3f}    collision_rate={val_stats['coll_rate']:.3f}")
    print(f"test.json:  n={test_stats['n']}   success_rate={test_stats['succ_rate']:.3f}   collision_rate={test_stats['coll_rate']:.3f}")

    leak_tv = train_stats["dims"] & val_stats["dims"] - {None}
    leak_tt = train_stats["dims"] & test_stats["dims"] - {None}
    leak_vt = val_stats["dims"]   & test_stats["dims"]  - {None}
    print(f"leaky buckets (train∩val): {len(leak_tv)}")
    print(f"leaky buckets (train∩test): {len(leak_tt)}")
    print(f"leaky buckets (val∩test): {len(leak_vt)}")

# =============================== PIPELINE ===================================
def run_pipeline():
    regroup_all_raw()
    build_rich_pairs()
    combine_processed_data()
    split_all_data()  # no expected_total needed for your 243-file pilot

if __name__ == "__main__":
    run_pipeline()
