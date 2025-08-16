# Minimal data pipeline: regroup -> rich pairs -> combine -> split
# Keep it tight and deterministic.

import os, json, glob, math, re, hashlib, random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import Counter

# ---------------------------- CONFIG (edit these) ----------------------------
DATA_DIR          = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection"
RAW_SUBDIR        = "raw_data"
FIXED_SUBDIR      = "fixed_data"        # regrouped (grasp-first)
PROCESSED_SUBDIR  = "processed_data"    # paired samples
COMBINED_SUBDIR   = "combined_data"     # all_data.json + splits
RAW_GLOB          = "object_*.json"

# Robust regroup rounding (5 is safe; use 6 only if tiny boxes collapse too much)
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

def _local_signature(item, nd=5):
    """
    item = [hand_pose7, obj_pose7, ...] with [x,y,z, qw,qx,qy,qz]
    Return a robust local (t, R) signature rounded to nd decimals.
    """
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

def verify_repacked_alignment(path, sample_grasps=10, tol_pos=1e-4, tol_deg=1e-3):
    """
    Spot-check that grasp-first packing preserved geometry:
    using pose 0 to define local (t,R), re-predict every other pose and compare.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        print(f"[regroup verify] {os.path.basename(path)}: empty, skipped")
        return {"ok": True, "skipped": True}

    # Object poses from any grasp (they all share poses)
    first_seq = next(iter(data.values()))
    R_objs = [_quat_wxyz_to_R(s[1][3:7]) for s in first_seq]
    t_objs = [np.array(s[1][:3], float) for s in first_seq]

    # sample a few grasp IDs
    rng = random.Random(0)
    keys = list(data.keys())
    sel = rng.sample(keys, min(sample_grasps, len(keys)))

    max_pos = 0.0
    max_deg = 0.0
    for k in sel:
        seq = data[k]
        t_h0 = np.array(seq[0][0][:3], float)
        R_h0 = _quat_wxyz_to_R(seq[0][0][3:7])
        R_o0, t_o0 = R_objs[0], t_objs[0]
        R_loc = R_o0.T @ R_h0
        t_loc = R_o0.T @ (t_h0 - t_o0)

        for i, s in enumerate(seq):
            R_pred = R_objs[i] @ R_loc
            t_pred = t_objs[i] + R_objs[i] @ t_loc
            R_true = _quat_wxyz_to_R(s[0][3:7])
            t_true = np.array(s[0][:3], float)

            pos_err = float(np.linalg.norm(t_pred - t_true))
            R_rel = R_pred.T @ R_true
            c = max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0))
            deg = math.degrees(math.acos(c))

            max_pos = max(max_pos, pos_err)
            max_deg = max(max_deg, deg)

    ok = (max_pos <= tol_pos) and (max_deg <= tol_deg)
    print(f"[regroup verify] {os.path.basename(path)}: "
          f"max_pos={max_pos:.2e} m, max_rot={max_deg:.2e} deg, ok={ok}")
    return {"ok": ok, "max_pos": max_pos, "max_rot_deg": max_deg}

def regroup_raw_v5_to_grasp_first(raw_path, out_path, nd=5):
    """
    INPUT (raw v5): keys are object-pose buckets ('grasp_0', ...), values = list of items
    OUTPUT (grasp-first): 'grasp_g' -> [item over every pose], aligned by local signature
    """
    with open(raw_path, "r") as f:
        data = json.load(f)

    # sort pose buckets by numeric index
    pose_keys = sorted([k for k in data if k.startswith("grasp_")],
                       key=lambda k: int(k.split("_")[1]))
    if not pose_keys:
        with open(out_path, "w") as f:
            json.dump({}, f)
        return {"num_grasps": 0, "num_poses": 0, "out": out_path}

    # per-pose map: sig -> item
    per_pose_maps = []
    for k in pose_keys:
        m = {}
        for itm in data[k]:
            sig = _local_signature(itm, nd=nd)
            m[sig] = itm  # last wins if duplicates
        per_pose_maps.append(m)

    # intersection of signatures across ALL poses -> rectangular set
    common = set(per_pose_maps[0].keys())
    for m in per_pose_maps[1:]:
        common &= set(m.keys())
    common = sorted(list(common))  # deterministic order

    out = {}
    for g, sig in enumerate(common):
        out[f"grasp_{g}"] = [m[sig] for m in per_pose_maps]

    with open(out_path, "w") as f:
        json.dump(out, f)
    return {"num_grasps": len(common), "num_poses": len(pose_keys), "out": out_path}

def regroup_all_raw(data_dir=DATA_DIR, nd=REGROUP_ROUND_DECIMALS):
    raw_dir = os.path.join(data_dir, RAW_SUBDIR)
    fix_dir = os.path.join(data_dir, FIXED_SUBDIR)
    os.makedirs(fix_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, RAW_GLOB)))
    print(f"[regroup] {len(files)} raw files in {raw_dir}")
    stats = []
    for p in tqdm(files, desc="regrouping"):
        out = os.path.join(fix_dir, f"repacked_{os.path.basename(p)}")
        info = regroup_raw_v5_to_grasp_first(p, out, nd=nd)
        # spot-verify a few files (cheap)
        if len(stats) < 10:
            verify_repacked_alignment(out)
        stats.append((os.path.basename(p), info["num_grasps"], info["num_poses"]))
    if stats:
        g_counts = [g for _,g,_ in stats]
        print(f"[regroup] grasps per pose: mean={np.mean(g_counts):.1f}, min={min(g_counts)}, max={max(g_counts)}")
    return stats


# ====================== 2) BUILD RICH (paired) ===============================
_ADJ = {1:{2,4,5,6}, 2:{1,3,5,6}, 3:{2,4,5,6}, 4:{1,3,5,6}, 5:{1,2,3,4}, 6:{1,2,3,4}}

def _face_id_from_q(w,x,y,z):
    # Isaac uses WXYZ; Rotation wants [x,y,z,w]
    rot = R.from_quat([x,y,z,w])
    up = np.array([0,0,1], float)
    # world-up projection of each object face normal
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
    # success = no collision at placement -> cases[j][3] == False
    pos = [j for j in indices if not cases[j][3]]
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

    global_counts = Counter()   # buckets
    total_pairs = 0
    succ_pairs  = 0            # non-collision at placement

    files = sorted(glob.glob(os.path.join(src_dir, source_glob)))
    print(f"[pairs] {len(files)} fixed files in {src_dir}")
    for path in tqdm(files, desc="building pairs"):
        with open(path, "r") as f:
            grasp_first = json.load(f)

        out = {k: [] for k in ("grasp_poses","initial_object_poses",
                               "final_object_poses","success_labels",
                               "collision_labels")}

        for g_key, cases in grasp_first.items():
            # meta per pose index for this grasp: from object pose quaternion
            meta = []
            for c in cases:
                qw,qx,qy,qz = c[1][3:7]
                meta.append({'quat':[qx,qy,qz,qw], 'face':_face_id_from_q(qw,qx,qy,qz)})

            # pick candidates: clean pick (success True AND no collision) + optional extras off
            clean_idx = []
            for i,c in enumerate(cases):
                ok = (bool(c[2]) and not bool(c[3]))
                # if extra flags exist (topple/slip/etc.), require them False for pick
                if len(c) > 6:
                    ok = ok and (not bool(c[4])) and (not bool(c[5])) and (not bool(c[6]))
                if ok:
                    clean_idx.append(i)

            produced = 0
            for i in clean_idx:
                if group_cap is not None and produced >= group_cap:
                    break

                # bucket all placements relative to this initial pose
                buckets = {b: [] for b in quotas.keys()}
                for j in range(len(cases)):
                    b = _bucket(meta[i], meta[j])
                    if b in buckets:
                        buckets[b].append(j)

                chosen = []
                leftovers = 0
                for b, k in quotas.items():
                    lst = buckets.get(b, [])
                    take = _sample_bucket(lst, k, cases, rng, TARGET_SUCCESS_RATE)
                    chosen.extend(take)
                    leftovers += max(0, k - len(take))

                if leftovers > 0:
                    pool = []
                    for b, lst in buckets.items():
                        pool.extend([j for j in lst if j not in chosen])
                    if pool:
                        chosen.extend(pool if len(pool) <= leftovers else rng.sample(pool, leftovers))

                # ensure at least one ADJACENT & one OPPOSITE if available
                for must in ("ADJACENT","OPPOSITE"):
                    lst = buckets.get(must, [])
                    if lst and not any(j in lst for j in chosen):
                        pool = [j for j in lst if j not in chosen]
                        if pool:
                            chosen.append(rng.sample(pool, 1)[0])

                # cap per-grasp output
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
        
        fname = f"processed_{os.path.basename(path)}"
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(out, f)
        # quick count to see balance
        col = sum(1 for x in out['collision_labels'] if x > 0.5)
        n = len(out['collision_labels'])
        print(f"  {fname}: {n} pairs | collision rate {col/n:.3f}")
        

    # Concise global summary across all files
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

                # end of array?
                if c == ']' and not in_str and depth == 0:
                    return

                # string handling
                if c == '"' and not esc:
                    in_str = not in_str
                esc = (c == '\\' and in_str and not esc)
                if in_str:
                    if depth > 0:
                        buf.append(c)
                    continue

                # structural handling
                if c == '{':
                    if depth == 0:
                        buf = ['{']
                        depth = 1
                    else:
                        buf.append('{')
                        depth += 1
                elif c == '}':
                    if depth > 0:
                        buf.append('}')
                        depth -= 1
                        if depth == 0:
                            yield json.loads(''.join(buf))
                            buf = []
                else:
                    if depth > 0:
                        buf.append(c)


def _count_split_file(path):
    """Stream-count size, success rate, collision rate and gather unique dims keys."""
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

from tqdm import tqdm  # already imported at top

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

    # open outputs and write array headers
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

    # stream the big array into the 3 splits, with a progress bar
    print(f"[split] streaming {in_path}")
    iterator = _stream_array(in_path)
    if show_progress:
        iterator = tqdm(iterator, total=expected_total, unit="obj", desc="Splitting", smoothing=0.05)

    for obj in iterator:
        key = _dims_key(obj.get("object_dimensions"))
        tgt = pick_split_by_key(key)
        if not first[tgt]:
            outs[tgt].write(",\n")
        else:
            first[tgt] = False
        outs[tgt].write(json.dumps(obj))

    # close the arrays/files
    for k, f in outs.items():
        f.write("\n]")
        f.close()
    print("[split] wrote train/val/test")

    # streaming sanity: priors + leakage by dims keys (uses _stream_array)
    train_stats = _count_split_file(train_p)
    val_stats   = _count_split_file(val_p)
    test_stats  = _count_split_file(test_p)

    print(f"train.json: n={train_stats['n']}  success_rate={train_stats['succ_rate']:.3f}  collision_rate={train_stats['coll_rate']:.3f}")
    print(f"val.json:   n={val_stats['n']}    success_rate={val_stats['succ_rate']:.3f}    collision_rate={val_stats['coll_rate']:.3f}")
    print(f"test.json:  n={test_stats['n']}   success_rate={test_stats['coll_rate']:.3f}   collision_rate={test_stats['coll_rate']:.3f}")

    leak_tv = train_stats["dims"] & val_stats["dims"] - {None}
    leak_tt = train_stats["dims"] & test_stats["dims"] - {None}
    leak_vt = val_stats["dims"]   & test_stats["dims"]  - {None}
    print(f"leaky buckets (train∩val): {len(leak_tv)}")
    print(f"leaky buckets (train∩test): {len(leak_tt)}")
    print(f"leaky buckets (val∩test): {len(leak_vt)}")
    if leak_tv or leak_tt or leak_vt:
        ex = list(leak_tv or leak_tt or leak_vt)[:5]
        print(f"examples: {ex}")

def split_all_data_fast_ijson(
    data_dir=DATA_DIR,
    out_subdir=COMBINED_SUBDIR,
    train_frac=0.8, val_frac=0.1, test_frac=0.1,
    seed=SEED,
    expected_total: int | None = None,
):
    """
    SUPER-FAST splitter using ijson (C backend) for streaming parse of a giant JSON array.
    Writes JSON arrays (not JSONL) so downstream stays unchanged.
    """
    import decimal
    import ijson
    try:
        import orjson as _fastjson
        def _default(o):
            if isinstance(o, decimal.Decimal):
                return float(o)
            raise TypeError
        _dumps = lambda o: _fastjson.dumps(
            o, default=_default  # convert any lingering Decimal -> float
        ).decode("utf-8")
    except Exception:
        _dumps = lambda o: json.dumps(
            o,
            default=lambda x: float(x) if isinstance(x, decimal.Decimal) else str(x)
        )

    out_dir = os.path.join(data_dir, out_subdir)
    in_path = os.path.join(out_dir, "all_data.json")
    train_p = os.path.join(out_dir, "train.json")
    val_p   = os.path.join(out_dir, "val.json")
    test_p  = os.path.join(out_dir, "test.json")

    # Open outputs & headers
    outs = {
        "train.json": open(train_p, "w"),
        "val.json":   open(val_p, "w"),
        "test.json":  open(test_p, "w"),
    }
    for f in outs.values():
        f.write("[\n")
    first = {k: True for k in outs}

    # Same split-by-dims-key logic you already use
    def _dims_key(dims, nd=6):
        if not dims: return None
        try:
            return tuple(round(float(x), nd) for x in dims)
        except Exception:
            return None

    import hashlib, random
    def pick_split_by_key(key_tuple):
        if key_tuple is None:
            r = random.Random(seed).random()
            return "train.json" if r < train_frac else ("val.json" if r < train_frac + val_frac else "test.json")
        h = hashlib.md5((str(key_tuple) + f"|{seed}").encode("utf-8")).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        return "train.json" if u < train_frac else ("val.json" if u < train_frac + val_frac else "test.json")

    # Small stats & leakage via dims sets (tiny)
    from collections import defaultdict
    succ = defaultdict(int)
    coll = defaultdict(int)
    count = defaultdict(int)
    dims_sets = defaultdict(set)

    print(f"[split-fast] streaming {in_path} with ijson")
    with open(in_path, "rb") as f:
        it = ijson.items(f, "item", use_float=True)  # iterates each object in the top-level array
        from tqdm import tqdm
        if expected_total is not None:
            it = tqdm(it, total=expected_total, unit="obj", smoothing=0.05, desc="Splitting (ijson)")
        for obj in it:
            key = _dims_key(obj.get("object_dimensions"))
            tgt = pick_split_by_key(key)

            # write
            if not first[tgt]:
                outs[tgt].write(",\n")
            else:
                first[tgt] = False
            outs[tgt].write(_dumps(obj))

            # stats
            count[tgt] += 1
            if obj.get("collision_label"):
                coll[tgt] += 1
            if obj.get("success_label") and not obj.get("collision_label"):
                succ[tgt] += 1
            dims_sets[tgt].add(key)

    # Close arrays
    for k, f in outs.items():
        f.write("\n]")
        f.close()
    print("[split-fast] wrote train/val/test")

    # Print concise summary
    for name in ["train.json","val.json","test.json"]:
        n = count[name]
        s = succ[name]
        c = coll[name]
        sr = (s / n) if n else 0.0
        cr = (c / n) if n else 0.0
        print(f"{name}: n={n}  success_rate={sr:.3f}  collision_rate={cr:.3f}")

    # Leakage by dims keys (tiny sets)
    tv = (dims_sets["train.json"] & dims_sets["val.json"]) - {None}
    tt = (dims_sets["train.json"] & dims_sets["test.json"]) - {None}
    vt = (dims_sets["val.json"]   & dims_sets["test.json"]) - {None}
    print(f"leaky buckets (train∩val): {len(tv)}")
    print(f"leaky buckets (train∩test): {len(tt)}")
    print(f"leaky buckets (val∩test): {len(vt)}")
    if tv or tt or vt:
        ex = list(tv or tt or vt)[:5]
        print(f"examples: {ex}")
# =============================== PIPELINE ===================================
def run_pipeline():
    # regroup_all_raw()
    source_subdir = "raw_data"
    source_glob = "object_*.json"
    build_rich_pairs(source_subdir=source_subdir, source_glob=source_glob)
    combine_processed_data()
    split_all_data(expected_total=28880120)
    # split_all_data_fast_ijson(expected_total=28880120)

if __name__ == "__main__":
    run_pipeline()
