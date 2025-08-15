import os
import json
import glob
import numpy as np
from tqdm import tqdm
import re
import random
import math
from scipy.spatial.transform import Rotation as R
import decimal
import concurrent.futures
import itertools

# Load data from JSON files
def completing_pairs(data_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    raw_data_path = os.path.join(data_folder, "raw_data")
    processed_data_path = os.path.join(data_folder, "processed_data")
    os.makedirs(processed_data_path, exist_ok=True)
    output = {}
    json_files = sorted(glob.glob(os.path.join(raw_data_path, "object_*.json")))
    # json_files.extend(sorted(glob.glob(os.path.join(data_path, "data_1.json"))))
    print(f"Found {len(json_files)} JSON files")

    for json_file in tqdm(json_files, desc="Loading data"):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Identify clearly initial valid poses (both success=True and collision=False)
        output = {
            "grasp_poses": [],
            "initial_object_poses": [],
            "final_object_poses": [],
            "success_labels": [],
            "collision_labels": [],
        }

        # Loop over each grasp key (one key = one grasp pose)
        for grasp_key, case_list in data.items():
            # Find all valid picking configs in this grasp group (success=True, collision=False)
            valid_pick_indices = [
                i for i, item in enumerate(case_list)
                if item[2] and not item[3]
            ]

            for pick_idx in valid_pick_indices:
                pick_case = case_list[pick_idx]
                pick_grasp_pose = pick_case[0]  # List or array
                pick_object_pose = pick_case[1] # List or array

                for place_idx, place_case in enumerate(case_list):
                    # Skip self-pairing (same pick and place pose)
                    if pick_idx == place_idx:
                        continue

                    place_object_pose = place_case[1]
                    success_label = float(place_case[2])
                    collision_label = float(place_case[3])

                    output["grasp_poses"].append(list(pick_grasp_pose))
                    output["initial_object_poses"].append(list(pick_object_pose))
                    output["final_object_poses"].append(list(place_object_pose))
                    output["success_labels"].append(success_label)
                    output["collision_labels"].append(collision_label)

        # Save output to JSON file
        final_path = os.path.join(processed_data_path, f"processed_{os.path.basename(json_file)}")
        with open(final_path, "w") as f:
            json.dump(output, f)
        print(f"Saved processed data: {final_path}")



def _load_json_fast(path):
    try:
        import orjson  # type: ignore
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except Exception:
        with open(path, "r") as f:
            return json.load(f)


def _dump_json_fast(obj, path):
    try:
        import orjson  # type: ignore
        with open(path, "wb") as f:
            f.write(orjson.dumps(obj))
    except Exception:
        with open(path, "w") as f:
            json.dump(obj, f)


def _process_single_file(json_file, processed_data_path):
    # Worker: replicate completing_pairs logic for a single file
    data = _load_json_fast(json_file)
    output = {
        "grasp_poses": [],
        "initial_object_poses": [],
        "final_object_poses": [],
        "success_labels": [],
        "collision_labels": [],
    }

    for grasp_key, case_list in data.items():
        valid_pick_indices = [
            i for i, item in enumerate(case_list)
            if item[2] and not item[3]
        ]

        for pick_idx in valid_pick_indices:
            pick_case = case_list[pick_idx]
            pick_grasp_pose = pick_case[0]
            pick_object_pose = pick_case[1]

            for place_idx, place_case in enumerate(case_list):
                if pick_idx == place_idx:
                    continue
                place_object_pose = place_case[1]
                success_label = float(place_case[2])
                collision_label = float(place_case[3])

                output["grasp_poses"].append(list(pick_grasp_pose))
                output["initial_object_poses"].append(list(pick_object_pose))
                output["final_object_poses"].append(list(place_object_pose))
                output["success_labels"].append(success_label)
                output["collision_labels"].append(collision_label)

    final_path = os.path.join(processed_data_path, f"processed_{os.path.basename(json_file)}")
    _dump_json_fast(output, final_path)
    return final_path


def completing_pairs_parallel(
    data_folder,
    quotas: dict = None,
    balance_collision: bool = True,
    group_cap: int = 1000,
    source_subdir: str = "fixed_data",
    source_glob: str = "reordered_object_*.json",
    out_subdir: str = "processed_data",
    seed: int = 42,
):
    """
    Build rich, bounded pairs for ALL files by delegating to build_rich_pairs.

    Defaults assume you've reordered raw files into <data_folder>/fixed_data/reordered_*.json.
    """
    os.makedirs(data_folder, exist_ok=True)
    # Tighter, diversity-focused default quotas; can be overridden by caller
    quotas = quotas or {'SAME':0,'SMALL':1,'MEDIUM':2,'ADJACENT':3,'OPPOSITE':4}
    print(f"Building rich pairs from {os.path.join(data_folder, source_subdir)} matching {source_glob}")
    build_rich_pairs(
        data_folder=data_folder,
        seed=seed,
        quotas=quotas,
        balance_collision=balance_collision,
        group_cap=group_cap,
        source_subdir=source_subdir,
        source_glob=source_glob,
        out_subdir=out_subdir,
    )
    print("Finished building rich processed_data files")


def combine_processed_data(data_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    processed_data_path = os.path.join(data_folder, "processed_data")
    files = sorted(glob.glob(os.path.join(processed_data_path, "processed_*.json")))
    
    combined_data_path = os.path.join(data_folder, "combined_data")
    os.makedirs(combined_data_path, exist_ok=True)
    output_file = os.path.join(combined_data_path, "all_data.json")
    
    print("Writing all_data.json incrementally...")
    with open(output_file, "w") as f:
        f.write("[\n")  # Start JSON array
        
        first_sample = True
        for file in tqdm(files, desc="Processing files"):
            # ----- Extract object dimensions from filename -----
            m = re.search(r'object_([0-9.]+)_([0-9.]+)_([0-9.]+)\.json', file)
            if m:
                dims = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            else:
                dims = [None, None, None]  # or raise an error if preferred
            
            with open(file, "r") as data_f:
                data = json.load(data_f)

            # Process each sample in the file
            num_samples = len(data["grasp_poses"])
            for i in range(num_samples):
                sample = {
                    "grasp_pose": data["grasp_poses"][i],
                    "initial_object_pose": data["initial_object_poses"][i],
                    "final_object_pose": data["final_object_poses"][i],
                    "success_label": data["success_labels"][i],
                    "collision_label": data["collision_labels"][i],
                    "object_dimensions": dims  # <--- Add dimensions here
                }
                
                # Add comma separator between samples
                if not first_sample:
                    f.write(",\n")
                else:
                    first_sample = False
                
                # Write the sample
                f.write(json.dumps(sample))
        
        f.write("\n]")  # End JSON array
    
    print("Finished writing.")

    


def reorder_single_file_to_grasp_first(raw_json_path: str, output_dir: str = None) -> str:
    """
    Read ONE raw JSON where keys are object-pose buckets (e.g., 'grasp_0', 'grasp_1', ...),
    and rewrite it into the previous grasp-first layout:
      - Keys become 'grasp_{g}' where g is the global grasp index
      - Each value is a list over all object poses for that same index g

    This function does NOT create grasp–place pairs. It only reorders the data.

    Args:
        raw_json_path: path to original raw JSON file.
        output_dir: optional directory to write reordered file. Defaults to
                    sibling directory 'processed_data' next to 'raw_data'.

    Returns:
        Path to the written reordered file.
    """
    raw_json_path = os.path.abspath(raw_json_path)
    with open(raw_json_path, "r") as f:
        data = json.load(f)

    # Default output directory: sibling processed_data
    raw_dir = os.path.dirname(raw_json_path)
    if output_dir is None:
        parent = os.path.dirname(raw_dir)
        output_dir = os.path.join(parent, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    # Collect and sort object-pose keys by numeric index
    keyed = []
    for k in data.keys():
        m = re.match(r"grasp_(\d+)$", k)
        if m:
            keyed.append((int(m.group(1)), k))
    keyed.sort(key=lambda x: x[0])
    sorted_keys = [k for _, k in keyed]

    if not sorted_keys:
        # Nothing to reorder; write empty structure
        out = {}
        out_path = os.path.join(output_dir, f"reordered_{os.path.basename(raw_json_path)}")
        with open(out_path, "w") as f:
            json.dump(out, f)
        return out_path

    # Use minimum length across groups to stay safe
    N = min(len(data[k]) for k in sorted_keys)

    reordered = {}
    for g in range(N):
        key = f"grasp_{g}"
        reordered[key] = [data[k][g] for k in sorted_keys]

    out_path = os.path.join(output_dir, f"reordered_{os.path.basename(raw_json_path)}")
    with open(out_path, "w") as f:
        json.dump(reordered, f)
    return out_path

def split_all_data(data_folder: str,
                   train_frac: float = 0.8,
                   val_frac:   float = 0.1,
                   test_frac:  float = 0.1,
                   seed:       int   = 42,
                   show_progress: bool = True,
                   pre_count: bool = False):
    """
    Stream-split combined_data/all_data.json into train/val/test without loading
    the entire file into memory. Writes JSON arrays: train.json, val.json, test.json
    in the same combined_data/ directory.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    out_dir = os.path.join(data_folder, "combined_data")
    in_path = os.path.join(out_dir, "all_data.json")
    os.makedirs(out_dir, exist_ok=True)

    # Open outputs
    outs = {
        "train.json": open(os.path.join(out_dir, "train.json"), "w"),
        "val.json":   open(os.path.join(out_dir, "val.json"),   "w"),
        "test.json":  open(os.path.join(out_dir, "test.json"),  "w"),
    }
    try:
        import hashlib
        for f in outs.values():
            f.write("[\n")
        first = {k: True for k in outs.keys()}
        rng = random.Random(seed)

        def _norm_dims(dims, nd=6):
            # dims is [x,y,z]; round to avoid float jitter
            if dims is None: 
                return None
            try:
                return tuple(round(float(d), nd) for d in dims)
            except Exception:
                return None

        def pick_split_by_key(key_tuple, train_frac=0.8, val_frac=0.1, seed=42):
            """
            Map an entire key (e.g., one object_dimensions tuple) to a single split.
            Uses a seeded hash so buckets are assigned stably and approximately
            proportional to the requested fractions.
            """
            if key_tuple is None:
                # fallback: random row-level (rare if combine wrote dims)
                r = random.Random(seed).random()
                return "train.json" if r < train_frac else ("val.json" if r < train_frac + val_frac else "test.json")
            # salted, stable hash -> [0,1)
            h = hashlib.md5((str(key_tuple) + f"|{seed}").encode("utf-8")).digest()
            u = int.from_bytes(h[:8], "big") / 2**64
            if u < train_frac:
                return "train.json"
            elif u < train_frac + val_frac:
                return "val.json"
            else:
                return "test.json"


        # Try fast streaming with ijson; fallback to manual streaming
        use_ijson = False
        try:
            import ijson  # type: ignore
            use_ijson = True
        except Exception:
            use_ijson = False

        # Optional pre-count for progress bar / ETA
        total_items = None
        if show_progress and pre_count:
            try:
                if use_ijson:
                    c = 0
                    with open(in_path, "rb") as f:
                        for _ in ijson.items(f, "item"):
                            c += 1
                    total_items = c
                else:
                    # reuse manual stream to count
                    def _stream_for_count(fp):
                        buf = []
                        depth = 0
                        in_str = False
                        esc = False
                        started = False
                        while True:
                            chunk = fp.read(8192)
                            if not chunk:
                                break
                            for ch in chunk:
                                c = ch if isinstance(ch, str) else chr(ch)
                                if not started:
                                    if c == '[':
                                        started = True
                                    continue
                                if c == ']' and not in_str and depth == 0:
                                    return
                                if c == '"' and not esc:
                                    in_str = not in_str
                                esc = (c == '\\' and in_str and not esc)
                                if not in_str:
                                    if c == '{':
                                        depth += 1
                                    elif c == '}':
                                        depth -= 1
                                if depth == 0 and c == '}':
                                    yield 1
                    total_items = 0
                    with open(in_path, "rb") as f:
                        for _ in _stream_for_count(f):
                            total_items += 1
            except Exception:
                total_items = None

        pbar = None
        if show_progress:
            try:
                pbar = tqdm(total=total_items, desc="Splitting all_data.json", unit="items")
            except Exception:
                pbar = None

        count = 0
        # JSON default to handle Decimal → float
        def _json_default(o):
            if isinstance(o, decimal.Decimal):
                return float(o)
            return str(o)

        if use_ijson:
            with open(in_path, "rb") as f:
                for obj in ijson.items(f, "item"):
                    dims_key = _norm_dims(obj.get("object_dimensions"))
                    tgt = pick_split_by_key(dims_key, train_frac=train_frac, val_frac=val_frac, seed=seed)
                    if not first[tgt]:
                        outs[tgt].write(",\n")
                    else:
                        first[tgt] = False
                    outs[tgt].write(json.dumps(obj, default=_json_default))
                    count += 1
                    if pbar is not None:
                        pbar.update(1)
        else:
            # Minimal manual streaming for a JSON array of objects
            def stream_objects(fp):
                buf = []
                depth = 0
                in_str = False
                esc = False
                started = False
                while True:
                    chunk = fp.read(8192)
                    if not chunk:
                        break
                    for ch in chunk:
                        if isinstance(ch, str):
                            c = ch
                        else:
                            c = chr(ch)
                        if not started:
                            if c == '[':
                                started = True
                            continue
                        if c == ']' and not in_str and depth == 0:
                            # end of array
                            yield from ()
                            return
                        if c == '"' and not esc:
                            in_str = not in_str
                        esc = (c == '\\' and in_str and not esc)
                        if not in_str:
                            if c == '{':
                                depth += 1
                            elif c == '}':
                                depth -= 1
                        if depth > 0:
                            buf.append(c)
                            if depth == 0:
                                # finished an object
                                try:
                                    yield json.loads(''.join(buf))
                                finally:
                                    buf.clear()
                        elif c == '{':
                            buf.append(c)
                            # depth increment already handled
                        else:
                            continue

            with open(in_path, "rb") as f:
                for obj in stream_objects(f):
                    dims_key = _norm_dims(obj.get("object_dimensions"))
                    tgt = pick_split_by_key(dims_key, train_frac=train_frac, val_frac=val_frac, seed=seed)
                    if not first[tgt]:
                        outs[tgt].write(",\n")
                    else:
                        first[tgt] = False
                    outs[tgt].write(json.dumps(obj, default=_json_default))
                    count += 1
                    if pbar is not None:
                        pbar.update(1)

        for name, f in outs.items():
            f.write("\n]")
            f.close()
        if pbar is not None:
            pbar.close()
        print(f"Stream-split {count} samples to {out_dir}/train.json, val.json, test.json")
    except Exception:
        for f in outs.values():
            try:
                f.close()
            except Exception:
                pass
        raise


def extract_data_from_json(data_path, output_path):
    """
    Extract a dataset balanced only on the collision label:
    - Take all non-collision samples (collision_label == 0)
    - Randomly sample an equal number of collision samples (collision_label == 1)
    - Combine and shuffle for a balanced dataset
    
    Args:
        data_path: Path to the input JSON file
        output_path: Directory to save the output file
    """
    import os
    import json
    import random

    target_path = os.path.join(output_path, "collision_balanced_samples.json")

    print(f"Extracting collision-balanced samples from {data_path} to {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with open(data_path, "r") as f:
        full_data = json.load(f)
    print(f"Loaded {len(full_data)} total samples")

    # Separate by collision label
    non_collision = [s for s in full_data if s["collision_label"] <= 0.5]
    collision = [s for s in full_data if s["collision_label"] > 0.5]

    print(f"Non-collision samples: {len(non_collision)}")
    print(f"Collision samples: {len(collision)}")

    n = len(non_collision)
    if n == 0:
        print("No non-collision samples found. Cannot balance.")
        return
    if len(collision) == 0:
        print("No collision samples found. Cannot balance.")
        return

    # Randomly sample collision samples to match non-collision count
    if len(collision) > n:
        collision_balanced = random.sample(collision, n)
    else:
        collision_balanced = collision
        print(f"Warning: Not enough collision samples to fully balance. Using {len(collision)}.")
        n = len(collision)  # adjust n to match available collision samples
        non_collision = random.sample(non_collision, n)

    balanced_data = non_collision + collision_balanced
    random.shuffle(balanced_data)

    with open(target_path, "w") as f:
        json.dump(balanced_data, f, indent=2)

    print(f"Successfully saved {len(balanced_data)} collision-balanced samples to {target_path}")
    print(f"Breakdown: {len(non_collision)} non-collision, {len(collision_balanced)} collision samples")


# ──────────────────────────────────────────────────────────────────────────
def build_rich_pairs(
    data_folder,
    seed: int = 42,
    quotas: dict = None,
    balance_collision: bool = True,
    group_cap: int = 1000,
    source_subdir: str = "raw_data",
    source_glob: str = "*.json",
    out_subdir: str = "processed_data",
):
    """
    Convert every JSON in <data_folder>/<source_subdir>/ into a processed_*.json
    with rich but bounded initial–placement pairs using bucketed quotas.

    Tweaks:
      - quotas: dict like {'SAME':1,'SMALL':2,'MEDIUM':2,'ADJACENT':2,'OPPOSITE':3}
      - balance_collision: within each bucket, sample ~50/50 by collision label when possible
      - group_cap: optional max pairs per grasp group (cap output per key)
      - source_subdir/source_glob: allows using reordered files (e.g., fixed_data/reordered_*.json)
      - out_subdir: output directory name (default 'processed_data')
      - Always ensure at least one ADJACENT and one OPPOSITE placement per clean pick when available
    """
    rng = random.Random(seed)

    quotas = quotas or {'SAME':0, 'SMALL':1, 'MEDIUM':2, 'ADJACENT':3, 'OPPOSITE':4}

    raw_dir  = os.path.join(data_folder, source_subdir)
    proc_dir = os.path.join(data_folder, out_subdir)
    os.makedirs(proc_dir, exist_ok=True)

    # ------------------ helpers ------------------------------------------------
    ADJ = {1:{2,4,5,6}, 2:{1,3,5,6}, 3:{2,4,5,6},
           4:{1,3,5,6}, 5:{1,2,3,4}, 6:{1,2,3,4}}

    def face_id(qw,qx,qy,qz):
        rot = R.from_quat([qx,qy,qz,qw])
        up  = np.array([0,0,1])
        faces = {1:[0,0,1], 2:[1,0,0], 3:[0,0,-1], 4:[-1,0,0], 5:[0,-1,0], 6:[0,1,0]}
        return max(faces, key=lambda k: np.dot(rot.apply(faces[k]), up))

    def bucket(meta_i, meta_j):
        fi, fj = meta_i['face'], meta_j['face']
        if fi == fj:
            dq = R.from_quat(meta_j['quat']) * R.from_quat(meta_i['quat']).inv()
            ang = dq.magnitude() * 180 / math.pi
            if ang < 1:   return 'SAME'
            if ang <= 30: return 'SMALL'
            if ang <= 90: return 'MEDIUM'
            return 'MEDIUM'
        return 'ADJACENT' if fj in ADJ[fi] else 'OPPOSITE'

    def sample_bucket(indices, k, cases):
        if not indices or k <= 0:
            return []
        if not balance_collision:
            return indices if len(indices) <= k else rng.sample(indices, k)
        # balance by collision label at placement
        pos = [j for j in indices if not cases[j][3]]  # no collision
        neg = [j for j in indices if cases[j][3]]      # collision
        take_pos = min(len(pos), k // 2)
        take_neg = min(len(neg), k - take_pos)
        chosen = []
        if pos:
            chosen.extend(pos if take_pos >= len(pos) else rng.sample(pos, take_pos))
        if neg:
            chosen.extend(neg if take_neg >= len(neg) else rng.sample(neg, take_neg))
        rem = k - len(chosen)
        if rem > 0:
            pool = [j for j in indices if j not in chosen]
            if pool:
                chosen.extend(pool if len(pool) <= rem else rng.sample(pool, rem))
        return chosen

    # ------------------ iterate files -----------------------------------------
    raw_files = sorted(glob.glob(os.path.join(raw_dir, source_glob)))
    print(f"Found {len(raw_files)} files in {raw_dir} matching {source_glob}")
    for raw_path in tqdm(raw_files, desc="building rich pairs"):
        with open(raw_path, "r") as f:
            raw = json.load(f)

        out = {k: [] for k in ("grasp_poses","initial_object_poses",
                               "final_object_poses","success_labels",
                               "collision_labels")}

        # ------- per grasp group ---------------------------------------------
        for grasp_key, cases in raw.items():
            # pre‑compute per‑orientation metadata (quat + face ID)
            meta = []
            for c in cases:
                qw,qx,qy,qz = c[1][3:7]
                meta.append({'quat':[qx,qy,qz,qw], 'face':face_id(qw,qx,qy,qz)})

            # candidate initial: fully clean
            clean_idx = [i for i,c in enumerate(cases)
                         if not c[3] and not c[4] and not c[5] and not c[6]]

            produced_for_group = 0
            for i in clean_idx:
                if group_cap is not None and produced_for_group >= group_cap:
                    break

                # build buckets
                buckets = {b: [] for b in quotas}
                for j in range(len(cases)):
                    b = bucket(meta[i], meta[j])
                    if b in buckets:
                        buckets[b].append(j)

                # primary pass with quotas
                chosen = []
                leftovers = 0
                for b, k in quotas.items():
                    lst = buckets.get(b, [])
                    take = sample_bucket(lst, k, cases)
                    chosen.extend(take)
                    leftovers += max(0, k - len(take))

                # redistribute unused quota to any remaining items
                if leftovers > 0:
                    pool = []
                    for b, lst in buckets.items():
                        pool.extend([j for j in lst if j not in chosen])
                    if pool:
                        extra = pool if len(pool) <= leftovers else rng.sample(pool, leftovers)
                        chosen.extend(extra)

                # enforce at least one ADJACENT and one OPPOSITE when available
                for must in ("ADJACENT", "OPPOSITE"):
                    lst = buckets.get(must, [])
                    if lst and not any(j in lst for j in chosen):
                        pool = [j for j in lst if j not in chosen]
                        if pool:
                            chosen.append(rng.sample(pool, 1)[0])

                # cap pairs produced per group
                remaining_group_slots = None
                if group_cap is not None:
                    remaining_group_slots = max(0, group_cap - produced_for_group)
                    if remaining_group_slots == 0:
                        break
                    if len(chosen) > remaining_group_slots:
                        # Randomly subsample to remove ordering bias under the cap
                        chosen = rng.sample(chosen, remaining_group_slots)

                pick = cases[i]
                for j in chosen:
                    place = cases[j]
                    out['grasp_poses'].append(pick[0])
                    out['initial_object_poses'].append(pick[1])
                    out['final_object_poses'].append(place[1])
                    out['success_labels'].append(float(place[2]))
                    out['collision_labels'].append(float(place[3]))
                produced_for_group += len(chosen)

        # -------- save -------------------------------------------------------
        fname = f"processed_{os.path.basename(raw_path)}"
        with open(os.path.join(proc_dir, fname), "w") as f:
            json.dump(out, f)
        print(f"  {fname}: {len(out['grasp_poses'])} pairs")
    



if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data-folder", type=str, default="/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection")
    # parser.add_argument("--build", action="store_true", help="Build processed pairs from fixed_data")
    # parser.add_argument("--combine", action="store_true", help="Combine processed files into one JSON array")
    # parser.add_argument("--split", action="store_true", help="Split combined all_data.json into train/val/test")
    # parser.add_argument("--group-cap", type=int, default=100)
    # parser.add_argument("--balance", action="store_true", default=True)
    # parser.add_argument("--source-subdir", type=str, default="fixed_data")
    # parser.add_argument("--source-glob", type=str, default="reordered_object_*.json")
    # args = parser.parse_args()

    # # Default to --build if no action flags provided (helpful for IDE Run)
    # if not (args.build or args.combine or args.split):
    #     print("No action flags provided. Defaulting to --build.")
    #     args.build = False
    #     args.combine = False
    #     args.split = True

    # if args.build:
    #     print("Building processed pairs from fixed_data")
    #     completing_pairs_parallel(
    #         data_folder=args.data_folder,
    #         quotas={'SAME':0,'SMALL':1,'MEDIUM':2,'ADJACENT':3,'OPPOSITE':4},
    #         balance_collision=args.balance,
    #         group_cap=args.group_cap,
    #         source_subdir=args.source_subdir,
    #         source_glob=args.source_glob,
    #         out_subdir="processed_data",
    #         seed=42,
    #     )
    # if args.combine:
    #     combine_processed_data(args.data_folder)
    # if args.split:
    #     split_all_data(args.data_folder)

    import json, os
    dd="/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/combined_data"
    for name in ["train.json","val.json","test.json"]:
        d=json.load(open(os.path.join(dd,name)))
        pos=sum(int(bool(s["success_label"]) and not bool(s["collision_label"])) for s in d)
        n=len(d); print(name, {"n":n,"pos_rate": pos/n})

    import json, os

    dd = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/combined_data"
    seen = {}
    for name in ["train.json","val.json","test.json"]:
        for obj in json.load(open(os.path.join(dd,name))):
            k = tuple(round(float(x),6) for x in obj["object_dimensions"])
            seen.setdefault(k, set()).add(name)

    leaky = {k:v for k,v in seen.items() if len(v)>1}
    print("leaky buckets:", len(leaky))   # should be 0


