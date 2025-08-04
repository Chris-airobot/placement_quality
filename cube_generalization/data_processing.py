import os
import json
import glob
import numpy as np
from tqdm import tqdm
import re
import random
import math
from scipy.spatial.transform import Rotation as R

# Load data from JSON files
def completing_pairs(data_folder):
    # Create the output directory if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    raw_data_path = os.path.join(data_folder, "raw_data/old")
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

    


def split_all_data(data_folder: str,
                   train_frac: float = 0.8,
                   val_frac:   float = 0.1,
                   test_frac:  float = 0.1,
                   seed:       int   = 42):
    """
    Read all_data.json, shuffle, split into train/val/test by fractions,
    and write train.json, val.json, test.json in output_dir.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    all_data_path = os.path.join(data_folder, "combined_data")
    os.makedirs(all_data_path, exist_ok=True)

    # 1) Count total samples first
    with open(all_data_path+"/all_data.json", 'r') as f:
        data = json.load(f)
    N = len(data)
    print(f"Found {N} samples in {all_data_path}")

    # 2) Shuffle indices
    indices = list(range(N))
    random.seed(seed)
    random.shuffle(indices)

    # 3) Compute split sizes
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    n_test  = N - n_train - n_val
    print(f"Splitting into {n_train} train / {n_val} val / {n_test} test samples")

    # 4) Create sets for fast lookup
    train_idxs = set(indices[:n_train])
    val_idxs   = set(indices[n_train:n_train + n_val])
    test_idxs  = set(indices[n_train + n_val:])

    # 5) Process and write incrementally
    splits = {
        'train.json': train_idxs,
        'val.json':   val_idxs,
        'test.json':  test_idxs
    }

    # Open all output files
    split_files = {}
    for fname, idxs in splits.items():
        out_path = os.path.join(all_data_path, fname)
        split_files[fname] = open(out_path, 'w')
        split_files[fname].write("[\n")  # Start JSON array

    # Process each sample and write to appropriate file
    first_samples = {'train.json': True, 'val.json': True, 'test.json': True}
    
    for i, sample in enumerate(data):
        if i in train_idxs:
            if not first_samples['train.json']:
                split_files['train.json'].write(",\n")
            else:
                first_samples['train.json'] = False
            split_files['train.json'].write(json.dumps(sample))
        elif i in val_idxs:
            if not first_samples['val.json']:
                split_files['val.json'].write(",\n")
            else:
                first_samples['val.json'] = False
            split_files['val.json'].write(json.dumps(sample))
        elif i in test_idxs:
            if not first_samples['test.json']:
                split_files['test.json'].write(",\n")
            else:
                first_samples['test.json'] = False
            split_files['test.json'].write(json.dumps(sample))

    # Close all files and add closing brackets
    for fname, f in split_files.items():
        f.write("\n]")  # End JSON array
        f.close()
        print(f"Wrote {len(splits[fname])} samples → {os.path.join(all_data_path, fname)}")


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
def build_rich_pairs(data_folder, seed: int = 42):
    """
    Convert every raw JSON in <data_folder>/raw_data/ into a processed_*.json
    with rich but bounded initial–placement pairs.

    Directory layout created/expected:
        data_folder/raw_data/......   (your original files)
        data_folder/processed_data/.. (new files go here)
    """
    rng = random.Random(seed)

    raw_dir  = os.path.join(data_folder, "raw_data")
    proc_dir = os.path.join(data_folder, "processed_data")
    os.makedirs(proc_dir, exist_ok=True)

    # ------------------ helpers ------------------------------------------------
    ADJ = {1:{2,4,5,6}, 2:{1,3,5,6}, 3:{2,4,5,6},
           4:{1,3,5,6}, 5:{1,2,3,4}, 6:{1,2,3,4}}

    def face_id(qw,qx,qy,qz):
        """Which cuboid face is +Z after applying quaternion (w,x,y,z)."""
        rot = R.from_quat([qx,qy,qz,qw])
        up  = np.array([0,0,1])
        faces = {
            1: [0,0, 1], 2:[ 1,0,0], 3:[0,0,-1],
            4:[-1,0,0], 5:[0,-1,0], 6:[0, 1,0]}
        return max(faces, key=lambda k: np.dot(rot.apply(faces[k]), up))

    def bucket(meta_i, meta_j):
        """Return SAME, SMALL, MEDIUM, ADJACENT, OPPOSITE."""
        fi, fj = meta_i['face'], meta_j['face']
        if fi == fj:
            dq = R.from_quat(meta_j['quat']) * R.from_quat(meta_i['quat']).inv()
            ang = dq.magnitude() * 180 / math.pi
            if ang <   1:  return 'SAME'
            if ang <= 30:  return 'SMALL'
            if ang <= 90:  return 'MEDIUM'
            return 'MEDIUM'                # >90°, same face
        return 'ADJACENT' if fj in ADJ[fi] else 'OPPOSITE'

    want = {'SAME':1, 'SMALL':2, 'MEDIUM':2, 'ADJACENT':2, 'OPPOSITE':3}

    # ------------------ iterate raw files -------------------------------------
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    print(f"Found {len(raw_files)} raw files")
    for raw_path in tqdm(raw_files, desc="processing"):
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
                meta.append({'quat':[qx,qy,qz,qw],
                             'face':face_id(qw,qx,qy,qz)})

            # indices of rows that are fully clean -> candidate initials
            clean_idx = [i for i,c in enumerate(cases)
                         if not c[3] and not c[4] and not c[5] and not c[6]]

            # for every clean initial state build <=10 placements
            for i in clean_idx:
                buckets = {b: [] for b in want}
                for j in range(len(cases)):
                    buckets[bucket(meta[i], meta[j])].append(j)

                chosen = []
                for b,k in want.items():
                    lst = buckets[b]
                    if not lst: continue
                    take = lst if len(lst)<=k else rng.sample(lst, k)
                    chosen.extend(take)

                pick = cases[i]
                for j in chosen:
                    place = cases[j]
                    out['grasp_poses'].append(pick[0])
                    out['initial_object_poses'].append(pick[1])
                    out['final_object_poses'].append(place[1])
                    out['success_labels'].append(float(place[2]))  # unchanged
                    out['collision_labels'].append(float(place[3]))

        # -------- save -------------------------------------------------------
        fname = f"processed_{os.path.basename(raw_path)}"
        with open(os.path.join(proc_dir, fname), "w") as f:
            json.dump(out, f)
        print(f"  {fname}: {len(out['grasp_poses'])} pairs")
    



if __name__ == "__main__":
    data_folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection"
    # completing_pairs(data_folder)
    # combine_processed_data(data_folder)
    split_all_data(data_folder)
    # data_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v3/data_collection/combined_data/test.json"
    # folder_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v3/data_collection/combined_data"
    # extract_data_from_json(data_path, folder_path)

    # build_rich_pairs(data_folder)