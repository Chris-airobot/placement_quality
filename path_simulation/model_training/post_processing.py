import json
import glob
import os
from tqdm import tqdm
import numpy as np
import random

def all_data(data_path, output_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(data_path, "processed_*.json")))
    all_data = []

    for file in tqdm(files):
        with open(file, "r") as f:
            data = json.load(f)

        # Make sure all keys have the same length
        num_samples = len(data["grasp_poses"])
        for i in range(num_samples):
            sample = {
                "grasp_pose": data["grasp_poses"][i],
                "initial_object_pose": data["initial_object_poses"][i],
                "final_object_pose": data["final_object_poses"][i],
                "success_label": data["success_labels"][i],
                "collision_label": data["collision_labels"][i],
            }
            all_data.append(sample)

    # Save combined list
    print("Writing all_data.json...")
    with open(os.path.join(output_path, "all_data.json"), "w") as f:
        json.dump(all_data, f)
    print("Finished writing.")



def split_all_data(all_data_path: str,
                   train_frac: float = 0.8,
                   val_frac:   float = 0.1,
                   test_frac:  float = 0.1,
                   seed:       int   = 42):
    """
    Read all_data.json, shuffle, split into train/val/test by fractions,
    and write train.json, val.json, test.json in output_dir.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
    os.makedirs(all_data_path, exist_ok=True)

    # 1) Load full list
    with open(all_data_path+"/all_data.json", 'r') as f:
        data = json.load(f)
    N = len(data)
    print(f"Loaded {N} samples from {all_data_path}")

    # 2) Shuffle indices
    indices = list(range(N))
    random.seed(seed)
    random.shuffle(indices)

    # 3) Compute split sizes
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)
    # remainder goes to test
    n_test  = N - n_train - n_val
    print(f"Splitting into {n_train} train / {n_val} val / {n_test} test samples")

    # 4) Partition indices
    train_idxs = indices[:n_train]
    val_idxs   = indices[n_train:n_train + n_val]
    test_idxs  = indices[n_train + n_val:]

    splits = {
        'train.json': train_idxs,
        'val.json':   val_idxs,
        'test.json':  test_idxs
    }

    # 5) Write out each split
    for fname, idxs in splits.items():
        out_path = os.path.join(all_data_path, fname)
        with open(out_path, 'w') as out_f:
            # write only the selected samples
            subset = [data[i] for i in idxs]
            json.dump(subset, out_f)
        print(f"Wrote {len(idxs)} samples → {out_path}")
    

# Load data from JSON files
def load_data_from_json(data_path, output_path):
    output = {}
    json_files = sorted(glob.glob(os.path.join(data_path, "data_*.json")))
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

        initial_valid_samples = [v for v in data.values() if v["success"] and not v["collision"]]

        # For each initial valid sample, explicitly pair it with every other pose as final pose
        for init_sample in tqdm(initial_valid_samples, desc="Processing samples"):
            init_grasp_pose = init_sample["grasp_pose"][0] + init_sample["grasp_pose"][1]
            init_object_pose =  [init_sample["z_position"]] + list(init_sample["object_orientation"])
            
            for final_sample in data.values():
                final_object_pose = [final_sample["z_position"]] + list(final_sample["object_orientation"])
                
                # Skip if initial and final poses are identical (optional filtering clearly stated)
                if np.allclose(init_object_pose, final_object_pose):
                    continue
                
                output["grasp_poses"].append(init_grasp_pose)
                output["initial_object_poses"].append(list(init_object_pose))
                output["final_object_poses"].append(list(final_object_pose))
                
                # Labels here represent whether the FINAL pose itself was feasible initially.
                # You may choose to recompute feasibility explicitly later.
                output["success_labels"].append(float(final_sample["success"]))
                output["collision_labels"].append(float(final_sample["collision"]))
        # Save output to JSON file
        final_path = f"{output_path}/processed_{json_file.split('/')[-1]}"
        if not os.path.exists(final_path): 
            with open(final_path, "w") as f:
                json.dump(output, f)


    return output
    

def extract_data_from_json(data_path, output_path, samples_per_category: int = 200):
    """
    Extract balanced samples from each category:
    - Success + No collision
    - Success + Collision
    - Failure + No collision
    - Failure + Collision
    
    Args:
        data_path: Path to the input JSON file
        output_path: Directory to save the output file
        samples_per_category: Number of samples to extract from each category
    """
    target_path = os.path.join(output_path, "balanced_samples.json")

    print(f"Extracting {samples_per_category} samples per category from {data_path} to {output_path}")
    # Ensure the target directory exists
    os.makedirs(output_path, exist_ok=True)

    # Load the full dataset
    with open(data_path, "r") as f:
        full_data = json.load(f)
    
    print(f"Loaded {len(full_data)} total samples")
    
    # Create a set to track scenario identifiers to prevent overlaps
    used_scenarios = set()
    
    # Function to create a unique identifier for a scenario
    def get_scenario_id(sample):
        # Create identifier from initial and final object poses to ensure uniqueness
        initial_pose_str = '_'.join([str(x) for x in sample["initial_object_pose"]])
        final_pose_str = '_'.join([str(x) for x in sample["final_object_pose"]])
        return f"{initial_pose_str}_{final_pose_str}"
    
    # Categorize samples while avoiding duplicates
    success_no_collision = []
    success_with_collision = []
    failure_no_collision = []
    failure_with_collision = []
    
    # First pass - categorize all samples and track unique scenarios
    for sample in full_data:
        scenario_id = get_scenario_id(sample)
        
        # Skip if this scenario has already been seen
        if scenario_id in used_scenarios:
            continue
            
        used_scenarios.add(scenario_id)
        
        success = sample["success_label"] > 0.5  # Assuming 1.0 means success
        collision = sample["collision_label"] > 0.5  # Assuming 1.0 means collision
        
        if success and not collision:
            success_no_collision.append(sample)
        elif success and collision:
            success_with_collision.append(sample)
        elif not success and not collision:
            failure_no_collision.append(sample)
        else:  # not success and collision
            failure_with_collision.append(sample)
    
    # Print category statistics after deduplication
    print(f"After deduplication:")
    print(f"Success + No Collision: {len(success_no_collision)} unique samples")
    print(f"Success + Collision: {len(success_with_collision)} unique samples")
    print(f"Failure + No Collision: {len(failure_no_collision)} unique samples")
    print(f"Failure + Collision: {len(failure_with_collision)} unique samples")
    
    # Sample from each category (or take all if less than requested)
    balanced_data = []
    
    for category_name, category in [
        ("Success + No Collision", success_no_collision),
        ("Success + Collision", success_with_collision),
        ("Failure + No Collision", failure_no_collision),
        ("Failure + Collision", failure_with_collision)
    ]:
        category_samples = min(samples_per_category, len(category))
        if category_samples < samples_per_category:
            print(f"Warning: Could only extract {category_samples} samples from {category_name}")
        
        # Randomly sample without replacement
        selected = random.sample(category, category_samples)
        balanced_data.extend(selected)
    
    # Shuffle the final dataset
    random.shuffle(balanced_data)
    
    # Save the samples to a new file
    with open(target_path, "w") as f:
        json.dump(balanced_data, f, indent=2)
    
    print(f"Successfully saved {len(balanced_data)} balanced samples to {target_path}")
    print(f"Breakdown: Up to {samples_per_category} samples per category")

if __name__ == "__main__":
    # Step 1: Load raw data into the processed individual files
    raw_data_path = "/home/chris/Chris/placement_ws/src/data/path_simulation/raw_data_v1"
    processed_data_path = "/home/chris/Chris/placement_ws/src/data/path_simulation/processed_data"
    load_data_from_json(raw_data_path, processed_data_path)
    
    # Step 2: Combine all the processed files into one
    combined_data_path = "/home/chris/Chris/placement_ws/src/data/path_simulation"
    all_data(processed_data_path, combined_data_path)

    # Step 3: Split the combined data into train, val, test
    split_all_data(combined_data_path)

    # data_path = "/media/chris/OS2/Users/24330/Desktop/placement_quality/unseen/all_data.json"
    # output_path = "/media/chris/OS2/Users/24330/Desktop/placement_quality/unseen"
    # extract_data_from_json(data_path, output_path)    