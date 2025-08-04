import numpy as np
import random
import json
import copy
import torch

from scipy.spatial.transform import Rotation as R
from utils import sample_object_poses, sample_fixed_surface_poses_increments, flip_pose_keep_delta
from collections import defaultdict

# Import your model and dataset
from model import CollisionPredictionNet
from dataset import cuboid_corners_local_ordered  # Fixed: correct function name
from dataset import EnhancedWorldFrameDataset  # or whatever your dataset class is

def generate_grasps(pos_w, quat_w, dims_total,
                    tilt_degs=[60, 90, 120],
                    enable_yaw_tilt=True):
    rot_obj       = R.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])

    # 5) Generate and filter contact points
    all_meta      = generate_contact_metadata(dims_total)


    local_normals = {
        '+X': np.array([ 1,0,0]), '-X': np.array([-1,0,0]),
        '+Y': np.array([ 0,1,0]), '-Y': np.array([ 0,-1,0]),
        '+Z': np.array([ 0,0,1]), '-Z': np.array([ 0,0,-1]),
    }
    dots = {f: float(np.dot(rot_obj.apply(n), [0,0,1])) for f,n in local_normals.items()}
    down_face = max(dots, key=lambda f: -dots[f])
    contact_meta = [m for m in all_meta if m['face'] != down_face]
    for m in contact_meta:
        m['p_world'] = rot_obj.apply(m['p_local']) + pos_w

    # define your tilt angles (degrees) and convert to radians
    tilt_rads = [np.deg2rad(t) for t in tilt_degs]
    # 6) Visualize top-down grasps only (no tilt/yaw)
    final_grasps = []
    for cp in contact_meta:
        # normalize the three contact‐frame directions
        z = cp['approach'] / np.linalg.norm(cp['approach'])  # tool Z (red)
        x = cp['axis']     / np.linalg.norm(cp['axis'])      # tool X (red)
        y = np.cross(z, x)                                   # tool Y (green)
        y /= np.linalg.norm(y)

        # build a rotation whose columns are [X, Y, Z]
        M = np.column_stack([y, x, z])

        # if it’s left‑handed, flip the X‑column so det>0
        if np.linalg.det(M) < 0:
            M[:,1] *= -1

        R_base = R.from_matrix(M)

        if enable_yaw_tilt:
            # Full yaw sweep (with a 20 % safety margin) while keeping the
            # tilt angles passed by the caller.  If the caller sets
            # tilt_degs=[90] this will give yaw-only variations.
            yaw_margin_factor = 0.4          # keep 40 % of theoretical range
            safe_psi = cp['psi_max'] * yaw_margin_factor
            yaw_rads = np.linspace(-safe_psi, safe_psi, 2)
        else:
            # single orientation, no yaw & no tilt
            yaw_rads  = [0.0]
            # tilt_rads = [np.deg2rad(90)]   # 90 ° keeps the tool Z aligned with the contact normal

        # for each tilt then yaw, compute a variant
        for tilt_rad in tilt_rads:
            # tilt about the binormal ⇒ prim‑local X axis
            delta_tilt = tilt_rad - (np.pi / 2)  
            R_tilt = R.from_rotvec(delta_tilt * np.array([1, 0, 0]))

            for yaw in yaw_rads:
                # yaw about the approach ⇒ prim‑local Z axis
                R_yaw = R.from_rotvec(yaw * np.array([0, 0, 1]))

                # apply tilt first, then yaw, then base alignment
                R_variant = R_base * R_yaw * R_tilt

                # world‐frame quaternion
                q_raw = (rot_obj * R_variant).as_quat()  # [x,y,z,w]
                quat_wxyz = [q_raw[3], q_raw[0], q_raw[1], q_raw[2]]

                final_grasps.append((cp['p_world'].tolist() + quat_wxyz))

    return final_grasps

def load_trained_model(checkpoint_path, device):
    """Load and prepare the trained model using the same method as model_eval.py"""
    try:
        # Use the same loading function as model_eval.py
        from model_eval import load_checkpoint_with_original_architecture
        model, _ = load_checkpoint_with_original_architecture(checkpoint_path, device)
        model.eval()
        
        print(f"Successfully loaded model from {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def model_prediction(model, data, device):
    """Fixed model prediction with proper normalization"""
    try:
        dx, dy, dz = data['object_dimensions']
        corners = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)

        # CRITICAL FIX: Load normalization stats and apply them
        # You need to find where your training stats were saved
        stats_path = "/path/to/your/normalization_stats.json"  # Find this file!
        norm_stats = EnhancedWorldFrameDataset.load_stats(stats_path)
        
        # Normalize corners (this is crucial!)
        corners_normalized = norm_stats.zscore_corners(corners)
        
        # Normalize pose XYZ components
        poses = np.stack([data["grasp_pose"][:3], 
                         data["initial_object_pose"][:3], 
                         data["final_object_pose"][:3]])
        poses_normalized = norm_stats.zscore_pose_xyz(poses)
        
        # Reconstruct full poses with normalized XYZ + original quaternions
        grasp_normalized = np.concatenate([poses_normalized[0], data["grasp_pose"][3:]])
        init_normalized = np.concatenate([poses_normalized[1], data["initial_object_pose"][3:]])
        final_normalized = np.concatenate([poses_normalized[2], data["final_object_pose"][3:]])

        # Convert to tensors
        corners_tensor = torch.tensor(corners_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        grasp_tensor = torch.tensor(grasp_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        init_tensor = torch.tensor(init_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        final_tensor = torch.tensor(final_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        
        # TODO: Replace with real embeddings if you have them
        dummy_embeddings = torch.zeros(1, 1024, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            collision_logits = model(dummy_embeddings, corners_tensor, grasp_tensor, init_tensor, final_tensor)
            pred_collision = torch.sigmoid(collision_logits)
            pred_no_collision = 1 - pred_collision.item()
        
        return pred_no_collision
        
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None

#============================
# Helper Functions
#============================
def compute_psi_max(du, dv, G_max):
    thetas = np.linspace(0, np.pi/2, 500)
    spans  = du * np.cos(thetas) + dv * np.sin(thetas)
    valid  = thetas[spans <= G_max]
    return valid.max() if valid.size else 0.0


def align(src, tgt):
    """
    Returns minimal rotation that aligns vector src to vector tgt.
    """
    u = src / np.linalg.norm(src)
    v = tgt / np.linalg.norm(tgt)
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    if dot > 1 - 1e-6:
        return R.identity()
    if dot < -1 + 1e-6:
        # 180-degree turn: pick an orthogonal axis
        ortho = np.array([1, 0, 0])
        if abs(np.dot(u, ortho)) > 0.9:
            ortho = np.array([0, 1, 0])
        axis = np.cross(u, ortho)
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis)
    axis = np.cross(u, v)
    axis /= np.linalg.norm(axis)
    return R.from_rotvec(np.arccos(dot) * axis)


def generate_contact_metadata(dims, approach_offset=0.01, G_max=0.08):
    """
    Computes up to 18 local contact points on a cuboid's 6 faces (3 per face),
    with metadata for approach, binormal, psi_max, and outward normal.
    """
    metadata = []
    face_axes = {
        '+X': (0,1,2), '-X': (0,1,2),
        '+Y': (1,0,2), '-Y': (1,0,2),
        '+Z': (2,0,1), '-Z': (2,0,1),
    }
    fractions = [0.25, 0.50, 0.75]
    half = dims * 0.5

    for face, (i, j, k) in face_axes.items():
        sign = 1 if face[0] == '+' else -1
        normal   = sign * np.eye(3)[i]      # outward face normal
        approach = -normal                  # gripper approach direction

        ej = np.eye(3)[j]
        ek = np.eye(3)[k]
        du, dv = dims[j], dims[k]
        long_vec, long_len = (ej, du) if du >= dv else (ek, dv)
         # —— NEW: pick the shorter edge as our X‐axis “axis” ——
        axis_vec = ek if du >= dv else ej
        axis     = axis_vec / np.linalg.norm(axis_vec)

        # —— NEW: binormal = approach × axis ——
        binormal = np.cross(axis, approach)
        binormal /= np.linalg.norm(binormal)

        psi = compute_psi_max(du, dv, G_max)
        if psi <= 0:
            continue

        base = normal * (half[i] + approach_offset)
        for frac in fractions:
            offset = (frac - 0.5) * long_len
            p_local = base + long_vec * offset
            metadata.append({
                'face':     face,
                'fraction': frac,
                'p_local':  p_local,
                'approach': approach,
                'binormal': binormal,
                'psi_max':  psi,
                'normal':   normal,
                'axis':     axis
            })
    return metadata


if __name__ == "__main__":
    # Configuration
    box_dim_lists = [
        [0.15, 0.15, 0.05],
        [0.05, 0.05, 0.2],
        [0.15, 0.05, 0.08],
        [0.05, 0.05, 0.05],
    ]
    
    # Model configuration
    checkpoint_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/training/models/model_20250804_001543/best_model_roc_20250804_001543.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    model = load_trained_model(checkpoint_path, device)
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Validate model is in eval mode
    model.eval()
    print(f"Model loaded successfully and set to evaluation mode")
    
    test_data = []
    prediction_results = []

    print("Generating experiment scenarios...")
    for box_dims in box_dim_lists:
        print(f"Processing box dimensions: {box_dims}")
        object_initial_poses = sample_fixed_surface_poses_increments(box_dims, base_position=(0.2, -0.3, 0.1))
        
        for i, object_initial_pose in enumerate(object_initial_poses):
            for num_turns in range(0, 4):
                object_final_pose = flip_pose_keep_delta(object_initial_pose, box_dims, num_turns=num_turns)
                grasps = generate_grasps(object_initial_pose[:3], object_initial_pose[3:], np.array(box_dims), enable_yaw_tilt=False)
                
                for grasp in grasps:  
                    final_pose = copy.deepcopy(object_final_pose)
                    final_pose[2] += 0.1
                    
                    current_trial = {
                        "grasp_pose": grasp,
                        "initial_object_pose": object_initial_pose,
                        "final_object_pose": final_pose,
                        "object_dimensions": box_dims
                    }
                    test_data.append(current_trial)
                    
                    # Make prediction immediately
                    pred_no_collision = model_prediction(model, current_trial, device)
                    
                    if pred_no_collision is not None:
                        prediction_result = {
                            "object_dimensions": box_dims,
                            "initial_pose": object_initial_pose,
                            "final_pose": final_pose,
                            "grasp_pose": grasp,
                            "num_turns": num_turns,
                            "pred_no_collision": float(pred_no_collision),
                            "pred_collision": float(1 - pred_no_collision)
                        }
                        prediction_results.append(prediction_result)
                        
                        # Print progress
                        if len(prediction_results) % 100 == 0:
                            print(f"Processed {len(prediction_results)} trials...")

    # Save generated trials to JSON
    experiment_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_generation.json"
    with open(experiment_path, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Generated {len(test_data)} trials")

    # Save prediction results
    output_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions.jsonl"
    with open(output_path, "w") as f_out:
        for rec in prediction_results:
            f_out.write(json.dumps(rec) + "\n")

    print(f"Saved {len(prediction_results)} prediction records to {output_path}")
    
    # Print summary statistics
    if prediction_results:
        pred_values = [r["pred_no_collision"] for r in prediction_results]
        print(f"\nPrediction Summary:")
        print(f"Total predictions: {len(pred_values)}")
        print(f"Mean no-collision probability: {np.mean(pred_values):.4f}")
        print(f"Std no-collision probability: {np.std(pred_values):.4f}")
        print(f"Min no-collision probability: {np.min(pred_values):.4f}")
        print(f"Max no-collision probability: {np.max(pred_values):.4f}")
        
        # Group by object dimensions
        by_dimensions = defaultdict(list)
        for result in prediction_results:
            dim_key = tuple(result["object_dimensions"])
            by_dimensions[dim_key].append(result["pred_no_collision"])
        
        print(f"\nBy Object Dimensions:")
        for dims, preds in by_dimensions.items():
            print(f"  {dims}: mean={np.mean(preds):.4f}, std={np.std(preds):.4f}, count={len(preds)}")

