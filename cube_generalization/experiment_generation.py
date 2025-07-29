import numpy as np
import random
import json
import copy

from scipy.spatial.transform import Rotation as R
from utils import sample_object_poses, sample_fixed_surface_poses_increments, flip_pose_keep_delta
from collections import defaultdict


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

def model_prediction(model, data, device):
    dx, dy, dz = data['object_dimensions']
    from dataset import cuboid_corners_local
    corners = cuboid_corners_local(dx, dy, dz).astype(np.float32)

    # Preprocess the initial and final poses as done in the dataset
    corners_tensor = torch.tensor(corners, dtype=torch.float32).unsqueeze(0).to(device)
    grasp_tensor = torch.tensor(data["grasp_pose"], dtype=torch.float32).unsqueeze(0).to(device)
    init_tensor = torch.tensor(data["initial_object_pose"], dtype=torch.float32).unsqueeze(0).to(device)
    final_tensor = torch.tensor(data["final_object_pose"], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Now use the preprocessed tensors with batch dimension

    with torch.no_grad():
            raw_success, raw_collision = model(corners_tensor, grasp_tensor, init_tensor, final_tensor)
    
    # Apply sigmoid to convert logits to probabilities
    pred_success = torch.sigmoid(raw_success)
    pred_collision = torch.sigmoid(raw_collision)
    
    # Extract scalar values from tensors
    pred_success_val = pred_success.item()
    pred_collision_val = pred_collision.item()
    
    # Get binary predictions based on threshold of 0.5
    pred_success_binary = pred_success > 0.5
    pred_collision_binary = pred_collision > 0.5  # True means "collision predicted"
    
    return pred_success_val, 1-pred_collision_val

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
    box_dim_lists = [
        [0.15, 0.15, 0.05],
        [0.05, 0.05, 0.2],
        [0.15, 0.05, 0.08],
        [0.05, 0.05, 0.05],

    ]
    test_data = []

    for box_dims in box_dim_lists:
        object_initial_poses = sample_fixed_surface_poses_increments(box_dims, base_position=(0.2, -0.3, 0.1))
        for object_initial_pose in object_initial_poses:
            for num_turns in range(0, 4):
                object_final_pose = flip_pose_keep_delta(object_initial_pose, box_dims, num_turns=num_turns)
                grasps = generate_grasps(object_initial_pose[:3], object_initial_pose[3:], np.array(box_dims), enable_yaw_tilt=False)
                for grasp in grasps:  
                    final_pose = copy.deepcopy(object_final_pose)
                    final_pose[2] += 0.1
                    current_trail = {
                        "grasp_pose": grasp,
                        "initial_object_pose": object_initial_pose,
                        "final_object_pose": final_pose,
                        "object_dimensions": box_dims
                    }
                    test_data.append(current_trail)

    # Save generated trials to JSON
    with open("/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_generation.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Generated {len(test_data)} trials")

    import torch
    from model_eval import GraspObjectFeasibilityNet
    checkpoint = "/home/chris/Chris/placement_ws/src/data/box_simulation/v3/training/models/model_20250723_042907/best_model_0_4566_pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

     # Load the checkpoint 
    checkpoint_data = torch.load(checkpoint, map_location=device)
    raw_state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    
    # Strip any unwanted prefix
    prefix_to_strip = "_orig_mod."
    cleaned_state_dict = {}
    for key, tensor in raw_state_dict.items():
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = tensor
    
    # Initialize model without static feature yet
    model = GraspObjectFeasibilityNet().to(device)
    # Register this feature with the model
    # Load the state dict but don't compute point cloud embedding yet
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    with open("/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_generation.json", "r") as f:
        test_data: list[dict] = json.load(f)


    # Collect and persist prediction results
    result_records = []
    for data in test_data:
        pred_success_val, pred_collision_val = model_prediction(model, data, device)
        # Show on console for immediate feedback
        print(f"object dimensions: {data['object_dimensions']}, no collision: {pred_collision_val:.4f}")

        # Accumulate for file output
        result_records.append({
            "object_dimensions": data["object_dimensions"],
            "pred_no_collision": float(pred_collision_val)
        })

    # Write results to a JSON-Lines file so it can grow incrementally or be easily parsed
    output_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions.json"
    with open(output_path, "w") as f_out:
        for rec in result_records:
            f_out.write(json.dumps(rec) + "\n")

    print(f"Saved {len(result_records)} prediction records to {output_path}")

