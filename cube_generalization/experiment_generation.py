import numpy as np
import random
import json
import copy
import torch

from scipy.spatial.transform import Rotation as R
from utils import sample_object_poses, sample_fixed_surface_poses_increments, flip_pose_keep_delta, sample_dims
from collections import defaultdict

# Import your model and dataset
from model import CollisionPredictionNet
from dataset import cuboid_corners_local_ordered  # Fixed: correct function name
from dataset import WorldFrameDataset  # or whatever your dataset class is


WORLD_UP = np.array([0., 0., 1.])          # gravity
PEDESTAL_TOP_Z = 0.10          # adjust if pedestal is elevated
MIN_CLEARANCE = 0.02         # minimal clearance above pedestal to allow grasp
PALM_DEPTH     = 0.038     # metres –- distance from contact point to palm
FINGER_THICK   = 0.10     # metres –- half-thickness of your finger tip
GRIPPER_OPEN_MAX = 0.08   # metres –- your jaw limit
MIN_PALM_CLEAR = 0.01     # metres –- clearance you insist on


def choose_yaw_clearance(contact_p_w,          # 3-vector world coords
                         approach_w,           # 3-vector, already normalised
                         dims_total,           # [L,W,H] full sizes (m)
                         rot_obj):             # world→object rotation (scipy R)
    """
    Pick the yaw (X-axis) giving the better palm clearance.
    Returns (x_world, y_world) or None if neither fits & clears.
    """
    a = approach_w / np.linalg.norm(approach_w)        # tool Z

    # Two orthogonal X candidates in the plane ⟂ a
    xA = np.cross(WORLD_UP, a)
    if np.linalg.norm(xA) < 1e-6:                      # approach ∥ up
        xA = np.cross([1., 0., 0.], a)
    xA /= np.linalg.norm(xA)
    xB = np.cross(a, xA)
    xB /= np.linalg.norm(xB)

    half = 0.5 * np.asarray(dims_total)

    def jaw_span(x_dir):
        local = rot_obj.inv().apply(x_dir)
        return 2.0 * np.sum(np.abs(local) * half)

    best  = None
    best_clear = -np.inf

    for x in (xA, xB):
        if jaw_span(x) > GRIPPER_OPEN_MAX:         # jaws can't open that wide
            continue

        y = np.cross(a, x);  y /= np.linalg.norm(y)

        # Palm centre and its lowest edge along y
        palm_center = contact_p_w - PALM_DEPTH * a
        palm_bottom_z = palm_center[2] - FINGER_THICK * abs(y[2])

        clearance = palm_bottom_z - PEDESTAL_TOP_Z
        if clearance >= MIN_PALM_CLEAR and clearance > best_clear:
            best_clear = clearance
            best = (x, y)

    return best  

def generate_grasps(pos_w, quat_w, dims_total,
                    tilt_degs=[75, 90, 105],
                    enable_yaw_tilt=True,
                    debug_indices=None,
                    current_base_index=0):

    rot_obj = R.from_quat([quat_w[1], quat_w[2], quat_w[3], quat_w[0]])

    # ---------- contact metadata ----------
    all_meta      = generate_contact_metadata(dims_total)

    local_normals = {
        '+X': np.array([ 1,0,0]), '-X': np.array([-1,0,0]),
        '+Y': np.array([ 0,1,0]), '-Y': np.array([ 0,-1,0]),
        '+Z': np.array([ 0,0,1]), '-Z': np.array([ 0,0,-1]),
    }
    dots = {f: float(np.dot(rot_obj.apply(n), WORLD_UP)) for f,n in local_normals.items()}
    down_face = max(dots, key=lambda f: -dots[f])

    # discard the face in contact with pedestal
    contact_meta = [m for m in all_meta if m['face'] != down_face]

    # --------------------------------------  CL-1:  fingertip clearance
    valid_cp = []
    for m in contact_meta:
        p_w = rot_obj.apply(m['p_local']) + pos_w
        if (p_w[2] - PEDESTAL_TOP_Z) >= MIN_CLEARANCE:
            m['p_world'] = p_w
            valid_cp.append(m)

    # ---------- grasp variants ----------
    tilt_rads = [np.deg2rad(t) for t in tilt_degs]
    final_grasps = []
    grasp_counter = current_base_index  # Track which overall trial index we're at
    
    for cp in valid_cp:
        z = cp['approach'] / np.linalg.norm(cp['approach'])  # tool Z

        xy = choose_yaw_clearance(cp['p_world'], z, dims_total, rot_obj)
        if xy is None:
            continue                    # no feasible yaw
        v1_w, v2_w = xy
        

        # --- NEW: project the face’s short edge into the approach plane ----------
        face_x_w = rot_obj.apply(cp['axis'])
        face_x_w = face_x_w - np.dot(face_x_w, z) * z     # remove any Z component
        norm = np.linalg.norm(face_x_w)
        if norm < 1e-6:                                   # edge ~parallel to Z
            face_x_w = v1_w                               # fall back, rare
        else:
            face_x_w /= norm

        # choose whichever candidate yaw axis gives the smaller jaw span
        jaw_span = lambda a: 2*np.sum(np.abs(rot_obj.inv().apply(a))*0.5*dims_total)
        span_v1 = jaw_span(v1_w)
        span_v2 = jaw_span(v2_w)
        
        if span_v1 <= span_v2:
            x_w, y_w = v1_w, v2_w
        else:
            x_w, y_w = v2_w, v1_w

        # ——— DEBUG: if we ever pick the larger‐span axis, dump & exit ———
        span_x = jaw_span(x_w)
        span_y = jaw_span(y_w)
        if span_x > span_y + 1e-6:
            print("‼️  MISORIENTED GRASP ‼️")
            continue
            # import sys; sys.exit(0)

        # re-orthonormalise (numerical safety)
        x_w = x_w - np.dot(x_w, z) * z;  x_w /= np.linalg.norm(x_w)
        y_w = np.cross(z, x_w)                   # right-handed frame

        # Try both frame constructions and calculate jaw spans
        R_option1 = R.from_matrix(np.column_stack([-y_w, x_w, z]))  # Original line 139
        R_option2 = R.from_matrix(np.column_stack([x_w, y_w, z]))   # Original line 140
        
        # Calculate jaw spans for both options
        fingers1 = R_option1.as_matrix()[:, 0]  # finger closing direction for option 1
        fingers2 = R_option2.as_matrix()[:, 0]  # finger closing direction for option 2
        span1 = 2*np.sum(np.abs(rot_obj.inv().apply(fingers1))*0.5*dims_total)
        span2 = 2*np.sum(np.abs(rot_obj.inv().apply(fingers2))*0.5*dims_total)
        
        # Choose gripper frame based on face type and orientation
        face_normal_world = rot_obj.apply(cp['normal'])
        
        # For axis-aligned faces (pure X, Y, or Z normals), use Option 1
        # For angled faces, use Option 2
        is_axis_aligned = (abs(face_normal_world[0]) > 0.9 or 
                          abs(face_normal_world[1]) > 0.9 or 
                          abs(face_normal_world[2]) > 0.9)
        
        if is_axis_aligned:
            R_base = R_option1
            chosen_option = 1
        else:
            R_base = R_option2
            chosen_option = 2
            
        # Store debug info for later use
        debug_info = {
            'face': cp['face'],
            'face_normal_world': rot_obj.apply(cp['normal']),
            'approach_world': rot_obj.apply(cp['approach']),
            'dims_total': dims_total,
            'face_axis_local': cp['axis'],
            'face_axis_world': rot_obj.apply(cp['axis']),
            'v1_w': v1_w,
            'v2_w': v2_w,
            'span_v1': span_v1,
            'span_v2': span_v2,
            'x_w': x_w,
            'y_w': y_w,
            'z': z,
            'fingers1': fingers1,
            'fingers2': fingers2,
            'span1': span1,
            'span2': span2,
            'chosen_option': chosen_option
        }


        # yaw sweep (still useful to sample a few)
        if enable_yaw_tilt:
            psi_safe  = cp['psi_max'] * 0.4
            yaw_rads  = np.linspace(-psi_safe, psi_safe, 2)
        else:
            yaw_rads  = [0.0]

        for tilt in tilt_rads:
            R_tilt = R.from_rotvec((tilt - np.pi/2) * np.array([1,0,0]))
            for yaw in yaw_rads:
                R_yaw = R.from_rotvec(yaw * np.array([0,0,1]))

                R_tool = R_base * R_yaw * R_tilt
                q_raw  = (rot_obj * R_tool).as_quat()  # [x,y,z,w]
                quat_w = [q_raw[3], q_raw[0], q_raw[1], q_raw[2]]

                grasp_with_debug = {
                    'grasp': cp['p_world'].tolist() + quat_w,
                    'debug_info': debug_info,
                    'trial_index': grasp_counter
                }
                final_grasps.append(grasp_with_debug)
                grasp_counter += 1

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
        norm_stats = WorldFrameDataset.load_stats(stats_path)
        
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
    original_box_dim_lists = sample_dims(n=4, min_s=0.05, max_s=0.2, seed=980579)
    box_dim_lists = [original_box_dim_lists[0], 
                     original_box_dim_lists[76],
                     original_box_dim_lists[152],
                     original_box_dim_lists[228],
                     original_box_dim_lists[399],
                     ]


    test_data = []
    prediction_results = []
    
    # DEBUG: Add a global counter and specify which indices to debug
    global_trial_index = 0
    debug_indices = {0, 16, 146}  # Add your problematic indices here, e.g. {42, 157, 892}
    
    print("Generating experiment scenarios...")
    for box_dims in box_dim_lists:
        print(f"Processing box dimensions: {box_dims}")
        object_initial_poses = sample_fixed_surface_poses_increments(box_dims, base_position=(0.2, -0.3, 0.1))
        
        for i, object_initial_pose in enumerate(object_initial_poses):
            for num_turns in range(0, 4):
                object_final_pose = flip_pose_keep_delta(object_initial_pose, box_dims, num_turns=num_turns)
                grasps = generate_grasps(object_initial_pose[:3], object_initial_pose[3:], np.array(box_dims), 
                                      enable_yaw_tilt=False, debug_indices=debug_indices, current_base_index=global_trial_index)
                
                for grasp_data in grasps:  
                    final_pose = copy.deepcopy(object_final_pose)
                    final_pose[2] += 0.1
                    
                    current_trial = {
                        "grasp_pose": grasp_data['grasp'],
                        "initial_object_pose": object_initial_pose,
                        "final_object_pose": final_pose,
                        "object_dimensions": box_dims,
                        "debug_info": grasp_data['debug_info']
                    }
                    test_data.append(current_trial)
                    global_trial_index += 1
                    


    # DEBUG: Print info for specified indices after all generation is complete
    print("\n" + "="*60)
    print("DEBUG INFO FOR SPECIFIED INDICES")
    print("="*60)
    
    for i, trial in enumerate(test_data):
        if i in debug_indices:
            debug = trial['debug_info']
            print(f"\n=== TRIAL INDEX {i} ===")
            print(f"Face: {debug['face']}")
            print(f"Face normal (world): {debug['face_normal_world']}")
            print(f"Approach dir (world): {debug['approach_world']}")
            print(f"Object dims: {debug['dims_total']}")
            print(f"Face axis (local): {debug['face_axis_local']}")
            print(f"Face axis (world): {debug['face_axis_world']}")
            print(f"v1_w: {debug['v1_w']} (span: {debug['span_v1']:.4f})")
            print(f"v2_w: {debug['v2_w']} (span: {debug['span_v2']:.4f})")
            print(f"x_w (chosen): {debug['x_w']}")
            print(f"y_w: {debug['y_w']}")
            print(f"z (approach): {debug['z']}")
            print(f"Option 1 finger-closing dir: {debug['fingers1']}")
            print(f"Option 2 finger-closing dir: {debug['fingers2']}")
            print(f"Option 1 jaw span: {debug['span1']:.4f}")
            print(f"Option 2 jaw span: {debug['span2']:.4f}")
            print(f"CHOSEN: Option {debug['chosen_option']}")
            print(f"Grasp pose: {trial['grasp_pose']}")
            print("-" * 40)
    
    print("="*60 + "\n")

    # Remove debug info before saving to keep JSON clean
    clean_test_data = []
    for trial in test_data:
        clean_trial = {k: v for k, v in trial.items() if k != 'debug_info'}
        clean_test_data.append(clean_trial)

    # Save generated trials to JSON
    experiment_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_generation.json"
    with open(experiment_path, "w") as f:
        json.dump(clean_test_data, f, indent=2)

    print(f"Generated {len(test_data)} trials")
