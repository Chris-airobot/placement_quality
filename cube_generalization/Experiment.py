import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to path 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Create an alias for model_training.pointnet2 as pointnet2
# This needs to happen before any imports that use pointnet2

from isaacsim import SimulationApp

DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON   = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height":1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

simulation_app = SimulationApp(CONFIG)

import os
import datetime 
from omni.isaac.core.utils import extensions
from simulator import Simulator
import numpy as np
import torch
import json
import time
from cube_generalization.utils import local_transform
from model import EnhancedCollisionPredictionNet  # Fixed: use the correct model class
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from cube_generalization.utils import *
# from omni.isaac.core import SimulationContext
import copy
# # Before running simulation
# sim = SimulationContext(physics_dt=1.0/240.0)  # 240 Hz
PEDESTAL_SIZE = np.array([0.09, 0.11, 0.1])   # X, Y, Z in meters

simulation_app.update()

base_dir = "/home/riot/Chris/data/box_simulation/v4"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"experiments")
# Define color codes
GREEN = '\033[92m'  # Green text
RED = '\033[91m'    # Red text
RESET = '\033[0m'   # Reset to default color
GRIP_HOLD_STEPS = 60
GRIPPER_STEP_SIZE = 0.002  # Step size for gradual gripper movement
GRIPPER_OPEN_POS = 0.04    # Fully open position
GRIPPER_CLOSE_POS = 0.0    # Fully closed position

def gradual_gripper_control(env: Simulator, target_open, max_steps=100):
    """
    Gradually control the gripper to open or close position with force detection
    Args:
        env: Simulator instance
        target_open: True to open gripper, False to close gripper
        max_steps: Maximum number of steps to prevent infinite loops
    Returns:
        bool: True if gripper has reached target position or object grasped, False otherwise
    """
    current_positions = env.gripper.get_joint_positions()
    
    # Add step counter to prevent infinite loops
    if not hasattr(env, 'gripper_step_counter'):
        env.gripper_step_counter = 0
    else:
        env.gripper_step_counter += 1
    
    # Check for timeout
    if env.gripper_step_counter >= max_steps:
        print(f"⚠️ Gripper control timeout after {max_steps} steps - forcing completion")
        env.gripper_step_counter = 0
        env.forced_completion = True
        return True
    
    if target_open:
        # Gradually open gripper (move toward 0.04)
        new_positions = [
            min(current_positions[0] + GRIPPER_STEP_SIZE, GRIPPER_OPEN_POS),
            min(current_positions[1] + GRIPPER_STEP_SIZE, GRIPPER_OPEN_POS)
        ]
        # Fix: Use AND instead of OR - both fingers must reach target
        target_reached = (abs(current_positions[0] - GRIPPER_OPEN_POS) < 1e-3 and 
                         abs(current_positions[1] - GRIPPER_OPEN_POS) < 1e-3)
        
        # Add progress tracking
        if env.gripper_step_counter % 100 == 0:  # Print every 100 steps
            print(f"Opening gripper: pos1={current_positions[0]:.4f}, pos2={current_positions[1]:.4f}, target={GRIPPER_OPEN_POS}")
            
    else:
        # Check if object contact force exceeds threshold BEFORE moving
        force_grasped = env.contact_force > env.force_threshold
        
        if force_grasped:
            print(f"Force threshold exceeded ({env.contact_force:.3f}N > {env.force_threshold}N), stopping gripper closing")
            env.gripper_step_counter = 0  # Reset counter
            return True
        
        # Gradually close gripper (move toward 0.0)
        new_positions = [
            max(current_positions[0] - GRIPPER_STEP_SIZE, GRIPPER_CLOSE_POS),
            max(current_positions[1] - GRIPPER_STEP_SIZE, GRIPPER_CLOSE_POS)
        ]
        
        # Fix: Use AND instead of OR - both fingers must reach target
        target_reached = (abs(current_positions[0] - GRIPPER_CLOSE_POS) < 1e-3 and 
                         abs(current_positions[1] - GRIPPER_CLOSE_POS) < 1e-3)
        
        # Print force information for debugging
        if env.contact_force > 0:
            print(f"Contact force: {env.contact_force:.3f}N (threshold: {env.force_threshold}N)")
        
        # Add progress tracking
        if env.gripper_step_counter % 100 == 0:  # Print every 100 steps
            print(f"Closing gripper: pos1={current_positions[0]:.4f}, pos2={current_positions[1]:.4f}, target={GRIPPER_CLOSE_POS}")
    
    # Apply the gradual movement
    env.gripper.apply_action(ArticulationAction(joint_positions=new_positions))
    
    # Reset counter when target is reached
    if target_reached:
        env.gripper_step_counter = 0
    
    return target_reached

# Global variable to store normalization stats (loaded once)
global_normalization_stats = None

def model_prediction(model, data, env: Simulator, device):
    """Updated model prediction with proper normalization and embeddings - FAST VERSION"""
    try:
        dx, dy, dz = data['object_dimensions']
        from dataset import cuboid_corners_local_ordered, _zscore_normalize
        corners = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)

        # Load real embeddings for this experiment
        embeddings_file = "/home/riot/Chris/placement_quality/cube_generalization/experiment_embeddings.npy"
        
        experiments = env.test_data
        
        # Find matching experiment by dimensions and poses
        exp_index = None
        for i, exp in enumerate(experiments):
            if (np.allclose(exp['object_dimensions'], data['object_dimensions'], atol=1e-6) and
                np.allclose(exp['initial_object_pose'], data['initial_object_pose'], atol=1e-6)):
                exp_index = i
                break
        
        if exp_index is None:
            print("⚠️ Experiment not found in embeddings, using zero embedding")
            embedding = np.zeros(1024, dtype=np.float32)
        else:
            # Load the corresponding embedding
            embeddings = np.load(embeddings_file)
            if exp_index < len(embeddings):
                embedding = embeddings[exp_index]
            else:
                print(f"⚠️ Embedding index {exp_index} out of range, using zero embedding")
                embedding = np.zeros(1024, dtype=np.float32)

        # Extract poses
        grasp_pos = np.array(data['grasp_pose'][:3])
        grasp_ori = np.array(data['grasp_pose'][3:])
        init_pos = np.array(data['initial_object_pose'][:3])
        init_ori = np.array(data['initial_object_pose'][3:])
        final_pos = np.array(data['final_object_pose'][:3])
        final_ori = np.array(data['final_object_pose'][3:])

        # ✨ FAST NORMALIZATION USING PRELOADED STATS ✨
        normalization_stats = global_normalization_stats  # Use global variable
        
        # Normalize corners
        corners_normalized, _, _ = _zscore_normalize(
            corners.reshape(1, -1),
            normalization_stats['corners_mean'],
            normalization_stats['corners_std']
        )
        corners_normalized = corners_normalized.reshape(8, 3)
        
        # Normalize positions using unified stats
        pos_mean = normalization_stats['pos_mean']
        pos_std = normalization_stats['pos_std']
        
        # Convert to numpy arrays if they're tensors
        if isinstance(pos_mean, torch.Tensor):
            pos_mean = pos_mean.cpu().numpy()
        if isinstance(pos_std, torch.Tensor):
            pos_std = pos_std.cpu().numpy()
        
        grasp_pos_norm = (grasp_pos - pos_mean) / pos_std
        init_pos_norm = (init_pos - pos_mean) / pos_std
        final_pos_norm = (final_pos - pos_mean) / pos_std
        
        # Normalize orientations
        grasp_ori_norm, _, _ = _zscore_normalize(
            grasp_ori.reshape(1, -1),
            normalization_stats['grasp_ori_mean'],
            normalization_stats['grasp_ori_std']
        )
        
        init_ori_norm, _, _ = _zscore_normalize(
            init_ori.reshape(1, -1),
            normalization_stats['init_ori_mean'],
            normalization_stats['init_ori_std']
        )
        
        final_ori_norm, _, _ = _zscore_normalize(
            final_ori.reshape(1, -1),
            normalization_stats['final_ori_mean'],
            normalization_stats['final_ori_std']
        )
        
        # Reconstruct normalized poses
        grasp_normalized = np.concatenate([grasp_pos_norm.flatten(), grasp_ori_norm.flatten()])
        init_normalized = np.concatenate([init_pos_norm.flatten(), init_ori_norm.flatten()])
        final_normalized = np.concatenate([final_pos_norm.flatten(), final_ori_norm.flatten()])

        # Convert to tensors
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        corners_tensor = torch.tensor(corners_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        grasp_tensor = torch.tensor(grasp_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        init_tensor = torch.tensor(init_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        final_tensor = torch.tensor(final_normalized, dtype=torch.float32).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            collision_logits = model(embedding_tensor, corners_tensor, grasp_tensor, init_tensor, final_tensor)
            collision_prob = torch.sigmoid(collision_logits).item()

        return 1-collision_prob

    except Exception as e:
        print(f"❌ Error in model prediction: {e}")
        return 0.5  # Default neutral prediction



def robot_action(env: Simulator, grasp_pose, current_state, next_state, pred_collision_val=None):
    grasp_position, grasp_orientation = grasp_pose[0], grasp_pose[1]
    env.task._frame.set_world_pose(np.array(grasp_position), 
                                    np.array(grasp_orientation))

    actions = env.controller.forward(
        target_end_effector_position=np.array(grasp_position),
        target_end_effector_orientation=np.array(grasp_orientation),
    )
    
    if env.controller.ik_check:
        kps, kds = env.task.get_custom_gains()
        env.articulation_controller.set_gains(kps, kds)
        env.articulation_controller.apply_action(actions)

        if env.check_for_collisions():
            # Only count collisions after grasping (post-grasp stages)
            post_grasp_stages = ["PREPLACE_ONE", "PREPLACE_TWO", "PLACE", "END"]
            if current_state in post_grasp_stages:
                env.collision_counter += 1
                print(f"⚠️ Collision during {current_state} (strike {env.collision_counter}/3)")

                # NEW: Abort this attempt if too many collisions
                if env.collision_counter >= 3:
                    print(f"🛑 Too many collisions during {current_state} — skipping to next grasp")
                    # Record failure due to collisions
                    env.results.append({
                        "index": env.data_index,
                        "grasp": True,  # Always True for post-grasp stages
                        "collision_counter": env.collision_counter,
                        "prediction_score": pred_collision_val,
                        "reason": "collision_limit",
                        "forced_completion": env.forced_completion
                    })
                    # Reset counters/state and move on
                    env.collision_counter = 0
                    env.state = "FAIL"
                    env.controller.reset()
                    return False
            else:
                # For pre-grasp stages, just print the collision but don't count it
                print(f"⚠️ Collision detected during {current_state} (not counted - pre-grasp stage)")
        # if env.collision_counter >= 300000000000000:
        #     print(f"🛑 Too many collisions during {current_state} — trying next grasp")
        #     env.collision_counter = 0
        #     env.state = "FAIL"
        #     env.results.append({
        #         "index": env.data_index,
        #         "grasp": True,
        #         "collision_counter": env.collision_counter,
        #         "reason": pose_diff,
        #         "forced_completion": env.forced_completion
        #     })
        #     return False
        if env.controller.is_done():
            print(f"----------------- {current_state} Plan Complete -----------------")
            env.state = next_state
            env.controller.reset()
            return True
    else:
        print(f"----------------- RRT cannot find a path for {current_state}, going to next grasp -----------------")
        env.state = "FAIL"
        env.results.append({
            "index": env.data_index,
            "grasp": True,
            "collision_counter": env.collision_counter,
            "prediction_score": pred_collision_val,
            "reason": "IK_fail",
            "forced_completion": env.forced_completion
        })
        return False
    


def write_results_to_file(results, file_path, mode='a'):
    """Append results to a JSONL file, ensuring the directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

def main(checkpoint, use_physics):
    global global_normalization_stats  # Declare we're modifying the global variable
    
    env = Simulator(use_physics=use_physics)
    env.start()
    
    object_position = env.current_data["initial_object_pose"][:3]
    object_orientation = env.current_data["initial_object_pose"][3:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load the checkpoint using the latest architecture
    try:
        from model_eval import load_checkpoint_with_original_architecture
        model, checkpoint_data = load_checkpoint_with_original_architecture(checkpoint, device, original_architecture=False)
        model.eval()
        print("✅ Loaded model with latest architecture")
        
        # Extract normalization stats from checkpoint (MUCH FASTER!)
        global_normalization_stats = checkpoint_data.get('normalization_stats', None)
        if global_normalization_stats is None:
            print("❌ No normalization stats in checkpoint! Please retrain model with normalization stats.")
            return
        else:
            print("✅ Loaded normalization stats from checkpoint")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("Model loaded from checkpoint.\n")

    
    results_file = os.path.join(DIR_PATH, "experiment_results_test_data.jsonl")
    write_interval = 1
    last_written = 0

    # Track how many times we have failed to GRASP for a given initial object pose
    env.grasp_fail_counts = {}

    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with during the step function")
            env.reset()
            continue
        
        if env.state == "SETUP":            
            # Skip samples that have already failed to GRASP many times (not movement failures)
            pose_key = tuple(env.current_data["initial_object_pose"].tolist()) if isinstance(env.current_data["initial_object_pose"], np.ndarray) else tuple(env.current_data["initial_object_pose"])
            
            # Debug: Print current grasp fail counts
            print(f"🔍 Debug - Current pose key: {pose_key}")
            print(f"🔍 Debug - Fail count for this pose: {env.grasp_fail_counts.get(pose_key, 0)}")
            print(f"🔍 Debug - Total poses with failures: {len(env.grasp_fail_counts)}")
            if env.grasp_fail_counts:
                print(f"🔍 Debug - Sample fail counts: {dict(list(env.grasp_fail_counts.items())[:3])}")
            
            while env.grasp_fail_counts.get(pose_key, 0) >= 10:
                print(f"🔁 Skipping sample index {env.data_index} – initial pose has already failed to GRASP {env.grasp_fail_counts[pose_key]} times")
                env.data_index += 1
                if env.data_index >= len(env.test_data):
                    print("✅ Reached end of dataset while skipping.")
                    break
                env.current_data = env.test_data[env.data_index]
                pose_key = tuple(env.current_data["initial_object_pose"].tolist()) if isinstance(env.current_data["initial_object_pose"], np.ndarray) else tuple(env.current_data["initial_object_pose"])

            object_position = env.current_data["initial_object_pose"][:3]
            object_orientation = env.current_data["initial_object_pose"][3:]
            # env.task._ycb.set_world_pose(object_position, object_orientation)
            
            grasp_position = env.current_data["grasp_pose"][:3]
            grasp_orientation = env.current_data["grasp_pose"][3:]
            grasp_pose = grasp_position + grasp_orientation
            # Update the grasp pose to be in the tool_center frame of the object
            fixed_grasp_pose = local_transform(grasp_pose, [0, 0, 0])
            grasp_position = fixed_grasp_pose[:3]
            grasp_orientation = fixed_grasp_pose[3:]


            # pregrasp_position = copy.deepcopy(grasp_position)
            # pregrasp_position[2] = 0.5
            pregrasp_position, pregrasp_orientation = get_reachable_prepose(grasp_position, grasp_orientation, env, max_offset=0.10)
            if pregrasp_position is None:
                print("No reachable pre-grasp for this grasp – skipping")
                env.state = "FAIL"
                continue

            placement_position, placement_orientation = env.calculate_placement_pose(fixed_grasp_pose, 
                                                                                     env.current_data["initial_object_pose"], 
                                                                                     env.current_data["final_object_pose"])

            
            pre_placement_position, pre_placement_orientation = get_reachable_prepose(grasp_position, grasp_orientation, env, max_offset=0.10)
            r_grasp = R.from_quat([grasp_orientation[1], grasp_orientation[2], grasp_orientation[3], grasp_orientation[0]])
            r_place = R.from_quat([placement_orientation[1], placement_orientation[2], placement_orientation[3], placement_orientation[0]])
            r_diff = r_grasp.inv() * r_place
            angle_diff_rad = r_diff.magnitude()
            angle_diff_deg = np.rad2deg(angle_diff_rad)
            skip_preplace_two = angle_diff_deg < 10.0  # skip PREPLACE_TWO if very similar

            # This one is object's placement preview
            final_object_position = env.current_data["final_object_pose"][:3]
            final_object_orientation = env.current_data["final_object_pose"][3:]


            env.task.set_params(
                object_position=object_position,
                object_orientation=object_orientation,
                preview_box_position=final_object_position,
                preview_box_orientation=final_object_orientation,
                object_scale=np.array(env.current_data["object_dimensions"]),
            )

            pred_collision_val = model_prediction(model, env.current_data, env, device)
            # Since we only have collision prediction, use it as the score
            score = pred_collision_val
            # Print prediction results
            print(f"Prediction: no collision probability: {pred_collision_val:.4f}")
            
            # Reset controller to clear any state from IK checks during setup
            env.controller.reset()
            
            env.state = "PREGRASP"
            env.contact_force = 0.0

        elif env.state == "FAIL":
            # Don't count movement failures - only grasp failures are counted elsewhere
            env.data_index += 1
            env.current_data = env.test_data[env.data_index]
            
            env.state = "SETUP"
            env.reset()
            continue

        elif env.state == "PREGRASP":
            # On first entry to PREGRASP, initialize collision counter
            env.gripper.open()
            pregrasp_pose = [pregrasp_position, pregrasp_orientation]
            robot_action(env, pregrasp_pose, "PREGRASP", "GRASP", pred_collision_val)
            
        elif env.state == "GRASP":
            env.gripper.open()
            if robot_action(env, [grasp_position, grasp_orientation], "GRASP", "GRIPPER", pred_collision_val):
                env.open = False

        elif env.state == "PREPLACE_ONE":
            preplace_one_pose = [pre_placement_position, grasp_orientation]
            next_state = "PLACE" if skip_preplace_two else "PREPLACE_TWO"
            robot_action(env, preplace_one_pose, "PREPLACE_ONE", next_state, pred_collision_val)
        
        elif env.state == "PREPLACE_TWO":
            preplace_two_pose = [pre_placement_position, placement_orientation]
            robot_action(env, preplace_two_pose, "PREPLACE_TWO", "PLACE", pred_collision_val)
            
        elif env.state == "PLACE":
            place_pose = [placement_position, placement_orientation]
            if robot_action(env, place_pose, "PLACE", "GRIPPER", pred_collision_val):
                env.open = True 

        elif env.state == "END":
            _, gripper_current_orientation = env.gripper.get_world_pose()
   
            env.task._frame.set_world_pose(np.array(pre_placement_position), np.array(gripper_current_orientation))
            actions = env.controller.forward(
                target_end_effector_position=np.array(pre_placement_position),
                target_end_effector_orientation=np.array(gripper_current_orientation),
            )
            
            if env.controller.ik_check:
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                env.articulation_controller.apply_action(actions)

                if env.check_for_collisions():
                    env.collision_counter += 1
                    print(f"⚠️ Collision during END (strike {env.collision_counter}/3)")

                    # NEW: Too many collisions while returning – treat as failure
                    if env.collision_counter >= 3:
                        print("🛑 Too many collisions during END — skipping to next grasp")
                        env.results.append({
                            "index": env.data_index,
                            "grasp": True,
                            "collision_counter": env.collision_counter,
                            "prediction_score": pred_collision_val,
                            "reason": "collision_limit",
                            "forced_completion": env.forced_completion
                        })
                        env.collision_counter = 0
                        env.state = "FAIL"
                        continue

                if env.controller.is_done():
                    print(f"----------------- END Plan Complete -----------------")
                    object_current_position, object_current_orientation = env.task._ycb.get_world_pose()
                    object_target_position, object_target_orientation = env.task.preview_box.get_world_pose()
                    initial_pose = np.concatenate([object_current_position, object_current_orientation])
                    final_pose = np.concatenate([object_target_position, object_target_orientation])
                    env.results.append({
                        "index": env.data_index,
                        "grasp": True,
                        "collision_counter": env.collision_counter,
                        "prediction_score": pred_collision_val,
                        "target_object_pose": final_pose.tolist(),
                        "actual_object_pose": object_current_position.tolist(),
                        "reason": "success",
                        "forced_completion": env.forced_completion
                    })
                    remaining = write_interval - ((len(env.results) - last_written) % write_interval)
                    if remaining == write_interval:
                        remaining = write_interval
                    print(f"    {remaining} results until next data write.")
                    # reset state & counter for a retry, this is the success case
                    env.data_index += 1
                    env.current_data = env.test_data[env.data_index]
                    env.state = "SETUP"
                    env.reset()
                    continue
            else:
                print(f"----------------- RRT cannot find a path for END, going to next grasp -----------------")
                env.state = "FAIL"
                env.results.append({
                    "index": env.data_index,
                    "grasp": True,
                    "collision_counter": env.collision_counter,
                    "prediction_score": pred_collision_val,
                    "reason": "IK_fail",
                    "forced_completion": env.forced_completion
                })


        elif env.state == "GRIPPER":
            # Use gradual gripper control instead of instant open/close
            if env.open:
                # Open gripper
                while not gradual_gripper_control(env, target_open=True, max_steps=100):
                    if env.forced_completion:
                        break
                    env.world.step(render=True)
                print("Gripper opening completed")
            else:
                # Close gripper
                while not gradual_gripper_control(env, target_open=False, max_steps=100):
                    if env.forced_completion:
                        break
                    env.world.step(render=True)
            
            env.step_counter += 1
            
            # Only proceed to next state when gripper has reached target position
            # and we've held it for the required number of steps
            if env.step_counter >= GRIP_HOLD_STEPS:
                # After grasp
                if not env.open:
                    if env.check_grasp_success() and not env.forced_completion:
                        print("Successfully grasped the object")
                        # Reset collision counter for post-grasp stages
                        env.collision_counter = 0
                        env.state = "PREPLACE_ONE"
                        env.controller.reset()
                        env.task.preview_box.set_world_pose(final_object_position, final_object_orientation)
                    else:
                        print("----------------- Grasp failed -----------------")
                        
                        # Count this as a grasp failure for this pose
                        pose_key = tuple(env.current_data["initial_object_pose"].tolist()) if isinstance(env.current_data["initial_object_pose"], np.ndarray) else tuple(env.current_data["initial_object_pose"])
                        env.grasp_fail_counts[pose_key] = env.grasp_fail_counts.get(pose_key, 0) + 1
                        
                        # Debug: Print grasp failure info
                        print(f"🔍 Debug - GRASP FAILED for pose: {pose_key}")
                        print(f"🔍 Debug - New fail count: {env.grasp_fail_counts[pose_key]}")
                        print(f"🔍 Debug - Moving from data_index {env.data_index} to {env.data_index + 1}")
                        
                        # reset state & counter for a retry, this is the failure case
                        env.results.append({
                            "index": env.data_index,
                            "grasp": False,
                            "collision_counter": env.collision_counter,
                            "prediction_score": pred_collision_val,
                            "reason": "grasp_fail",
                            "forced_completion": env.forced_completion
                        })
                        
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        object_next_orientation = env.current_data["initial_object_pose"][1:]
                        # while object_next_orientation == object_orientation:

                        #     env.data_index += 1
                        #     env.current_data = env.test_data[env.data_index]
                        #     object_next_orientation = env.current_data["initial_object_pose"][1:]

                        env.current_data = env.test_data[env.data_index]
                        env.state = "SETUP"
                        remaining = write_interval - ((len(env.results) - last_written) % write_interval)
                        if remaining == write_interval:
                            remaining = write_interval
                        print(f"[RESULT] index: {env.data_index}, grasp: failure, collision_counter: {env.collision_counter}")
                        print(f"    {remaining} results until next data write.")
                        env.reset()
                        continue

                # After placement
                else:
                    # # Wait for 2 seconds (120 steps at 60 FPS) for object to stabilize
                    stabilization_steps = 60 # 2 seconds at 60 FPS
                    for _ in range(stabilization_steps):
                        env.world.step(render=True)
                    
                    print("----------------- Object stabilization complete, Going to END -----------------")
                    env.state = "END"


            else:
                continue   # stay here until gripper reaches target and we've done enough steps


        # After appending a result to env.results (success or failure), check if we should write:
        if len(env.results) - last_written >= write_interval:
            write_results_to_file(env.results[last_written:], results_file, mode='a')
            last_written = len(env.results)

    # Cleanup when simulation ends
    simulation_app.close()

    # Write any remaining results not yet saved
    if len(env.results) > last_written:
        write_results_to_file(env.results[last_written:], results_file, mode='a')

if __name__ == "__main__":
    model_path = "/home/riot/Chris/placement_quality/cube_generalization/best_model_roc_20250804_175834.pth"
    use_physics = True
    main(model_path, use_physics)