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
import model_training.pointnet2
sys.modules['pointnet2'] = model_training.pointnet2

# Create an alias for model_training.dataset as dataset
import model_training.dataset
sys.modules['dataset'] = model_training.dataset

import model_training.model
sys.modules['model'] = model_training.model

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
import rclpy
import torch
import open3d as o3d
import json
from placement_quality.cube_simulation import helper
from rclpy.executors import SingleThreadedExecutor
import time
from model_training.dataset import KinematicFeasibilityDataset
from model_training.pointnet2 import *
from model_training.model import GraspObjectFeasibilityNet, PointNetEncoder
from model_training.train import load_pointcloud
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from utils import *
# from omni.isaac.core import SimulationContext
import copy
# # Before running simulation
# sim = SimulationContext(physics_dt=1.0/240.0)  # 240 Hz
PEDESTAL_SIZE = np.array([0.09, 0.11, 0.1])   # X, Y, Z in meters

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2"
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


def model_prediction(model, data, device):
    # Define constant values as in the dataset
    const_xy = torch.tensor([0.2, -0.3], dtype=torch.float32)
    
    # Preprocess the initial and final poses as done in the dataset
    initial_pose = torch.cat([const_xy, torch.tensor(data["initial_object_pose"], dtype=torch.float32)])
    final_pose = torch.cat([const_xy, torch.tensor(data["final_object_pose"], dtype=torch.float32)])
    
    # Now use the preprocessed tensors with batch dimension
    # print(f"input 1: {torch.tensor(data['grasp_pose'], dtype=torch.float32).unsqueeze(0).to(device)}")
    # print(f"input 2: {initial_pose.unsqueeze(0).to(device)}")
    # print(f"input 3: {final_pose.unsqueeze(0).to(device)}")
    raw_success, raw_collision = model(None, 
                                     torch.tensor(data["grasp_pose"], dtype=torch.float32).unsqueeze(0).to(device), 
                                     initial_pose.unsqueeze(0).to(device), 
                                     final_pose.unsqueeze(0).to(device))
    
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
    return pred_success_binary.item(), pred_collision_binary.item()


def robot_action(env: Simulator, grasp_pose, current_state, next_state):
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
            env.collision_counter += 1
            print(f"⚠️ Collision during {current_state} (strike {env.collision_counter}/3)")

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
            "reason": "IK",
            "forced_completion": env.forced_completion
        })
        return False
    


def write_results_to_file(results, file_path, mode='a'):
    """Append results to a JSONL file, ensuring the directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode) as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

def main(checkpoint, use_physics, test_mode=False):
    test_mode = True
    object_frame_path = "/World/Ycb_object"
    pcd_topic = "/cam0/depth_pcl"

    # Initialize ROS2 node
    rclpy.init()
    sim_subscriber = helper.TFSubscriber(pcd_topic)

    # Create an executor# Create an executor
    executor = SingleThreadedExecutor()
    executor.add_node(sim_subscriber)

    env = Simulator(use_physics=use_physics)
    env.start()
    

    object_position = [0.2, -0.3, env.current_data["initial_object_pose"][0]]
    object_orientation = env.current_data["initial_object_pose"][1:]
    if not test_mode:   
        helper.tf_graph_generation(object_frame_path)
        helper.set_cameras(object_position)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Add a flag to check if we have computed the static object feature
    static_feature_computed = False

    results_file = os.path.join(DIR_PATH, "experiment_results_origin_box.jsonl")
    write_interval = 1
    last_written = 0

    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with during the step function")
            env.reset()
            continue
        
        # Process ROS callbacks
        rclpy.spin_once(sim_subscriber, timeout_sec=0)

        # Wait for initial TF data
        if not test_mode and (sim_subscriber.latest_tf is None or sim_subscriber.latest_pcd is None):
            continue
            
        # Calculate the static object feature once we have a valid pointcloud
        if not static_feature_computed:
            env.gripper.open()
            print("Computing static point-cloud embedding from live pointcloud...")
            
            # saved_path = obtain_pcd(sim_subscriber)
            # saved_path = "/home/chris/Chris/placement_ws/src/box_cube_0.031_0.096_0.190.pcd"
            saved_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/experiments/run_20250701_010620/pointcloud.pcd"

            # Convert pointcloud to tensor
            object_pcd_np = load_pointcloud(saved_path)
            object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
            print(f"Loaded point cloud with {object_pcd.shape[0]} points...")
            
            # Forward once through PointNetEncoder
            with torch.no_grad():
                pn = PointNetEncoder(global_feat_dim=256).to(device)
                static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
            print("Done.\n")

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
            model = GraspObjectFeasibilityNet(use_static_obj=True).to(device)
            # Register this feature with the model
            model.register_buffer('static_obj_feat', static_obj_feat)
            # Load the state dict but don't compute point cloud embedding yet
            model.load_state_dict(cleaned_state_dict)
            model.eval()

            
            static_feature_computed = True
            print("Model loaded from checkpoint.\n")

        if env.state == "SETUP":
            # Step 6: Find all grasps with their scores and sort them
            
            object_position = [0.2, -0.3, env.current_data["initial_object_pose"][0]]
            object_orientation = env.current_data["initial_object_pose"][1:]
            # env.task._ycb.set_world_pose(object_position, object_orientation)
            
            grasp_position = env.current_data["grasp_pose"][:3]
            grasp_orientation = env.current_data["grasp_pose"][3:]
            pregrasp_position = copy.deepcopy(grasp_position)
            pregrasp_position[2] = 0.5
            # pregrasp_position, pregrasp_orientation = get_prepose(grasp_position, grasp_orientation)
            placement_position, placement_orientation = env.calculate_placement_pose(env.current_data["grasp_pose"], 
                                                                                     env.current_data["initial_object_pose"], 
                                                                                     env.current_data["final_object_pose"])
            
            # pre_placement_position, pre_placement_orientation = get_prepose(placement_position, placement_orientation)
            pre_placement_position = copy.deepcopy(placement_position)
            pre_placement_position[2] = 0.5
            pre_placement_orientation = euler_angles_to_quat(np.array([180, 0, -90]), degrees=True)

            # This one is object's placement preview
            final_object_position = np.array([0.2, -0.3, env.current_data["final_object_pose"][0]])
            final_object_orientation = env.current_data["final_object_pose"][1:]
            print(f"initial object orientation: {object_orientation}")
            print(f"final object orientation: {final_object_orientation}")

            env.task.set_params(
                object_position=object_position,
                object_orientation=object_orientation,
                preview_box_position=final_object_position,
                preview_box_orientation=final_object_orientation,
            )

            pred_success_val, pred_collision_val = model_prediction(model, env.current_data, device)
            score = pred_success_val * pred_collision_val
            # Print prediction results
            print(f"Prediction: score: {score:.4f}, success: {pred_success_val:.4f}, no collision: {pred_collision_val:.4f}")
            env.state = "PREGRASP"
            env.contact_force = 0.0

        elif env.state == "FAIL":
            env.data_index += 1
            env.current_data = env.test_data[env.data_index]
            
            env.state = "SETUP"
            env.reset()
            continue

        elif env.state == "PREGRASP":
            # On first entry to PREGRASP, initialize collision counter
            env.gripper.open()
            pregrasp_pose = [pregrasp_position, grasp_orientation]
            robot_action(env, pregrasp_pose, "PREGRASP", "GRASP")
            
        elif env.state == "GRASP":
            env.gripper.open()
            if robot_action(env, [grasp_position, grasp_orientation], "GRASP", "GRIPPER"):
                env.open = False

        elif env.state == "PREPLACE_ONE":
            preplace_one_pose = [pre_placement_position, grasp_orientation]
            robot_action(env, preplace_one_pose, "PREPLACE_ONE", "PREPLACE_TWO")
        
        elif env.state == "PREPLACE_TWO":   
            preplace_two_pose = [pre_placement_position, placement_orientation]
            robot_action(env, preplace_two_pose, "PREPLACE_TWO", "PLACE")
            
        elif env.state == "PLACE":
            place_pose = [placement_position, placement_orientation]
            if robot_action(env, place_pose, "PLACE", "GRIPPER"):
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


                if env.controller.is_done():
                    print(f"----------------- END Plan Complete -----------------")
                    object_current_position, object_current_orientation = env.task._ycb.get_world_pose()
                    object_target_position, object_target_orientation = env.task.preview_box.get_world_pose()
                    initial_pose = np.concatenate([object_current_position, object_current_orientation])
                    final_pose = np.concatenate([object_target_position, object_target_orientation])
                    pose_diff = pose_difference(initial_pose, final_pose)
                    env.results.append({
                        "index": env.data_index,
                        "grasp": True,
                        "collision_counter": env.collision_counter,
                        "prediction_score": pred_collision_val,
                        "reason": pose_diff,
                        "forced_completion": env.forced_completion
                    })
                    remaining = write_interval - ((len(env.results) - last_written) % write_interval)
                    if remaining == write_interval:
                        remaining = write_interval
                    print(f"[RESULT] index: {env.data_index}, grasp: success, pose_difference: {pose_diff}")
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
                    "reason": "IK",
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
                        env.state = "PREPLACE_ONE"
                        env.controller.reset()
                        env.task.preview_box.set_world_pose(final_object_position, final_object_orientation)
                    else:
                        print("----------------- Grasp failed -----------------")
                        # reset state & counter for a retry, this is the failure case
                        env.results.append({
                            "index": env.data_index,
                            "grasp": False,
                            "collision_counter": env.collision_counter,
                            "prediction_score": pred_collision_val,
                            "reason": "gripper",
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
    model_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/models/model_20250626_220307/best_model_0_1045_pth"
    use_physics = True
    main(model_path, use_physics)