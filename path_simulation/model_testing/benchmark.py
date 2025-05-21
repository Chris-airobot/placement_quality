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
# from omni.isaac.core import SimulationContext

# # Before running simulation
# sim = SimulationContext(physics_dt=1.0/240.0)  # 240 Hz


# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/benchmark"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")
# Define color codes
GREEN = '\033[92m'  # Green text
RED = '\033[91m'    # Red text
RESET = '\033[0m'   # Reset to default color
GRIP_HOLD_STEPS = 40


def model_prediction(model, data, device):
    
    # Define constant values as in the dataset
    const_xy = torch.tensor([0.35, 0.0], dtype=torch.float32)
    
    # Preprocess the initial and final poses as done in the dataset
    initial_pose = torch.cat([const_xy, torch.tensor(data["initial_object_pose"], dtype=torch.float32)])
    final_pose = torch.cat([const_xy, torch.tensor(data["final_object_pose"], dtype=torch.float32)])
    
    # Now use the preprocessed tensors with batch dimension
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


def main(checkpoint, use_physics, test_mode=False):
    test_mode = False
    object_frame_path = "/World/Ycb_object"
    pcd_topic = "/cam0/depth_pcl"
    pcd_path = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/pointcloud.pcd"

    # Initialize ROS2 node
    rclpy.init()
    sim_subscriber = helper.TFSubscriber(pcd_topic)

    # Create an executor# Create an executor
    executor = SingleThreadedExecutor()
    executor.add_node(sim_subscriber)

    env = Simulator(use_physics=use_physics)
    env.start()
    

    object_position = [0.35, 0, env.current_data["initial_object_pose"][0]]
    object_orientation = env.current_data["initial_object_pose"][1:]
    if not test_mode:   
        helper.tf_graph_generation(object_frame_path)
        helper.set_cameras(object_position)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Add a flag to check if we have computed the static object feature
    static_feature_computed = False
    collision_detected = False
    grasp_poses = []
    # logger = Logger(env, DIR_PATH)
    IK = False

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
            if not test_mode:
                # Step 1: Transform from camera frame to robot base frame
                transformed_pcd_robot = helper.transform_pointcloud_to_frame(sim_subscriber.latest_pcd, 
                                                                            sim_subscriber.buffer, 
                                                                            "panda_link0")
            
                # Step 2: Process the pointcloud (filtering/downsampling)
                processed_points = helper.process_pointcloud(transformed_pcd_robot)
                
                # Step 3: Transform from robot base frame to object frame
                # Create a PointCloud2 message for the processed points
                # processed_ros_msg = helper.numpy_to_pointcloud2(processed_points, "panda_link0")
                
                # # Transform the processed pointcloud to object frame
                # object_frame_pcd_msg = helper.transform_pointcloud_to_frame(processed_ros_msg, 
                #                                                             sim_subscriber.buffer, 
                #                                                             "Ycb_object")
                
                # Step 4: Save the object-frame pointcloud directly
                saved_path = helper.save_pointcloud(processed_points, DIR_PATH)
                # saved_path = helper.save_pointcloud(object_frame_pcd_msg, DIR_PATH)
                
                # Step 5: Send to grasper
                grasp_poses = helper.obtain_grasps(saved_path, 12345)
                print(f"Obtained {len(grasp_poses)} grasp poses")
            else:
                raw_grasps = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation/test_grasps.json"
                with open(raw_grasps, "r") as f:
                    raw_grasps = json.load(f)
                for key in sorted(raw_grasps.keys(), key=int):
                    item = raw_grasps[key]
                    position = item["position"]
                    orientation = item["orientation_wxyz"]
                    # Each grasp: [ [position], [orientation] ]
                    grasp_poses.append([position, orientation])
                saved_path = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/pointcloud.pcd"
            
            # Convert pointcloud to tensor
            object_pcd_np = load_pointcloud(pcd_path)
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
            grasp_scores = []  # List to store (grasp, score, success_val, collision_val)
            
            object_position = [0.35, 0, env.current_data["initial_object_pose"][0]]
            object_orientation = env.current_data["initial_object_pose"][1:]
            env.task._ycb.set_world_pose(object_position, object_orientation)
            for grasp_pose in grasp_poses:
                grasp_pose_local = [grasp_pose[0], grasp_pose[1]]
                grasp_pose_world = helper.transform_relative_pose(grasp_pose_local, 
                                                                  object_position,
                                                                  object_orientation)
                grasp_pose_center = helper.local_transform(grasp_pose_world, [0, 0, -0.065])
                # grasp_pose_center = grasp_pose_world
                env.current_data["grasp_pose"] = grasp_pose_center[0] + grasp_pose_center[1]
                pred_success_val, pred_collision_val = model_prediction(model, env.current_data, device)
                # pred_success_val = 0.5
                # pred_collision_val = 0.5
                
                # Define a score that prioritizes high success and low collision probability
                # collision_val is the probability of NO collision, so higher is better
                score = pred_success_val * 0.3 + pred_collision_val * 0.7  # Weighted combination
                
                # Add this grasp and its score to our list
                grasp_scores.append((grasp_pose_center, score, pred_success_val, pred_collision_val))
            
            # Sort grasps by score in descending order
            grasp_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Print top grasps
            print("\nTop 5 grasps:")
            for i, (grasp, score, success, no_collision) in enumerate(grasp_scores[:5]):
                print(f"#{i+1}: Score: {score:.4f} (success: {success:.4f}, no collision: {no_collision:.4f})")
            
            # Use the best grasp for execution
            if grasp_scores:
                env.state = "PREGRASP"  
                current_grasp = grasp_scores.pop(0)

        if env.state == "PREGRASP":
            # On first entry to PREGRASP, initialize collision counter
            if not hasattr(env, 'pregrasp_collision_counter'):
                env.pregrasp_collision_counter = 0
                env.gripper.open()
                
                # Calculate pregrasp pose (offset from grasp position)
                env.pregrasp_position = np.array(current_grasp[0][0]).copy()
                env.pregrasp_orientation = np.array(current_grasp[0][1]).copy()
                
                # Move the pregrasp position 10cm above the grasp point (assuming z is up)
                env.pregrasp_position[2] += 0.1
                
                # Visualize pregrasp target
                env.task._frame.set_world_pose(env.pregrasp_position, env.pregrasp_orientation)
                
                # Reset the controller for new planning
                env.controller.reset()
                print("Planning movement to pregrasp position...")
            
            # Generate actions for moving to pregrasp pose
            actions = env.controller.forward(
                target_end_effector_position=env.pregrasp_position,
                target_end_effector_orientation=env.pregrasp_orientation,
            )
            
            # Apply the actions if planning successful
            if env.controller.ik_check:
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                env.articulation_controller.apply_action(actions)
            else:
                print(f"----------------- RRT cannot find path to pregrasp, trying next grasp -----------------")
                # Clean up state before trying next grasp
                delattr(env, 'pregrasp_collision_counter')
                if hasattr(env, 'pregrasp_position'):
                    delattr(env, 'pregrasp_position')
                if hasattr(env, 'pregrasp_orientation'):
                    delattr(env, 'pregrasp_orientation')
                
                # Try next grasp if available
                if grasp_scores:    
                    current_grasp = grasp_scores.pop(0)
                    env.state = "PREGRASP"  # Restart with new grasp
                else:
                    print("No more grasps to try")
                    env.data_index += 1
                    env.current_data = env.test_data[env.data_index]
                    env.state = "SETUP"
                    env.reset()
                continue
            
            # Check for collisions during pregrasp movement
            if env.check_for_collisions():
                env.pregrasp_collision_counter += 1
                print(f"⚠️ Collision during PREGRASP (strike {env.pregrasp_collision_counter}/3)")
                
                # If we've hit 3 strikes, try next grasp
                if env.pregrasp_collision_counter >= 3:
                    print("🛑 Too many collisions during pregrasp — trying next grasp")
                    # Clean up state
                    delattr(env, 'pregrasp_collision_counter')
                    if hasattr(env, 'pregrasp_position'):
                        delattr(env, 'pregrasp_position')
                    if hasattr(env, 'pregrasp_orientation'):
                        delattr(env, 'pregrasp_orientation')
                    
                    # Try next grasp if available
                    if grasp_scores: 
                        current_grasp = grasp_scores.pop(0)
                        env.state = "PREGRASP"  # Restart with new grasp
                    else:
                        print("No more grasps to try")
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        env.state = "SETUP"
                    
                    env.reset()
                    continue
            
            # When pregrasp movement is complete, transition to GRASP state
            if env.controller.is_done():
                print("----------------- Pregrasp Complete, Moving to Final Grasp -----------------")
                # Clean up pregrasp state
                delattr(env, 'pregrasp_collision_counter')
                # Keep pregrasp_position and pregrasp_orientation for reference
                
                # Reset controller for the next planning phase
                env.controller.reset()
                
                # Update visualization frame to show actual grasp target
                env.task._frame.set_world_pose(np.array(current_grasp[0][0]), 
                                              np.array(current_grasp[0][1]))
                
                # Transition to GRASP state
                env.state = "GRASP"

        if env.state == "GRASP":
            env.gripper.open()

            # on first entry to GRASP, open the gripper and zero the collision counter
            if not hasattr(env, 'grasp_collision_counter'):
                env.grasp_collision_counter = 0

            # tmp_grasp = grasp_scores[1]
            env.task._frame.set_world_pose(np.array(current_grasp[0][0]), 
                                            np.array(current_grasp[0][1]))
            
            actions = env.controller.forward(
                target_end_effector_position=np.array(current_grasp[0][0]),
                target_end_effector_orientation=np.array(current_grasp[0][1]),
            )
            if env.controller.ik_check:
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                # Apply the actions to the robot
                env.articulation_controller.apply_action(actions)
            else:
                print(f"----------------- RRT cannot find a path, going to next grasp -----------------")
                if grasp_scores:    
                    current_grasp = grasp_scores.pop(0)
                    env.state = "PREGRASP"
                    continue
                else:
                    print("No more grasps to try")
                    env.data_index += 1
                    env.current_data = env.test_data[env.data_index]
                    env.state = "SETUP"
                    env.reset()
                    continue
            
            # check for collision this step
            if env.check_for_collisions():
                env.grasp_collision_counter += 1
                print(f"⚠️ Collision during GRASP (strike {env.grasp_collision_counter}/3)")
                # if we’ve hit 3 strikes, reset entire episode
                if env.grasp_collision_counter >= 3:
                    print("🛑 Too many collisions — resetting environment")
                    # clear the counter before resetting
                    delattr(env, 'grasp_collision_counter')
                    if grasp_scores: 
                        current_grasp = grasp_scores.pop(0)
                        env.state = "PREGRASP"
                    else:
                        print("No more grasps to try")
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        env.state = "SETUP"
                    
                    env.reset()
                    continue
            # else:
            #     # clean step → reset the strike counter
            #     env.grasp_collision_counter = 0
            
            if env.controller.is_done():
                print("----------------- RRT Grasp Plan Complete -----------------")
                env.state = "GRIP"
                IK = True
                env.open = False
                delattr(env, 'grasp_collision_counter')
                
            
        elif env.state == "GRIP":
            # if this is the first time we enter GRIP, reset our counter
            if not hasattr(env, "grip_step_counter"):
                env.grip_step_counter = 0
                if env.open:
                    env.gripper.open()  
                else:
                    env.gripper.close()
                

            env.grip_step_counter += 1
            if env.grip_step_counter < GRIP_HOLD_STEPS:
                continue   # stay here until we've done enough steps

            if not env.open:
                if env.check_grasp_success():
                    print("Successfully grasped the object")
                    env.state = "PLACE"
                    env.controller.reset()
                else:
                    print("----------------- Grasp failed -----------------")
                    # reset state & counter for a retry
                    delattr(env, "grip_step_counter")
                    env.task.set_params(
                        object_position=np.array([0.35, 0, env.current_data["initial_object_pose"][0]]),
                        object_orientation=np.array(env.current_data["initial_object_pose"][1:]),
                    )
                    if grasp_scores: 
                        current_grasp = grasp_scores.pop(0)
                        env.state = "PREGRASP"
                    else:
                        print("No more grasps to try")
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        env.state = "SETUP"
                    env.reset()
                    env.gripper.open()
            else:
                print("----------------- Gripper Open -----------------")



        elif env.state == "PLACE":
            if not hasattr(env, 'place_collision_counter'):
                env.place_collision_counter = 0
            # Wait for replay to finish if planner is not at event 4 (post-grasp)
            final_grasp_position, final_grasp_orientation = env.calculate_placement_pose(env.current_data["grasp_pose"], 
                                                                                         env.current_data["initial_object_pose"], 
                                                                                         env.current_data["final_object_pose"])
            env.task._frame.set_world_pose(np.array(final_grasp_position), 
                                           np.array(final_grasp_orientation))

            # final_grasp_position = np.array(current_grasp[0][0]).copy()
            # final_grasp_position[2] += 0.2
            # Generate actions for placement
            actions = env.controller.forward(
                target_end_effector_position=final_grasp_position,
                target_end_effector_orientation=final_grasp_orientation,
            )
            
            # Apply the actions to the robot
            if env.controller.ik_check:
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                env.articulation_controller.apply_action(actions)
            else:
                print(f"----------------- RRT cannot find a path, going to next grasp -----------------")
                if grasp_scores:    
                    current_grasp = grasp_scores.pop(0)
                    env.state = "PREGRASP"
                    env.reset()
                    continue
                else:
                    print("No more grasps to try")
                    env.data_index += 1
                    env.current_data = env.test_data[env.data_index]
                    env.state = "SETUP"
                    env.reset()
                    continue
            
            if env.check_for_collisions():
                env.place_collision_counter += 1
                print(f"⚠️ Collision during PLACE (strike {env.place_collision_counter}/3)")
                # if we’ve hit 3 strikes, reset entire episode
                if env.place_collision_counter >= 3:
                    print("🛑 Too many collisions — resetting environment")
                    # clear the counter before resetting
                    delattr(env, 'place_collision_counter')
                    if grasp_scores: 
                        current_grasp = grasp_scores.pop(0)
                        env.state = "GRASP"
                    else:
                        print("No more grasps to try")
                        env.data_index += 1
                        env.current_data = env.test_data[env.data_index]
                        env.state = "SETUP"
                    
                    env.reset()
                    continue
            # else:
            #     # clean step → reset the strike counter
            #     env.grasp_collision_counter = 0
            
            if env.controller.is_done():
                print("----------------- Placement Plan Complete -----------------")
                env.open = True
                env.state = "GRIP"
                delattr(env, 'place_collision_counter')
            
    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    model_path = "/home/chris/Chris/placement_ws/src/data/path_simulation/models/model_20250520_210301/best_model_0_1069_pth"
    use_physics = True
    main(model_path, use_physics)