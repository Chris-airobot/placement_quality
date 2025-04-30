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
import torch
from scipy.spatial.transform import Rotation as R
from model_training.dataset import KinematicFeasibilityDataset
from model_training.pointnet2 import *
from model_training.model import GraspObjectFeasibilityNet, PointNetEncoder
from model_training.train import load_pointcloud

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")
# Define color codes
GREEN = '\033[92m'  # Green text
RED = '\033[91m'    # Red text
RESET = '\033[0m'   # Reset to default color

def draw_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    scale: float = 0.1,
):
    # Isaac Sim's debug draw interface
    from omni.isaac.debug_draw import _debug_draw
    from carb import Float3, ColorRgba
    # Acquire the debug draw interface once at startup or script init
    draw = _debug_draw.acquire_debug_draw_interface()
    # Clear previous lines so we have only one, moving frame.
    draw.clear_lines()
    """
    Draws a coordinate frame with colored X, Y, Z axes at the specified pose.
    
    Args:
        frame_name: A unique name for this drawing (used by Debug Draw).
        position:   (3,) array-like of [x, y, z] for frame origin in world coordinates.
        orientation:(4,) array-like quaternion [x, y, z, w].
        scale:      Length of each axis line.
        duration:   How long (in seconds) the lines remain in the viewport
                    before disappearing. If you keep calling draw_frame each
                    physics step, it will appear continuously.
    """

    def convert_wxyz_to_xyzw(q_wxyz):
        """Convert a quaternion from [w, x, y, z] format to [x, y, z, w] format."""
        return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


    # Convert the quaternion into a 3x3 rotation matrix
    rot_mat = R.from_quat(convert_wxyz_to_xyzw(orientation)).as_matrix()
    
    # Extract the basis vectors for x, y, z from the rotation matrix
    # and scale them to draw lines
    x_axis = rot_mat[:, 0] * scale
    y_axis = rot_mat[:, 1] * scale
    z_axis = rot_mat[:, 2] * scale
    
    # Convert position to a numpy array if needed
    origin = np.array(position, dtype=float)
    
    # Create carb.Float3 objects for start and end points.
    start_points = [
        Float3(origin[0], origin[1], origin[2]),  # for x-axis
        Float3(origin[0], origin[1], origin[2]),  # for y-axis
        Float3(origin[0], origin[1], origin[2])   # for z-axis
    ]
    end_points = [
        Float3(*(origin + x_axis)),
        Float3(*(origin + y_axis)),
        Float3(*(origin + z_axis))
    ]

    # Create carb.ColorRgba objects for each axis.
    colors = [
        ColorRgba(1.0, 0.0, 0.0, 1.0),  # red for x-axis
        ColorRgba(0.0, 1.0, 0.0, 1.0),  # green for y-axis
        ColorRgba(0.0, 0.0, 1.0, 1.0)   # blue for z-axis
    ]

    # Specify line thicknesses as a list of floats.
    sizes = [2.0, 2.0, 2.0]

    # Draw the three axes.
    draw.draw_lines(start_points, end_points, colors, sizes)


def model_prediction(model, data, device):
    i_s, f_s = map(int, data['surfaces'].split('_'))
    
    # Define constant values as in the dataset
    const_xy = torch.tensor([0.4, 0.0], dtype=torch.float32)
    
    # Preprocess the initial and final poses as done in the dataset
    initial_pose = torch.cat([const_xy, torch.tensor(data["initial_object_pose"], dtype=torch.float32)])
    final_pose = torch.cat([const_xy, torch.tensor(data["final_object_pose"], dtype=torch.float32)])
    
    # Now use the preprocessed tensors with batch dimension
    raw_success, raw_collision = model(None, 
                                     torch.tensor(data["grasp_pose"], dtype=torch.float32).unsqueeze(0).to(device), 
                                     initial_pose.unsqueeze(0).to(device), 
                                     final_pose.unsqueeze(0).to(device), 
                                     torch.tensor([i_s, f_s], dtype=torch.long).unsqueeze(0).to(device))
    
    # Apply sigmoid to convert logits to probabilities
    pred_success = torch.sigmoid(raw_success)
    pred_collision = torch.sigmoid(raw_collision)
    
    # Extract scalar values from tensors
    pred_success_val = pred_success.item()
    pred_collision_val = pred_collision.item()
    
    # Get binary predictions based on threshold of 0.5
    pred_success_binary = pred_success > 0.5
    pred_collision_binary = pred_collision > 0.5  # True means "no collision predicted"
    
    
    return pred_success_binary.item(), pred_collision_binary.item()

def main(checkpoint, use_physics):
    env = Simulator(use_physics=use_physics)
    env.start()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    print("Computing static point-cloud embedding …")
    pcd_path      = '/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/perfect_pointcloud.pcd'
    object_pcd_np = load_pointcloud(pcd_path)
    object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
    print(f"Loaded point cloud with {object_pcd.shape[0]} points...")

    # forward once through PointNetEncoder
    with torch.no_grad():
        pn = PointNetEncoder(global_feat_dim=256).to(device)
        static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
    print("Done.\n")

    # 1) Load the raw checkpoint
    checkpoint_data = torch.load(checkpoint, map_location=device)
    # if you saved via Lightning it may be under 'state_dict'
    raw_state_dict = checkpoint_data.get("state_dict", checkpoint_data)

    # 2) Strip any unwanted prefix (e.g. "_orig_mod.")
    prefix_to_strip = "_orig_mod."
    cleaned_state_dict = {}
    for key, tensor in raw_state_dict.items():
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = tensor

    model = GraspObjectFeasibilityNet(use_static_obj=True).to(device)
    model.register_buffer('static_obj_feat', static_obj_feat)  # now model.static_obj_feat is available
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    collision_detected = False
    # logger = Logger(env, DIR_PATH)
    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with during the step function")
            env.reset()
            continue
 
        if env.state == "GRASP":
            pred_success, pred_collision = model_prediction(model, env.current_data, device)
            
            draw_frame(np.array(env.current_data["grasp_pose"][0:3]), 
                    np.array(env.current_data["grasp_pose"][3:]))

            # Perform grasping
            try:
                # Generate actions for the robot
                actions = env.controller.forward(
                    target_end_effector_position=np.array(env.current_data["grasp_pose"][0:3]),
                    target_end_effector_orientation=np.array(env.current_data["grasp_pose"][3:]),
                )

                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                # Apply the actions to the robot
                env.articulation_controller.apply_action(actions)
                if env.check_for_collisions():
                    collision_detected = True
                # Check if we've completed all phases of the planner
                if env.controller.is_done():
                    print("----------------- Grasp Plan Complete -----------------")
                    env.state = "PLACE"
                    env.controller.reset()
                    env.task.set_params(
                        object_position=np.array([0.4, 0, env.current_data["final_object_pose"][0]]),
                        object_orientation=np.array(env.current_data["final_object_pose"][1:]),
                    )
            except Exception as e:
                print("Something went wrong with the movement, recording data and restarting")
                # logger.record_grasping(message="ERROR")
                env.reset()
                continue


        elif env.state == "PLACE":
            # Wait for replay to finish if planner is not at event 4 (post-grasp)
            final_grasp_position, final_grasp_orientation = env.calculate_final_grasp_pose()

            draw_frame(np.array(final_grasp_position), 
                       np.array(final_grasp_orientation))

            try:
                # Generate actions for placement
                actions = env.controller.forward(
                    target_end_effector_position=final_grasp_position,
                    target_end_effector_orientation=final_grasp_orientation,
                )
                
                # Apply the actions to the robot
                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                env.articulation_controller.apply_action(actions)
                if env.check_for_collisions():
                    collision_detected = True
                                            
                # Check if the entire planning sequence is complete
                if env.controller.is_done():
                    print("----------------- Placement Plan Complete -----------------")

                    # Check correctness - fixed comparison for collision
                    success_correct = pred_success == env.controller.ik_check
                    collision_correct = pred_collision == collision_detected
                    
                    pred_success_message = f"{GREEN}IK should be successful{RESET}" if pred_success else f"{RED}IK should fail{RESET}"
                    pred_collision_message = f"{GREEN}there will be a collision{RESET}" if pred_collision else f"{RED}there will not be a collision{RESET}"

                    actual_success_message = f"{GREEN}IK is indeed successful{RESET}" if env.controller.ik_check else f"{RED}IK should fail{RESET}"
                    actual_collision_message = f"{GREEN}there is a collision{RESET}" if collision_detected else f"{RED}there is no collision{RESET}"
                    
                    
                    # Create colored text
                    success_result = f"{GREEN}CORRECT{RESET}" if success_correct else f"{RED}INCORRECT{RESET}"
                    collision_result = f"{GREEN}CORRECT{RESET}" if collision_correct else f"{RED}INCORRECT{RESET}"
                    print(f"You are on sample {env.data_index}")
                    print(f"Predicted: {pred_success_message}, and {pred_collision_message}")
                    print(f"Actual: {actual_success_message}, and {actual_collision_message}")
                    print(f"Correctness: success prediction is {success_result}, collision prediction is {collision_result}")

                    env.data_index += 1
                    input("Press Enter to continue...")
                    env.reset()
            except Exception as e:
                print(f"Error during placement: {e}")
                env.reset()

    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    model_path = "/media/chris/OS2/Users/24330/Desktop/placement_quality/models/model_20250427_224148/best_model_0_0520_pth"
    use_physics = False
    main(model_path, use_physics)