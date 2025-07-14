import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from isaacsim import SimulationApp

# Display options for the simulation viewport
DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

# Simulation configuration
CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

# Initialize the simulation application
simulation_app = SimulationApp(CONFIG)

import os
import rclpy
import datetime 
from omni.isaac.core.utils import extensions
# from path_simulation.simulation.logger import Logger
# from path_simulation.simulation.simulator import Simulator
# from ycb_simulation.utils.helper import draw_frame
import rclpy


from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import omni

import numpy as np
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import json
import tf2_ros
from rclpy.time import Time
import struct
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, quat2euler
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import random
import math
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# from path_simulation.simulation.RRT_controller import RRTController
# from path_simulation.simulation.RRT_task import RRTTask
# from ycb_simulation.utils.helper import tf_graph_generation
import numpy as np

import carb
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux


# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")


class StandaloneIK:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None
        self.world = None

    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        # set_camera_view(eye=[1.5, 1.25, 2], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")
        
        # Create a world instance
        self.world = World()
        
        # Load robot and target
        self._articulation, self._target = self.load_assets()
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        
        # Set up the physics scene
        self.world.reset()
        
        # Set up the kinematics solver
        self.setup_kinematics()

    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def load_assets(self):
        """Load the Franka robot and target frame"""
        # Add the Franka robot to the stage
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        articulation = Articulation(robot_prim_path)
        
        # Add the target frame to the stage
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        
        return articulation, target

    def setup_kinematics(self):
        """Set up the kinematics solver for the Franka robot"""
        # Load kinematics configuration for the Franka robot
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)
        
        # Print valid frame names for debugging
        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
        
        # Configure the articulation kinematics solver with the end effector
        end_effector_name = "right_gripper"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, 
            self._kinematics_solver, 
            end_effector_name
        )

    def update(self, step):
        """Update the robot's position based on the target's position"""
        # Get the target position and orientation
        target_position, target_orientation = self._target.get_world_pose()
        
        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        
        # Compute inverse kinematics to find joint positions
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, 
            target_orientation
        )
        
        # Apply the joint positions if IK was successful
        if success:
            self._articulation.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken")

    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()
        
        print("Starting simulation loop. Press Ctrl+C to exit.")
        
        # Main simulation loop
        while simulation_app.is_running():
            # Step the simulation
            self.world.step(render=True)
            
            # Update the robot's position
            self.update(step=1.0/60.0)

        # Cleanup when simulation ends
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env = StandaloneIK()
    env.run()
    
    # Close the simulation application
    simulation_app.close()