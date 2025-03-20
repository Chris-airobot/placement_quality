import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
import rclpy
import time

from rclpy.executors import SingleThreadedExecutor

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.sensor import ContactSensor

from simulation.task import YcbTask
from simulation.planner import YcbPlanner

from utils.vision import *
from utils.helper import *

class YcbCollection:
    def __init__(self):
        # Basic variables
        self.world = None
        self.object = None
        self.planner = None
        self.controller = None
        self.robot = None
        self.cameras = None
        self.task = None
        self.data_logger = None
        self.contact_sensors = None

        # Task variables
        self.sim_subscriber = None
        self.contact_message = None
        self.object_target_orientation = None  
        self.ee_grasping_orientation = None
        self.ee_placement_orientation = None 
        self.grasp_poses = []
        self.current_grasp_pose = None

        # Counters and timers
        self.grasp_counter = 0
        self.placement_counter = 0
        self.pcd_counter = 0
        self.setup_start_time = None
        self.tf_wait_start_time = None
        
        # State tracking
        self.state = "INIT"  # States: INIT, SETUP, GRASP, PLACE, REPLAY, RESET
        
        # Object state tracking
        self.object_grasped = False
        self.object_collision = False
        self.grasping_failure = False
        
        # Data logging state
        self.logging_active = False
        self.data_recorded = False

    def start(self):
        # Initialize ROS2 node
        rclpy.init()
        self.sim_subscriber = SimSubscriber()

        # Create an executor
        executor = SingleThreadedExecutor()
        executor.add_node(self.sim_subscriber)

        # Simulation Environment Setup
        self.world: World = World(stage_units_in_meters=1.0)
        self.data_logger = self.world.get_data_logger()
        self.task = YcbTask(set_camera=True)
        self.world.add_task(self.task)
        self.world.reset()
        
        # Robot and planner setup
        self.robot: Franka = self.world.scene.get_object(self.task.get_params()["robot_name"]["value"])
        self.controller = self.robot.get_articulation_controller()
        self.planner = YcbPlanner(
            name="ycb_planner",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
        )
        
        # Initially hide the robot
        self.robot.prim.GetAttribute("visibility").Set("invisible")

        # Robot movement setup
        self.ee_grasping_orientation = np.array([0, np.pi, 0])
        self.ee_placement_orientation = np.array([0, np.pi, 0])

        # TF setup
        tf_graph_generation()
        
        # Update state to begin setup
        self.state = "SETUP"
        
    def reset(self):
        """Reset the simulation environment"""
        print("Resetting simulation environment, robot invisible")
        self.world.reset()
        self.planner.reset()
        self.task.object_init(False)
        
        # Reset state flags
        self.state = "SETUP"
        self.robot.prim.GetAttribute("visibility").Set("invisible")
        self.logging_active = False
        self.data_recorded = False
        
        # Keep counters as they are - they need to be handled by the main loop
        
    def setup_environment(self):
        """Handle environment setup phase"""
        # Initialize setup timer if needed
        if not self.setup_start_time:
            self.setup_start_time = time.time()
            return False
            
        # Wait a moment before proceeding with setup
        if time.time() - self.setup_start_time < 1:
            return False
            
        # Initialize cameras and object
        self.task.object_pose_finalization()
        self.cameras = self.task.set_camera(self.task.get_params()["object_current_position"]["value"])
        start_cameras(self.cameras)
        
        self.tf_wait_start_time = time.time()
        print("Cameras started, waiting for point clouds and TF data...")
        
        # Reset setup timer
        self.setup_start_time = None
        return True
        
    def wait_for_tf_buffer(self):
        """Wait for TF buffer to be populated"""
        if self.sim_subscriber.latest_tf is None:
            return False
        
        # Ensure we wait at least 2 seconds for TF buffer
        current_time = time.time()
        if current_time - self.tf_wait_start_time < 2.0:
            return False
            
        try:
            camera1_exists = self.sim_subscriber.check_transform_exists("world", "camera_side1")
            camera2_exists = self.sim_subscriber.check_transform_exists("world", "camera_side2")
            camera3_exists = self.sim_subscriber.check_transform_exists("world", "camera_top")
            
            if camera1_exists and camera2_exists and camera3_exists:
                return True
        except Exception:
            pass
            
        return False
        
    def wait_for_point_clouds(self, path):
        """Wait for and process point clouds from all cameras"""
        pcds = self.sim_subscriber.get_latest_pcds()
        if pcds["pcd1"] is None or pcds["pcd2"] is None or pcds["pcd3"] is None:
            return False
            
        print("All point clouds received!")
        pcd_path = path + f"Pcd_{self.pcd_counter}/pointcloud.pcd"
        raw_pcd, processed_pcd = merge_and_save_pointclouds(pcds, self.sim_subscriber.buffer, pcd_path)
        
        if raw_pcd is not None and processed_pcd is not None:
            print(f"Raw point cloud saved to: {pcd_path}")
            self.pcd_counter += 1
            self.grasp_poses = obtain_grasps(raw_pcd, 12346)
            self.current_grasp_pose = self.grasp_poses.pop(0)
            draw_frame(self.current_grasp_pose[0], self.current_grasp_pose[1])
            self.grasp_counter = 1
        
        # Make the robot visible again after setup
        self.robot.prim.GetAttribute("visibility").Set("inherited")
        print("Setup finished, robot visible again")
        return True
        
    def check_grasp_success(self):
        """Check if the grasp was successful"""
        if self.planner.get_current_event() == 4 and self.placement_counter <= 1:
            # Gripper is fully closed but didn't grasp the object
            if np.floor(self.robot._gripper.get_joint_positions()[0] * 100) == 0:
                self.grasping_failure = True
                return False
        return True