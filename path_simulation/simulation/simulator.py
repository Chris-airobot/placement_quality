import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import rclpy
import time

from rclpy.executors import SingleThreadedExecutor

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
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

from path_simulation.simulation.RRT_controller import RRTController
from path_simulation.simulation.RRT_task import RRTTask
from ycb_simulation.utils.helper import tf_graph_generation

class SimSubscriber(Node):
    def __init__(self, buffer_size=100.0, visualization=False, pcd_path=None):
        super().__init__("Sim_subscriber")
        self.latest_tf = None

        self.buffer = Buffer(rclpy.duration.Duration(seconds=buffer_size))
        self.listener = TransformListener(self.buffer, self)

        self.tf_subscription = self.create_subscription(
            TFMessage,
            "/tf",
            self.tf_callback,
            10
        )

        
        if visualization:
            # Add publishers for visualization
            self.pcd_pub = self.create_publisher(PointCloud2, '/visualization/pcd', 10)
            self.grasp_pub = self.create_publisher(MarkerArray, '/visualization/grasp_pose', 10)
        
            # Timer for continuous publishing
            self.viz_timer = self.create_timer(0.1, self.publish_visualization)
            
            # Store raw point cloud
            self.pcd = None
            self.grasp_pose = None


    def tf_callback(self, msg):
        self.latest_tf = msg
        if self.latest_tf is not None and len(self.latest_tf.transforms) > 0:
            if not hasattr(self, 'all_transforms'):
                self.all_transforms = {}
            for transform in self.latest_tf.transforms:
                try:
                    self.buffer.set_transform(transform, "default_authority")
                    self.all_transforms[transform.child_frame_id] = transform
                except Exception:
                    pass
        
    def check_transform_exists(self, target_frame, source_frame):
        try:
            self.buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return True
        except Exception:
            return False
        




class Simulator:
    def __init__(self):
        # Basic variables
        self.world = None
        self.object = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.stage = omni.usd.get_context().get_stage()
        
        # Task variables
        self.sim_subscriber = None
        self.object_target_orientation = None  
        self.ee_grasping_orientation = None
        self.ee_placement_orientation = None 
        self.grasp_poses = []
        self.current_grasp_pose = None

        # Counters and timers
        self.grasp_counter = 0
        self.placement_counter = 0
        self.tf_wait_start_time = None
        
        # State tracking
        self.state = "INIT"  # States: INIT, SETUP, MOVE, REPLAY, RESET
    
        # Data logging state
        self.start_logging = True
        self.data_recorded = False


    def start(self):
        # Initialize ROS2 node
        rclpy.init()
        self.sim_subscriber = SimSubscriber(visualization=True)

        # Create an executor
        executor = SingleThreadedExecutor()
        executor.add_node(self.sim_subscriber)

        # Simulation Environment Setup
        self.world: World = World(stage_units_in_meters=1.0)
        self.task = RRTTask()
        self.world.add_task(self.task)
        self.world.reset()
        
        # Robot and planner setup
        self.robot: Franka = self.world.scene.get_object(self.task.get_params()["robot_name"]["value"])
        self.articulation_controller = self.robot.get_articulation_controller()
        self.controller = RRTController(
            name="RRT_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot,
        )

        # Robot movement setup
        euler_angles = [random.uniform(0, 360) for _ in range(3)]
        self.ee_placement_orientation = R.from_euler('xyz', euler_angles, degrees=True).as_quat()

        # TF setup
        tf_graph_generation()
        
        # Update state to begin setup
        self.state = "SETUP"


    def reset(self):
        """Reset the simulation environment"""
        print("Resetting simulation environment, robot invisible")
        self.world.reset()
        self.controller.reset()
        self.task.object_init(False)
        
        # Reset state flags
        self.state = "SETUP"
        self.start_logging = True
        self.data_recorded = False

    def setup(self):
        """Handle environment setup phase"""
        # Initialize setup timer if needed
        if not self.setup_start_time:
            self.setup_start_time = time.time()
            return False
            
        # Wait a moment before proceeding with setup
        if time.time() - self.setup_start_time < 1:
            return False
            
        # Initialize cameras and object
        self.task.init()
        
        self.tf_wait_start_time = time.time()
        print("Cameras started, waiting for TF data...")
        
        # Reset setup timer
        self.setup_start_time = None
        return True

    def wait_for_tf_buffer(self):
        """Wait for TF buffer to be populated"""
        if self.sim_subscriber.check_transform_exists("world", "Ycb_object"):
            return True
        else:
            return False

            
        
            