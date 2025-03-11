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
import numpy as np
import asyncio
import json
import os
import glob
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
import time
import re
from functools import partial
from typing import List
from rclpy.executors import SingleThreadedExecutor

from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.sensor import ContactSensor

from task import YcbTask
from planner import YcbPlanner
from vision import *
from helper import *

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

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
        self.setup_start_time = None

        # Condition variables 
        self.enable_pcd = True
        self.object_collision = False
        self.setup_finished = False
        self.object_grasped = False
        self.waiting_for_pcds = False
        self.cameras_started = False
        self.tf_buffer_ready = False
        self.tf_wait_start_time = None

    def start(self):
        # Initialize ROS2 node
        rclpy.init()
        self.sim_subscriber = SimSubscriber()

        # Create an executor (SingleThreadedExecutor is the simplest choice)
        executor = SingleThreadedExecutor()
        executor.add_node(self.sim_subscriber)

        # Simulation Environment Setup
        self.world: World = World(stage_units_in_meters=1.0)
        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default
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

        # Robot movment setup
        self.ee_grasping_orientation = np.array([0, np.pi, 0])
        self.ee_placement_orientation = np.array([0, np.pi, 0])

        # TF setup
        tf_graph_generation()
        
    def reset(self):
        self.world.reset()
        self.planner.reset()
        self.task.object_init(False)
        self.setup_finished = False

def main():
    env = YcbCollection()
    env.start()
    
    # One grasp corresponding to many placements
    reset_needed = False        # Used when grasp is done 
    # pcd_finished = False

    while simulation_app.is_running():
        # Process any pending ROS callbacks
        rclpy.spin_once(env.sim_subscriber, timeout_sec=0)
        
        env.world.step(render=True)
        if env.sim_subscriber.latest_tf is None:
            continue

        if env.world.is_stopped() and not reset_needed:
            reset_needed = True
        if env.world.is_playing():
            if reset_needed:
                env.world.reset()
                reset_needed = False

            if not env.setup_finished:
                
                if not env.setup_start_time:
                    env.setup_start_time = time.time()

                if time.time() - env.setup_start_time < 1:
                    # Skip sending robot commands, letting other parts of the simulation continue.
                    continue
                else:
                    # Initialize setup sequence if not already started
                    if not env.cameras_started:
                        env.task.object_pose_finalization()
                        # Set camera to the object
                        env.cameras = env.task.set_camera(env.task.get_params()["object_current_position"]["value"])
                        start_cameras(env.cameras)
                        
                        # Make the robot visible again after setup
                        env.robot.prim.GetAttribute("visibility").Set("inherited")
                        env.cameras_started = True
                        env.waiting_for_pcds = True
                        env.tf_wait_start_time = time.time()
                        print("Cameras started, waiting for point clouds and TF data...")
                        continue

                    # First ensure TF buffer is populated
                    if not env.tf_buffer_ready:
                        # Wait for TF buffer to be populated (give it a few seconds)
                        if env.sim_subscriber.latest_tf is not None:
                            current_time = time.time()
                            # Wait at least 2 seconds for TF buffer to be populated
                            if current_time - env.tf_wait_start_time > 2.0:
                                # Check if the required transforms exist
                                try:
                                    # Print available frames in TF buffer for debugging
                                    print(f"Available frames in TF buffer: {env.sim_subscriber.buffer.all_frames_as_string()}")
                                    
                                    # Check if the required transforms exist
                                    camera1_exists = env.sim_subscriber.check_transform_exists("world", "camera_1")
                                    camera2_exists = env.sim_subscriber.check_transform_exists("world", "camera_2")
                                    camera3_exists = env.sim_subscriber.check_transform_exists("world", "camera_3")
                                    
                                    if camera1_exists and camera2_exists and camera3_exists:
                                        env.tf_buffer_ready = True
                                        print("TF buffer is now populated with required transforms")
                                    else:
                                        print("Not all required transforms are available yet")
                                        continue
                                except Exception as e:
                                    print(f"Your tf info: {env.sim_subscriber.latest_tf}")
                                    print(f"Still waiting for TF data: {e}")
                                    # Continue to next iteration to process more TF messages
                                    continue
                            else:
                                print(f"Waiting for TF buffer to be populated ({current_time - env.tf_wait_start_time:.1f}/20.0 seconds)")
                                continue
                        else:
                            print("Waiting for initial TF message...")
                            continue

                    # Check if we're waiting for point clouds
                    if env.waiting_for_pcds:
                        pcds = env.sim_subscriber.get_latest_pcds()
                        if pcds["pcd1"] is not None and pcds["pcd2"] is not None and pcds["pcd3"] is not None:
                            print("All point clouds received!")
                            # Print TF buffer frames for debugging
                            print(f"Available frames in TF buffer: {env.sim_subscriber.buffer.all_frames_as_string()}")
                            merged_pcd = merge_and_save_pointclouds(pcds, env.sim_subscriber.buffer)
                            print("Merged point cloud saved successfully")
                            env.waiting_for_pcds = False
                            env.setup_finished = True
                            env.setup_start_time = None
                            print("Setup finished")
                        else:
                            print(f"Still waiting for point clouds: {pcds}")
                            continue

            observations = env.world.get_observations()
            task_params = env.task.get_params()
            picking_position = observations[task_params["object_name"]["value"]]["object_current_position"]
            picking_position[2] += 0.1

            actions = env.planner.forward(
                picking_position=picking_position,
                placing_position=observations[task_params["object_name"]["value"]]["object_target_position"],
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=None,
                placement_orientation=env.ee_placement_orientation,  
                grasping_orientation=env.ee_grasping_orientation,
            )

            # env.controller.apply_action(actions)
        
        if env.planner.is_done():
            print("----------------- done picking and placing ----------------- \n\n")
            reset_needed = True
            env.reset()
    simulation_app.close()

if __name__ == "__main__":
    main()