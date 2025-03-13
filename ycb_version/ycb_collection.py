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
import datetime 
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

from simulation.task import YcbTask
from simulation.planner import YcbPlanner
from simulation.logger import YcbLogger
from vision import *
from helper import *

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")

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
        
        # Counters and timers
        self.grasp_counter = 0
        self.placement_counter = 0
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
        self.world.reset()
        self.planner.reset()
        self.task.object_init(False)
        
        # Reset state flags
        self.state = "SETUP"
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
            camera1_exists = self.sim_subscriber.check_transform_exists("world", "camera_1")
            camera2_exists = self.sim_subscriber.check_transform_exists("world", "camera_2")
            camera3_exists = self.sim_subscriber.check_transform_exists("world", "camera_3")
            
            if camera1_exists and camera2_exists and camera3_exists:
                return True
        except Exception:
            pass
            
        return False
        
    def wait_for_point_clouds(self):
        """Wait for and process point clouds from all cameras"""
        pcds = self.sim_subscriber.get_latest_pcds()
        if pcds["pcd1"] is None or pcds["pcd2"] is None or pcds["pcd3"] is None:
            return False
            
        print("All point clouds received!")
        raw_pcd, processed_pcd = merge_and_save_pointclouds(pcds, self.sim_subscriber.buffer)
        
        if raw_pcd is not None and processed_pcd is not None:
            print("Point clouds merged and processed successfully!")
            print("Raw point cloud saved to: merged_pointcloud_raw.pcd")
            print("Processed point cloud saved to: merged_pointcloud.pcd")
        
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

def main():
    env = YcbCollection()
    env.start()
    logger = YcbLogger(env, DIR_PATH)

    setup_phase = 0  # Track which setup step we're on

    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            if env.placement_counter <= 1:
                env.reset()
                continue
            elif env.placement_counter > 1 and env.placement_counter < 200:
                env.reset()
                env.state = "REPLAY"
                continue
        
        # Process ROS callbacks
        rclpy.spin_once(env.sim_subscriber, timeout_sec=0)
        
        # Wait for initial TF data
        if env.sim_subscriber.latest_tf is None and env.state != "INIT":
            continue
            
        # Check if simulation is stopped unexpectedly
        if env.world.is_stopped():
            print("Simulation stopped unexpectedly, resetting...")
            env.reset()
            
            # Reset counters for new grasp attempt
            if env.state != "REPLAY" or env.placement_counter >= 200:
                env.grasp_counter += 1
                env.placement_counter = 0
            continue
            
        # Skip if simulation is paused
        if not env.world.is_playing():
            continue
            
        # State machine for simulation flow
        if env.state == "SETUP":
            # Handle the different phases of setup sequentially
            if setup_phase == 0:
                # Environment and camera setup
                if env.setup_environment():
                    setup_phase = 1
                    
            elif setup_phase == 1:
                # Wait for TF buffer
                if env.wait_for_tf_buffer():
                    setup_phase = 2
                    
            elif setup_phase == 2:
                # Wait for point clouds
                if env.wait_for_point_clouds():
                    setup_phase = 0  # Reset for next time
                    
                    # Move to grasp state if we're just starting
                    if env.placement_counter == 0:
                        env.state = "GRASP"
                    else:
                        env.state = "REPLAY"
                        
        elif env.state == "GRASP":
            # Start logging for the grasp attempt
            if not env.logging_active:
                logger.log_grasping()
                env.logging_active = True
                
            # Perform grasping
            try:
                observations = env.world.get_observations()
                task_params = env.task.get_params()
                
                # For first attempts, use same orientation for grasp and place
                orientation = env.ee_grasping_orientation
                
                picking_position = observations[task_params["object_name"]["value"]]["object_current_position"]
                picking_position[2] += 0.1
                
                # Generate actions for the robot
                actions = env.planner.forward(
                    picking_position=picking_position,
                    placing_position=observations[task_params["object_name"]["value"]]["object_target_position"],
                    current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=orientation,  
                    grasping_orientation=orientation,
                )
                
                # Apply the actions to the robot
                env.controller.apply_action(actions)
                
                # Check if grasp failed
                if not env.check_grasp_success():
                    print("Grasp failed, resetting environment")
                    logger.record_grasping(DIR_PATH)
                    env.reset()
                    env.grasp_counter += 1  # Increment grasp counter for new attempt
                    continue
                
                # Check current planner event phase
                current_event = env.planner.get_current_event()
                
                # Phases 0-3 are the approach and grasp, Phase 4 is lifting the object
                if current_event < 4:
                    # Still in grasping phase, continue
                    pass
                elif current_event >= 4 and current_event < 10:
                    # Object has been grasped and is being lifted (phase 4)
                    # or we're already in placement phases (5-9)
                    # Let the planner continue through all phases to complete the entire operation
                    if current_event == 4 and env.object_grasped == False:
                        print(f"----------------- Object grasped, continuing to placement (event {current_event}) -----------------")
                        env.object_grasped = True
                    elif current_event > 4 and current_event < 9:
                        # Print progress through placement phases
                        print(f"Placement in progress - event {current_event}/9")
                        
                    # Check if we've completed all phases of the planner
                    if env.planner.is_done():
                        print("----------------- First grasp and placement complete -----------------")
                        # Record the successful trajectory
                        logger.record_grasping(DIR_PATH)
                        env.data_recorded = True
                        
                        # Increment the placement counter and move to replay state
                        env.placement_counter += 1
                        
                        # Reset for next placement
                        env.reset()
                        env.state = "REPLAY"
            
            except Exception as e:
                print(f"Error during grasping: {e}")
                env.reset()
                continue
                
        elif env.state == "REPLAY":
            # Start the replay of the grasping trajectory
            print(f"Replaying grasp {env.grasp_counter}, placement {env.placement_counter}")
            logger.replay_grasping()
            
            # Move to place state - the replay will continue in the background
            # and the planner will pick up after the replay is done
            env.state = "PLACE"
            env.logging_active = False
            
        elif env.state == "PLACE":
            # Start logging for the placement if not already logging
            if not env.logging_active:
                logger.log_grasping()
                env.logging_active = True
            
            # Wait for replay to finish if planner is not at event 4 (post-grasp)
            # Event 4 means the object has been grasped and lifted
            if env.planner.get_current_event() < 4:
                continue
                
            # Replay is finished, proceed with placement
            try:
                observations = env.world.get_observations()
                task_params = env.task.get_params()
                
                # Use orientation for placement
                picking_position = observations[task_params["object_name"]["value"]]["object_current_position"]
                picking_position[2] += 0.1
                
                # Generate actions for placement
                actions = env.planner.forward(
                    picking_position=picking_position,
                    placing_position=observations[task_params["object_name"]["value"]]["object_target_position"],
                    current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=env.ee_placement_orientation,  
                    grasping_orientation=env.ee_grasping_orientation,
                )
                
                # Apply the actions to the robot
                env.controller.apply_action(actions)
                
                # Track the current planner event
                current_event = env.planner.get_current_event()
                
                # Add progress tracking
                if current_event > 4 and current_event < 9:
                    # Print progress through placement phases
                    print(f"Placement in progress - event {current_event}/9")
                
                # Check if the entire planning sequence is complete
                if env.planner.is_done():
                    print(f"----------------- Placement {env.placement_counter} complete ----------------- \n\n")
                    # Record the placement data
                    logger.record_grasping(DIR_PATH)
                    env.data_recorded = True
                    
                    # Reset object state for next iteration
                    env.object_grasped = False
                    
                    # Increment placement counter
                    env.placement_counter += 1
                    
                    # Check if we've reached the maximum placements
                    if env.placement_counter >= 200:
                        print(f"Maximum placements reached for grasp {env.grasp_counter}")
                        env.grasp_counter += 1
                        env.placement_counter = 0
                        env.reset()
                        env.state = "SETUP"
                    else:
                        # Reset for next placement with the same grasp
                        env.reset()
                        env.state = "REPLAY"
                
            except Exception as e:
                print(f"Error during placement: {e}")
                env.reset()
                env.state = "REPLAY"
                continue
                
    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    main()