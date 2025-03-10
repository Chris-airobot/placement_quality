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
        env.world.step(render=True)
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

                    env.task.object_pose_finalization()
                    # Set camera to the object
                    env.cameras = env.task.set_camera(env.task.get_params()["object_current_position"]["value"])
                    start_cameras(env.cameras)
                    
                    # Make the robot visible again after setup
                    # env.robot.prim.GetAttribute("visibility").Set("inherited")
                    
                    # Set up finished       
                    env.setup_finished = True
                    env.setup_start_time = None
                    print(f"set up finished")

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