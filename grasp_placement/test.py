# import carb
from isaacsim import SimulationApp
CONFIG = {"headless": True}
simulation_app = SimulationApp(CONFIG)

import numpy as np
import asyncio
import json
import os
import glob
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from carb import Float3

from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from utilies.camera_utility import *
from omni.isaac.sensor import ContactSensor
from functools import partial
from typing import List
from omni.isaac.dynamic_control import _dynamic_control
import datetime
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()


class TFSubscriber(Node):
    def __init__(self):
        super().__init__("tf_subscriber")
        self.latest_tf = None  # Store the latest TFMessage here
        self.latest_pcd = None # Store the latest Pointcloud here

        # Create the subscription
        self.tf_subscription = self.create_subscription(
            TFMessage,           # Message type
            "/tf",               # Topic name
            self.tf_callback,    # Callback function
            10                   # QoS
        )



    def tf_callback(self, msg):
        # This callback is triggered for every new TF message on /tf
        self.latest_tf = msg

class StartSimulation:
    def __init__(self):
        self.world = None
        self.cube = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.task_params = None
        self.grasping_orientation = None
        self.placement_orientation = None  # Gripper orientation when the cube is about to be placed
        self.camera = None
        self.contact = None
        self.cube_grasped = None
        self.contact_sensors = None
        # self.dc = None
        # self.ee_body_handle = None

        
        self.data_logger = None

        self.grasp_counter = 0
        self.placement_counter = 0

        self.replay_finished = True
        self.grasping_failure = False
        self.cube_contacted = False


    def start(self):

        # Orientations creation
        self.grasping_orientation = orientation_creation()

        # Set up the world
        self.world: World = World(stage_units_in_meters=1.0)

        ranges = [(-0.3, -0.1), (0.1, 0.3)]
        range_choice = ranges[np.random.choice(len(ranges))]
        
        # Generate x and y as random values between -π and π
        x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
        p, q = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])


        self.task = PickPlaceCamera(set_camera=False)
        

        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default

        self.world.add_task(self.task)

        self.world.reset()

        panda_prim_names = [
            "Franka/panda_link0",
            "Franka/panda_link1",
            "Franka/panda_link2",
            "Franka/panda_link3",
            "Franka/panda_link4",
            "Franka/panda_link5",
            "Franka/panda_link6",
            "Franka/panda_link7",
            "Franka/panda_link8",
            "Franka/panda_hand",
            "Cube"
        ]

        self.contact_sensors = []
        for i, link_name in enumerate(panda_prim_names):
            sensor: ContactSensor = self.world.scene.add(
                ContactSensor(
                    prim_path=f"/World/{link_name}/contact_sensor",
                    name=f"contact_sensor_{i}",
                    min_threshold=0.0,
                    max_threshold=1e7,
                    radius=0.1,
                )
            )
            # Use raw contact data if desired
            sensor.add_raw_contact_data_to_frame()
            self.contact_sensors.append(sensor)

        self.world.reset()

        # Contact report for links
        self.world.add_physics_callback("contact_sensor_callback", partial(self.on_sensor_contact_report, sensors=self.contact_sensors))

        self.task_params = self.task.get_params()
        self.robot: Franka =  self.world.scene.get_object(self.task_params["robot_name"]["value"])
        
        # Set up the robot
        self.controller = DataCollectionController(
            name = "data_collection_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )
        self.articulation_controller = self.robot.get_articulation_controller() 

        self.task.set_params(
            cube_position=np.array([x, y, 0]),
            target_position=np.array([p, q, 0.075])
        )

        self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        
        # external force set up
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        robot_prim_path = "/World/Franka"    
        robot_art = self.dc.get_articulation(robot_prim_path)
        ee_body_name = "panda_leftfinger"  # <-- The link name in your URDF / USD
        self.ee_body_handle = self.dc.find_articulation_body(robot_art, ee_body_name)


        tf_graph_generation()
        # start_camera(self.task._camera)




    def on_sensor_contact_report(self, dt, sensors: List[ContactSensor]):
        """Physics-step callback: checks all sensors, sets self.contact accordingly."""
        any_contact = False  # track if at least one sensor had contact
        self.cube_contacted = False
        self.cube_grasped = None

        for sensor in sensors:
            frame_data = sensor.get_current_frame()
            if frame_data["in_contact"]:
                # We have contact! Extract the bodies, force, etc.
                for c in frame_data["contacts"]:
                    body0 = c["body0"]
                    body1 = c["body1"]
                    if "panda" in body0 + body1 and "Cube" in body0 + body1:
                        # print("Cube in the ground")
                        self.cube_grasped = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"

                    if "GroundPlane" in body0 + body1 and "Cube" in body0 + body1:
                        # print("Cube in the ground")
                        self.cube_contacted = True
                    elif ("GroundPlane" in body0) or ("GroundPlane" in body1):
                        print("Robot hits the ground, and it will be recorded")
                        any_contact = True
                        self.contact = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"
                        

        # If, after checking all sensors, none had contact, reset self.contact to None
        if not any_contact:
            self.contact = None



    
    def reset(self):
        self.world.reset()
        self.controller.reset()
        self.grasping_failure = False

        ranges = [(-0.3, -0.1), (0.1, 0.3)]
        range_choice = ranges[np.random.choice(len(ranges))]
        
        # Generate x and y as random values between -π and π
        x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
        p, q = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])

        # Create the cube position with z fixed at 0
        self.task.set_params(
            cube_position=np.array([x, y, 0]),
            target_position=np.array([p, q, 0.075])
        )
        print(f"cube_position is :{np.array([x, y, 0])}")
        self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        



def main():
    rclpy.init()
    tf_node = TFSubscriber()
    
    # Create an executor (SingleThreadedExecutor is the simplest choice)
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(tf_node)

    env = StartSimulation()
    env.start()

    # One grasp corresponding to many placements
    reset_needed = False        # Used when grasp is done 
    replay = False              # Used for replay data     
    grasp_collected = False     # Use when the grasping is done
    

    while simulation_app.is_running():
        try:
            env.world.step(render=True)
            
        except:
            print("Something wrong with hitting the floor in step")
            env.reset()
            # env.world.add_physics_callback("contact_sensor_callback", partial(env.on_sensor_contact_report, sensors=env.contact_sensors))
            if env.placement_counter <= 1:
                reset_needed = True
                continue
            elif env.placement_counter > 1 and env.placement_counter < 200:
                # record_grasping(False, env)
                env.reset()
                replay = True
                continue
            
        # 2) Spin ROS for a short time so callbacks are processed
        executor.spin_once(timeout_sec=0.01)
        # Technically only pcd ready should be sufficient because it seems pcd takes longer to be prepared
        if tf_node.latest_tf is None:
            continue

        if env.world.is_playing():
            # The grasp is done, no more placement 
            if reset_needed:
                env.reset()
                reset_needed = False
                replay = False
                grasp_collected = False
                env.grasp_counter += 1
                env.placement_counter = 0
            # Replaying Session
            if replay:
                # This function should only be played once
                env.replay_finished = False
                env.placement_counter += 1
                replay = False
            elif env.replay_finished:
                try:
                    observations = env.world.get_observations()
                except:
                    print("Something wrong with hitting the floor in observation")
                    if env.placement_counter <= 1:
                        reset_needed = True
                        continue
                    elif env.placement_counter > 1 and env.placement_counter < 200:
                        # record_grasping(False, env)
                        env.reset()
                        replay = True
                        continue

                # Use random orientation only after the grasp part trajectory has been collected  
                placement_orientation = env.placement_orientation if grasp_collected else env.grasping_orientation[env.grasp_counter]

                actions = env.controller.forward(
                    picking_position=observations[env.task_params["cube_name"]["value"]]["position"],
                    placing_position=observations[env.task_params["cube_name"]["value"]]["target_position"],
                    current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=None,
                    placement_orientation=placement_orientation,  
                    grasping_orientation=env.grasping_orientation[env.grasp_counter],    
                )
                # if env.controller.get_current_event() > 4 and env.controller.get_current_event() < 7:
                #     position_vec = Float3(np.random.uniform(0, 5, 3))
                #     env.dc.apply_body_force(env.ee_body_handle, force_vec, position_vec, False)

                # Gripper fully closed, did not grasp the object
                if env.controller.get_current_event() == 4 and env.placement_counter <= 1:
                    if np.floor(env.robot._gripper.get_joint_positions()[0] * 100) == 0 : 
                        env.grasping_failure = True
                        reset_needed = True

                env.articulation_controller.apply_action(actions)

            if env.controller.is_done():
                print("----------------- done picking and placing ----------------- \n\n")
                grasp_collected = True

                # Maximum placement has been reached
                if env.placement_counter >= 200:
                    reset_needed = True
                else: 
                    env.reset()
                    replay = True

            
    simulation_app.close()

if __name__ == "__main__":
    main()
    

    


