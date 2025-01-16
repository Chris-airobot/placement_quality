# import carb
from isaacsim import SimulationApp

CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

import carb
import omni
import math
import numpy as np
import asyncio
import json
import os
from omni.isaac.core import World
from omni.isaac.core.utils import stage, extensions
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.franka.tasks import PickPlace
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
# from graph_initialization import joint_graph_generation, gripper_graph_generation
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

GRIPPER_MAX = 0.04
GRIPPER_SPEED = 0.005
GRASP_PATH = "/home/chris/Chris/placement_ws/src/grasp_placement/grasp_placement/data/picking.json"
DIR_PATH = "/home/chris/Chris/placement_ws/src/grasp_placement/data/"

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
        self.placement_orientation = None
        
        self.data_logger = None

        self.grasp_counter = 0
        self.placement_counter = 0

        self.replay_finished = True

        self.grasping_failure = False
        self.placement_failure = False


    def start(self):

        # Orientations creation
        self.grasping_orientation = orientation_creation()

        # Set up the world
        self.world: World = World(stage_units_in_meters=1.0)
        self.task = PickPlace()
        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default

        self.world.add_task(self.task)
        self.world.reset()

        self.task_params = self.task.get_params()
        self.robot: Franka =  self.world.scene.get_object(self.task_params["robot_name"]["value"])
        
        # Set up the robot
        self.controller = DataCollectionController(
            name = "data_collection_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )
        self.articulation_controller = self.robot.get_articulation_controller() 

    def _on_logging_event(self, val):
        print(f"----------------- Grasping {self.grasp_counter} Placement {self.placement_counter} Start -----------------")
        print(f'Cube position is: {self.task.get_params()["cube_position"]["value"]}')
        print(f'Cube orientation is: {self.task.get_params()["cube_orientation"]["value"]}\n')
        if not self.world.get_data_logger().is_started():
            robot_name = self.task_params["robot_name"]["value"]
            cube_name = self.task_params["cube_name"]["value"]
 
            # A data logging function is called at every time step index if the data logger is started already.
            # We define the function here. The tasks and scene are passed to this function when called.

            def frame_logging_func(tasks, scene: Scene):
                cube_position, cube_orientation =  scene.get_object(cube_name).get_world_pose()
                ee_position, ee_orientation =  scene.get_object(robot_name).end_effector.get_world_pose()
                surface = surface_detection(quat_to_euler_angles(ee_orientation))

                return {
                    "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),# save data as lists since its a json file.
                    "applied_joint_positions": scene.get_object(robot_name).get_applied_action().joint_positions.tolist(),
                    "ee_position": ee_position.tolist(),
                    "ee_orientation": ee_orientation.tolist(),
                    "cube_position": cube_position.tolist(),
                    "cube_orientation": cube_orientation.tolist(),
                    "stage": self.controller.get_current_event(),
                    "surface": surface,
                    "grasp_failure": self.grasping_failure,
                    "placement_failure": self.placement_failure
                }

            self.data_logger.add_data_frame_logging_func(frame_logging_func) # adds the function to be called at each physics time step.
        if val:
            self.data_logger.start() # starts the data logging
        else:
            self.data_logger.pause()
        return



    def _on_save_data_event(self, log_path=GRASP_PATH):
        print("----------------- Saving Start -----------------\n")
        self.data_logger.save(log_path=log_path) # Saves the collected data to the json file specified.
        print(f"----------------- Successfully saved it to {log_path} -----------------\n")
        self.data_logger.reset() # Resets the DataLogger internal state so that another set of data can be collected and saved separately.
        return


    # This is for replying the whole scene
    async def _on_replay_scene_event_async(self, data_file=GRASP_PATH):
            self.data_logger.load(log_path=data_file)
            await self.world.play_async()
            self.world.add_physics_callback("replay_scene", self._on_replay_scene_step)
            return 


    def _on_replay_scene_step(self, step_size):

        if self.world.current_time_step_index < self.data_logger.get_num_of_data_frames():
            cube_name = self.task_params["cube_name"]["value"]
            data_frame = self.data_logger.get_data_frame(data_frame_index=self.world.current_time_step_index)
            self.articulation_controller.apply_action(
                ArticulationAction(joint_positions=data_frame.data["applied_joint_positions"])
            )
            # Sets the world position of the goal cube to the same recoded position
            self.world.scene.get_object(cube_name).set_world_pose(
                position=np.array(data_frame.data["cube_position"]),
                orientation=np.array(data_frame.data["cube_orientation"])
            )

        elif self.world.current_time_step_index == self.data_logger.get_num_of_data_frames():
            print("----------------- Replay Finished, now moving to Placement Phase -----------------\n")
            self.replay_finished = True
            self.controller._event = 4
            self.world.clear_physics_callbacks()
        
        
        return

    def reset(self):
        self.world.reset()
        self.controller.reset()
        self.placement_failure = False
        self.grasping_failure = False

        # Generate x and y as random values between -π and π
        x, y = np.random.uniform(low=-0.3, high=0.3, size=2)
        p, q = np.random.uniform(low=-np.pi, high=np.pi, size=2)
        # Create the cube position with z fixed at 0
        self.task.set_params(
            cube_position=np.array([x, y, 0]),
            target_position=np.array([p, q, 0])
        )
        self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)


def log_grasping(start_logging, env: StartSimulation):
    # Logging sections
    if start_logging:
        env._on_logging_event(True)
        start_logging = False

    return start_logging


def record_grasping(recorded, env: StartSimulation):
    # Recording section
    if not recorded:
        file_path = DIR_PATH + f"Grasping_{env.grasp_counter}/placement_{env.placement_counter}.json"

        # Ensure the parent directories exist
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)

        # Check if the file exists; if not, create it
        if not os.path.exists(file_path):
            # Open the file in write mode and create an empty JSON structure
            with open(file_path, 'w') as file:
                json.dump({}, file)

        env._on_save_data_event(file_path)
        recorded = True
    return recorded


def replay_grasping(env: StartSimulation):
    print(f"----------------- Replaying Grasping {env.grasp_counter} ----------------- \n")

    file_path = DIR_PATH + f"Grasping_{env.grasp_counter}/grasping.json"

    # If the replay data does not exist, create one
    if not os.path.exists(file_path):
        extract_grasping(DIR_PATH + f"Grasping_{env.grasp_counter}/placement_{env.placement_counter}.json")

    asyncio.ensure_future(env._on_replay_scene_event_async(file_path))
    return True


def main():

    env = StartSimulation()
    env.start()

    # One grasp corresponding to many placements
    reset_needed = False # Used when grasp is done 
    start_logging = True # Used for start logging
    recorded = False     # Used to check if the data has been recorded
    replay = False       # Used for replay data     
    placement_finished = False     # Use when placement is done

    while simulation_app.is_running():
        env.world.step(render=True)
        if env.world.is_playing():
            # The grasp is done, no more placement 
            if reset_needed:
                env.reset()
                reset_needed = False
                start_logging = True
                recorded = False
                replay = False
                placement_finished = False
                env.grasp_counter += 1
                env.placement_counter = 0
                
            
            # Replaying Session
            if replay:
                # This function should only be played once
                env.replay_finished = False
                replay_grasping(env)
                env.placement_counter += 1
                replay = False
                start_logging = True
                recorded = False
                env.placement_failure = False

            # One placement is done
            elif placement_finished:
                env.world.reset()
                env.controller.reset()
                replay = True
                placement_finished = False
                env.placement_failure = False
                
            elif env.replay_finished:

                

                # Recording Session
                start_logging = log_grasping(start_logging, env)

                observations = env.world.get_observations()
                actions = env.controller.forward(
                    picking_position=observations[env.task_params["cube_name"]["value"]]["position"],
                    placing_position=observations[env.task_params["cube_name"]["value"]]["target_position"],
                    current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                    end_effector_offset=np.array([0, 0.005, 0]),
                    placement_orientation=env.placement_orientation,  
                    grasping_orientation=env.grasping_orientation[env.grasp_counter],           
                    # grasping_orientation=np.array([0, np.pi, 0]),           
                    # placement_orientation=np.array([0, np.pi, 0]),  
                )

                # Gripper release, but could not place the object into the ground
                if env.controller.get_current_event() == 7:
                    position, _ = env.world.scene.get_object(env.task_params["cube_name"]["value"]).get_world_pose()
                    if position[2] != 0:
                        env.placement_failure = True

                # Gripper fully closed, did not grasp the object
                if env.controller.get_current_event() == 4:
                    if np.floor(env.robot._gripper.get_joint_positions()[0] * 100) == 0 : 
                        env.grasping_failure = True
                        reset_needed = True
                        recorded = True

                env.articulation_controller.apply_action(actions)

            if env.controller.is_done():
                print("----------------- done picking and placing ----------------- \n\n")
                placement_finished = True
                recorded = record_grasping(recorded, env)

            
    simulation_app.close()

if __name__ == "__main__":
    main()
    
    # Global upward direction
    
















