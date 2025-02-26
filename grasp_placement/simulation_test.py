# import carb
from isaacsim import SimulationApp
CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

import numpy as np

from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from helper import *
from utilies.camera_utility import *
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.tasks import PickPlace

# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

class TestRobotMovement:
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
        self.camera = None
        self.data_logger = None




    def start(self):
        # Set up the world
        self.world: World = World(stage_units_in_meters=1.0)
        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default

        self.task = PickPlaceCamera(set_camera=False)
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

        # self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        self.grasping_orientation = np.array([0, np.pi, 0])
        self.placement_orientation = np.array([0, np.pi, 0])
        
    

    
    def reset(self):
        self.world.reset()
        self.controller.reset()

        cube_initial_position, cube_initial_orientation, target_position = task_randomization()

        # Create the cube position with z fixed at 0
        self.task.set_params(
            cube_position=cube_initial_position,
            cube_orientation=cube_initial_orientation,
            target_position=target_position
        )
        # print(f"cube_position is :{np.array([x, y, 0])}")
        # self.placement_orientation = np.random.uniform(low=-np.pi, high=np.pi, size=3)
        



def main():
    env = TestRobotMovement()
    env.start()

    # One grasp corresponding to many placements
    reset_needed = False        # Used when grasp is done 
    
    while simulation_app.is_running():
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with hitting the floor in step")
            env.reset()
            continue
        if env.world.is_stopped() and not reset_needed:
            reset_needed = True
        if env.world.is_playing():
            # The grasp is done, no more placement 
            if reset_needed:
                env.reset()
                env.controller.reset()
                reset_needed = False
            try:
                observations = env.world.get_observations()
            except:
                print("Something wrong with hitting the floor in observation")
                env.reset()
                continue

            # Use random orientation only after the grasp part trajectory has been collected  
            placement_orientation = env.placement_orientation 

            actions = env.controller.forward(
                picking_position=observations[env.task_params["cube_name"]["value"]]["position"],
                placing_position=observations[env.task_params["cube_name"]["value"]]["target_position"],
                current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=None,
                placement_orientation=placement_orientation,  
                grasping_orientation=env.grasping_orientation,    
            )


            env.articulation_controller.apply_action(actions)

        if env.controller.is_done():
            print("----------------- done picking and placing ----------------- \n\n")
            reset_needed = True
            env.reset()


            
    simulation_app.close()

if __name__ == "__main__":
    main()
    

    


