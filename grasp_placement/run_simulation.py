# import carb
from isaacsim import SimulationApp

CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

import carb
import omni
import math
import numpy as np
from omni.isaac.core import SimulationContext, World
from omni.isaac.core.utils import stage, extensions
from omni.isaac.franka import KinematicsSolver
from omni.isaac.franka import Franka
from omni.isaac.franka.tasks import PickPlace
from data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
# from graph_initialization import joint_graph_generation, gripper_graph_generation
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

GRIPPER_MAX = 0.04
GRIPPER_SPEED = 0.005

class StartSimulation:

    def __init__(self):
        self.world = None
        self.cube = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.task_params = None
        self._gripper = None

    def start(self):
        # Set up the world
        self.world = World(stage_units_in_meters=1.0)
        self.task = PickPlace()
        self.world.add_task(self.task)
        self.world.reset()

        self.task_params = self.task.get_params()
        self.robot =  self.world.scene.get_object(self.task_params["robot_name"]["value"])
        
        # Set up the robot
        self.controller = DataCollectionController(
            name = "data_collection_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )
        self.articulation_controller = self.robot.get_articulation_controller() 



    

def main():

    env = StartSimulation()
    env.start()
    reset_needed = False
    while simulation_app.is_running():
        env.world.step(render=True)
        if env.world.is_stopped() and not reset_needed:
            reset_needed = True
        if env.world.is_playing():
            if reset_needed:
                env.world.reset()
                reset_needed = False
            observations = env.world.get_observations()
            actions = env.controller.forward(
                picking_position=observations[env.task_params["cube_name"]["value"]]["position"],
                placing_position=observations[env.task_params["cube_name"]["value"]]["target_position"],
                current_joint_positions=observations[env.task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=np.array([0, 0.005, 0]),
                end_effector_orientation_offset=np.array([0, -np.pi/4, 0]),
                end_effector_orientation=np.array([0, np.pi, 0]),
            )
            
            
            env.articulation_controller.apply_action(actions)
        
    simulation_app.close()

if __name__ == "__main__":
    main()
















