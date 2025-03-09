from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World

from omni.isaac.core.utils import extensions
from omni.isaac.franka import Franka
from task import YcbTask
import time
from helper import *
# from grasp_placement.utilies.camera_utility import *
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

setup_start_time = None

my_world: World = World(stage_units_in_meters=1.0)
my_task = YcbTask(set_camera=True)
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_franka: Franka = my_world.scene.get_object(task_params["robot_name"]["value"])
# my_controller = DataCollectionController(
#     name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
# )
articulation_controller = my_franka.get_articulation_controller()

my_world.reset()

reset_needed = False
setup_finished = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False


        if not setup_finished:
            if setup_start_time is None:
                setup_start_time = time.time()

            if time.time() - setup_start_time < 1:
                # Skip sending robot commands, letting other parts of the simulation continue.
                continue
            else:
                setup_finished = True
                my_task.object_pose_finalization()
                setup_start_time = None
                print(f"set up finished")
                my_task.set_camera(my_task.get_params()["object_current_position"]["value"])

        observations = my_world.get_observations()
        
simulation_app.close()