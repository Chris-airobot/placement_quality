from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import numpy as np
from tasks.pick_place_task import PickPlace
from omni.isaac.manipulators import SingleManipulator
from controllers.pick_place_controller import PickPlaceController
my_world: World = World(stage_units_in_meters=1.0)


target_position = np.array([-0.3, 0.6, 0])
target_position[2] = 0.0515 / 2.0
my_task = PickPlace(name="xarm_pick_place", target_position=target_position)
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("xarm_pick_place").get_params()
xarm_name = task_params["robot_name"]["value"]
my_xarm: SingleManipulator = my_world.scene.get_object(xarm_name)
#initialize the controller
my_controller = PickPlaceController(name="controller", robot_articulation=my_xarm, gripper=my_xarm.gripper)
task_params = my_world.get_task("xarm_pick_place").get_params()
articulation_controller = my_xarm.get_articulation_controller()


while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()   
        observations = my_world.get_observations()
        print(f"You are in stage: {my_controller._event}")
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            # This offset needs tuning as well
            end_effector_offset=np.array([0, 0, 0.25]),
        )

        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
simulation_app.close()