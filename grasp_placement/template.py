from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World

from omni.isaac.core.utils import extensions
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from utilies.camera_utility import *
from omni.isaac.dynamic_control import _dynamic_control
from carb import Float3
from omni.physx import get_physx_interface
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

# def physics_update(dt: float):
#     physxIFace = get_physx_interface()
#     physxIFace.apply_force_at_pos(prims[-1], Float3([0,0,1000]),Float3([0,0,0]))
#     physxIFace.apply_force_at_pos(prims[-1], Float3([1000,0,0]),Float3([0,100,0]))
#     physxIFace.apply_force_at_pos(prims[-1], Float3([-1000,0,0]),Float3([0,-100,0]))

my_world: World = World(stage_units_in_meters=1.0)
my_task = PickPlaceCamera(set_camera=False)
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_franka: Franka = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = DataCollectionController(
    name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()
# Acquire the dynamic control interface
dc = _dynamic_control.acquire_dynamic_control_interface()
robot_prim_path = "/World/Franka"    
robot_art = dc.get_articulation(robot_prim_path)

# 2) Find the end-effector body within the articulation
ee_body_name = "panda_hand"  # <-- The link name in your URDF / USD
ee_body_handle = dc.find_articulation_body(robot_art, ee_body_name)
if ee_body_handle == _dynamic_control.INVALID_HANDLE:
    raise RuntimeError(f"Could not find end-effector body: {ee_body_name}")



reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        force_vec = Float3(50, 50, 50)
        # Apply the force at the body's center of mass (zero offset).
        position_vec = Float3(0.0, 0.0, 0.0)
        dc.apply_body_force(ee_body_handle, force_vec, position_vec, False)

        articulation_controller.apply_action(actions)
simulation_app.close()