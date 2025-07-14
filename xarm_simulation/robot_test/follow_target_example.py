from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from tasks.follow_target import FollowTarget
import numpy as np
from ik_solver import KinematicsSolver
# from controllers.rmpflow import RMPFlowController
from omni.isaac.manipulators import SingleManipulator
import carb

my_world: World = World(stage_units_in_meters=1.0)
#Initialize the Follow Target task with a target location for the cube to be followed by the end effector
my_task = FollowTarget(name="xarm_follow_target", target_position=np.array([0.3989998996257782, 0, 0.29350119829177856]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("xarm_follow_target").get_params()
target_name = task_params["target_name"]["value"]
xarm_name = task_params["robot_name"]["value"]
my_xarm: SingleManipulator = my_world.scene.get_object(xarm_name)
#initialize the controller
my_controller = KinematicsSolver(my_xarm)
articulation_controller = my_xarm.get_articulation_controller()
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
        observations = my_world.get_observations()
        actions, succ = my_controller.compute_inverse_kinematics(
            target_position=observations[target_name]["position"],
            target_orientation=observations[target_name]["orientation"],
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            print("IK failed")
        
simulation_app.close()