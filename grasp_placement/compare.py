# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from omni.isaac.core import World
from omni.isaac.franka import KinematicsSolver
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka.tasks import FollowTarget
from omni.isaac.sensor import ContactSensor, _sensor
import omni.kit.commands
import numpy as np
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Usd, UsdLux, UsdGeom, UsdShade, Sdf, Gf, Tf, Vt, UsdPhysics, PhysxSchema
from omni.physx.scripts.physicsUtils import *
import omni.usd

def get_current_stage() -> Usd.Stage:
    return omni.usd.get_context().get_stage()


my_world = World(stage_units_in_meters=1.0)
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)
my_world.reset()



task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()
reset_needed = False


# apply contact report
stage = get_current_stage()
spherePrim = stage.GetPrimAtPath("/World/Franka")
add_rigid_xform(stage,"/World/Franka")
contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(spherePrim)
contactReportAPI.CreateThresholdAttr().Set(200000)



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

        contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()
        print(len(contact_headers))
        for contact_header in contact_headers:
            print("Got contact header type: " + str(contact_header.type))
            print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)))
            print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)))
            print("Collider0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0)))
            print("Collider1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1)))
            print("StageId: " + str(contact_header.stage_id))
            print("Number of contacts: " + str(contact_header.num_contact_data))
        if succ:
            articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")
simulation_app.close()
