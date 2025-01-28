from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.sensor import ContactSensor

from omni.isaac.franka.tasks import FollowTarget
from omni.isaac.franka import KinematicsSolver

def on_sensor_contact_report(dt):
    """Physics-step callback: checks all sensors, prints only if in_contact==True."""
    for sensor in panda_sensors:
        frame_data = sensor.get_current_frame()
        if frame_data["in_contact"]:
            # We have contact! Extract whatever you want
            print(f"Contact on sensor '{sensor.name}':")
            for c in frame_data["contacts"]:
                body0 = c["body0"]
                body1 = c["body1"]
                print(f"  Bodies: {body0} <-> {body1}")
            print(f"  Force: {frame_data['force']:.3f}, #Contacts: {frame_data['number_of_contacts']}")
            print()  # blank line for readability


#
# Create your world and task
#
my_world = World(stage_units_in_meters=1.0)
my_task = FollowTarget(name="follow_target_task")
my_world.add_task(my_task)

my_world.reset()

#
# Create the contact sensors
#
panda_prim_names = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_link8",
    "panda_hand",
]

panda_sensors = []
for i, link_name in enumerate(panda_prim_names):
    sensor = my_world.scene.add(
        ContactSensor(
            prim_path=f"/World/Franka/{link_name}/contact_sensor",
            name=f"panda_contact_sensor_{i}",
            min_threshold=0.0,
            max_threshold=1e7,
            radius=0.1,
        )
    )
    # Use raw contact data if desired
    print(sensor)
    sensor.add_raw_contact_data_to_frame()
    panda_sensors.append(sensor)


my_world.reset()

#
# Register the sensor check as a physics callback
#
my_world.add_physics_callback("contact_sensor_callback", on_sensor_contact_report)

#
# Usual control setup
#
task_params = my_world.get_task("follow_target_task").get_params()
franka_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_franka = my_world.scene.get_object(franka_name)
my_controller = KinematicsSolver(my_franka)
articulation_controller = my_franka.get_articulation_controller()

reset_needed = False

#
# Main simulation loop
#
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
            carb.log_warn("IK did not converge to a solution.  No action is being taken.")

simulation_app.close()
