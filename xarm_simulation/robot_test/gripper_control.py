from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.manipulators.manipulators import SingleManipulator
from omni.isaac.sensor import ContactSensor
from functools import partial
from typing import List
import numpy as np
from omni.isaac.nucleus import get_assets_root_path 

# World setup with physics enabled
my_world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)

# Use Isaac Sim provided asset
asset_path = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"  # Built-in asset (relative to nucleus mount)
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")

# Define the Panda gripper
# Joint names and limits for Panda (Franka)
panda_gripper = ParallelGripper(
    end_effector_prim_path="/World/Franka/panda_hand",  # effector link
    joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
    joint_opened_positions=np.array([0.04, 0.04]),    # fully open (max 0.04)
    joint_closed_positions=np.array([0.0, 0.0]),      # fully closed
    action_deltas=np.array([-0.04, -0.04]),           # not used, but set for completeness
)

# Define the manipulator
my_franka = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/Franka",
        name="franka_robot",
        end_effector_prim_name="panda_hand",
        gripper=panda_gripper
    )
)

# Set default joint positions (all zeros is fine for demo)
joints_default_positions = np.zeros(9)
my_franka.set_joints_default_state(positions=joints_default_positions)

my_world.scene.add_default_ground_plane()

# Contact state tracking
contact_force = 0.0
finger_contact_detected = False
contact_sensors = []

def setup_contact_sensors():
    """Set up contact sensors after world is properly initialized"""
    global contact_sensors
    
    # Clear any existing sensors
    contact_sensors.clear()
    
    # Contact sensor setup for gripper fingers - try different prim paths
    finger_prim_paths = [
        "/World/Franka/panda_leftfinger",
        "/World/Franka/panda_rightfinger",
        # Alternative paths that might work
        "/World/Franka/panda_hand/panda_leftfinger",
        "/World/Franka/panda_hand/panda_rightfinger"
    ]

    for i, prim_path in enumerate(finger_prim_paths):
        try:
            sensor: ContactSensor = my_world.scene.add(
                ContactSensor(
                    prim_path=prim_path,
                    name=f"finger_contact_sensor_{i}",
                    min_threshold=0.0,
                    max_threshold=1e7,
                    radius=0.05,  # Smaller radius for more precise detection
                )
            )
            # Use raw contact data for detailed contact information
            sensor.add_raw_contact_data_to_frame()
            contact_sensors.append(sensor)
            print(f"Contact sensor {i} created for {prim_path}")
        except Exception as e:
            print(f"Failed to create contact sensor for {prim_path}: {e}")

def on_sensor_contact_report(dt, sensors: List[ContactSensor]):
    """Physics-step callback: checks finger contact sensors"""
    global contact_force, finger_contact_detected
    
    finger_contact_detected = False
    contact_force = 0.0
    
    for i, sensor in enumerate(sensors):
        try:
            frame_data = sensor.get_current_frame()
            
            # Debug: Print the structure of frame_data
            if i == 0 and my_world.current_time_step_index % 200 == 0:  # Print every 200 steps
                print(f"Frame data keys: {list(frame_data.keys()) if isinstance(frame_data, dict) else 'Not a dict'}")
                print(f"Frame data: {frame_data}")
            
            # Check if frame_data has the expected structure
            if not isinstance(frame_data, dict):
                continue
                
            # Check for different possible keys in frame_data
            in_contact = False
            if "in_contact" in frame_data:
                in_contact = frame_data["in_contact"]
            elif "contact" in frame_data:
                in_contact = frame_data["contact"]
            elif "contacts" in frame_data and len(frame_data["contacts"]) > 0:
                in_contact = True
            
            if in_contact:
                print(f"Contact detected on sensor {i}!")
                # We have contact! Extract the bodies, force, etc.
                contacts = frame_data.get("contacts", [])
                if not contacts:
                    # Try alternative key names
                    contacts = frame_data.get("contact_data", [])
                
                for c in contacts:
                    if isinstance(c, dict):
                        body0 = c.get("body0", "")
                        body1 = c.get("body1", "")
                        force = c.get("force", 0.0)
                        
                        print(f"Contact details: {body0} | {body1} | Force: {force:.3f}N")
                        
                        # Check if fingers are touching each other or other objects
                        if "panda_leftfinger" in body0 + body1 and "panda_rightfinger" in body0 + body1:
                            # Fingers touching each other
                            finger_contact_detected = True
                            contact_force = max(contact_force, force)
                            print(f"Fingers touching each other! Force: {force:.3f}N")
                        elif "panda_leftfinger" in body0 + body1 or "panda_rightfinger" in body0 + body1:
                            # Finger touching something else
                            finger_contact_detected = True
                            contact_force = max(contact_force, force)
                            print(f"Finger contact detected: {body0} | {body1} | Force: {force:.3f}N")
                            
        except Exception as e:
            # Print errors for debugging
            if my_world.current_time_step_index % 200 == 0:  # Print every 200 steps
                print(f"Error in sensor {i}: {e}")

# Initialize the world first
my_world.reset()

# Now set up contact sensors after world is initialized
setup_contact_sensors()

# Add physics callback for contact monitoring (only if sensors were created successfully)
if contact_sensors:
    my_world.add_physics_callback("contact_sensor_callback", 
                                 partial(on_sensor_contact_report, sensors=contact_sensors))
    print(f"Contact sensor callback added for {len(contact_sensors)} sensors")
else:
    print("No contact sensors created, skipping callback")

i = 0

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
        i += 1
        gripper_positions = my_franka.gripper.get_joint_positions()
        
        # Print current gripper state and contact information
        if i % 100 == 0:  # Print every 100 steps
            print(f"Step {i}: Gripper positions: [{gripper_positions[0]:.4f}, {gripper_positions[1]:.4f}]")
            print(f"Contact force: {contact_force:.3f}N, Finger contact: {finger_contact_detected}")
            
            # Debug: Check if gripper is actually closed
            if gripper_positions[0] < 0.001 and gripper_positions[1] < 0.001:
                print("Gripper appears to be fully closed - checking for contact...")
        
        if i < 500:
            # Close the gripper slowly (both joints move toward 0)
            my_franka.gripper.apply_action(
                ArticulationAction(joint_positions=[
                    max(gripper_positions[0] - 0.002, 0.0),  # step -0.002 per frame, clamp at 0
                    max(gripper_positions[1] - 0.002, 0.0)
                ])
            )
            
            # Stop closing if fingers are touching (optional safety feature)
            if finger_contact_detected and contact_force > 5.0:  # 5N threshold
                print(f"Stopping gripper closing - high contact force detected: {contact_force:.3f}N")
                # You can add a break here if you want to stop closing
                # break
                
        if i > 500:
            # Open the gripper slowly (both joints move toward 0.04)
            my_franka.gripper.apply_action(
                ArticulationAction(joint_positions=[
                    min(gripper_positions[0] + 0.002, 0.04),  # step +0.002 per frame, clamp at 0.04
                    min(gripper_positions[1] + 0.002, 0.04)
                ])
            )
        if i == 1000:
            i = 0

simulation_app.close()
