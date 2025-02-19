import numpy as np
from omni.isaac.dynamic_control import _dynamic_control
from carb import Float3

# Acquire the dynamic control interface
dc = _dynamic_control.acquire_dynamic_control_interface()

robot_prim_path = "/panda"    
robot_art = dc.get_articulation(robot_prim_path)
print("Articulation handle:", robot_art)

# Get the number of bodies in the articulation.
num_bodies = dc.get_articulation_body_count(robot_art)
print("Number of bodies:", num_bodies)

if num_bodies > 0:
    # Try to find the body handle by its name.
    # Adjust the body name if necessary. For a simple cube, it might be "Cube".
    body_handle = dc.find_articulation_body(robot_art, "panda_leftfinger")
    
    if body_handle == _dynamic_control.INVALID_HANDLE:
        print("Could not find the body handle for 'Cube'.")
    else:
        # Define a force vector (e.g., 10 N upward).
        force_vec = Float3(20, 20,20)
        # Apply the force at the body's center of mass (zero offset).
        position_vec = Float3(0, 5, 5)
        is_global = True

        success = dc.apply_body_force(body_handle, force_vec, position_vec, is_global)
        if success:
            print("Force applied successfully.")
        else:
            print("Failed to apply force.")
else:
    print("No bodies found in the articulation.")
