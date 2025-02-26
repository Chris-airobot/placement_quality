import numpy as np

# Recommended workspace (from Panda datasheet, see :contentReference[oaicite:0]{index=0})
workspace_center = np.array([0.515, 0.0, 0.226])  # in meters
workspace_size = np.array([0.4, 0.4, 0.4])         # dimensions in meters

# Compute workspace boundaries:
workspace_min = workspace_center - workspace_size / 2.0
workspace_max = workspace_center + workspace_size / 2.0

print("Recommended workspace bounds:")
print("Min:", workspace_min)
print("Max:", workspace_max)

# When generating a cube, ensure its position falls within these bounds:
def is_position_within_workspace(position):
    return np.all(position >= workspace_min) and np.all(position <= workspace_max)

# Example cube position:
cube_position = np.array([0.55, 0.05, 0.23])
if is_position_within_workspace(cube_position):
    print("Cube is within the Panda's workspace.")
else:
    print("Cube is out of reach. Adjust its position.")
