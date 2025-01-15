import os

# Full file path
file_path = "/home/chris/Chris/placement_ws/src/grasp_placement/data/Grasping_0/grasping_0.json"

# Extract the directory path
directory_path = os.path.dirname(file_path)

print(directory_path)