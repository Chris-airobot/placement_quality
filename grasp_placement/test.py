import os
import glob


DIR_PATH = "/home/chris/Chris/placement_ws/src/grasp_placement/data/"
file_path = DIR_PATH + f"Grasping_33/grasping.json"

    # If the replay data does not exist, create one
if not os.path.exists(file_path):
    file_pattern = os.path.join(DIR_PATH, "Grasping_33/placement_*.json")
    file_list = glob.glob(file_pattern)
    
print(file_list)
