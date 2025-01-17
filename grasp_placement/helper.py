import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json
DIR_PATH = "/home/chris/Chris/placement_ws/src/grasp_placement/data/"

def surface_detection(rpy):
    local_normals = {
        "1": np.array([0, 0, 1]),   # +z going up (0, 0, 0)
        "2": np.array([1, 0, 0]),   # +x going up (-90, -90, -90)
        "3": np.array([0, 0, -1]),  # -z going up (180, 0, -180)
        "4": np.array([-1, 0, 0]),  # -x going up (90, 90, -90)
        "5": np.array([0, -1, 0]),  # -y going up (-90, 0, 0)
        "6": np.array([0, 1, 0]),   # +y going up (90, 0, 0)
        }
    
    global_up = np.array([0, 0, 1]) 

      # Replace with your actual quaternion x,y,z,w
    rotation = R.from_euler('xyz', rpy)

    # Transform normals to the world frame
    world_normals = {face: rotation.apply(local_normal) for face, local_normal in local_normals.items()}

    # Find the face with the highest dot product with the global up direction
    upward_face = max(world_normals, key=lambda face: np.dot(world_normals[face], global_up))
    
    return int(upward_face)


def extract_grasping(input_file_path):
    # Load the original JSON data
  output_file_path = os.path.dirname(input_file_path) + '/grasping.json'  # Path to save the filtered file

  with open(input_file_path, 'r') as file:
      data = json.load(file)

  # Extract data until the stage number hits 4
  filtered_data = []
  for entry in data.get("Isaac Sim Data", []):
      if entry["data"]["stage"] == 4:
          break
      filtered_data.append(entry)

  output_data = {"Isaac Sim Data": filtered_data}

  # Save the filtered data into a new JSON file
  with open(output_file_path, 'w') as output_file:
      json.dump(output_data, output_file)


def orientation_creation():

    # Define the range and step in radians
    start = -np.pi
    end = np.pi
    step = np.deg2rad(36)  # Convert 10 degrees to radians

    # Create the list using nested loops
    result = []
    for i in np.arange(start, end + step, step):
        for j in np.arange(start, end + step, step):
            for k in np.arange(start, end + step, step):
                result.append(np.array([i, j, k]))
    
    return result



def count_files_in_subfolders(directory):
    """
    Count the number of files within all subfolders of a directory.

    Args:
        directory (str): Path to the main directory.

    Returns:
        dict: A dictionary where keys are subfolder paths and values are the file counts.
        int: Total number of files across all subfolders.
    """
    file_count_per_subfolder = {}
    total_file_count = 0

    for root, dirs, files in os.walk(directory):
        # Only consider subfolders (not the main folder)
        if root != directory:
            file_count = len(files)
            file_count_per_subfolder[root] = file_count
            total_file_count += file_count

    return file_count_per_subfolder, total_file_count


# # Example Usage
# main_directory = "/home/chris/Chris/placement_ws/src/grasp_placement/data"  # Replace with your directory path
# file_counts, total_files = count_files_in_subfolders(main_directory)

# print("File counts in each subfolder:")
# for subfolder, count in file_counts.items():
#     print(f"{subfolder}: {count} files")

# print(f"Total files across all subfolders: {total_files}")
