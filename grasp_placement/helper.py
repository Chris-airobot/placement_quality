import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
import omni.graph.core as og

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
    step = np.deg2rad(36)  # Convert 36 degrees to radians

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




def projection(q_current_cube, q_current_ee, q_desired_ee):
    """
    Args:
        q_current_cube: orientation of current cube
        q_current_ee: orientation of current end effector
        q_desired_ee: orientation of desired end effector

        all in w,x,y,z format
    """
    # --- Step 1: Compute the relative rotation from EE to cube ---
    q_rel = qmult(qinverse(q_current_ee), q_current_cube)

    # --- Step 2: Compute the initial desired cube orientation based on desired EE ---
    q_desired_cube = qmult(q_desired_ee, q_rel)

    # Convert this quaternion to a rotation matrix for projection
    R_cube = quat2mat(q_desired_cube)

    # --- Step 3: Project the orientation so the cube's designated face is up ---
    # Define world up direction
    u = np.array([0, 0, 1])

    # Assuming the cube's local z-axis should point up:
    v = R_cube[:, 2]  # Current "up" direction of the cube according to its orientation

    # Compute rotation axis and angle to align v with u
    axis = np.cross(v, u)
    axis_norm = np.linalg.norm(axis)

    # Handle special cases: if axis is nearly zero, v is aligned or anti-aligned with u
    if axis_norm < 1e-6:
        # If anti-aligned, choose an arbitrary perpendicular axis
        if np.dot(v, u) < 0:
            axis = np.cross(v, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v, np.array([0, 1, 0]))
        else:
            # v is already aligned with u
            axis = np.array([0, 0, 1])
        axis_norm = np.linalg.norm(axis)

    axis = axis / axis_norm  # Normalize the axis
    angle = np.arccos(np.clip(np.dot(v, u), -1.0, 1.0))  # Angle between v and u

    # Compute corrective rotation matrix
    R_align = axangle2mat(axis, angle)

    # Apply the corrective rotation to project the cube's orientation
    R_cube_projected = np.dot(R_align, R_cube)

    # Convert the projected rotation matrix back to quaternion form
    q_cube_projected = mat2quat(R_cube_projected)

    return q_cube_projected





def joint_graph_generation():
    import omni.graph.core as og
    keys = og.Controller.Keys

    robot_frame_path= "/World/Franka"
    cube_frame = "/World/Cube"
    graph_path = "/Graphs/TF"

    (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
        {
            "graph_path": graph_path, 
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("IsaacClock", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("RosContext", "omni.isaac.ros2_bridge.ROS2Context"),
                ("TF_Tree", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
            ],

            keys.SET_VALUES: [
                ("TF_Tree.inputs:topicName", "/tf"),
                ("TF_Tree.inputs:targetPrims", [robot_frame_path, cube_frame]),
                # ("TF_Tree.inputs:targetPrims", cube_frame),
                ("TF_Tree.inputs:queueSize", 10),
 
            ],

            keys.CONNECT: [
                ("OnTick.outputs:tick", "TF_Tree.inputs:execIn"),
                ("IsaacClock.outputs:simulationTime", "TF_Tree.inputs:timeStamp"),
                ("RosContext.outputs:context", "TF_Tree.inputs:context"),
            ]
        }
    )


if __name__ == "__main__":
    # # # Example Usage
    main_directory = "/home/chris/Chris/placement_ws/src/data"  # Replace with your directory path
    file_counts, total_files = count_files_in_subfolders(main_directory)

    sorted_subfolders = sorted(
        ((subfolder, count) for subfolder, count in file_counts.items() if count > 1 
        and os.path.basename(subfolder).startswith("Grasping_")),
        key=lambda item: int(os.path.basename(item[0]).split('_')[-1])
    )

    for subfolder, count in sorted_subfolders:
        print(f"{subfolder}: {count} files")

    print(f"Total files across all subfolders: {total_files}")