import os
import json
import re
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf_transformations as tft 

def reformat_json(file_path): 
    # Parse the JSON string into a Python dictionary
    
    with open(file_path, "r") as file:
        raw_data = json.load(file)  # Parse JSON into a Python dictionary
    
    
    
    # Pretty-print the JSON
    pretty_json = json.dumps(raw_data, indent=4)
    
    directory, base_name = os.path.split(file_path)
    pretty_name = "pretty_" + base_name
    output_path = os.path.join(directory, pretty_name)
    
    with open(output_path, "w") as pretty_file:
        pretty_file.write(pretty_json)
    
def convert_wxyz_to_xyzw(q_wxyz):
    """Convert a quaternion from [w, x, y, z] format to [x, y, z, w] format."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def pose_to_homogeneous(position, quaternion_wxyz):
    """
    Convert a pose (position and quaternion as [x, y, z] and [x, y, z, w])
    into a 4x4 homogeneous transformation matrix.
    """
    quat_xyzw = convert_wxyz_to_xyzw(quaternion_wxyz)
    T = tft.quaternion_matrix(quat_xyzw)  # Returns a 4x4 transformation matrix
    T[0:3, 3] = position
    return T



def get_transform_to_world(tf_list, target_frame):
    """
    Recursively computes the transform from the 'world' frame to target_frame.
    
    Each TF entry in tf_list is assumed to have:
      - "parent_frame": The parent frame name.
      - "child_frame": The child frame name.
      - "translation": A dict with keys "x", "y", "z".
      - "rotation": A dict with keys "x", "y", "z", "w" (in wxyz order in the data).
    
    Returns a 4x4 homogeneous transformation matrix representing the pose of 
    target_frame in the world coordinate system, or None if no chain can be built.
    """
    if target_frame == "world":
        return np.eye(4)
    
    # Look for an entry whose child_frame is the target_frame.
    for tf_entry in tf_list:
        if tf_entry.get("child_frame") == target_frame:
            # Get the transform from its parent to this target_frame.
            translation = tf_entry["translation"]
            rotation = tf_entry["rotation"]
            pos = np.array([translation["x"], translation["y"], translation["z"]])
            # Convert quaternion from the tf data order (wxyz) to (x,y,z,w) order.
            quat_wxyz = np.array([rotation["w"], rotation["x"], rotation["y"], rotation["z"]])
            T_target_given_parent = pose_to_homogeneous(pos, quat_wxyz)
            
            parent_frame = tf_entry["parent_frame"]
            # Recursively get the transform from world to the parent_frame.
            T_parent = get_transform_to_world(tf_list, parent_frame)
            if T_parent is None:
                # Could not resolve parent transform.
                return None
            # The complete transform is the chain: T_world->target = T_world->parent * T_parent->target.
            return np.dot(T_parent, T_target_given_parent)
    # If no entry is found for the target_frame, return None.
    return None

def get_relative_transform(tf_list, source_frame, target_frame):
    """
    Computes the relative transform from source_frame to target_frame.
    
    The transformation is computed as:
       T_target_in_source = (T_source_in_world)^{-1} * T_target_in_world
       
    Returns the 4x4 homogeneous transformation matrix representing the pose 
    of target_frame in the coordinate system of source_frame.
    """
    T_source = get_transform_to_world(tf_list, source_frame)
    T_target = get_transform_to_world(tf_list, target_frame)
    
    if T_source is None:
        print(f"Transform chain to source frame '{source_frame}' could not be resolved.")
        return None
    if T_target is None:
        print(f"Transform chain to target frame '{target_frame}' could not be resolved.")
        return None
    
    # Compute the relative transform.
    T_relative = np.dot(np.linalg.inv(T_source), T_target)
    return T_relative


def transform_pose_to_frame(world_position, world_orientation, T_frame_inv):
    """
    Transform a world-frame pose (position & orientation in wxyz) into a new frame.
    Returns (relative_position, relative_orientation in wxyz).
    """
    T_world = pose_to_homogeneous(world_position, world_orientation)
    T_rel = np.dot(T_frame_inv, T_world)
    rel_pos = T_rel[0:3, 3].tolist()
    rel_orient = tft.quaternion_from_matrix(T_rel)
    return rel_pos, [rel_orient[3], rel_orient[0], rel_orient[1], rel_orient[2]]



def compute_cube_target_orientation(data):
    """
    Compute the target orientation for the cube given:
      - the EE pose at the moment of grasp,
      - the cube pose at the moment of grasp, and
      - the target EE pose (which provides the target orientation for the EE).

    Parameters:
      data: data of the important time steps

    Returns:
      cube_target_orientation: Quaternion (as [w, x, y, z]) representing the predicted target orientation of the cube.
    """
    ee_position_at_grasp = data["ee_position"]
    ee_orientation_at_grasp = convert_wxyz_to_xyzw(data["ee_orientation"])
    cube_position_at_grasp = data["cube_position"]
    cube_orientation_at_grasp = convert_wxyz_to_xyzw(data["cube_orientation"])
    ee_target_orientation = euler2quat(data["ee_target_orientation"][0],
                                       data["ee_target_orientation"][1],
                                       data["ee_target_orientation"][2],)

    ee_target_orientation_xyzw = [ee_target_orientation[1], ee_target_orientation[2], ee_target_orientation[3], ee_target_orientation[0]]
    # Compute the homogeneous transforms at grasp time
    T_ee = pose_to_homogeneous(ee_position_at_grasp, ee_orientation_at_grasp)
    T_cube = pose_to_homogeneous(cube_position_at_grasp, cube_orientation_at_grasp)
    
    # Compute the relative transform (which is assumed fixed during the grasp)
    T_relative = np.linalg.inv(T_ee) @ T_cube
    relative_quat = tft.quaternion_from_matrix(T_relative)
    
    # Compose the target EE orientation with the relative rotation to predict the cube's target orientation.
    cube_target_quat = tft.quaternion_multiply(ee_target_orientation_xyzw, relative_quat)
    
    return [cube_target_quat[3], cube_target_quat[0], cube_target_quat[1], cube_target_quat[2]]


def pose_difference(final_pos, final_quat, target_pos, target_quat):
    # 1. Position difference
    d_pos = np.linalg.norm(np.array(final_pos) - np.array(target_pos))

    # 2. Orientation difference
    # relative_quat = target_quat * inverse(final_quat)
    # Then angle = 2 * acos(|relative_quat.w|)
    relative_quat = qmult(target_quat, qinverse(final_quat))

    norm = np.sqrt(sum(q*q for q in relative_quat))
    normalize = [q / norm for q in relative_quat]


    w = abs(normalize[0])
    w = np.clip(w, -1.0, 1.0) 
    angle = 2.0 * np.arccos(w)  # [0] is 'w' if your quat is [w, x, y, z]

    return float(d_pos), float(angle)


def process_file(file_path):
    """
    Process a single file to extract relevant information for training.

    """
    with open(file_path, "r") as file:
        data_all = json.load(file)

    isaac_sim_data = data_all["Isaac Sim Data"]
    if not isaac_sim_data:
        raise ValueError(f"No data entries found under 'Isaac Sim Data' in {file_path}")
    

    # Match the first boolean after "placement_XX_"
    match = re.search(r"Placement_\d+_(False|True)", file_path)
    grasp_unsuccessful = (match.group(1) == "True") if match else None

    # Initialize variables
    contact_count = 0

    ###############################
    ####### INPUT EXTRACTION
    ###############################
        # --- Assemble inputs (return both world-frame and relative poses as desired) ---
    inputs = {
        "grasp_position": None,
        "grasp_orientation": None,
        "cube_initial_position": isaac_sim_data[0]["data"]["cube_position"],
        "cube_initial_orientation": isaac_sim_data[0]["data"]["cube_orientation"],
        "cube_target_position": isaac_sim_data[0]["data"]["target_position"],
        "cube_target_orientation": list(euler2quat(*np.random.uniform(low=-np.pi, high=np.pi, size=3))),
        # "cube_target_surface": int(np.random.randint(1, 7)),
        "cube_initial_rel_position": None,          # Relative to the gripper frame when the robot just grasped the cube.
        "cube_initial_rel_orientation": None,       # Relative to the gripper frame when the robot just grasped the cube.
        "cube_target_rel_position": None,           # Relative to the gripper frame when the robot just grasped the cube.
        "cube_target_rel_orientation": None,        # Relative to the gripper frame when the robot just grasped the cube.
    }

    # --- Extract grasp pose from stage 3 entries (look for consecutive cube_grasped=True) ---
    stage3 = [d for d in isaac_sim_data if d["data"]["stage"] == 3]
    if not stage3:
        raise ValueError(f"No stage=3 entries found in {file_path}")
    
    # Find the entry with the maximum current_time
    first_stage3 = min(stage3, key=lambda x: x["current_time"])
    data_after_stage3 = [entry for entry in isaac_sim_data if entry["current_time"] >= first_stage3["current_time"]]

    # Loop safely over indices (i, i+1, i+2) to find consecutive entries with cube_grasped==True.
    for i in range(len(data_after_stage3) - 2):
        if data_after_stage3[i]["data"]["cube_grasped"] and \
           data_after_stage3[i+1]["data"]["cube_grasped"] and \
           data_after_stage3[i+2]["data"]["cube_grasped"]:
            inputs["grasp_position"] = data_after_stage3[i]["data"]["ee_position"]
            inputs["grasp_orientation"] = data_after_stage3[i]["data"]["ee_orientation"]
            # --- Determine cube target orientation  ---
            if not grasp_unsuccessful:
                # Use your projection function (assumed defined elsewhere)
                inputs["cube_target_orientation"] = compute_cube_target_orientation(data_after_stage3[i]["data"])
                inputs["cube_initial_position"] = data_after_stage3[i]["data"]["cube_position"]
                inputs["cube_initial_orientation"] = data_after_stage3[i]["data"]["cube_orientation"]
                inputs["cube_target_position"] = data_after_stage3[i]["data"]["target_position"]
                T_grasp = get_relative_transform(data_after_stage3[i]["data"]["tf"], "panda_hand", "world")
                # T_grasp = None
                # if grasp_before_release and "tf" in grasp_before_release["data"]:
                #     T_grasp = get_relative_transform(grasp_before_release["data"]["tf"], "panda_hand", "world")
                    
                # if T_grasp is None:
                #     T_grasp = pose_to_homogeneous(inputs["grasp_position"], inputs["grasp_orientation"])
            break
        else:
            grasp_unsuccessful = True

    # Count contacts after stage 3.
    for d in data_after_stage3:
        if d["data"]["contact"]:
            contact_count += 1


    # --- Extract cube initial and target (world frame) ---
    # inputs["cube_initial_position"] = isaac_sim_data[0]["data"]["cube_position"]
    # inputs["cube_initial_orientation"] = isaac_sim_data[0]["data"]["cube_orientation"]
    # inputs["cube_target_position"] = isaac_sim_data[0]["data"]["target_position"]


    # --- Find the grasp data right before release ---
    # for d in reversed(isaac_sim_data):
    #     if d["data"]["cube_grasped"]:
    #         grasp_before_release = d
    #         break

    # --- Compute the grasp (end-effector) transform ---
    
    if not grasp_unsuccessful:
        T_grasp_inv = np.linalg.inv(T_grasp)

        # --- Transform cube initial pose into the grasp frame ---
        inputs["cube_initial_rel_position"], inputs["cube_initial_rel_orientation"] = transform_pose_to_frame(inputs["cube_initial_position"],
                                                                                                            inputs["cube_initial_orientation"],
                                                                                                            T_grasp_inv)
        


        
            # inputs["cube_target_surface"] = grasp_before_release["data"]["surface"]

        # Transform cube target pose into the grasp frame.
        inputs["cube_target_rel_position"], inputs["cube_target_rel_orientation"] = transform_pose_to_frame(inputs["cube_target_position"],
                                                                                                            inputs["cube_target_orientation"],
                                                                                                            T_grasp_inv)
    






    ###############################
    ####### Outputs
    ###############################
    outputs = {
            "cube_final_position": None,
            "cube_final_orientation": None,
            "cube_final_surface": None,
            "position_difference": None,
            "orientation_difference": None,
            "shift_position": None,
            "shift_orientation": None,
            "cube_final_rel_position": None,        # Relative to the gripper frame when the cube just reached it's final pose.
            "cube_final_rel_orientation": None,     # Relative to the gripper frame when the cube just reached it's final pose.
            "contacts": contact_count,
            "grasp_unsuccessful": grasp_unsuccessful,
            "bad": False
    }
    

    if not grasp_unsuccessful: 

        delta_pose = None
        final_pose = None
        stage7 = [d for d in isaac_sim_data if d["data"]["stage"] == 7]
        if stage7:
            first_stage7 = min(stage7, key=lambda x: x["current_time"])
            for i in range(len(isaac_sim_data)-2):
                if isaac_sim_data[i]["current_time"] >= first_stage7["current_time"]:
                    if isaac_sim_data[i]["data"]["cube_in_ground"] and (delta_pose is None):
                        delta_pose = isaac_sim_data[i]

                    if np.allclose(np.array(isaac_sim_data[i]["data"]["cube_position"]), 
                                   np.array(isaac_sim_data[i+2]["data"]["cube_position"]), 
                                   atol=1e-6):
                        final_pose = isaac_sim_data[i]
                        break
                    
        if final_pose is None:
            final_pose = isaac_sim_data[-1]
            outputs["bad"] = True

        T_release = get_relative_transform(final_pose["data"]["tf"], "panda_hand", "world")
        T_release_inv = np.linalg.inv(T_release)

        # --- Cube final pose (world frame) ---
        outputs["cube_final_position"] = final_pose["data"]["cube_position"]
        outputs["cube_final_orientation"] = final_pose["data"]["cube_orientation"]
        outputs["cube_final_surface"] = final_pose["data"]["surface"]

        # Compute the differences (in world frame) between the final cube pose and the target pose.
        outputs["position_difference"], outputs["orientation_difference"] = pose_difference(
            outputs["cube_final_position"], outputs["cube_final_orientation"],
            inputs["cube_target_position"], inputs["cube_target_orientation"]
        )

        # Determine the delta pose from when the cube has settled (after stage 7).
        

        if delta_pose is not None:
            outputs["shift_position"], outputs["shift_orientation"] = pose_difference(
                outputs["cube_final_position"], outputs["cube_final_orientation"],
                delta_pose["data"]["cube_position"], delta_pose["data"]["cube_orientation"]
            )


        # Transform cube final pose into the grasp frame.
        outputs["cube_final_rel_position"], outputs["cube_final_rel_orientation"] = transform_pose_to_frame(outputs["cube_final_position"],
                                                                                                            outputs["cube_final_orientation"],
                                                                                                            T_release_inv)
        
 
    return {
        "file_path": file_path,
        "inputs": inputs,
        "outputs": outputs
    }

def data_analysis(file_path):
    # Load your JSON file
    with open(file_path, "r") as file:
        trajectories = json.load(file)
    
    # Count trajectories with at least one null (None) value in the outputs
    null_trajectory_count = sum(
        1 for traj in trajectories 
        if any(value is None for value in traj.get("outputs", {}).values())
    )
    print(f"Total number of trajectories: {len(trajectories)}")
    print("Number of trajectories with null outputs:", null_trajectory_count)

def process_folder(root_folder, output_file):
    """
    Process all JSON files in a folder and its subfolders, and save the extracted data.
    """
    results = []

    for subdir, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith(".json") and file_name != "Grasping.json":
                file_path = os.path.join(subdir, file_name)
                try:
                    print(f"Starting to process {file_path}")
                    data = process_file(file_path)
                    results.append(data)
                    print(f"Successfully processed {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    if 'quat' in str(e):
                        os.remove(file_path)
                        print(f"Removed {file_path}")
                    if 'Eigenvalues' in str(e):
                        os.remove(file_path)
                        print(f"Removed {file_path}")

    # Save results to an output file (JSON format)
    with open(output_file, "w") as out_file:
        json.dump(results, out_file, indent=4)
    print(f"Processed data saved to {output_file}")

# Usage Example


if __name__ == "__main__":
    # process_folder("/home/chris/Chris/placement_ws/src/random_data", "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json")
    # process_file("/home/chris/Chris/placement_ws/src/random_data/Grasping_159/Placement_68_False.json")
    # reformat_json("/home/chris/Chris/placement_ws/src/random_data/Grasping_159/Placement_68_False.json")
    data_analysis("/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json")