import os
import json
import re
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat


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

    # Initialize variables
    last_stage3 = None
    last_stage6 = None
    contact_count_after_stage3 = 0
    # earliest_stage7 = None
    # found_earliest_stage7 = False

    lowest_z_value = float("inf")
    first_lowest_z_entry = None

    last_entry = isaac_sim_data[-1]

    # Iterate through the data
    for entry in isaac_sim_data:
        stage = entry["data"]["stage"]
        current_time = entry["current_time"]
        contact = entry["data"].get("contact")

        # Last stage=3
        if stage == 3:
            if (last_stage3 is None) or (current_time > last_stage3["current_time"]):
                last_stage3 = entry

        # Last stage=6
        if stage == 6:
            if (last_stage6 is None) or (current_time > last_stage6["current_time"]):
                last_stage6 = entry

        if last_stage3 is not None and current_time > last_stage3["current_time"]:
            if contact is not None:
                contact_count_after_stage3 += 1


    # Match the first boolean after "placement_XX_"
    match = re.search(r"placement_\d+_(False|True)", file_path)
    grasp_unsuccessful = match.group(1) == "True" if match else None
    # Extract required information
    # if not last_stage3:
    #     raise ValueError(f"No stage=3 found in {file_path}")
    # if not last_stage6:
    #     raise ValueError(f"No stage=6 found in {file_path}")

    

    # Inputs
    Grasp_position = last_stage3["data"]["ee_position"]
    Grasp_orientation = last_stage3["data"]["ee_orientation"]
    cube_initial_position = isaac_sim_data[0]["data"]["cube_position"]
    cube_initial_orientation = isaac_sim_data[0]["data"]["cube_orientation"]
    cube_target_position = isaac_sim_data[0]["data"]["target_position"]

    if not grasp_unsuccessful: 
        
        cube_target_orientation = projection(
            last_stage6["data"]["cube_orientation"],
            last_stage6["data"]["ee_orientation"],
            euler2quat(*last_stage6["data"]["ee_target_orientation"])
        )
        cube_target_surface = last_stage6["data"]["surface"]

        # Outputs
        cube_final_position = last_entry["data"]["cube_position"]
        cube_final_orientation = last_entry["data"]["cube_orientation"]
        cube_final_surface = last_entry["data"]["surface"]

        

        d_pos, angle = pose_difference(
            cube_final_position, cube_final_orientation,
            cube_target_position, cube_target_orientation
        )
        d_threshold = 0.01
        angle_threshold = 0.1

        position_successful = d_pos < d_threshold
        orientation_successful = angle < angle_threshold

        
        stage7_entries = [d for d in isaac_sim_data if d["data"]["stage"] == 7]
        if stage7_entries:
            # Find the entry with the maximum current_time
            first_stage7 = min(stage7_entries, key=lambda x: x["current_time"])

        # Step 2: Filter entries starting from the first stage 7
        start_time = first_stage7["current_time"]
        filtered_data = [entry for entry in isaac_sim_data if entry["current_time"] >= start_time]

        # Step 3: Track the lowest z value and find the first time it hits
        lowest_z_value = float("inf")
        first_lowest_z_entry = None

        for entry in filtered_data:
            cube_z = entry["data"]["cube_position"][2]
            if cube_z < lowest_z_value:
                # if cube_z < 0.05: # It hits the ground
                # Update the lowest z and store the first time it hits
                lowest_z_value = cube_z
                first_lowest_z_entry = entry
                # else:
                    
            elif cube_z > lowest_z_value:
                # If the z starts increasing, stop searching
                break

        # Step 4: Find the last entry in the data
        if first_lowest_z_entry is not None:
            last_entry = isaac_sim_data[-1]
            # Step 5: Compute the pose difference (cube_position difference)
            lowest_z_position = first_lowest_z_entry["data"]["cube_position"]
            lowest_z_orientation = first_lowest_z_entry["data"]["cube_orientation"]

            pose_shift_position, pose_shift_orientation = pose_difference(
                cube_final_position, cube_final_orientation,
                lowest_z_position, lowest_z_orientation
            )

        else:
            pose_shift_position, pose_shift_orientation = None, None
        # Create a dictionary with inputs and outputs

        return {
            "file_path": file_path,
            "inputs": {
                "Grasp_position": Grasp_position,
                "Grasp_orientation": Grasp_orientation,
                "cube_initial_position": cube_initial_position,
                "cube_initial_orientation": cube_initial_orientation,
                "cube_target_position": cube_target_position,
                "cube_target_orientation": list(cube_target_orientation),
                "cube_target_surface": cube_target_surface
            },
            "outputs": {
                "cube_final_position": cube_final_position,
                "cube_final_orientation": cube_final_orientation,
                "cube_final_surface": cube_final_surface,
                "grasp_unsuccessful": grasp_unsuccessful,
                "position_successful": bool(position_successful),
                "orientation_successful": bool(orientation_successful),
                "position_differece": d_pos,
                "orientation_differece": angle,
                "pose_shift_position": pose_shift_position,
                "pose_shift_orientation": pose_shift_orientation,
                "contacts": contact_count_after_stage3
            }
        }
    else:
        return {
            "file_path": file_path,
            "inputs": {
                "Grasp_position": Grasp_position,
                "Grasp_orientation": Grasp_orientation,
                "cube_initial_position": cube_initial_position,
                "cube_initial_orientation": cube_initial_orientation,
                "cube_target_position": cube_target_position,
                "cube_target_orientation": list(euler2quat(*np.random.uniform(low=-np.pi, high=np.pi, size=3))),
                "cube_target_surface": np.random.randint(1,7)
            },
            "outputs": {
                "cube_final_position": None,
                "cube_final_orientation": None,
                "cube_final_surface": None,
                "position_differece": None,
                "orientation_differece": None,
                "grasp_unsuccessful": grasp_unsuccessful,
                "position_successful": None,
                "orientation_successful": None,
                "pose_shift_position": None,
                "pose_shift_orientation": None,
                "contacts": None
            }
        }


def process_folder(root_folder, output_file):
    """
    Process all JSON files in a folder and its subfolders, and save the extracted data.
    """
    results = []

    for subdir, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith(".json") and file_name != "grasping.json":
                file_path = os.path.join(subdir, file_name)
                try:
                    data = process_file(file_path)
                    results.append(data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Save results to an output file (JSON format)
    with open(output_file, "w") as out_file:
        json.dump(results, out_file, indent=4)
    print(f"Processed data saved to {output_file}")

# Usage Example


if __name__ == "__main__":
    reformat_json("/home/chris/Sim/data/placement_1_False.json")
    # root_folder = "/home/chris/Downloads/python/data"  # Root folder containing subfolders and JSON files
    # output_file = "processed_data.json"  # Output dataset file
    # process_folder(root_folder, output_file)


    # with open("processed_data.json", "r") as file:
    #     data_all = json.load(file)