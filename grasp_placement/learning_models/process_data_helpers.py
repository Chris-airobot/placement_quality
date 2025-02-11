import os
import json
import re
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
import numpy as np
from scipy.spatial.transform import Rotation as R

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




def projection(grasp_before_release):


    ee_target_quat = convert_wxyz_to_xyzw(grasp_before_release['data']['ee_orientation'])
    ee_quat = convert_wxyz_to_xyzw(grasp_before_release['data']['ee_orientation'])
    cube_quat = convert_wxyz_to_xyzw(grasp_before_release['data']['cube_orientation'])


    r_offset = R.from_quat(ee_quat).inv() * R.from_quat(cube_quat)

    # Convert the offset rotation back to a quaternion.
    ee_rot = R.from_quat(ee_target_quat)
    offset_rot = R.from_quat(r_offset.as_quat())



    cube_target_rot = (ee_rot * offset_rot).as_quat()
 
    # Return the quaternion in [x, y, z, w] format.
    return [cube_target_rot[3], cube_target_rot[0], cube_target_rot[1], cube_target_rot[2]]


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
    contact_count_after_stage3 = 0
    # earliest_stage7 = None
    # found_earliest_stage7 = False
    
    grasp_position = None
    grasp_orientation = None
    grasp_before_release = None
    last_entry = isaac_sim_data[-1]






    # Match the first boolean after "placement_XX_"
    match = re.search(r"Placement_\d+_(False|True)", file_path)
    grasp_unsuccessful = match.group(1) == "True" if match else None
    # Extract required information
    # if not last_stage3:
    #     raise ValueError(f"No stage=3 found in {file_path}")
    # if not last_stage6:
    #     raise ValueError(f"No stage=6 found in {file_path}")

    
    ###############################
    ####### Inputs
    ###############################
    # grasp pose calculation
    stage3_entries = [d for d in isaac_sim_data if d["data"]["stage"] == 3]
    if stage3_entries:
        # Find the entry with the maximum current_time
        first_stage3 = min(stage3_entries, key=lambda x: x["current_time"])

    # Step 2: Filter entries starting from the first stage 3
    start_time = first_stage3["current_time"]
    data_after_stage3 = [entry for entry in isaac_sim_data if entry["current_time"] >= start_time]
    for i in range(len(data_after_stage3)):
        if data_after_stage3[i]["data"]["cube_grasped"]:
            if data_after_stage3[i+1]["data"]["cube_grasped"] and data_after_stage3[i+2]["data"]["cube_grasped"]:
                grasp_position = data_after_stage3[i]["data"]["ee_position"]
                grasp_orientation = data_after_stage3[i]["data"]["ee_orientation"]
                break

    for data in data_after_stage3:
        if data["data"]["contact"]:
            contact_count_after_stage3 += 1

    cube_initial_position = isaac_sim_data[0]["data"]["cube_position"]
    cube_initial_orientation = isaac_sim_data[0]["data"]["cube_orientation"]
    cube_target_position = isaac_sim_data[0]["data"]["target_position"]


    # Find the grasp data right before release
    for data in reversed(isaac_sim_data):
        if data["data"]["cube_grasped"]:
            grasp_before_release = data
            break

    
    
    if not grasp_unsuccessful: 
        # Calculate the target orientation
        cube_target_orientation = projection(grasp_before_release)
        # Pose difference between the final cube pose and the target pose
        
        
        cube_target_surface = grasp_before_release["data"]["surface"]



        ###############################
        ####### Outputs
        ###############################
        cube_final_position = last_entry["data"]["cube_position"]
        cube_final_orientation = last_entry["data"]["cube_orientation"]
        cube_final_surface = last_entry["data"]["surface"]

        d_pos, angle = pose_difference(
            cube_final_position, cube_final_orientation,
            cube_target_position, cube_target_orientation
        )

        delta_pose_initial = None
        first_stage7 = None
        stage7_entries = [d for d in isaac_sim_data if d["data"]["stage"] == 7]
        if stage7_entries:
            # Find the entry with the maximum current_time
            first_stage7 = min(stage7_entries, key=lambda x: x["current_time"])

        # Step 2: Filter entries starting from the first stage 7
        if first_stage7:
            start_time = first_stage7["current_time"]
            filtered_data = [entry for entry in isaac_sim_data if entry["current_time"] >= start_time]

            for entry in filtered_data:
                if entry["data"]["cube_in_ground"]:
                    delta_pose_initial = entry

        # Step 4: Find the last entry in the data
        if delta_pose_initial is not None:
            # Step 5: Compute the pose difference (cube_position difference)
            shift_position, shift_orientation = pose_difference(
                cube_final_position, cube_final_orientation,
                delta_pose_initial["data"]["cube_position"], delta_pose_initial["data"]["cube_orientation"]
            )

        else:
            shift_position, shift_orientation = None, None
        # Create a dictionary with inputs and outputs

        return {
            "file_path": file_path,
            "inputs": {
                "grasp_position": grasp_position,
                "grasp_orientation": grasp_orientation,
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
                "position_differece": d_pos,
                "orientation_differece": angle,
                "shift_position": shift_position,
                "shift_orientation": shift_orientation,
                "contacts": contact_count_after_stage3
            }
        }
    else:
        return {
            "file_path": file_path,
            "inputs": {
                "grasp_position": grasp_position,
                "grasp_orientation": grasp_orientation,
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
                    if 'quat' in str(e):
                        os.remove(file_path)
                        print(f"Removed {file_path}")

    # Save results to an output file (JSON format)
    with open(output_file, "w") as out_file:
        json.dump(results, out_file, indent=4)
    print(f"Processed data saved to {output_file}")

# Usage Example


if __name__ == "__main__":
    process_folder("//home/chris/Chris/placement_ws/src/random_data", "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json")
    # reformat_json("/home/chris/Sim/data/placement_1_False.json")
    # root_folder = "/home/chris/Downloads/python/data"  # Root folder containing subfolders and JSON files
    # output_file = "processed_data.json"  # Output dataset file
    # process_folder(root_folder, output_file)


    # with open("processed_data.json", "r") as file:
    #     data_all = json.load(file)