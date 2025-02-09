import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import open3d as o3d
import json
import tf2_ros
from rclpy.time import Time
import struct
from network_client import GraspClient
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import omni.graph.core as og
DIR_PATH = "/home/chris/Chris/placement_ws/src/data/"

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
  output_file_path = os.path.dirname(input_file_path) + '/Grasping.json'  # Path to save the filtered file

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


    sorted_subfolders = sorted(
        ((subfolder, count) for subfolder, count in file_count_per_subfolder.items() if count > 1 
        and os.path.basename(subfolder).startswith("Grasping_")),
        key=lambda item: int(os.path.basename(item[0]).split('_')[-1])
    )

    for subfolder, count in sorted_subfolders:
        print(f"{subfolder}: {count} files")

    print(f"Total files across all subfolders: {total_file_count}")




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





def tf_graph_generation():
    keys = og.Controller.Keys

    robot_frame_path= "/World/Franka"
    cube_frame = "/World/Cube"
    graph_path = "/Graphs/TF"
    # test_cube = "/World/Franka/test_cube"

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




def process_tf_message(tf_message: TFMessage):
    # Extract the frames and transformations
    tf_data = []
    for transform in tf_message.transforms:
        frame_data = {
            "parent_frame": transform.header.frame_id,
            "child_frame": transform.child_frame_id,
            "translation": {
                "x": transform.transform.translation.x,
                "y": transform.transform.translation.y,
                "z": transform.transform.translation.z
            },
            "rotation": {
                "x": transform.transform.rotation.x,
                "y": transform.transform.rotation.y,
                "z": transform.transform.rotation.z,
                "w": transform.transform.rotation.w
            }
        }
        tf_data.append(frame_data)
    return tf_data


def downsample_points(points: np.ndarray, voxel_size: float = 0.0025) -> np.ndarray:
    """
    Downsample a Nx4 or Nx3 point cloud using Open3D's voxel grid filter.
    This reduces the file size and solves many 'voxelized cloud: 0' issues in PCL.
    """
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    if points.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    else:  # Nx4, ignoring last column for geometry
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Downsample
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # Convert back to NumPy (only x,y,z)
    down_xyz = np.asarray(down_pcd.points)

    # If original data had 4 columns, reattach the 'rgb' (0.0f or something):
    # We'll do a simple approach: everything gets the same 4th column
    if points.shape[1] == 4:
        # For now set all 'rgb' to 0.0. If you want to preserve color,
        # you'd need a more advanced approach, e.g. approximate nearest neighbors
        # in the original data.
        downsampled = np.zeros((down_xyz.shape[0], 4), dtype=np.float32)
        downsampled[:, :3] = down_xyz
        return downsampled
    else:
        return down_xyz


def filter_points_in_bounding_box(points, box_center, box_size):
    """
    Removes points that lie **inside** a specified axis-aligned bounding box.

    Args:
        points (np.ndarray): Nx4 array of [x, y, z, rgb].
        box_center (list/tuple): [cx, cy, cz] for the center of the box.
        box_size (list/tuple): [sx, sy, sz] for the full box dimensions.
                               (orientation = 0,0,0, so axis-aligned)

    Returns:
        np.ndarray: Filtered Nx4 array, with points inside the box removed.
    """
    c = np.array(box_center, dtype=np.float32)
    s = np.array(box_size, dtype=np.float32) * 0.5  # half-extends
    box_min = c - s
    box_max = c + s

    pts_xyz = points[:, :3]  # Nx3
    inside_mask = np.all((pts_xyz >= box_min) & (pts_xyz <= box_max), axis=1)

    # Keep points that are **outside** the box
    return points[~inside_mask]


def save_pointcloud(msg: PointCloud2, pcd_counter: int) -> str:
    """
    Saves a ROS PointCloud2 (with x,y,z,[rgb]) to an ASCII PCD file.
    The file includes a voxel-based downsampling step to reduce size.

    The output path is:  DIR_PATH/Pcd_{pcd_counter}/pointcloud.pcd
    """
    # 1) Create the directory
    dir_name = os.path.join(DIR_PATH, f"Pcd_{pcd_counter}")
    os.makedirs(dir_name, exist_ok=True)

    # 2) Fixed filename: "pointcloud.pcd"
    file_path = os.path.join(dir_name, "pointcloud.pcd")

    # 3) Identify offsets for x,y,z,rgb
    offset_x = offset_y = offset_z = None
    offset_rgb = None
    for field in msg.fields:
        if field.name == "x":
            offset_x = field.offset
        elif field.name == "y":
            offset_y = field.offset
        elif field.name == "z":
            offset_z = field.offset
        elif field.name in ("rgb", "rgba"):
            offset_rgb = field.offset

    if offset_x is None or offset_y is None or offset_z is None:
        raise ValueError("PointCloud2 does not contain x, y, z fields!")

    # 4) Convert data to (N,4) float32 array: [x y z rgb]
    num_points = len(msg.data) // msg.point_step
    points = np.zeros((num_points, 4), dtype=np.float32)

    for i in range(num_points):
        start = i * msg.point_step
        x = struct.unpack_from("f", msg.data, start + offset_x)[0]
        y = struct.unpack_from("f", msg.data, start + offset_y)[0]
        z = struct.unpack_from("f", msg.data, start + offset_z)[0]

        if offset_rgb is not None:
            # read raw 32 bits as float or int
            rgb_uint = struct.unpack_from("I", msg.data, start + offset_rgb)[0]
            # store as float so we can put it in 'rgb' column
            points[i] = [x, y, z, float(rgb_uint)]
        else:
            points[i] = [x, y, z, 0.0]


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # [A] FILTER OUT POINTS INSIDE THE ROBOT'S BOUNDING BOX
    # NOTE: Make sure points are in the same coordinate frame as the bounding box
    box_center = [-0.04, 0.0, 0.0]  # center of the robot bounding box in "world" frame
    box_size   = [0.24, 0.20, 1] # size in x,y,z
    points_filtered = filter_points_in_bounding_box(points, box_center, box_size)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




    # 5) Downsample to reduce size (voxel_size=0.01 => 1cm grid; adjust as needed)
    points_down = downsample_points(points_filtered, voxel_size=0.005)

    # 6) Build ASCII PCD header
    # NOTE: We now do "TYPE F F F F" for the 4 fields, so that PCL's voxel/crop
    #       filter doesn't reject them. (Instead of 'I' for rgb.)
    header = (
        "# .PCD v.7 - Point Cloud Data file format\n"
        "VERSION .7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"  # <--- we changed the last field to F
        "COUNT 1 1 1 1\n"
        f"WIDTH {points_down.shape[0]}\n"
        "HEIGHT 1\n"
        f"POINTS {points_down.shape[0]}\n"
        "DATA ascii\n"
    )

    # 7) Write ASCII data: x, y, z, rgb
    with open(file_path, "w") as f:
        f.write(header)
        for i in range(points_down.shape[0]):
            x, y, z, rgbf = points_down[i]
            # Here we treat the last column as float
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgbf:.6f}\n")

    print(f"[INFO] Saved ASCII PCD to {file_path}: {points_down.shape[0]} points (down from {num_points}).")
    return file_path



def transform_pointcloud_to_frame(
    cloud_in: PointCloud2,
    tf_buffer: tf2_ros.Buffer,
    target_frame: str="panda_link0"
) -> PointCloud2:
    """
    Transforms a PointCloud2 from its current frame (cloud_in.header.frame_id)
    into 'target_frame', using the transform data in tf_msg.

    Args:
        cloud_in (PointCloud2): The input point cloud.
        tf_msg (TFMessage): A TFMessage containing one or more transforms.
        target_frame (str): Name of the desired target frame, e.g. "panda_link0".

    Returns:
        PointCloud2: A new point cloud in the target frame.
    """
    # ---------------------------------------------------
    # 1. Find the Transform from cloud_in's frame to target_frame
    # ---------------------------------------------------
    source_frame = cloud_in.header.frame_id

    # We assume tf_msg.transforms has the needed transform directly
    transform_stamped: TransformStamped = tf_buffer.lookup_transform(
        target_frame=target_frame,
        source_frame=source_frame,
        time=Time())  # "latest" transform

    if transform_stamped is None:
        raise ValueError(f"No direct transform from '{source_frame}' to '{target_frame}' found in TFMessage.")

    # Extract translation
    tx = transform_stamped.transform.translation.x
    ty = transform_stamped.transform.translation.y
    tz = transform_stamped.transform.translation.z

    # Extract rotation (quaternion)
    qx = transform_stamped.transform.rotation.x
    qy = transform_stamped.transform.rotation.y
    qz = transform_stamped.transform.rotation.z
    qw = transform_stamped.transform.rotation.w

    # ---------------------------------------------------
    # 2. Build a 4x4 transform matrix
    # ---------------------------------------------------
    # Rotation from quaternion
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # Construct the rotation part of the matrix:
    rx = 1 - 2*(qy**2 + qz**2)
    ry = 2*(qx*qy - qz*qw)
    rz = 2*(qx*qz + qy*qw)

    ux = 2*(qx*qy + qz*qw)
    uy = 1 - 2*(qx**2 + qz**2)
    uz = 2*(qy*qz - qx*qw)

    fx = 2*(qx*qz - qy*qw)
    fy = 2*(qy*qz + qx*qw)
    fz = 1 - 2*(qx**2 + qy**2)

    transform_mat = np.array([
        [rx, ry, rz, tx],
        [ux, uy, uz, ty],
        [fx, fy, fz, tz],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)

    # ---------------------------------------------------
    # 3. Parse points from cloud_in into a Nx? array
    #    We'll handle x,y,z at least, and keep other fields as-is.
    # ---------------------------------------------------
    # Identify offsets for x, y, z
    offset_x = offset_y = offset_z = None
    # We keep track of other fields too, so we can copy them unmodified
    fields_dict = {}  # {field_name: (offset, datatype, count)}

    for f in cloud_in.fields:
        fields_dict[f.name] = (f.offset, f.datatype, f.count)
        if f.name == 'x':
            offset_x = f.offset
        elif f.name == 'y':
            offset_y = f.offset
        elif f.name == 'z':
            offset_z = f.offset

    if offset_x is None or offset_y is None or offset_z is None:
        raise ValueError("PointCloud2 is missing x, y, or z fields.")

    point_step = cloud_in.point_step
    row_step = cloud_in.row_step
    data_bytes = cloud_in.data

    num_points = cloud_in.width * cloud_in.height

    # We'll build a new bytearray for the output data
    out_data = bytearray(len(data_bytes))

    # For each point, transform x,y,z
    for i in range(num_points):
        point_offset = i * point_step

        # Read x,y,z from the input
        x = struct.unpack_from('f', data_bytes, point_offset + offset_x)[0]
        y = struct.unpack_from('f', data_bytes, point_offset + offset_y)[0]
        z = struct.unpack_from('f', data_bytes, point_offset + offset_z)[0]

        # Make it homogeneous
        pt_hom = np.array([x, y, z, 1.0], dtype=np.float64)

        # Apply the transform
        pt_trans = transform_mat @ pt_hom
        x_out = pt_trans[0]
        y_out = pt_trans[1]
        z_out = pt_trans[2]

        # Store x_out,y_out,z_out into the new data
        struct.pack_into('f', out_data, point_offset + offset_x, x_out)
        struct.pack_into('f', out_data, point_offset + offset_y, y_out)
        struct.pack_into('f', out_data, point_offset + offset_z, z_out)

    # We'll do a small fix: first copy everything, then transform x,y,z in the loop above
    out_data[:] = data_bytes[:]  # copy entire buffer
    # Then the loop modifies x,y,z in place

    for i in range(num_points):
        point_offset = i * point_step
        x = struct.unpack_from('f', data_bytes, point_offset + offset_x)[0]
        y = struct.unpack_from('f', data_bytes, point_offset + offset_y)[0]
        z = struct.unpack_from('f', data_bytes, point_offset + offset_z)[0]

        pt_hom = np.array([x, y, z, 1.0], dtype=np.float64)
        pt_trans = transform_mat @ pt_hom
        x_out, y_out, z_out = pt_trans[0], pt_trans[1], pt_trans[2]

        struct.pack_into('f', out_data, point_offset + offset_x, x_out)
        struct.pack_into('f', out_data, point_offset + offset_y, y_out)
        struct.pack_into('f', out_data, point_offset + offset_z, z_out)

    # ---------------------------------------------------
    # 4. Construct a new PointCloud2 with the same metadata
    #    but updated frame & data
    # ---------------------------------------------------
    cloud_out = PointCloud2()
    cloud_out.header.stamp = cloud_in.header.stamp
    cloud_out.header.frame_id = target_frame  # new frame
    cloud_out.height = cloud_in.height
    cloud_out.width = cloud_in.width
    cloud_out.fields = cloud_in.fields
    cloud_out.is_bigendian = cloud_in.is_bigendian
    cloud_out.point_step = cloud_in.point_step
    cloud_out.row_step = cloud_in.row_step
    cloud_out.is_dense = cloud_in.is_dense

    # set the new data
    cloud_out.data = bytes(out_data)

    return cloud_out


def obtain_grasps(file_path):
    node = GraspClient()
    raw_grasps = node.request_grasps(file_path)

    grasps = []
    for key in sorted(raw_grasps.keys(), key=int):
        item = raw_grasps[key]
        position = item["position"]
        orientation = item["orientation_wxyz"]
        # Each grasp: [ [position], [orientation] ]
        grasps.append([position, orientation])

    return grasps






def view_pcd(file_path):
    # Load and visualize the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # # # Example Usage
    file_path = "/home/chris/Chris/placement_ws/src/data/pcd_0/pointcloud.pcd"
    view_pcd(file_path)

    
    # Suppose you have an Nx3 cloud (world frame), e.g.:
    # cloud_world = np.random.rand(1000, 3) - 0.5  # random points in [-0.5, 0.5]^3

    # # Your robot bounding box info:
    # robot_center = [-0.04, 0.0, 0.0]
    # robot_size   = [0.24, 0.20, 0.01]

    # filtered_cloud = filter_robot_bounding_box(cloud_world, robot_center, robot_size)
    # print("Original cloud size:", len(cloud_world))
    # print("Filtered cloud size:", len(filtered_cloud))