import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import open3d as o3d
import json
import tf2_ros
from rclpy.time import Time
import struct
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
from transforms3d.euler import euler2quat, quat2euler
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import random
import math
from omni.isaac.core.utils.prims import is_prim_path_valid
from rclpy.node import Node
import rclpy

class SimSubscriber(Node):
    def __init__(self, buffer_size=100.0):
        super().__init__("Sim_subscriber")
        self.latest_tf = None  # Store the latest TFMessage here
        
        # Store the latest pointclouds here
        self.latest_pcd1 = None
        self.latest_pcd2 = None
        self.latest_pcd3 = None

        self.buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=buffer_size))
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        # Create the TF subscription
        self.tf_subscription = self.create_subscription(
            TFMessage,           # Message type
            "/tf",               # Topic name
            self.tf_callback,    # Callback function
            10                   # QoS
        )

        # Import message_filters for synchronization
        import message_filters
        
        # Create subscribers for the three PCD topics
        self.pcd_sub1 = message_filters.Subscriber(self, PointCloud2, "/cam0/depth_pcl")
        self.pcd_sub2 = message_filters.Subscriber(self, PointCloud2, "/cam1/depth_pcl")
        self.pcd_sub3 = message_filters.Subscriber(self, PointCloud2, "/cam2/depth_pcl")
        
        # Create a time synchronizer with a queue size of 10 and a time tolerance of 0.1 seconds
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pcd_sub1, self.pcd_sub2, self.pcd_sub3], 
            queue_size=10, 
            slop=0.1
        )
        
        # Register the callback for synchronized messages
        self.ts.registerCallback(self.synchronized_pcd_callback)

        # Flag to control whether to process new pointcloud messages.
        self.pcd_callback_enabled = True

    def synchronized_pcd_callback(self, pcd1_msg, pcd2_msg, pcd3_msg):
        """Callback for synchronized pointcloud messages from all three topics"""
        if not self.pcd_callback_enabled:
            return

        self.latest_pcd1 = pcd1_msg
        self.latest_pcd2 = pcd2_msg
        self.latest_pcd3 = pcd3_msg
        
        # You can process the synchronized point clouds here
        # self.get_logger().info("Received synchronized point clouds")
        
        # Optionally merge the point clouds if needed
        # self.merge_point_clouds(pcd1_msg, pcd2_msg, pcd3_msg)
        
    def tf_callback(self, msg):
        # This callback is triggered for every new TF message on /tf
        self.latest_tf = msg
        
        # The TransformListener should automatically handle adding transforms to the buffer,
        # but we can try to explicitly add them as well
        if self.latest_tf is not None and len(self.latest_tf.transforms) > 0:
            for transform in self.latest_tf.transforms:
                print(f"Received transform from {transform.header.frame_id} to {transform.child_frame_id}")
                try:
                    # Explicitly set the transform in the buffer
                    self.buffer.set_transform(transform, "default_authority")
                    print(f"Successfully added transform from {transform.header.frame_id} to {transform.child_frame_id} to buffer")
                except Exception as e:
                    print(f"Error adding transform to buffer: {e}")
        
    def get_latest_pcds(self):
        """Return the latest synchronized point clouds"""
        return {
            "pcd1": self.latest_pcd1,
            "pcd2": self.latest_pcd2,
            "pcd3": self.latest_pcd3
        }
        
    def check_transform_exists(self, target_frame, source_frame):
        """Check if a transform exists in the buffer"""
        try:
            self.buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            return True
        except Exception as e:
            print(f"Transform from {source_frame} to {target_frame} does not exist: {e}")
            return False



def convert_wxyz_to_xyzw(q_wxyz):
    """Convert a quaternion from [w, x, y, z] format to [x, y, z, w] format."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def cube_orientation_to_ee_orientation(q_cube_init, q_ee_init, q_cube_target):
    """
    Compute the target end-effector orientation in quaternion form 
    to preserve the same relative orientation (offset).
    
    Parameters:
    -----------
    q_cube_init  : (4,) array_like
        Quaternion (x, y, z, w) for the initial cube orientation (world frame).
    q_ee_init    : (4,) array_like
        Quaternion for the initial end-effector orientation.
    q_cube_target: (4,) array_like
        Quaternion for the target cube orientation.
        
    Returns:
    --------
    q_ee_target : (4,) np.ndarray
        Quaternion (x, y, z, w) for the target end-effector orientation.
    """


    R_cube_init  = R.from_quat(convert_wxyz_to_xyzw(q_cube_init))
    R_ee_init    = R.from_quat(convert_wxyz_to_xyzw(q_ee_init))
    R_cube_target= R.from_quat(convert_wxyz_to_xyzw(q_cube_target))
    
    # Offset is the rotation taking the cube frame to the ee frame
    # offset = R_cube_init⁻¹ * R_ee_init
    offset = R_cube_init.inv() * R_ee_init
    
    # Target end-effector orientation = R_cube_target * offset
    R_ee_target = R_cube_target * offset
    
    return R_ee_target.as_quat()



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

    """
    Generate a list of random orientations for the grasping pose.
    Each orientation is represented as an array of three random angles (Euler angles)
    in the range [-pi, pi].
    
    Parameters:
        num_orientations (int): Number of random orientations to generate.
        
    Returns:
        List[np.ndarray]: A list containing `num_orientations` arrays, each with 3 random angles.
    """
    result = []
    for _ in range(1000):
        # Generate three random angles in the range [-pi, pi]
        orientation = np.random.uniform(-np.pi, np.pi, size=3)
        result.append(orientation)

    return result





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
    import omni.graph.core as og

    keys = og.Controller.Keys

    robot_frame_path= "/World/Franka"
    ycb_frame = "/World/Ycb_object"
    graph_path = "/Graphs/TF"
    # test_cube = "/World/Franka/test_cube"
    if is_prim_path_valid(graph_path):
        return
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
                ("TF_Tree.inputs:targetPrims", [robot_frame_path, ycb_frame]),
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






def draw_frame(
    position: np.ndarray,
    orientation: np.ndarray,
    scale: float = 0.1,
):
    # Isaac Sim's debug draw interface
    from omni.isaac.debug_draw import _debug_draw
    from carb import Float3, ColorRgba
    # Acquire the debug draw interface once at startup or script init
    draw = _debug_draw.acquire_debug_draw_interface()
    # Clear previous lines so we have only one, moving frame.
    draw.clear_lines()
    """
    Draws a coordinate frame with colored X, Y, Z axes at the specified pose.
    
    Args:
        frame_name: A unique name for this drawing (used by Debug Draw).
        position:   (3,) array-like of [x, y, z] for frame origin in world coordinates.
        orientation:(4,) array-like quaternion [x, y, z, w].
        scale:      Length of each axis line.
        duration:   How long (in seconds) the lines remain in the viewport
                    before disappearing. If you keep calling draw_frame each
                    physics step, it will appear continuously.
    """
    # Convert the quaternion into a 3x3 rotation matrix
    rot_mat = R.from_quat(convert_wxyz_to_xyzw(orientation)).as_matrix()
    
    # Extract the basis vectors for x, y, z from the rotation matrix
    # and scale them to draw lines
    x_axis = rot_mat[:, 0] * scale
    y_axis = rot_mat[:, 1] * scale
    z_axis = rot_mat[:, 2] * scale
    
    # Convert position to a numpy array if needed
    origin = np.array(position, dtype=float)
    
    # Create carb.Float3 objects for start and end points.
    start_points = [
        Float3(origin[0], origin[1], origin[2]),  # for x-axis
        Float3(origin[0], origin[1], origin[2]),  # for y-axis
        Float3(origin[0], origin[1], origin[2])   # for z-axis
    ]
    end_points = [
        Float3(*(origin + x_axis)),
        Float3(*(origin + y_axis)),
        Float3(*(origin + z_axis))
    ]

    # Create carb.ColorRgba objects for each axis.
    colors = [
        ColorRgba(1.0, 0.0, 0.0, 1.0),  # red for x-axis
        ColorRgba(0.0, 1.0, 0.0, 1.0),  # green for y-axis
        ColorRgba(0.0, 0.0, 1.0, 1.0)   # blue for z-axis
    ]

    # Specify line thicknesses as a list of floats.
    sizes = [2.0, 2.0, 2.0]

    # Draw the three axes.
    draw.draw_lines(start_points, end_points, colors, sizes)



def process_tf_message(tf_message: TFMessage):
    allowed_frames = {"world", "panda_link0", "panda_hand", "Cube"}
    # Extract the frames and transformations
    tf_data = []
    for transform in tf_message.transforms:
        parent_frame = transform.header.frame_id
        child_frame = transform.child_frame_id
        if parent_frame in allowed_frames and child_frame in allowed_frames:
            frame_data = {
                "parent_frame": parent_frame,
                "child_frame": child_frame,
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


def save_pointcloud(msg: PointCloud2, root) -> str:
    """
    Saves a ROS PointCloud2 (with x,y,z,[rgb]) to an ASCII PCD file.
    The file includes a voxel-based downsampling step to reduce size.

    The output path is:  root/Pcd_{pcd_counter}/pointcloud.pcd
    """
    # 1) Create the directory
    # dir_name = os.path.join(root, f"Pcd_{pcd_counter}")
    # os.makedirs(dir_name, exist_ok=True)

    # 2) Fixed filename: "pointcloud.pcd"
    file_path = os.path.join(root, "pointcloud.pcd")

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

def pointcloud2_to_o3d(pcd_ros):
    import sensor_msgs_py.point_cloud2 as pc2
    # Extract the structured points as a list of tuples (x, y, z)
    points_list = list(pc2.read_points(pcd_ros, field_names=("x", "y", "z"), skip_nans=True))
    
    # Convert the list of tuples to a standard NumPy array of floats
    points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float64)
    
    if points.shape[0] == 0:
        raise ValueError("Empty point cloud received!")

    # Create an Open3D PointCloud object and assign the points
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    
    return pcd_o3d

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
    print(f"your source frame is {source_frame}")
    print(f"your target frame is {target_frame}")
    # We assume tf_msg.transforms has the needed transform directly
    transform_stamped: TransformStamped = tf_buffer.lookup_transform(
        target_frame=target_frame,
        source_frame=source_frame,
        time=Time())  # "latest" transform
    print(f"your transform stamped is {transform_stamped}")
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
    print("Transformation complete")
    return cloud_out



def get_current_end_effector_pose() -> np.ndarray:
    """
    Return the current end-effector (grasp center, i.e. outside the robot) pose.
    return: (3,) np.ndarray: [x, y, z] position of the grasp center.
            (4,) np.ndarray: [w, x, y, z] orientation of the grasp center.
    """

    offset = np.array([0.0, 0.0, 0.1034])
    from omni.isaac.dynamic_control import _dynamic_control
    """Return the current end-effector (grasp center) position."""
    # Acquire dynamic control interface
    dc = _dynamic_control.acquire_dynamic_control_interface()
    # Get rigid body handles for the two gripper fingers
    ee_body = dc.get_rigid_body("/World/Franka/panda_hand")
    # Query the world poses of each finger
    ee_pose = dc.get_rigid_body_pose(ee_body)
    panda_hand_translation = ee_pose.p 
    panda_hand_quat = ee_pose.r

    # Create a Rotation object from the panda_hand quaternion.
    hand_rot = R.from_quat(panda_hand_quat)
    
    # Rotate the local offset into world coordinates.
    offset_world = hand_rot.apply(offset)
    
    # Compute the tool_center position.
    tool_center_translation = panda_hand_translation + offset_world
    
    # Since the relative orientation is identity ([0,0,0]), the tool_center's orientation
    # remains the same as the panda_hand's.
    tool_center_quat = [panda_hand_quat[3], panda_hand_quat[0], panda_hand_quat[1], panda_hand_quat[2]]
    
    return tool_center_translation, tool_center_quat


def get_upward_facing_marker(cube_prim_path):
    import omni
    """
    Determines which marker (attached to a cube) is the highest in world space.
    This marker indicates which face of the cube is currently facing upward.
    
    Args:
        cube_prim_path (str): The prim path of the cube.
    
    Returns:
        A tuple (marker_name, world_position) for the marker with the highest Z.
    """
    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath(cube_prim_path)
    if not cube_prim:
        print(f"Cube prim not found at {cube_prim_path}")
        return None

    def get_world_translation(prim):
        """
        Computes the world translation of a prim by computing its local-to-world
        transformation matrix and extracting its translation.
        """
        from pxr import UsdGeom, Gf, Usd

        time=Usd.TimeCode(0)
        xformable = UsdGeom.Xformable(prim)
        # Compute the local-to-world transform matrix at the default time.
        world_matrix = xformable.ComputeLocalToWorldTransform(time)
        return world_matrix.ExtractTranslation()

    upward_marker = None
    max_z = -float('inf')
    
    # Assume the markers are children of the cube with names starting with "marker_"
    for child in cube_prim.GetChildren():
        if not child.GetName().startswith("marker_"):
            continue

        # Use XformCommonAPI to get the marker's world translation.
        world_trans = get_world_translation(child)
        if world_trans[2] > max_z:
            max_z = world_trans[2]
            upward_marker = (child.GetName(), world_trans)

    
    return upward_marker






    




def view_pcd(file_path):
    # Load and visualize the point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

def merge_and_save_pointclouds(pcds_dict, tf_buffer, output_path="/home/chris/Chris/placement_ws/src/merged_pointcloud.pcd"):
    """
    Merges point clouds from multiple cameras after transforming them to the world frame.
    
    Args:
        pcds_dict (dict): Dictionary containing the three point clouds from different cameras
        tf_buffer (tf2_ros.Buffer): TF buffer containing transform information
        output_path (str): Path where to save the merged point cloud
        
    Returns:
        o3d.geometry.PointCloud: The merged point cloud in world frame
    """
    print("Now you are about to merge point clouds")
    import open3d as o3d
    
    # First transform all point clouds to the world frame
    transformed_pcds = []
    
    # Print available frames in TF buffer for debugging
    print(f"Available frames in TF buffer: {tf_buffer.all_frames_as_string()}")
    
    for name, pcd_msg in pcds_dict.items():
        if pcd_msg is None:
            print(f"Warning: {name} is None, skipping")
            continue
            
        # Get the source frame from the point cloud message
        source_frame = pcd_msg.header.frame_id
        print(f"your source frame is {source_frame}")
        print(f"your target frame is world")
        
        # Check if the transform exists before attempting to use it
        try:
            # First check if the transform exists
            transform = tf_buffer.lookup_transform("world", source_frame, rclpy.time.Time())
            print(f"your transform stamped is {transform}")
            
            # Transform the point cloud to world frame
            world_pcd_msg = transform_pointcloud_to_frame(pcd_msg, tf_buffer, target_frame="world")
            
            # Convert ROS PointCloud2 to Open3D point cloud
            o3d_pcd = pointcloud2_to_o3d(world_pcd_msg)
            
            # Downsample to reduce size and noise
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=0.005)
            
            transformed_pcds.append(o3d_pcd)
            print(f"Transformed {name} to world frame: {len(o3d_pcd.points)} points")
            print("Transformation complete")
        except Exception as e:
            print(f"Error transforming {name}: {e}")
    
    if not transformed_pcds:
        print("No valid point clouds to merge")
        # Create an empty point cloud to avoid errors
        empty_pcd = o3d.geometry.PointCloud()
        o3d.io.write_point_cloud(output_path, empty_pcd)
        print(f"Empty point cloud saved to {output_path}")
        return empty_pcd
    
    # Merge all point clouds
    merged_pcd = transformed_pcds[0]
    for pcd in transformed_pcds[1:]:
        merged_pcd += pcd
    
    # Remove statistical outliers
    try:
        filtered_pcd, _ = merged_pcd.remove_statistical_outlier(
            nb_neighbors=20, 
            std_ratio=1.5
        )
    except Exception as e:
        print(f"Error removing outliers: {e}")
        filtered_pcd = merged_pcd
    
    # Filter out points inside the robot's bounding box
    # The robot is at the origin in world frame
    box_center = [0.0, 0.0, 0.5]  # Center of robot in world frame
    box_size = [0.6, 0.6, 1.0]    # Size of box to remove robot points
    
    # Convert Open3D point cloud to numpy array for filtering
    points_np = np.asarray(filtered_pcd.points)
    
    # Create a mask for points outside the robot bounding box
    c = np.array(box_center)
    s = np.array(box_size) * 0.5  # half-extends
    box_min = c - s
    box_max = c + s
    
    # Find points outside the box
    outside_mask = ~np.all((points_np >= box_min) & (points_np <= box_max), axis=1)
    
    # Create a new point cloud with only the outside points
    filtered_points = points_np[outside_mask]
    filtered_pcd_no_robot = o3d.geometry.PointCloud()
    filtered_pcd_no_robot.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Save the merged and filtered point cloud
    try:
        o3d.io.write_point_cloud(output_path, filtered_pcd_no_robot)
        print(f"Merged point cloud saved to {output_path}")
    except Exception as e:
        print(f"Error saving point cloud: {e}")
    
    return filtered_pcd_no_robot

if __name__ == "__main__":
    # # # Example Usage
    # file_path = "/home/chris/Chris/placement_ws/src/data/pcd_0/pointcloud.pcd"
    # view_pcd(file_path)   
    # data_analysis("/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json")
    print(len(orientation_creation()))
    
    # Suppose you have an Nx3 cloud (world frame), e.g.:
    # cloud_world = np.random.rand(1000, 3) - 0.5  # random points in [-0.5, 0.5]^3

    # # Your robot bounding box info:
    # robot_center = [-0.04, 0.0, 0.0]
    # robot_size   = [0.24, 0.20, 0.01]

    # filtered_cloud = filter_robot_bounding_box(cloud_world, robot_center, robot_size)
    # print("Original cloud size:", len(cloud_world))
    # print("Filtered cloud size:", len(filtered_cloud))