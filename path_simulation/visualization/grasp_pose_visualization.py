#!/usr/bin/env python3
"""
ROS2 node to visualize a point cloud and grasp poses.

This node:
  - Loads a PCD file (your gelatin box) using Open3D.
  - Publishes the point cloud as a sensor_msgs/PointCloud2 message.
  - Reads grasp poses (in the object's local frame) defined as a dictionary.
  - Transforms each grasp pose by the object's pose (which can later be randomized in simulation).
  - Creates a MarkerArray to show the coordinate axes (X/red, Y/green, Z/blue) for each grasp.
  - Publishes the MarkerArray so they are visible in RViz.
  
You can then verify that when the object's pose changes in the simulator, the transformed grasp axes (visualized via MarkerArray)
appear correctly around the object.
"""
import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add the placement workspace root to the Python path so we can import ycb_simulation module
placement_ws_root = "/home/chris/Chris/placement_ws/src/placement_quality/"
if placement_ws_root not in sys.path:
    sys.path.insert(0, placement_ws_root)

from isaacsim import SimulationApp
import json

DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON   = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height":1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
    "physics_dt": 1.0/60.0,  # Physics timestep (60Hz)
    "rendering_dt": 1.0/30.0,  # Rendering timestep (30Hz)
}

simulation_app = SimulationApp(CONFIG)

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from transforms3d.euler import euler2quat
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from ycb_simulation.utils.helper import draw_frame, transform_relative_pose, local_transform
from scipy.spatial.transform import Rotation as R
import pyquaternion
ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"



class GraspVisualizer(Node):
    def __init__(self):
        super().__init__('grasp_visualizer')
        
        ################################
        ##### ROS parameters 
        ################################

        # Publishers for the point cloud and the grasp pose markers
        self.pc_pub = self.create_publisher(PointCloud2, 'point_cloud_viz', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'pose_viz', 10)

        # Timer to regularly publish messages (10Hz here)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Grasp Visualizer Node Initialized (ROS2)")

        # --- Load the PCD file ---
        # NOTE: Update the file path to your actual PCD file!
        pcd_file = '/home/chris/Chris/placement_ws/src/perfect_cube.pcd'
        grasps_file = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation/box_grasps.json"
        object_poses_file = "/home/chris/Chris/placement_ws/src/object_poses_box.json"
        
        # Load the original PCD data once
        self.original_pcd = o3d.io.read_point_cloud(pcd_file)
        
        self.all_object_poses = json.load(open(object_poses_file))
        
        # --- Read Grasp Poses in Object Local Frame ---
        with open(grasps_file, 'r') as f:
            self.grasp_poses = json.load(f)


        
        # --- Object Pose in the Global Frame ---
        # This pose is normally provided by your simulator (or set randomly). For now, we use the identity (no translation/rotation).
        # When the simulator sets a random object pose, update this dictionary.
        self.object_pose = self.all_object_poses[1]

        self.transformed_grasp_pose = None


        # Transform and publish with current pose
        self.pcd_msg = self.create_transformed_pc_msg(self.original_pcd, 
                                                     self.object_pose['position'],
                                                     self.object_pose['orientation_quat'])

        ################################
        ##### Isaac Sim parameters 
        ################################

         # Basic variables
        self.world = None
        self.scene = None
        self.object = None  # Changed to list of objects
        self.object_name = "009_gelatin_box.usd"  # Fixed to one object
        self.grasp_offset = [0, 0, -0.035] 
        self.box_dims = np.array([0.143, 0.0915, 0.051])



        # Store the last used pose for comparison
        self.last_transform_pose = self.object_pose.copy()

    def get_pregrasp(self,grasp_pos, grasp_quat, offset=0.15):
        # grasp_quat: [w, x, y, z]
        # scipy uses [x, y, z, w]!
        grasp_quat_xyzw = [grasp_quat[1], grasp_quat[2], grasp_quat[3], grasp_quat[0]]
        rot = R.from_quat(grasp_quat_xyzw)
        # Get approach direction (z-axis of gripper in world frame)
        approach_dir = rot.apply([0, 0, 1])  # [0, 0, 1] is z-axis
        # Compute pregrasp position (move BACK along approach vector)
        pregrasp_pos = np.array(grasp_pos) - offset * approach_dir
        return pregrasp_pos, grasp_quat  # Same orientation

    def setup(self):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.scene = self.world.scene
        self.scene.add_default_ground_plane()
        self.world.reset()
        self.object_pose["position"][0] = 0.2
        self.object_pose["position"][1] = -0.3
        self.create_object(self.object_pose["position"], 
                           self.object_pose["orientation_quat"])
        
        # grasp pose draw frame
        self.local_grasp_pose = [self.grasp_poses["1"]["position"], 
                                   self.grasp_poses["1"]["orientation_wxyz"]]
        

        self.world_grasp_pose = transform_relative_pose(self.local_grasp_pose, 
                                                        self.object_pose["position"], 
                                                        self.object_pose["orientation_quat"])
        
        
        self.transformed_grasp_pose = local_transform(self.world_grasp_pose, self.grasp_offset)
        draw_frame(self.transformed_grasp_pose[0], self.transformed_grasp_pose[1])

        pregrasp_pos, pregrasp_quat = self.get_pregrasp(self.transformed_grasp_pose[0], self.transformed_grasp_pose[1], offset=0.15)
        print(f"The pregrasp pose is: {pregrasp_pos}, {pregrasp_quat}")

        self.target.set_world_pose(pregrasp_pos, pregrasp_quat)



    def create_object(self, pos, quat):
        """Create one YCB object as rigid bodies with physics enabled"""

        # # Create the specified one object
        # prim_path = f"/World/Ycb_object"
        # name = f"ycb_object"
        
        # unique_prim_path = find_unique_string_name(
        #     initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        # )
        # unique_name = find_unique_string_name(
        #     initial_name=name, is_unique_fn=lambda x: not self.scene.object_exists(x)
        # )
        
        # usd_path = ROOT_PATH + self.object_name
        
        # # Add reference to stage and get the prim
        # object_prim = add_reference_to_stage(usd_path=usd_path, prim_path=unique_prim_path)
       
        #  # Use a wrapper to wrap the object in a XFormPrim.
        # object = XFormPrim(
        #     prim_path=unique_prim_path,
        #     name=unique_name,
        #     position=pos,
        #     orientation=quat,
        # )

        from omni.isaac.core.objects import DynamicCuboid   
        object = DynamicCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array(pos),
            orientation=quat,
            scale=self.box_dims.tolist(),  # [x, y, z]
            color=np.random.rand(3)  # Give each a random color if you want
        )

        self.scene.add(object)
        from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats
        # Add the target frame to the stage (for IK control)
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self.target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        self.target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        self.scene.add(self.target)

            



    def timer_callback(self):
        # Check if the object pose has changed
        if (self.object_pose['position'] != self.last_transform_pose['position'] or 
            self.object_pose['orientation_quat'] != self.last_transform_pose['orientation_quat']):
            
            # Update the point cloud with the new pose
            self.pcd_msg = self.create_transformed_pc_msg(self.original_pcd,
                                                         self.object_pose['position'],
                                                         self.object_pose['orientation_quat'])
            self.last_transform_pose = self.object_pose.copy()
        
        # Publish the current point cloud
        self.pc_pub.publish(self.pcd_msg)

        # Create a MarkerArray for all the grasp pose axes
        marker_array = MarkerArray()
        marker_id = 0

        # Convert the object's pose to a homogeneous transformation matrix
        T_obj = self.pose_to_matrix(self.object_pose['position'],
                                      self.object_pose['orientation_quat'])

        # Only visualize the current_grasp_pose
        if hasattr(self, 'current_grasp_pose'):
            # Get position and orientation from the current_grasp_pose
            grasp_position = self.current_grasp_pose[0]
            grasp_orientation = self.current_grasp_pose[1]
            
            # Create transformation matrix for the grasp pose
            T_grasp = self.pose_to_matrix(grasp_position, grasp_orientation)
            
            # Global grasp pose: T_global = T_object * T_grasp
            T_global = np.dot(T_obj, T_grasp)
            pos, quat = self.matrix_to_pose(T_global)
            
            # Update transformed_grasp_pose for debugging
            self.transformed_grasp_pose = [pos, quat]
            
            # Create visual markers (arrows) for this grasp pose
            markers = self.create_axes_markers(pos, quat, marker_id)
            print(f"The markers are: {pos}, {quat}")
            marker_array.markers.extend(markers)
        
        self.marker_pub.publish(marker_array)

    def pose_to_matrix(self, position, orientation):
        """
        Converts a pose (position + quaternion) into a 4x4 homogeneous transform.
        
        The orientation is assumed to be in [w, x, y, z] order.
        """
        T = np.eye(4)
        q = pyquaternion.Quaternion(orientation)
        T[:3, :3] = q.rotation_matrix
        T[:3, 3] = np.array(position)
        return T

    def matrix_to_pose(self, T):
        """
        Converts a 4x4 homogeneous transformation matrix into (position, quaternion)
        where quaternion is [w, x, y, z].
        """
        pos = T[:3, 3].tolist()
        q = pyquaternion.Quaternion(matrix=T[:3, :3])
        quat = [q.elements[0], q.elements[1], q.elements[2], q.elements[3]]
        return pos, quat

    def create_axes_markers(self, position, orientation, marker_id_start):
        """
        Creates three LINE_LIST markers to represent the coordinate axes at a given pose.
        The marker for the X axis will be red, for Y axis green, and for Z axis blue.
        Using LINE_LIST markers gives more consistent control over the line width.
        """
        markers = []
        # Parameters: adjust these until you get the desired look.
        axis_length = 0.05   # Length of each axis line (in meters)
        line_width = 0.002   # Thickness of the line

        # Convert orientation (quaternion) to a rotation matrix.
        q = pyquaternion.Quaternion(orientation)
        R = q.rotation_matrix

        # Each column of R is an axis direction.
        axes = {
            'x': (R[:, 0], (1.0, 0.0, 0.0)),
            'y': (R[:, 1], (0.0, 1.0, 0.0)),
            'z': (R[:, 2], (0.0, 0.0, 1.0))
        }

        for i, (axis_name, (axis_vec, color)) in enumerate(axes.items()):
            marker = Marker()
            marker.header.frame_id = "world"  # Adjust if your frame is different.
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "grasp_axes"
            marker.id = marker_id_start + i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            # The scale field for LINE_LIST is used as line width.
            marker.scale.x = line_width

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0

            # Create start point (grasp pose) and the corresponding end point along the axis.
            start_pt = Point()
            start_pt.x, start_pt.y, start_pt.z = position

            # Calculate the end point along the axis direction.
            end_vec = np.array(position) + axis_length * axis_vec
            end_pt = Point()
            end_pt.x, end_pt.y, end_pt.z = end_vec.tolist()

            # For LINE_LIST, the points list should contain a pair of points.
            marker.points = [start_pt, end_pt]
            markers.append(marker)
        return markers

    def create_transformed_pc_msg(self, pcd, position, orientation):
        """
        Creates a PointCloud2 message from Open3D point cloud, transformed by the given pose.
        """
        # Get points from the Open3D point cloud
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Create transformation matrix
        T = self.pose_to_matrix(position, orientation)
        
        # Apply transformation to all points (using homogeneous coordinates)
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_homog = np.hstack((points, ones))
        points_transformed = np.dot(points_homog, T.T)[:, :3]
        
        # Convert to ROS message (similar to your existing read_pcd_file method)
        # ... convert to PointCloud2 with the transformed points ...
        
        # For brevity, I'm not including the full color handling code here
        # But you would need to retain that from your existing read_pcd_file method
        
        # Create header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"
        
        # Create cloud message with transformed points
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points_transformed)
        return cloud_msg

    def read_pcd_file(self, file_path):
        """
        Reads a PCD file using Open3D and converts it to a sensor_msgs/PointCloud2 message.
        """
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # Check if the point cloud has colors. If so, convert them to a single uint32.
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors, dtype=np.float32) * 255).astype(np.uint8)
            # Pack r, g, b into a uint32
            rgb_uint32 = (colors[:, 0].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | colors[:, 2].astype(np.uint32)
            cloud_data = np.zeros(points.shape[0],
                                  dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            cloud_data['rgb'] = rgb_uint32

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
        else:
            cloud_data = np.zeros(points.shape[0],
                                  dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]

        # Create header (using the current node clock)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

        # Create the PointCloud2 message using the helper function
        cloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)
        return cloud_msg


def main(args=None):
    rclpy.init(args=args)
    node = GraspVisualizer()
    node.setup()
    try:
        while simulation_app.is_running():
            node.world.step(render=True)
            rclpy.spin_once(node, timeout_sec=0)
            print(f"The transformed grasp pose is: {node.transformed_grasp_pose}")
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
