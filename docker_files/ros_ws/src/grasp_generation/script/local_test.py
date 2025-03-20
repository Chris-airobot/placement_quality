#!/usr/bin/python3
# Add this to change the namespace, otherwise the robot model cannot be found
 
from typing import List, Union
import socket
import json
import struct
import rospy
from geometry_msgs.msg import PoseArray, Vector3, Point, Pose, Quaternion
from gpd_ros.msg import GraspConfig, GraspConfigList
 
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import open3d as o3d
# from pathlib import Path
import numpy as np
import pyquaternion
 
HOST = ''        # Typically bind to all interfaces inside the container
PORT = 12345     # Choose any free port
 
# For gpd_ros, the topic that outputs the grasp poses are "clustered_grasps"
 
class Grasp:
    def __init__(self):
        rospy.init_node("grasp_generation")
        
        # Call the point cloud data service
        # self.left_scan = rospy.ServiceProxy("/left_scan", AssembleScans2)
 
        self.grasp_process_timeout = rospy.Duration(5)
        
        # Publish the point cloud data into topic that GPD package which receives the input
        self.grasps_pub = rospy.Publisher("/gpd_input", PointCloud2, queue_size=1)
        
        # Add a publisher for point cloud visualization in RViz
        self.cloud_viz_pub = rospy.Publisher("/point_cloud_viz", PointCloud2, queue_size=1)
        
        # Subscribe to the gpd topic and get the grasp gesture data, save it into the grasp_list through call_back function
        self.gpd_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.save_grasps)
        self.grasp_list: Union[List[GraspConfig], None] = None
        
        # Topic for visualization of the grasp pose
        self.pose_pub = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)
        
        # Store the calculated pose array
        self.saved_pose_array = None

        rospy.sleep(1)
 
    
 
    def save_grasps(self, data: GraspConfigList):
        self.grasp_list = data.grasps
 
        
    def vector3ToNumpy(self, vec: Vector3) -> np.ndarray:
        return np.array([vec.x, vec.y, vec.z])        
        
        
    def find_grasps(self, start_time):
        grasp_dict = dict()
        i = 0
        while (self.grasp_list is None and
               rospy.Time.now() - start_time < self.grasp_process_timeout):
            rospy.loginfo("waiting for left grasps")
            rospy.sleep(0.5)
 
            if self.grasp_list is None:
                rospy.logwarn("No grasps detected on pcl, restarting!!!")
                continue
        # Sort all grasps based on the gpd_ros's msg: GraspConfigList.grasps.score.data            
        self.grasp_list.sort(key=lambda x: x.score.data, reverse=True)
 
        # Create a PoseArray for visualization in RViz
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"  # Changed from "world" to "map" to match RViz default
        pose_array.header.stamp = rospy.Time.now()

        ### Pay attention here, the axis of the gripper
        for grasp in self.grasp_list:
            
            i+=1
 
            rot = np.zeros((3, 3))
            # grasp approach direction
            rot[:, 2] = self.vector3ToNumpy(grasp.approach)
            # hand closing direction
            rot[:, 0] = self.vector3ToNumpy(grasp.binormal)
            # hand axis
            rot[:, 1] = self.vector3ToNumpy(grasp.axis)
            
            
            # Turn the roll pitch yaw thing into the quaternion axis
            quat = pyquaternion.Quaternion(matrix=rot)
            pos: Point = grasp.position
            
            grasp_dict[i] = {
                "position":[pos.x, pos.y, pos.z],
                "orientation_wxyz": [quat[0], quat[1], quat[2], quat[3]]
            }
            
            # Create a Pose for RViz visualization
            grasp_pose = Pose(
                position=pos,
                orientation=Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0]),
            )
            pose_array.poses.append(grasp_pose)
        
        # Store the pose array for continuous publishing
        self.saved_pose_array = pose_array
        
        # Publish the pose array for visualization
        self.pose_pub.publish(pose_array)
        
        return grasp_dict
 
            
 
    def read_pcd_file(self, file_path):
        """
        Reads a PCD file using Open3D and converts it to a ROS PointCloud2 message.

        Parameters:
            file_path (str): Path to the PCD file.

        Returns:
            PointCloud2: The converted ROS PointCloud2 message.
        """
        # Read the PCD file using Open3D
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Extract points and colors if available
        points = np.asarray(pcd.points, dtype=np.float32)
        
        # Check if we have colors
        if pcd.has_colors():
            # Convert RGB [0-1] to uint32 format
            colors = np.asarray(pcd.colors, dtype=np.float32)
            colors_uint32 = (colors * 255).astype(np.uint8)
            rgb_uint32 = colors_uint32[:, 0] << 16 | colors_uint32[:, 1] << 8 | colors_uint32[:, 2]
            rgb_uint32 = rgb_uint32.astype(np.uint32)
            
            # Combine points and colors
            cloud_data = np.zeros(points.shape[0], dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('rgb', np.uint32)
            ])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            cloud_data['rgb'] = rgb_uint32
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1)
            ]
        else:
            # Points only
            cloud_data = np.zeros(points.shape[0], dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32)
            ])
            cloud_data['x'] = points[:, 0]
            cloud_data['y'] = points[:, 1]
            cloud_data['z'] = points[:, 2]
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]

        # Create header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"  # Changed from "world" to "map" to match RViz default

        # Create PointCloud2 message
        cloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)
        
        return cloud_msg
    
 
    
        
    def main(self):
 
        rospy.loginfo("ROS 1 Container Server: Listening on port %d", PORT)
        
        file_path = "/home/pointcloud_raw.pcd"
        # file_path = "/home/gpd/tutorials/krylon.pcd"
        
        # file_path = "/home/pointcloud.pcd"
        pcd = self.read_pcd_file(file_path)

        start_time = rospy.Time.now()
        # Process point cloud data to generate grasps
        self.grasp_list = None

        # Publish point cloud for GPD processing
        self.grasps_pub.publish(pcd)
        
        # Also publish the point cloud for visualization in RViz
        self.cloud_viz_pub.publish(pcd)
        
        rospy.loginfo("Published point cloud to /gpd_input and /point_cloud_viz")
        
        # Set up rate for continuous publishing
        rate = rospy.Rate(10)  # 10Hz refresh rate
        
        grasp_detected = False
        
        while not rospy.is_shutdown():
            # Always publish the point cloud for visualization
            self.cloud_viz_pub.publish(pcd)
            
            # Check if we have grasp poses and calculate them once
            if self.grasp_list is not None and not grasp_detected:
                rospy.loginfo(f"Received {len(self.grasp_list)} grasp poses")
                # Calculate grasp poses and store them
                response_dict = self.find_grasps(start_time)
                grasp_detected = True
                rospy.loginfo("Grasp poses published to /pose_viz for RViz visualization")
            
            # Continue to publish already calculated grasp poses
            if grasp_detected and self.saved_pose_array is not None:
                print("publishing grasp poses")
                # Update the timestamp for visualization
                self.saved_pose_array.header.stamp = rospy.Time.now()
                # Republish the saved pose array
                self.pose_pub.publish(self.saved_pose_array)
            
            # Make sure to sleep to maintain the publishing rate
            rate.sleep()
        
        rospy.loginfo("Node is shutting down")

                    

 
   
if __name__ == '__main__':
    # params_path = "/home/jason/catkin_ws/src/ur_grasping/src/box_grasping/configs/base_config.yaml"
    
    try:
        grasper = Grasp()
        grasper.main()
        # grasper.robot.move_gripper(0.3)
        #[51 47 20]
    except KeyboardInterrupt:
        pass