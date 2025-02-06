#!/usr/bin/python3
# Add this to change the namespace, otherwise the robot model cannot be found
 
from typing import List, Union
import socket
import json
import struct
import rospy
from geometry_msgs.msg import PoseArray, Vector3, Point
from gpd_ros.msg import GraspConfig, GraspConfigList
 
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import open3d as o3d
# from pathlib import Path
import numpy as np
import pyquaternion
 
HOST = ''        # Typically bind to all interfaces inside the container
PORT = 12341     # Choose any free port
 
# For gpd_ros, the topic that outputs the grasp poses are "clustered_grasps"
 
class Grasp:
    def __init__(self):
        rospy.init_node("grasp_generation")
        
        # Call the point cloud data service
        # self.left_scan = rospy.ServiceProxy("/left_scan", AssembleScans2)
 
        self.grasp_process_timeout = rospy.Duration(5)
        
        # Publish the point cloud data into topic that GPD package which receives the input
        self.grasps_pub = rospy.Publisher("/gpd_input", PointCloud2, queue_size=1)
        
        # Subscribe to the gpd topic and get the grasp gesture data, save it into the grasp_list through call_back function
        self.gpd_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.save_grasps)
        self.grasp_list: Union[List[GraspConfig], None] = None
        
        # Topic for visualization of the grasp pose
        self.pose_pub = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)
 

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
            
            # grasp_pose = Pose(
            #     position=pos,
            #     orientation=Quaternion(x=quat[1], y=quat[2], z=quat[3], w=quat[0]),
            # )
        
        return grasp_dict
 
            
 
    def read_pcd_file(self, file_path):
        """
        Reads an ASCII PCD file with fields [x y z rgb] and converts it to a ROS PointCloud2 message.

        Parameters:
            file_path (str): Path to the PCD file.

        Returns:
            PointCloud2: The converted ROS PointCloud2 message.
        """
        # 1. Read the PCD file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 2. Parse header to find where the data starts
        data_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('DATA'):
                data_start_idx = i + 1
                break

        # 3. Parse data lines into a list of points
        points = []
        for line in lines[data_start_idx:]:
            values = line.strip().split()
            if len(values) < 4:
                continue  # Skip incomplete lines
            x, y, z = map(float, values[:3])
            rgb = int(float(values[3]))
            points.append([x, y, z, rgb])

        # Convert to NumPy array
        points_np = np.array(points, dtype=np.float32)  # Shape: (N, 4)

        # 4. Define PointCloud2 fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),  # 'rgb' as UINT32
        ]

        # 5. Create the header
        header = Header()
        header.stamp = rospy.Time.now()  # Requires rospy to be initialized
        header.frame_id = "map"  # Change to your appropriate TF frame

        # 6. Create the PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = points_np.shape[0]
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = False
        cloud_msg.fields = fields
        cloud_msg.point_step = 16  # 4 fields * 4 bytes each
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.data = points_np.tobytes()

        return cloud_msg
    
 
    
        
    def main(self):
 
        rospy.loginfo("ROS 1 Container Server: Listening on port %d", PORT)
        
            
            
        file_path = "/home/pointcloud.pcd"
        # file_path = "/home/gpd/tutorials/krylon.pcd"
        
        # file_path = "/home/pointcloud.pcd"
        pcd = self.read_pcd_file(file_path)

        start_time = rospy.Time.now()
        # Process point cloud data to generate grasps
        self.grasp_list = None

        self.grasps_pub.publish(pcd)
        
        # Build the response
        response_dict = self.find_grasps(start_time)
        rospy.spin()

                    

 
   
if __name__ == '__main__':
    # params_path = "/home/jason/catkin_ws/src/ur_grasping/src/box_grasping/configs/base_config.yaml"
    
    try:
        grasper = Grasp()
        grasper.main()
        # grasper.robot.move_gripper(0.3)
        #[51 47 20]
    except KeyboardInterrupt:
        pass