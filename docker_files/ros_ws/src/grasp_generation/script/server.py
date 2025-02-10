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
        
        # Subscribe to the gpd topic and get the grasp gesture data, save it into the grasp_list through call_back function
        self.gpd_sub = rospy.Subscriber("/detect_grasps/clustered_grasps", GraspConfigList, self.save_grasps)
        self.grasp_list: Union[List[GraspConfig], None] = None
        
        # Topic for visualization of the grasp pose
        self.pose_pub = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)
 
        # Below are TCP connections
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind((HOST, PORT))
        self.server_sock.listen(1)
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
 
            
 
    def msg_to_pcd2(self, msg):
        """
        Converts JSON-friendly point cloud data to a PointCloud2 ROS message.
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"  # Replace with your frame ID
 
        # Extract points from JSON
        points = [
            [point['x'], point['y'], point['z'], point['rgb']]
            for point in msg
        ]
 
        # Define PointCloud2 fields
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
 
        # Create PointCloud2 message
        pc2_msg = point_cloud2.create_cloud(header, fields, points)
        return pc2_msg
    
 
    
        
    def main(self):
 
        rospy.loginfo("ROS 1 Container Server: Listening on port %d", PORT)
        try:
            while not rospy.is_shutdown():
                conn, addr = self.server_sock.accept()
                rospy.loginfo("Connected by %s", addr)
                with conn:
                    data_len = struct.unpack('>I', conn.recv(4))[0]
                    data = b""
                    while len(data) < data_len:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
 
 
                    # Decode and parse the incoming JSON
                    incoming_data = json.loads(data.decode('utf-8'))
 
                    pcd = self.msg_to_pcd2(incoming_data["pointcloud"])
 
                    start_time = rospy.Time.now()
                    # Process point cloud data to generate grasps
                    self.grasp_list = None
                    self.grasps_pub.publish(pcd)
                    
                    # Build the response
                    response_dict = self.find_grasps(start_time)
                    response_str = json.dumps(response_dict)
                    send_data = response_str.encode('utf-8')
                    send_len = struct.pack('>I', len(send_data))
 
                    # Send back the response
                    conn.sendall(send_len + send_data)
                    rospy.loginfo("Sent grasp data to client.")
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.server_sock.close()
 
   
if __name__ == '__main__':
    # params_path = "/home/jason/catkin_ws/src/ur_grasping/src/box_grasping/configs/base_config.yaml"
    
    try:
        grasper = Grasp()
        grasper.main()
        # grasper.robot.move_gripper(0.3)
        #[51 47 20]
    except KeyboardInterrupt:
        pass