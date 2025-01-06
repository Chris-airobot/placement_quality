#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import rclpy
from rclpy.node import Node
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import JointState
from grasp_interfaces.srv import MoveGripper
from std_msgs.msg import Empty
import math
import numpy as np
import time

GRIPPER_MAX = 0.04
GRIPPER_SPEED = 0.005

class SimInitialization(Node):
    def __init__(self):

        super().__init__("sim_robot")

        self.current_joints = None

        # Topic 
        self.open_gripper = self.create_publisher(Empty, "panda/open_gripper", 10)
        self.close_gripper = self.create_publisher(Empty, "panda/close_gripper", 10)
        self.joint_publisher = self.create_publisher(JointState, "panda/joint_command", 10)
    

        self.joint_subscriber = self.create_subscription(JointState, "panda/joint_states", self.joint_state_callback, 10)




    def move_gripper(self, value):
        """
        To move the gripper, the request value should be a number from [0, 1], 
        0 means fully close, 1 means fully open

        """
        self.current_joints = wait_for_message(JointState, self, "panda/joint_states")[1].position
        frequency = abs(math.floor((self.current_joints[7] - value*GRIPPER_MAX) / 0.005))
        msg = Empty()

        if value*GRIPPER_MAX < self.current_joints[7]: # The gripper should be closing
            for _ in range(0,frequency):
                self.close_gripper.publish(msg)
            
        elif value*GRIPPER_MAX > self.current_joints[7]: # The gripper should be opening
           for _ in range(0,frequency):
                self.open_gripper.publish(msg)

        return 

    def joint_state_callback(self, msg: JointState):
        self.current_joints = msg.position
        # print(self.current_joints)


def main(args=None):
    rclpy.init(args=args)

    ros2_publisher = SimInitialization()






    rclpy.spin(ros2_publisher)
    # Destroy the node explicitly
    ros2_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
