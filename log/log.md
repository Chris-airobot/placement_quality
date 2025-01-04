## Dec 15 2024
Set up the environment, for Isaac Sim, Isaac Lab and Ufactory Arm robot


## Dec 16 2024
Learning Isaac Sim, should be ready as soon as possible


## Dec 17 2024
Input robot to the Isaac Sim:
1. Find the top level urdf.xacro file
2. Then run the command below to generate urdf file

``` bash
ros2 run xacro xacro xarm_device.urdf.xacro -o xarm7.urdf 
``` 
3. You can also check the file by 
``` bash
check_urdf your_robot.urdf
``` 


4. After that runthe code below, it has four arguments, the first one is the urdf file path, the second one is the output file path, third and forth seems default ones 

``` bash
./isaaclab.sh -p source/standalone/tools/convert_urdf.py /home/chris/Chris/placement_ws/src/xarm_ros2/xarm_description/urdf/xarm7.urdf source/extensions/omni.isaac.lab_assets/data/Robots/ANYbotics/xarm7.usd --merge-joints --make-instanceable 
```

## Jan 2 2025
A lot of happened basically, but changes are:
- Plan to use the **panda robot rather than xarm in simulation**, potentially believe that the robot in simulation should not matter too much for the model, and importing xarm to the simulator producing too many troubles


## Jan 3 2025
Determined what services and topics should be implemented in Isaacsim for the project

## Jan 4 2025
### Topics that I should build

| Name        | Description | Message Type | Input 
| ----------- | ----------- | ----------- | ----------- |
| /panda/joint_states      | Continuously provides the arm’s joint angles, velocities, and efforts.       | sensor_msgs/msg/JointState | Joint state values 
| /panda/gripper_state   | Provides real-time updates about the gripper’s state        | Not sure | Gripper values

### Services that I should build
| Name        | Description | Request | Response 
| ----------- | ----------- | ----------- | ----------- |
| /panda/compute_grasps      | compute grasp positions for a known object pose.      | Object pose | A list of candidate grasp poses | 
/panda/plan_joint_trajectory   | Perform path planning in a single call        | end-effector pose | A planned trajectory (list of waypoints) + success/failure| 
/panda/execute_joint_trajectory   | Actually move the robot along the planned path        | A trajectory | Success/failure once execution completes| 
/panda/move_gripper   | open the gripper        | a target width | Success/failure once execution completes
/panda/move_to_pose   | planning + execution        | Cartesian pose of the end-effector | Success/failure once execution completes
/reset_scene   | restore the simulation        | None | Success/failure once execution completes
/spawn_object   | Dynamically add objects for grasp testing        | None | Success/failure once execution completes
/save_data   | Store experiment outcomes        | None | Success/failure once execution completes