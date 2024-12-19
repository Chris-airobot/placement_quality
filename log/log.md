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




