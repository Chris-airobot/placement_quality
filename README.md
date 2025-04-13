# Grasping for placement project
## Project High-Level Pipeline
1. **Generate Candidate Grasps:** Use Dex-Net or another grasp planner.
2. **For Each Grasp, Simulate Placements:** Randomize placement strategies and record outcomes.
3. **Collect Data & Outcomes:** Store object descriptors (weight, geometry via TSDF), and placement results (pose shifts, success/failure, etc.). <span style="color:blue"> Maybe we can also find the enegry consumption?</span>.
4. **Aggregate & Model a Distribution:** From multiple placement attempts per grasp, estimate a distribution over outcomes.
5. **Train a Predictive Model:** Input: (object representation, target pose, grasp parameters) → Output: expected placement quality.
6. **Runtime Use:** For a new object, run grasp generation and use the trained model to select the best grasp.

A few notes:
- Maybe use some common shapes that could represent most objects
- Idea of the second step,:
    - After grasping the object, define a range of final poses where the object might be placed
    - For each of the final pose, use some common planner to place down the object, the strategy could be dropping the object from random distance, or directly touching the ground
- Idea of the forth step:
    - $Q = S_{success} ∗ (w_1 ∗ D + w_2 ∗ T)$ where $T$ is time taken, and $D$ is the pose shift
- For Sim-to-real gap, two potential methods:
    - **Domain randomization:** randomly vary fricition, angle of the ground, and add noises to sensors
    - **Real data fine-tuning:** After the model has been trained in simulation, also collects a small dataset in reality to fine-tune the model

## Discussions:
1. Approaches for grasping and placement should be the same for training and testing?
 - Yes. But it feels like a bit limited considering there are infinite ways to grasping and placing
2. Steps:
    - Random samples the object -> Use grasp planner to generate a bunch of grasps -> Place it down for a specific goal pose 
    - Input of the model: initial pose of the object, grasp pose, and the final pose of the object
    - Start with fixing the orientation of the object, simply varying the positions

3. When generating the goal pose, some poses may be not be feasible (like a donut is vertically placed). One way is to randomly drop the object to find discrete feasible poses.

4. Consider the absolute poses?



## Simulator Choices:
The simulator plan to use is IsaacSim

## Robot Connection 
The robot plan to use is UFactory xarm 7, for linux system, set up the wire connection with:
- Address: 192.168.1.12
- Netmask: 255.255.255.0
- Gateway: 192.168.1.1
- DNS: 192.168.1.1

For the robot in the lab, IP address is **192.168.1.209**, i.e. Type "192.168.1.209:18333" to use UFactory Studio



## Troubleshooting Guides
1. **Colcon build** might explode the ram so the PC freezes
- Solutions: **export MAKEFLAGS="-j 1" colcon build --executor sequential**, this command will use one core and one package to build the workspace

2. Ros2 extension may not be able to work when the simulator starts
- Solutions: **make sure to configure the ROS for Isaac sim**, basically use the fastdds.xml

3. The configuration may still not work, i.e. ros2 command pops up an error saying some configuration errors even after completing step 2
- Solutions: **open up one terminal that set the environment first (export FASTRTPS_DEFAULT_PROFILES_FILE=...), then start the isaac sim.** No idea why it's not working even I put that in the *extra arg* of isaac sim 
4. Docker:
- Docker image build
    - docker build --build-arg GITHUB_TOKEN=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 -t my_isaac_ros_image .


- Docker container command
    - docker run --name my_isaac_ros_container \
        --runtime=nvidia --gpus all \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        --network=host \
        -v /home/chris/Chris/placement_ws/src/data:/home/chris/Chris/placement_ws/src/data:rw \
        -it --entrypoint bash my_isaac_ros_image

- Docker for grasping
    - docker build -t my_ros_noetic_image .
    - docker run -it   -v /home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation:/home/ros_ws/src/grasp_generation:rw   my_ros_noetic_image


5. If Vscode does not recognize some certain packages
- Solutions:
    - import the package, then print(package.\_\_file__)

6. Cannot Plot in the docker for gpd:
- Solutions:
  - xhost +local:
  - docker run --gpus all -it \
        -v /home/chris/Chris/placement_ws/src/placement_quality/docker_files:/home \
        --name ros1 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        my-ros1-x11:latest

7. Cluster usage:
- Connections:
    - ssh tianyuanl@gandalf-dev.it.deakin.edu.au