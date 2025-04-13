import os, sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from isaacsim import SimulationApp

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
}

simulation_app = SimulationApp(CONFIG)

import os
import rclpy
import datetime 
from omni.isaac.core.utils import extensions
from path_simulation.simulation.logger import Logger
from path_simulation.simulation.simulator import Simulator
from ycb_simulation.utils.helper import draw_frame


# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")



def main():
    env = Simulator()
    env.start()
    logger = Logger(env, DIR_PATH)

    while simulation_app.is_running():
        # Handle simulation step
        try:
            env.world.step(render=True)
        except:
            print("Something wrong with during the step function")
            env.reset()
            continue
        # Process ROS callbacks
        rclpy.spin_once(env.sim_subscriber, timeout_sec=0)
        
        # Wait for initial TF data
        if env.sim_subscriber.latest_tf is None and env.state != "INIT":
            continue
    
        # State machine for simulation flow
        if env.state == "SETUP":
            env.setup()
                    
            if env.wait_for_tf_buffer():
                env.state = "GRASP"


        elif env.state == "GRASP":

            # Start logging for the grasp attempt
            logger.log_grasping()
            
            observations = env.world.get_observations()
            task_params = env.task.get_params()
            draw_frame(env.current_grasp_pose[0], env.current_grasp_pose[1])

            # Perform grasping
            try:
                # Generate actions for the robot
                actions = env.controller.forward(
                    target_end_effector_position=env.current_grasp_pose[0],
                    target_end_effector_orientation=env.current_grasp_pose[1],
                )

                kps, kds = env.task.get_custom_gains()
                env.articulation_controller.set_gains(kps, kds)
                # Apply the actions to the robot
                env.articulation_controller.apply_action(actions)
                    
                # Check if we've completed all phases of the planner
                if env.controller.is_done():
                    print("----------------- Controller Plan Complete -----------------")
                    # Record the successful trajectory
                    logger.record_grasping(message="SUCCESS")             
                    # Increment the placement counter and move to replay state
                    env.placement_counter += 1
                    # Reset for next placement
                    env.state = "REPLAY"
            
            except Exception as e:
                print("Something went wrong with the movement, recording data and restarting")
                logger.record_grasping(message="ERROR")
                env.reset()
                continue
                
        elif env.state == "REPLAY":
            # Start the replay of the grasping trajectory
            logger.replay_grasping()        
            if logger.data_logger.get_num_of_data_frames() >= 600:
                env.reset()
                continue
            # Move to place state - the replay will continue in the background
            env.state = "MOVE"
            env.start_logging = True

            # Generate random placement orientation
            ...
        elif env.state == "PLACE":
            # Wait for replay to finish if planner is not at event 4 (post-grasp)
            if env.controller.get_current_event() < 4:
                continue
            # Start logging for the placement if not already logging
            logger.log_grasping()
            observations = env.world.get_observations()
            task_params = env.task.get_params()
            # Replay is finished, proceed with placement
            try:                
                # Generate actions for placement
                actions = env.controller.forward(
                    target_end_effector_position=task_params["object_target_position"]["value"],
                    target_end_effector_orientation=task_params["object_target_orientation"]["value"],
                )
                
                # Apply the actions to the robot
                env.articulation_controller.apply_action(actions)
                                        
                # Check if the entire planning sequence is complete
                if env.controller.is_done():
                    print(f"----------------- Placement {env.placement_counter} complete ----------------- \n\n")
                    # Record the placement data
                    logger.record_grasping(message="SUCCESS")
                    # Increment placement counter
                    env.placement_counter += 1
                    
                    # Check if we've reached the maximum placements
                    if env.placement_counter >= 200:
                        print(f"Maximum placements reached for grasp {env.grasp_counter}")
                        # Go with next grasp
                        ...
                    else:
                        # Reset for next placement with the same grasp
                        env.soft_reset()
                
            except Exception as e:
                print(f"Error during placement: {e}")
                # if "Found zero norm quaternions in `quat`" in str(e):
                #     env.reset()
                #     continue
                env.soft_reset()
                continue
                
        

    # Cleanup when simulation ends
    simulation_app.close()

if __name__ == "__main__":
    main()