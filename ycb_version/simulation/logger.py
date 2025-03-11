from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
from helper import SimSubscriber
from ycb_collection import YcbCollection
from omni.isaac.core.scenes import Scene
import numpy as np
from omni.isaac.core.utils.types import ArticulationAction
import os
import glob
import asyncio



class YcbLogger:
    def __init__(self, sim_env: YcbCollection, dir_path: str):
        self.env = sim_env
        self.world = self.env.world
        self.task = self.env.task
        self.data_logger = self.world.get_data_logger()
        self.planner = self.env.planner
        self.controller = self.env.controller
        self.contact_sensors = self.env.contact_sensors
        self.dir_path = dir_path

    def _on_logging_event(self, val, tf_node: SimSubscriber):
            print(f"----------------- Grasping {self.env.grasp_counter} Placement {self.env.placement_counter} Start -----------------")

            if not self.world.get_data_logger().is_started():
                self.task_params = self.task.get_params()
                robot_name = self.task_params["robot_name"]["value"]
                cube_name = self.task_params["cube_name"]["value"]
                target_position = self.task_params["target_position"]["value"]
                if tf_node.latest_tf is not None:
                    tf_data = process_tf_message(tf_node.latest_tf)
                else:
                    tf_data = None
                # A data logging function is called at every time step index if the data logger is started already.
                # We define the function here. The tasks and scene are passed to this function when called.
                def frame_logging_func(tasks, scene: Scene):
                    cube_position, cube_orientation =  scene.get_object(cube_name).get_world_pose()
                    ee_position, ee_orientation =  scene.get_object(robot_name).end_effector.get_world_pose()
                    # surface = surface_detection(quat_to_euler_angles(cube_orientation))
                    surface, _ = get_upward_facing_marker("/World/Cube")
                    # camera_position, camera_orientation =  scene.get_object(camera_name).get_world_pose()

                    return {
                        "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),# save data as lists since its a json file.
                        "applied_joint_positions": scene.get_object(robot_name).get_applied_action().joint_positions.tolist(),
                        "ee_position": ee_position.tolist(),
                        "ee_orientation": ee_orientation.tolist(),
                        "target_position": target_position.tolist(), # Cube target position
                        "cube_position": cube_position.tolist(),
                        "cube_orientation": cube_orientation.tolist(),
                        "stage": self.planner.get_current_event(),
                        "surface": surface,
                        "ee_target_orientation":self.env.ee_placement_orientation.tolist(),
                        "tf": tf_data,
                        "contact_info": self.env.contact_message,
                        "object_in_ground": self.env.object_collision,
                        "object_grasped": self.env.object_grasped
                    }

                self.data_logger.add_data_frame_logging_func(frame_logging_func) # adds the function to be called at each physics time step.
            if val:
                self.data_logger.start() # starts the data logging
            else:
                self.data_logger.pause()
            return



    def _on_save_data_event(self, log_path):
        print("----------------- Saving Start -----------------\n")
        self.data_logger.save(log_path=log_path) # Saves the collected data to the json file specified.

        print(f"----------------- Successfully saved it to {log_path} -----------------\n")
        self.data_logger.reset() # Resets the DataLogger internal state so that another set of data can be collected and saved separately.
        return


    # This is for replying the whole scene
    async def _on_replay_scene_event_async(self, data_file):
            self.data_logger.load(log_path=data_file)

            await self.world.play_async()
            self.world.add_physics_callback("replay_scene", self._on_replay_scene_step)
            return 


    def _on_replay_scene_step(self, step_size):
        if self.world.current_time_step_index < self.data_logger.get_num_of_data_frames():
            cube_name = self.task_params["cube_name"]["value"]
            data_frame = self.data_logger.get_data_frame(data_frame_index=self.world.current_time_step_index)
            self.controller.apply_action(
                ArticulationAction(joint_positions=data_frame.data["applied_joint_positions"])
            )
            # Sets the world position of the goal cube to the same recoded position
            self.world.scene.get_object(cube_name).set_world_pose(
                position=np.array(data_frame.data["cube_position"]),
                orientation=np.array(data_frame.data["cube_orientation"])
            )

        elif self.world.current_time_step_index == self.data_logger.get_num_of_data_frames():
            print("----------------- Replay Finished, now moving to Placement Phase -----------------\n")
            self.replay_finished = True
            self.planner._event = 4
            self.world.remove_physics_callback("replay_scene")
        return
    
    def replay_grasping(self):
        print(f"----------------- Replaying Grasping {self.env.grasp_counter} ----------------- \n")

        file_path = self.dir_path + f"Grasping_{self.env.grasp_counter}/Grasping.json"

        # If the replay data does not exist, create one
        if not os.path.exists(file_path):
            file_pattern = os.path.join(self.dir_path, f"Grasping_{self.env.grasp_counter}/Placement_*.json")
            file_list = glob.glob(file_pattern)

            extract_grasping(file_list[0])
        asyncio.ensure_future(self._on_replay_scene_event_async(file_path))
        return True