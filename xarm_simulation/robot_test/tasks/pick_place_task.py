from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np


class PickPlace(tasks.PickPlace):
    def __init__(
        self,
        name: str = "xarm_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=np.array([0.0515, 0.0515, 0.0515]),
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        #TODO: change the asset path here
        asset_path = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper/xarm7_with_gripper.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/UF_ROBOT")
        gripper = ParallelGripper(
            end_effector_prim_path="/UF_ROBOT/link_tcp",
            joint_prim_names=["drive_joint", "right_outer_knuckle_joint"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0.0085, -0.0085]),
            action_deltas=np.array([0.0085, -0.0085]))
        manipulator = SingleManipulator(prim_path="/UF_ROBOT",
                                        name="xarm_robot",
                                        end_effector_prim_path="/UF_ROBOT/link_tcp",
                                        gripper=gripper)
        joints_default_positions = np.zeros(13)
        joints_default_positions[5] =-1.5708
        joints_default_positions[7] = 0.0085
        joints_default_positions[11] = -0.0085
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator