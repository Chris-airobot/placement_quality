from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from typing import Optional
import numpy as np

# Xarm7 with gripper
asset_path = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper/xarm7_with_gripper.usd"
prim_path = "/UF_ROBOT"
end_effector_prim_path = "/UF_ROBOT/link_tcp"
name = "xarm7_robot"
joint_prim_names = ["drive_joint", "right_outer_knuckle_joint"]
joints_default_positions = np.zeros(13)
joints_default_positions[5] =-1.5708
joints_default_positions[7] = 0.0085
joints_default_positions[11] = -0.0085

joint_opened_positions = np.array([0, 0])
joint_closed_positions = np.array([0.0085, -0.0085])


# # Gen3 lite 
# asset_path = "/home/chris/Chris/ros2_ws/src/ros2_kortex/kortex_description/robots/gen3_lite/gen3_lite.usd"
# prim_path = "/gen3_lite"
# end_effector_prim_path = "/gen3_lite/tool_frame"
# name = "gen3_lite_robot"
# joint_prim_names = ["left_finger_bottom_joint", "right_finger_bottom_joint"]
# joint_opened_positions = np.array([0, 0])
# joint_closed_positions = np.array([0.0085, -0.0085])

# joints_default_positions = np.zeros(10)


# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "xarm_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        #TODO: change this to the robot usd file.
        
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        gripper = ParallelGripper(
            end_effector_prim_path=end_effector_prim_path,
            joint_prim_names=joint_prim_names,
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=joint_closed_positions,)
        manipulator = SingleManipulator(prim_path=prim_path,
                                        name=name,
                                        end_effector_prim_path=end_effector_prim_path,
                                        gripper=gripper)
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator