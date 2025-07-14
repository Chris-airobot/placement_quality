from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.articulations import Articulation
from typing import Optional

# Xarm7 with gripper
urdf_path = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper.urdf"
robot_description_path = "/home/chris/Chris/placement_ws/src/placement_quality/xarm_simulation/robot_test/rmpflow/robot_descriptor.yaml"
xarm_end_effector_frame_name = "link_tcp"

# # Gen3 lite
# urdf_path = "/home/chris/Chris/ros2_ws/src/ros2_kortex/kortex_description/robots/gen3_lite.urdf"
# robot_description_path = "/home/chris/Chris/placement_ws/src/placement_quality/xarm_simulation/robot_test/rmpflow/gen3_lite.yaml"
# gen3_lite_end_effector_frame_name = "tool_frame"

class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        #TODO: change the config path
        self._kinematics = LulaKinematicsSolver(robot_description_path=robot_description_path,
                                                urdf_path=urdf_path)
        if end_effector_frame_name is None:
            end_effector_frame_name = xarm_end_effector_frame_name
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return