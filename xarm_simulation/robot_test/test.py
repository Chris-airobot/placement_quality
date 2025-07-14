# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from typing import Optional

import carb
import numpy as np
import omni.isaac.core.objects
import omni.isaac.motion_generation.interface_config_loader as interface_config_loader
from omni.isaac.franka import Franka
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.motion_generation import ArticulationTrajectory
from omni.isaac.motion_generation.lula import RRT
from omni.isaac.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from omni.isaac.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from omni.isaac.motion_generation.path_planning_interface import PathPlanner
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
class FrankaRrtExample(BaseController):
    def __init__(self,
            name: str,
            robot_articulation: Articulation,
            physics_dt: float = 1 / 60.0,
            rrt_interpolation_max_dist: float = 0.01,
        ):
        BaseController.__init__(self, name)

        path_to_robot_usd = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper/xarm7_with_gripper.usd"
        robot_prim_path = "/UF_ROBOT"
        robot_description_path = "/home/chris/Chris/placement_ws/src/placement_quality/xarm_simulation/robot_test/rmpflow/robot_descriptor.yaml"
        urdf_path = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper.urdf"
        end_effector_frame_name = "link_tcp"
        config_path = "/home/chris/Chris/placement_ws/src/placement_quality/xarm_simulation/robot_test/rmpflow/rrt_planner_config.yaml"

        #Initialize an RRT object
        self._rrt = RRT(
            robot_description_path = robot_description_path,
            urdf_path = urdf_path,
            rrt_config_path = config_path,
            end_effector_frame_name = end_effector_frame_name
        )


        self._rrt.set_max_iterations(5000)




        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        self._target = XFormPrim("/World/target", scale=[.04,.04,.04])
        self._target.set_default_state(np.array([.45,.5,.7]),euler_angles_to_quats([3*np.pi/4, 0, np.pi]))

        
        

        # Create a trajectory generator to convert RRT cspace waypoints to trajectories
        self._cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path
        )

        # It is important that the Robot Description File includes optional Jerk and Acceleration limits so that the generated trajectory
        # can be followed closely by the simulated robot Articulation
        for i in range(len(self._rrt.get_active_joints())):
            assert self._cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limit(i)
            assert self._cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limit(i)
        
        # Create a visualizer to visualize the path planner
        self._path_planner_visualizer = PathPlannerVisualizer(self._articulation, self._rrt)
        self._path_planner: RRT = self._path_planner_visualizer.get_path_planner()

        self._robot = self._path_planner_visualizer.get_robot_articulation() 

        self._last_solution = None
        self._action_sequence = None

        self._physics_dt = physics_dt
        self._rrt_interpolation_max_dist = rrt_interpolation_max_dist
        self.ik_check = None



  

    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        # This example uses the LulaCSpaceTrajectoryGenerator to convert RRT waypoints to a cspace trajectory.
        # In general this is not theoretically guaranteed to work since the trajectory generator uses spline-based
        # interpolation and RRT only guarantees that the cspace position of the robot can be linearly interpolated between
        # waypoints.  For this example, we verified experimentally that a dense interpolation of cspace waypoints with a maximum
        # l2 norm of .01 between waypoints leads to a good enough approximation of the RRT path by the trajectory generator.

        interpolated_path = self._path_planner_visualizer.interpolate_path(rrt_plan, self._rrt_interpolation_max_dist)
        trajectory = self._cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)

        return art_trajectory.get_action_sequence()

    def _make_new_plan(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ):
        self._path_planner.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._path_planner.update_world()

        # In the original script, they created a new PathPlannerVisualizer object here, not sure why
        # path_planner_visualizer = PathPlannerVisualizer(self._robot, self._path_planner)

        active_joints = self._path_planner_visualizer.get_active_joints_subset()
        if self._last_solution is None:
            start_pos = active_joints.get_joint_positions()
        else:
            start_pos = self._last_solution

        self._path_planner.set_max_iterations(5000)
        self._rrt_plan = self._path_planner.compute_path(start_pos, np.array([]))

        if self._rrt_plan is None or len(self._rrt_plan) <= 1:
            carb.log_warn("No plan could be generated to target pose: " + str(target_end_effector_position))
            self._action_sequence = []
            return False


        self._action_sequence = self._convert_rrt_plan_to_trajectory(self._rrt_plan)
        self._last_solution = self._action_sequence[-1].joint_positions
        return True
    def forward(
        self, 
        target_end_effector_position: np.ndarray, 
        target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> ArticulationAction:
        if self._action_sequence is None:
            # This will only happen the first time the forward function is used
            self.ik_check = self._make_new_plan(target_end_effector_position, target_end_effector_orientation)

        if len(self._action_sequence) == 0:
            # The plan is completed; return null action to remain in place
            return ArticulationAction()

        if len(self._action_sequence) == 1:
            final_positions = self._action_sequence[0].joint_positions
            return ArticulationAction(
                final_positions, np.zeros_like(final_positions), joint_indices=self._action_sequence[0].joint_indices
            )

        return self._action_sequence.pop(0)

    def add_obstacle(self, obstacle: omni.isaac.core.objects, static: bool = False) -> None:
        self._path_planner.add_obstacle(obstacle, static)

    def remove_obstacle(self, obstacle: omni.isaac.core.objects) -> None:
        self._path_planner.remove_obstacle(obstacle)

    def reset(self) -> None:
        # PathPlannerController will make one plan per reset
        self._path_planner.reset()
        self._action_sequence = None
        self._last_solution = None

    def get_path_planner(self) -> PathPlanner:
        return self._path_planner
    

    def is_done(self) -> bool:
        if len(self._action_sequence) <= 1:
            return True
        else:
            return False

