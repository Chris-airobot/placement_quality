# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.sensor import Camera
from pyquaternion import Quaternion
from omni.isaac.franka import Franka
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import omni
from helper import *

class MyPickPlace(ABC, BaseTask):
    """[summary]

    Args:
        name (str): [description]
        cube_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        set_camera: bool = True,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_cube = None
        self._cube = None
        self._camera = None
        self._cube_initial_position = cube_initial_position
        self._cube_initial_orientation = cube_initial_orientation
        self._target_position = target_position
        self._cube_size = cube_size
        
        if self._cube_initial_position is None:
            self._cube_initial_position, self._cube_initial_orientation, self._target_position = task_randomization()
        if self._cube_size is None:
            self._cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        self._set_camera = set_camera

        return

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        cube_prim_path = find_unique_string_name(
            initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="cube", is_unique_fn=lambda x: not self.scene.object_exists(x))
        self._cube = scene.add(
            DynamicCuboid(
                name=cube_name,
                position=self._cube_initial_position,
                orientation=self._cube_initial_orientation,
                prim_path=cube_prim_path,
                scale=self._cube_size,
                size=1.0,
                color=np.array([0, 0, 1]),
            )
        )
        self.attach_face_markers(cube_prim_path)
        self._task_objects[self._cube.name] = self._cube

        if self._set_camera:
            camera_position = self._cube_initial_position + np.array([0.0, 0.0, 1.1])
            # camera_orientation = np.array([1, 0, 0, 0])
            camera_orientation = np.array([0.5, -0.5, 0.5, 0.5])

            self._camera:Camera = self.set_camera(camera_position, camera_orientation)
            scene.add(self._camera)
            self._task_objects[self._camera.name] = self._camera

        self._robot: Franka = self.set_robot()
        self._robot.set_enabled_self_collisions(True)
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return

    @abstractmethod
    def set_robot(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def set_camera(self) -> Camera:
        raise NotImplementedError
    

    def attach_face_markers(self, cube_prim_path, cube_size=1.0, offset=0.05):
        """
        Attaches a small sphere marker (with a label in its name) to each face of the cube.
        These markers are visual-only and will not be used for collisions.
        
        Args:
            cube_prim_path (str): The prim path of the cube.
            cube_size (float): The overall (uniform) size of the cube.
            offset (float): Extra offset to push the marker outwards from the face.
        """
        stage = omni.usd.get_context().get_stage()
        half = cube_size / 2.0

        # Define for each face: a translation and a label.
        markers = {
            "front":  (Gf.Vec3d(half + offset, 0, 0),        "1"),
            "back":   (Gf.Vec3d(-half - offset, 0, 0),       "2"),
            "left":   (Gf.Vec3d(0, half + offset, 0),        "3"),
            "right":  (Gf.Vec3d(0, -half - offset, 0),       "4"),
            "top":    (Gf.Vec3d(0, 0, half + offset),        "5"),
            "bottom": (Gf.Vec3d(0, 0, -half - offset),       "6")
        }
        
        # Get the cube prim.
        cube_prim = stage.GetPrimAtPath(cube_prim_path)
        if not cube_prim:
            print(f"Cube prim not found at {cube_prim_path}")
            return

        # Loop over each face.
        for face, (translation, label) in markers.items():
            # Create a marker Xform as a child of the cube.
            marker_path = cube_prim.GetPath().AppendChild(f"marker_{face}")
            marker = UsdGeom.Xform.Define(stage, marker_path)
            marker.AddTranslateOp().Set(translation)
            
            # Under the marker, create a small sphere.
            sphere_path = marker_path.AppendChild("sphere")
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.03)
            
            # (Optional) You could also set a custom attribute on the marker or sphere
            # to indicate the face number, or name the prim accordingly.
            # print(f"Attached marker for face '{face}' with label {label} at {translation}")



    def set_params(
        self,
        cube_position: Optional[np.ndarray] = None,
        cube_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
    ) -> None:
        if target_position is not None:
            self._target_position = target_position
        if cube_position is not None or cube_orientation is not None:
            self._cube.set_local_pose(translation=cube_position, orientation=cube_orientation)

            if self._set_camera:
                camera_position = cube_position + np.array([0, 0.0, 1.1])
                camera_orientation = np.array([0.5, -0.5, 0.5, 0.5])
                self._camera.set_local_pose(translation=camera_position, orientation=camera_orientation)

        return

    def get_params(self) -> dict:
        params_representation = dict()
        position, orientation = self._cube.get_local_pose()

        params_representation["cube_position"] = {"value": position, "modifiable": True}
        params_representation["cube_orientation"] = {"value": orientation, "modifiable": True}

        params_representation["target_position"] = {"value": self._target_position, "modifiable": True}
        params_representation["cube_name"] = {"value": self._cube.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}

        if self._set_camera:
            camera_postion, camera_orientation = self._camera.get_local_pose()
            params_representation["camera_name"] = {"value": self._camera.name, "modifiable": False}
            params_representation["camera_position"] = {"value": camera_postion, "modifiable": True}
            params_representation["camera_orientation"] = {"value": camera_orientation, "modifiable": True}

        return params_representation

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        cube_position, cube_orientation = self._cube.get_local_pose()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()

        observations = {
            self._cube.name: {
                "position": cube_position,
                "orientation": cube_orientation,
                "target_position": self._target_position,
            },
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            }
        }

        if self._set_camera:
            camera_position, camera_orientation = self._camera.get_local_pose()
            
            observations[self._camera.name] = {
                "position": camera_position,
                "orientation": camera_orientation,
            }
        return observations



        

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        return

    def post_reset(self) -> None:
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper

        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return

    def calculate_metrics(self) -> dict:
        """[summary]"""
        raise NotImplementedError

    def is_done(self) -> bool:
        """[summary]"""
        raise NotImplementedError
    

