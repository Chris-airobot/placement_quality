# Combined pick and place task with camera functionality
from abc import ABC
from typing import Optional
import numpy as np
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
from omni.isaac.franka import Franka
import omni
import random
import carb
import omni.graph.core as og
import omni.syntheticdata
from pxr import Usd, UsdGeom, Gf, UsdPhysics
# Assuming these helper functions are defined in your project
from helper import pose_init, euler2quat, get_current_end_effector_pose
from scipy.spatial.transform import Rotation as R


# get_assets_root_path = "omniverse://localhost/NVIDIA/Assets/Isaac/4.2"
# /Isaac/Props/YCB/Axis_Aligned/ycb_object_0000.usd
ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"
with open("/home/chris/Chris/placement_ws/src/placement_quality/ycb_version/ycb_list.txt", "r") as f:
    usd_files = [line.strip() for line in f if line.strip()]

class YcbTask(BaseTask, ABC):
    """
    Combined class for a pick and place task that includes camera setup.
    
    This class merges functionality from both the base task and the PickPlaceCamera subclass.
    It creates the objects, sets up the robot and camera, and provides methods to update the task.
    """
    def __init__(
        self,
        name: str = "franka_pick_place",
        object_initial_position: Optional[np.ndarray] = None,
        object_initial_orientation: Optional[np.ndarray] = None,
        object_target_position: Optional[np.ndarray] = None,
        object_target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        set_camera: bool = True,
    ) -> None:
        # Initialize the BaseTask with the given name and offset.
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self._target_object = None
        self._object = None
        self._object_final = None
        self._cameras = None
        self._set_camera = set_camera
        self._object_initial_position = object_initial_position
        self._object_initial_orientation = object_initial_orientation
        self._object_target_position = object_target_position
        self._object_target_orientation = object_target_orientation
        # self._dc_interface = _dynamic_control.acquire_dynamic_control_interface()

        # Initialize object poses if not provided.
        if self._object_initial_position is None:
            self.object_init()
        return

    def object_init(self, first_time: bool = True) -> None:
        """Initialize the object poses (both initial and target) using a helper function."""
        self._object_initial_position, self._object_initial_orientation = pose_init()
        self._object_target_position, self._object_target_orientation = pose_init()
        if not first_time:
            self.set_params(
                object_position=self._object_initial_position,
                object_orientation=self._object_initial_orientation,
                object_target_position=self._object_target_position,
                object_target_orientation=self._object_target_orientation,
            )
        return

    def set_up_scene(self, scene: Scene) -> None:
        """Set up the scene by adding ground plane, objects, robot, and optionally the camera."""
        # Call the parent's scene setup if applicable and add a default ground plane.
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        # Select a random object from the YCB dataset.
        selected_object = random.choice(usd_files)
        usd_path = ROOT_PATH + selected_object

        def create_object_prim(prim_path, usd_path):
            object_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            UsdPhysics.RigidBodyAPI.Apply(object_prim)
            UsdPhysics.CollisionAPI.Apply(object_prim)
            # object_prim.GetAttribute("physics:collision:approximation").Set("convexHull")
            return 

        # Create the initial object.
        initial_object_prim_path = find_unique_string_name(
            initial_name="/World/Ycb_object", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        object_name = find_unique_string_name(
            initial_name="object", is_unique_fn=lambda x: not scene.object_exists(x)
        )
    
        create_object_prim(initial_object_prim_path, usd_path)

        # Use a wrapper to wrap the object in a XFormPrim.
        ## Attention: it probably has some problems with the pose settings, if it does:
        ## use something like: xform = UsdGeom.Xform.Define(stage, "/xform")
        ## xform.AddTranslateOp().Set(Gf.Vec3d(2.0, -2.0, 1.0))
        ## xform.AddRotateXYZOp().Set(Gf.Vec3d(20, 30, 40))
        ## xform.AddScaleOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
        self._object = XFormPrim(
            prim_path=initial_object_prim_path,
            name=object_name,
            translation=self._object_initial_position,
            orientation=self._object_initial_orientation,
        )

        # Create the final object.
        final_object_prim_path = find_unique_string_name(
            initial_name="/World/Ycb_final", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        final_object_name = find_unique_string_name(
            initial_name="object", is_unique_fn=lambda x: not scene.object_exists(x)
        )
        create_object_prim(final_object_prim_path, usd_path)

        self._object_final = XFormPrim(
            prim_path=final_object_prim_path,
            name=final_object_name,
            translation=self._object_target_position,
            orientation=self._object_target_orientation,
        )


        # Attach face markers to the initial object.
        # self.attach_face_markers(initial_object_prim_path)
        self._task_objects[self._object.name] = self._object
        self._task_objects[self._object_final.name] = self._object_final

        # Set up the robot.
        self._robot = self.set_robot()
        self._robot.set_enabled_self_collisions(True)
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot


        # Set up the camera if enabled.
        # if self._set_camera:
        #     self._cameras = self.set_camera()
        #     for cam in self._cameras:
        #         scene.add(cam)
                # self._task_objects[cam.name] = cam
        # orientation_degrees = np.deg2rad([0, -90, 180])
        # orientation = euler2quat(orientation_degrees[0], orientation_degrees[1], orientation_degrees[2])
        # if self._set_camera:
        #     self._camera = Camera(
        #         prim_path="/World/Franka/panda_hand/geometry/realsense/realsense_camera",
        #         name="realsense_camera",
        #         translation=[0.05, 0.0, 0.05],
        #         orientation=orientation,
        #         resolution=[640, 480],
        #     )
        #     # Set camera parameters similar to a RealSense D435.
        #     self._camera.set_focal_length(0.193)
        #     self._camera.set_clipping_range(0.0001, 1000.0)

        # Move task objects to their designated frames.
        self._move_task_objects_to_their_frame()
        return

    def set_robot(self) -> Franka:
        """
        Set up the Franka robot.
        
        Returns:
            Franka: A Franka robot instance with a unique prim path and name.
        """
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=franka_prim_path, name=franka_robot_name)
    

    def set_camera(self, object_position):
        def look_at(from_pos, to_pos, up=[0, 0, 1]):
            """
            Compute a quaternion (in WXYZ order) that rotates from the camera position 
            to look at the target position, using the given up vector.
            """
            from_pos = np.array(from_pos)
            to_pos = np.array(to_pos)
            up = np.array(up)
            
            # Compute forward direction (normalized)
            forward = to_pos - from_pos
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-6:
                forward_norm = 1.0
            forward = forward / forward_norm
            
            # Compute right and true up vectors
            right = np.cross(up, forward)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                right_norm = 1.0
            right = right / right_norm
            
            true_up = np.cross(forward, right)
            
            # Build a rotation matrix with columns [right, true_up, forward]
            rot_matrix = np.column_stack((right, true_up, forward))
            
            # Use SciPy to robustly convert the rotation matrix to a quaternion.
            # SciPy's as_quat() returns [x, y, z, w] by default.
            quat_xyzw = R.from_matrix(rot_matrix).as_quat()
            # Rearrange to get WXYZ order.
            quat_wxyz = np.concatenate(([quat_xyzw[3]], quat_xyzw[:3]))
            return quat_wxyz.tolist()

        distance = 3.0      # Distance of cameras from the object.
        offset_z = 1.0      # Height offset for the cameras.

        camera_positions = [
            [object_position[0], object_position[1] - distance, object_position[2] + offset_z],  # Front view
            [object_position[0] - distance, object_position[1], object_position[2] + offset_z],  # Left view
            [object_position[0] + distance, object_position[1], object_position[2] + offset_z]   # Right view
        ]

        # Define camera configuration (unique prim paths and names).
        camera_configs = [
            {"name": "camera_front", "prim_path": "/World/camera_front"},
            {"name": "camera_left",  "prim_path": "/World/camera_left"},
            {"name": "camera_right", "prim_path": "/World/camera_right"}
        ]

        cameras = []
        for pos, config in zip(camera_positions, camera_configs):
            # Compute the orientation (in WXYZ order) so the camera looks at the object.
            quat = look_at(pos, self._object_initial_position)
            
            # Create the camera with the computed position and orientation.
            cam = Camera(
                prim_path=config["prim_path"],
                name=config["name"],
                translation=pos,
                orientation=quat,  # Orientation in WXYZ order.
                resolution=[640, 480],
            )
            # Set a wide field of view (e.g., 90 degrees) to cover a large scene.
            # cam.set_field_of_view(90.0)
            # Set clipping range if necessary.
            cam.set_clipping_range(0.01, 1000.0)
            
            cameras.append(cam)
        return cameras




    def attach_face_markers(self, object_prim_path: str, object_size: float = 1.0, offset: float = 0.05) -> None:
        """
        Attaches a small sphere marker to each face of the object.
        
        Args:
            object_prim_path (str): The prim path of the object.
            object_size (float): The overall size of the object.
            offset (float): Offset to push the marker outwards.
        """
        stage = omni.usd.get_context().get_stage()
        half = object_size / 2.0
        markers = {
            "front":  (Gf.Vec3d(half + offset, 0, 0),        "1"),
            "back":   (Gf.Vec3d(-half - offset, 0, 0),       "2"),
            "left":   (Gf.Vec3d(0, half + offset, 0),        "3"),
            "right":  (Gf.Vec3d(0, -half - offset, 0),       "4"),
            "top":    (Gf.Vec3d(0, 0, half + offset),        "5"),
            "bottom": (Gf.Vec3d(0, 0, -half - offset),       "6")
        }
        object_prim = stage.GetPrimAtPath(object_prim_path)
        if not object_prim:
            print(f"Cube prim not found at {object_prim_path}")
            return

        for face, (translation, label) in markers.items():
            marker_path = object_prim.GetPath().AppendChild(f"marker_{face}")
            marker = UsdGeom.Xform.Define(stage, marker_path)
            marker.AddTranslateOp().Set(translation)
            sphere_path = marker_path.AppendChild("sphere")
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.03)
        return

    def object_pose_finalization(self) -> None:
        """
        Finalize the object poses by updating parameters and moving the final object off-screen.
        """
        object_position, object_orientation = self._object.get_world_pose()
        object_target_position, object_target_orientation = self._object_final.get_world_pose()
        self.set_params(
            object_position=object_position,
            object_orientation=object_orientation,
            object_target_position=object_target_position,
            object_target_orientation=object_target_orientation,
        )
        self._object_final.set_world_pose(position=[1000, 1000, 0.5], orientation=object_target_orientation)
        return

    def set_params(
        self,
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[np.ndarray] = None,
        object_target_position: Optional[np.ndarray] = None,
        object_target_orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Set the object parameters."""
        if object_target_position is not None:
            self._object_target_position = object_target_position
        if object_target_orientation is not None:
            self._object_target_orientation = object_target_orientation
        if object_position is not None or object_orientation is not None:
            self._object.set_world_pose(position=object_position, orientation=object_orientation)
        return

    def get_params(self) -> dict:
        """Return current task parameters."""
        params_representation = dict()
        position, orientation = self._object.get_world_pose()
        params_representation["object_current_position"] = {"value": position, "modifiable": True}
        params_representation["object_current_orientation"] = {"value": orientation, "modifiable": True}
        params_representation["object_target_position"] = {"value": self._object_target_position, "modifiable": True}
        params_representation["object_target_orientation"] = {"value": self._object_target_orientation, "modifiable": True}
        params_representation["object_name"] = {"value": self._object.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_observations(self) -> dict:
        """Collect observations from the robot and object states."""
        joints_state = self._robot.get_joints_state()
        object_position, object_orientation = self._object.get_world_pose()
        end_effector_position, end_effector_orientation = get_current_end_effector_pose()
        observations = {
            self._object.name: {
                "object_current_position": object_position,
                "object_current_orientation": object_orientation,
                "object_target_position": self._object_target_position,
                "object_target_orientation": self._object_target_orientation,
            },
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
                "end_effector_orientation": end_effector_orientation,
            }
        }
        return observations

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Pre-step callback (can be extended as needed)."""
        return

    def post_reset(self) -> None:
        """Reset robot gripper to opened positions after a reset."""
        from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        return

    def calculate_metrics(self) -> dict:
        """Calculate task metrics (to be implemented by the user)."""
        raise NotImplementedError

    def is_done(self) -> bool:
        """Determine if the task is finished (to be implemented by the user)."""
        raise NotImplementedError
