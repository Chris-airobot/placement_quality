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
from helper import euler2quat, get_current_end_effector_pose
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
        self._object_initial_position, self._object_initial_orientation = self.pose_init()
        self._object_target_position, self._object_target_orientation = self.pose_init()
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
        """
        Set up cameras in a circular pattern, each pointing at the object from different angles.
        Each camera is positioned 120 degrees apart around the object and tilted to look at the object center.
        
        Args:
            object_position: The position of the object to look at
            
        Returns:
            List of camera objects
        """
        # Store cameras for later use
        self._cameras = []
        
        # If camera setup is disabled, return early
        if not self._set_camera:
            return self._cameras

        # Camera parameters
        radius = 0.5       # Horizontal distance from object center
        height = 0.3       # Height above the object
        
        # Define camera positions in a circle around the object (120 degrees apart)
        angles = [0, 120, 240]  # in degrees, spaced 120 degrees apart
        
        # Define camera configuration
        camera_configs = [
            {"name": "camera_1", "prim_path": "/World/camera_1"},
            {"name": "camera_2", "prim_path": "/World/camera_2"},
            {"name": "camera_3", "prim_path": "/World/camera_3"},
        ]

        # Create cameras with positions and orientations that point at the object
        for i, (angle, config) in enumerate(zip(angles, camera_configs)):
            # Convert angle to radians
            angle_rad = np.radians(angle)
            
            # Calculate position on the circle with some height
            x = object_position[0] + radius * np.cos(angle_rad)
            y = object_position[1] + radius * np.sin(angle_rad)
            z = object_position[2] + height
            camera_position = [x, y, z]
            
            # Calculate the direction vector from camera to object
            direction = np.array(object_position) - np.array(camera_position)
            
            # Normalize the direction vector
            direction = direction / np.linalg.norm(direction)
            
            # Create a rotation matrix that aligns the camera's forward direction with the direction vector
            # We need to find a rotation that maps [0,0,-1] (camera forward in USD) to our direction vector
            
            # First, calculate the rotation axis and angle using cross product and dot product
            forward = np.array([0, 0, -1])  # Camera forward direction in USD
            rotation_axis = np.cross(forward, direction)
            
            # If the vectors are parallel, we need a different approach
            if np.allclose(rotation_axis, 0):
                if np.dot(forward, direction) > 0:  # Same direction
                    rotation_quat = [1, 0, 0, 0]  # Identity quaternion
                else:  # Opposite direction
                    rotation_quat = [0, 1, 0, 0]  # 180 degree rotation around X
            else:
                # Normalize the rotation axis
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                # Calculate the rotation angle
                cos_angle = np.dot(forward, direction)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                # Convert to quaternion (axis-angle to quaternion)
                sin_half_angle = np.sin(angle / 2)
                cos_half_angle = np.cos(angle / 2)
                rotation_quat = [
                    cos_half_angle,
                    rotation_axis[0] * sin_half_angle,
                    rotation_axis[1] * sin_half_angle,
                    rotation_axis[2] * sin_half_angle
                ]
            
            # Create the camera
            cam = Camera(
                prim_path=config["prim_path"],
                name=config["name"],
                position=camera_position,
                orientation=rotation_quat,
                resolution=[640, 480],
            )
            cam.set_world_pose(position=camera_position, orientation=rotation_quat, camera_axes="usd")
            
            # Set clipping range
            cam.set_clipping_range(0.01, 1000.0)
            
            self._cameras.append(cam)
            
            # Add camera to task objects for tracking
            self._task_objects[config["name"]] = cam
            
        return self._cameras




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


    
    def pose_init(self):
        ranges = [(-0.5, -0.15), (0.15, 0.5)]
        range_choice = ranges[np.random.choice(len(ranges))]
        
        # Generate x and y as random values between -π and π
        x, y = np.random.uniform(low=range_choice[0], high=range_choice[1]), np.random.uniform(low=range_choice[0], high=range_choice[1])
        z = np.random.uniform(1.0, 2.0)

        # Random Euler angles in degrees and convert to radians
        euler_angles_deg = [random.uniform(0, 360) for _ in range(3)]
        euler_angles_rad = np.deg2rad(euler_angles_deg)
        
        # Convert Euler angles to quaternion using your preferred function,
        # e.g., if you have a function euler2quat that takes 3 angles:
        quat = euler2quat(*euler_angles_rad)  # Make sure euler2quat returns in the correct order
        pos =  np.array([x, y, z])

        return pos, quat
    
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