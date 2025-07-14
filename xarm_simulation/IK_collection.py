import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from isaacsim import SimulationApp

# Display options for the simulation viewport
DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

# Simulation configuration
CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

# Initialize the simulation application
simulation_app = SimulationApp(CONFIG)

import datetime 
import numpy as np
import json
import carb
from copy import deepcopy
import omni
from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux
from scipy.spatial.transform import Rotation as R
# Import the collision detection functionality
from collision_check import GroundCollisionDetector
from pxr import UsdGeom, Gf, Usd
from placement_quality.cube_simulation import helper
from omni.physx import get_physx_scene_query_interface
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

base_dir = "/home/chris/Chris/placement_ws/src/data/YCB_data/"
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_PATH = os.path.join(base_dir, f"run_{time_str}/")


# Mapping of surface names to indices
surface_mapping ={
    # z up 
    "z_up": 0,
    # z down
    "z_down": 1,
    # y up
    "y_up": 2,
    # y down    
    "y_down": 3,
    # x up
    "x_up": 4,
    # x down
    "x_down": 5
}


class StandaloneIK:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None
        self.world = None
        self.collision_detector = None
        self.base_path = "/UF_ROBOT"
        self.robot_parts_to_check = []
        self.collision_detected = False


        # File reads
        object_poses_file = "/home/chris/Chris/placement_ws/src/144_6_box.json"
        self.all_object_poses = json.load(open(object_poses_file))
        
        self.grasp_poses_file = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation/test_grasps.json"
        self.grasp_poses = json.load(open(self.grasp_poses_file))
        start_index = 0
        self.grasp_poses = {k: v for k, v in self.grasp_poses.items() if int(k) >= start_index}

        self.data_folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/raw_data_v1/"

        # Offset of the grasp from the object center
        # Grasp poses are based on the gripper center, so we need to offset it to transform it to the tool_center, i.e. 
        # the middle of the gripper fingers
        self.grasp_offset = [0, 0, 0.06] 
        self.gripper_max_aperture = 0.05
        self.data_dict = {}
        self.count = 0
        self.episode_count = start_index

        self.box_dims = np.array([0.143, 0.0915, 0.051])

    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()
        
        # Load robot and target
        self._articulation, self._target= self.load_assets()
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        # self.world.scene.add(self._pedestal)
        
        # self._articulation.set_enabled_self_collisions(True)
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        # Set up the physics scene
        self.world.reset()

        # self._target.set_world_pose(self.object_pose_at_grasp["position"], 
        #                             self.object_pose_at_grasp["orientation_quat"])
        
        # Set up the kinematics solver
        self.setup_kinematics()




        

    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def load_assets(self):
        """Load the Franka robot and target frame"""
        # Add the Franka robot to the stage
        prim_path = "/UF_ROBOT"
        asset_path = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper/xarm7_with_gripper.usd"
        end_effector_prim_path = "/UF_ROBOT/link_tcp"
        name = "xarm7_robot"
        add_reference_to_stage(asset_path, prim_path)
        articulation = Articulation(prim_path)
        
        # Add the target frame to the stage
        # add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd", "/World/Ycb_object")
        # target = XFormPrim("/World/Ycb_object")
        from omni.isaac.core.objects import VisualCuboid
        target = VisualCuboid(
            prim_path="/World/Ycb_object",
            name="Ycb_object",
            position=np.array([0, 0, 0]),
            scale=self.box_dims.tolist(),  # [x, y, z]
            color=np.random.rand(3)  # Give each a random color if you want
        )

        from omni.isaac.core.objects import VisualCylinder
        # Cylinder params
        self.pedestal_radius = 0.08  # meters
        self.pedestal_height = 0.10  # meters

        # Place the pedestal so its top is at z=0 (i.e., flush with ground)
        self.pedestal_center_z = self.pedestal_height / 2

        # # Create the cylinder
        # pedestal = VisualCylinder(
        #     prim_path="/World/Pedestal",
        #     name="Pedestal",
        #     position=np.array([0.2, -0.3, self.pedestal_center_z]),
        #     radius=self.pedestal_radius,
        #     height=self.pedestal_height,
        #     color=np.array([0.6, 0.6, 0.6])  # light gray
        # )


        # target.set_default_state(np.array([0.35, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        
        
        return articulation, target

    def setup_kinematics(self):
        urdf_path = "/home/chris/Chris/ros2_ws/src/xarm_ros2/xarm_description/urdf/xarm7_with_gripper.urdf"
        robot_description_path = "/home/chris/Chris/placement_ws/src/placement_quality/xarm_simulation/robot_test/rmpflow/robot_descriptor.yaml"
        xarm_end_effector_frame_name = "link_tcp"
        # Load kinematics configuration for the Franka robot
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        self._kinematics_solver = LulaKinematicsSolver(robot_description_path=robot_description_path,
                                                urdf_path=urdf_path)
        
        # Print valid frame names for debugging
        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
        
        # Configure the articulation kinematics solver with the end effector
        end_effector_name = xarm_end_effector_frame_name
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, 
            self._kinematics_solver, 
            end_effector_name
        )

    def setup_collision_detection(self):
        """Set up the ground collision detector"""
        stage = omni.usd.get_context().get_stage()
        self.collision_detector = GroundCollisionDetector(stage, non_colliding_part="/UF_ROBOT/link_base")
        
        # Create a virtual ground plane at z=-0.05 (lowered to avoid false positives)
        self.collision_detector.create_virtual_ground(
            size_x=20.0, 
            size_y=20.0, 
            position=Gf.Vec3f(0, 0, -0.05)  # Lower the ground to avoid false positives
        )

        self.collision_detector.create_virtual_pedestal()
        
        # Define robot parts to check for collisions (explicitly excluding the base link0)
        self.robot_parts_to_check = [
            f"{self.base_path}/panda_link1",
            f"{self.base_path}/panda_link2",
            f"{self.base_path}/panda_link3",
            f"{self.base_path}/panda_link4",
            f"{self.base_path}/panda_link5",
            f"{self.base_path}/panda_link6",
            f"{self.base_path}/panda_link7",
            f"{self.base_path}/panda_hand",
            f"{self.base_path}/panda_leftfinger",
            f"{self.base_path}/panda_rightfinger"
        ]
        # Note: panda_link0 is deliberately excluded since it's expected to touch the ground

    def is_pose_reached(self, target_pos, target_quat, pos_thresh=0.002, quat_thresh=0.02):
        """
        Returns True if the current EE pose is close enough to the target.
        """
        # Get current EE pose (world frame)
        current_pos, current_quat = self.get_current_end_effector_pose()
        pos_error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        quat_error = np.abs(1 - np.abs(np.dot(current_quat, target_quat)))  # Quaternion similarity

        return (pos_error < pos_thresh) and (quat_error < quat_thresh)
    

    def check_for_collisions(self):
        """Check if any robot parts are colliding with ground or pedestal."""
        ground_hit = any(
            self.collision_detector.is_colliding_with_ground(part_path)
            for part_path in self.robot_parts_to_check
        )
        pedestal_hit = any(
            self.collision_detector.is_colliding_with_pedestal(part_path)
            for part_path in self.robot_parts_to_check
        )
        self.collision_detected = ground_hit or pedestal_hit
        return self.collision_detected

    def update(self, step):
        """Update the robot's position based on the target's position"""
        # object_position, object_orientation = self._target.get_world_pose()
        
        # Local grasp pose in gripper frame  
        grasp_pose_local = [self.grasp_pose["position"], self.grasp_pose["orientation_wxyz"]]
        # World grasp pose in gripper frame
        grasp_pose_world = helper.transform_relative_pose(grasp_pose_local, 
                                                   self.current_object_pose["position"], 
                                                   self.current_object_pose["orientation_quat"])
        # World grasp pose in tool center frame
        grasp_pose_center = helper.local_transform(grasp_pose_world, self.grasp_offset)
        
        # draw_frame(grasp_pose_center[0], grasp_pose_center[1])
        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        
        # Compute inverse kinematics to find joint positions
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            np.array(grasp_pose_center[0]), 
            np.array(grasp_pose_center[1])
        )
        
        # Apply the joint positions if IK was successful
        if success:
            self._articulation.apply_action(action)
            timeout_s = 10.0
            elapsed = 0
            dt = 1.0 / 60.0
            while not self.is_pose_reached(grasp_pose_center[0], grasp_pose_center[1]):
                self.world.step(render=True)
                elapsed += dt
                if elapsed > timeout_s:
                    print("Timeout waiting for robot to reach target pose")
                    success = False
                    break
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken")
        
        # Check for collisions with the ground
        collision = self.check_for_collisions()
        
        
        self.data_dict[self.count] = {
            "collision": collision, 
            "grasp_pose": grasp_pose_center,
            "z_position": self.current_object_pose["position"][2],
            "object_orientation": self.current_object_pose["orientation_quat"],
            "success": success,
            }

    def save_data(self):
        """Save the data to a file"""
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_folder), exist_ok=True)
        
        # Create the file path
        file_path = self.data_folder + f"data_{self.episode_count}.json"
        
        # Save the data to the file
        with open(file_path, "w") as f:
            json.dump(self.data_dict, f)

        self.data_dict = {}

    
    def get_current_end_effector_pose(self) -> np.ndarray:
        """
        Return the current end-effector (grasp center, i.e. outside the robot) pose.
        return: (3,) np.ndarray: [x, y, z] position of the grasp center.
                (4,) np.ndarray: [w, x, y, z] orientation of the grasp center.
        """

        offset = np.array([0.0, 0.0, 0.1034])
        # offset = np.array([0.0, 0.0, 0.0])
        from omni.isaac.dynamic_control import _dynamic_control
        """Return the current end-effector (grasp center) position."""
        # Acquire dynamic control interface
        dc = _dynamic_control.acquire_dynamic_control_interface()
        # Get rigid body handles for the two gripper fingers
        ee_body = dc.get_rigid_body("/UF_ROBOT/link_tcp")
        # Query the world poses of each finger
        ee_pose = dc.get_rigid_body_pose(ee_body)
        panda_hand_translation = ee_pose.p 
        panda_hand_quat = ee_pose.r

        # Create a Rotation object from the panda_hand quaternion.
        hand_rot = R.from_quat(panda_hand_quat)
        
        # Rotate the local offset into world coordinates.
        offset_world = hand_rot.apply(offset)
        
        # Compute the tool_center position.
        tool_center_translation = panda_hand_translation 
        
        # Since the relative orientation is identity ([0,0,0]), the tool_center's orientation
        # remains the same as the panda_hand's.
        tool_center_quat = [panda_hand_quat[3], panda_hand_quat[0], panda_hand_quat[1], panda_hand_quat[2]]
        
        return tool_center_translation, tool_center_quat
    
    def move_to_home(self, timeout_s=10.0):
        """Move the robot to the home position"""
        self.home_joint_positions = [0,0,0,1.57,0,1.57,0,0,0,0,0,0,0]
        self._articulation.set_joint_positions(self.home_joint_positions)
        # Step until joints have converged
        dt = 1.0 / 60.0
        elapsed = 0
        while True:
            # Step simulation
            self.world.step(render=True)
            elapsed += dt
            # Check if at home (tolerance in radians)
            current_joints = self._articulation.get_joint_positions()
            if np.allclose(current_joints, self.home_joint_positions, atol=0.01):
                break
            if elapsed > timeout_s:
                print("Timeout waiting for home position")
                break
        
        

    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()
        
        # If self.grasp_poses is a dict, we need to get the first key-value pair
        first_key = list(self.grasp_poses.keys())[0]
        self.grasp_pose = self.grasp_poses[first_key]
        # Remove the first item from the dict
        self.grasp_poses.pop(first_key)
        self.object_poses = deepcopy(self.all_object_poses)

        # Main simulation loop
        while simulation_app.is_running():
            print(f"The current progress is: {self.episode_count} / {len(self.grasp_poses)}: {self.count}/{len(self.all_object_poses)}")
            if self.object_poses:
                self.current_object_pose = self.object_poses.pop(0)
                self.current_object_pose['position'][0] = 0.2
                self.current_object_pose['position'][1] = -0.3
                self.current_object_pose['position'][2] += self.pedestal_height 
                self._target.set_world_pose(self.current_object_pose["position"], 
                                            self.current_object_pose["orientation_quat"])
                self.move_to_home()
            else:
                print("No more object poses, saving data and restarting simulation")
                self.save_data()
                self.episode_count += 1
                self.count = 0

                first_key = list(self.grasp_poses.keys())[0]
                self.grasp_pose = self.grasp_poses[first_key]
                # Remove the first item from the dict
                self.grasp_poses.pop(first_key)

                self.object_poses = deepcopy(self.all_object_poses)
                self.current_object_pose = self.object_poses.pop(0)
                self.current_object_pose['position'][0] = 0.2
                self.current_object_pose['position'][1] = -0.3
                self.current_object_pose['position'][2] += self.pedestal_height 
                self._target.set_world_pose(self.current_object_pose["position"], 
                                            self.current_object_pose["orientation_quat"])
            # Step the simulation
            self.world.step(render=True)

            # Update the robot's position
            self.update(step=1.0/60.0)
            self.count += 1
        
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env = StandaloneIK()
    env.run()
    
    # Close the simulation application
    simulation_app.close()