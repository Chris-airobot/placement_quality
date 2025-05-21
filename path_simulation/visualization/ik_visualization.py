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

from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux

# Import the collision detection functionality
from collision_check import GroundCollisionDetector
from pxr import UsdGeom, Gf, Usd
from placement_quality.cube_simulation import helper 

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


class IKVisualization:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None
        self.world = None
        self.collision_detector = None
        self.base_path = "/World/panda"
        self.robot_parts_to_check = []
        self.collision_detected = False


        # File reads
        # Test data
        with open("/media/chris/OS2/Users/24330/Desktop/placement_quality/unseen/balanced_samples.json", "r") as f:
            self.test_data: list[dict] = json.load(f)
        
        self.current_data = self.test_data.pop(0)
        # Offset of the grasp from the object center
        # Grasp poses are based on the gripper center, so we need to offset it to transform it to the tool_center, i.e. 
        # the middle of the gripper fingers
        # self.grasp_offset = [0, 0, -0.1034] 
        self.grasp_offset = [0, 0, -0.065] 

        self.data_dict = {}
        self.count = 0
        self.episode_count = 0
        self.gripper_max_aperture = 0.05
    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()
        
        # Load robot and target
        self._articulation, self._target, self._frame = self.load_assets()
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        self.world.scene.add(self._frame)
        
        
        # self._articulation.set_enabled_self_collisions(True)
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        # Set up the physics scene
        self.world.reset()
        self.added = False
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
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.prims import XFormPrim
        # Add the Franka robot to the stage
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        articulation = Articulation(robot_prim_path)
        
        # Add the target frame to the stage
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd", "/World/Ycb_object")
        target = XFormPrim("/World/Ycb_object", name="Ycb_object")
        # target.set_default_state(np.array([0.35, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))

        
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        frame = XFormPrim("/World/target", scale=[.04,.04,.04], name="target")
        frame.set_default_state(np.array([0.35, 0, 0.5]),
                                np.array([0, 0, 0, 1]))
        
        
        return articulation, target, frame

    def setup_kinematics(self):
        """Set up the kinematics solver for the Franka robot"""
        # Load kinematics configuration for the Franka robot
        print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)
        
        # Print valid frame names for debugging
        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
        
        # Configure the articulation kinematics solver with the end effector
        end_effector_name = "panda_hand"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, 
            self._kinematics_solver, 
            end_effector_name
        )

    def setup_collision_detection(self):
        """Set up the ground collision detector"""
        stage = omni.usd.get_context().get_stage()
        self.collision_detector = GroundCollisionDetector(stage, non_colliding_part="/World/panda/panda_link0")
        
        # Create a virtual ground plane at z=-0.05 (lowered to avoid false positives)
        self.collision_detector.create_virtual_ground(
            size_x=20.0, 
            size_y=20.0, 
            position=Gf.Vec3f(0, 0, -0.05)  # Lower the ground to avoid false positives
        )
        
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

    def check_for_collisions(self):
        """Check if any robot parts are colliding with the ground"""
        for part_path in self.robot_parts_to_check:
            if self.collision_detector.is_colliding_with_ground(part_path):
                # print(f"Collision detected")
                self.collision_detected = True
                return True
            else:
                # print(f"No collision detected")
                self.collision_detected = False
                return False

    def update(self, step):
        """Update the robot's position based on the target's position"""
        # object_position, object_orientation = self._target.get_world_pose()
        
        
        # World grasp pose in gripper frame
        grasp_pose_world = helper.transform_relative_pose(self.grasp_pose, 
                                                        [0.35, 0, self.current_data["initial_object_pose"][0]], 
                                                        self.current_data["initial_object_pose"][1:])
        
        self._frame.set_world_pose(grasp_pose_world[0], grasp_pose_world[1])
        # World grasp pose in tool center frame
        grasp_pose_center = helper.local_transform(grasp_pose_world, self.grasp_offset)
        # grasp_pose_center = grasp_pose_world
        self._frame.set_world_pose(grasp_pose_center[0], grasp_pose_center[1])
        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        
        # Compute inverse kinematics to find joint positions
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            np.array(grasp_pose_center[0]), 
            np.array(grasp_pose_center[1])
        )
        
        # print(f"input orientation: {grasp_pose_center[1]}")
        # print(f"input position: {grasp_pose_center[0]}")
        # Apply the joint positions if IK was successful
        if success:
            self._articulation.apply_action(action)
            # self.aperture_filter_mesh(self._target, grasp_pose_center)
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken")
        
        # Check for collisions with the ground
        collision = self.check_for_collisions()
        

    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()
        
        grasp_poses = []
        raw_grasps = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation/grasps.json"
        with open(raw_grasps, "r") as f:
            raw_grasps = json.load(f)
        for key in sorted(raw_grasps.keys(), key=int):
            item = raw_grasps[key]
            position = item["position"]
            orientation = item["orientation_wxyz"]
            # Each grasp: [ [position], [orientation] ]
            grasp_poses.append([position, orientation])
        
        # Initialize variables for grasp pose cycling
        grasp_index = 0
        self.grasp_pose = grasp_poses[grasp_index]
        wait_time = 2.0  # seconds to wait between pose changes
        elapsed_time = 0.0
        time_step = 1.0/60.0  # Simulation time step
        
        # Main simulation loop
        while simulation_app.is_running():
            self._target.set_world_pose([0.35, 0, self.current_data["initial_object_pose"][0]], 
                                        self.current_data["initial_object_pose"][1:])
            # Step the simulation
            self.world.step(render=True)

            # Update the robot's position
            self.update(step=time_step)
            self.count += 1
            
            # Update elapsed time
            elapsed_time += time_step
            
            # Check if it's time to change to the next grasp pose
            if elapsed_time >= wait_time:
                grasp_index = (grasp_index + 1) % len(grasp_poses)  # Loop back to the beginning when all poses are shown
                self.grasp_pose = grasp_poses[grasp_index]
                print(f"Switching to grasp pose {grasp_index + 1}/{len(grasp_poses)}")
                elapsed_time = 0.0  # Reset the timer
        
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env = IKVisualization()
    env.run()
    
    # Close the simulation application
    simulation_app.close()