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
from omni.isaac.franka import Franka
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.motion_generation import interface_config_loader
from pxr import Sdf, UsdLux

# Import the collision detection functionality
from collision_check import GroundCollisionDetector
from RRT_controller import RRTController
from pxr import Gf, UsdPhysics
from ycb_simulation.utils.helper import draw_frame, transform_relative_pose, local_transform

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


class PhysicsCollection:
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
        object_poses_file = "/home/chris/Chris/placement_ws/src/object_poses_test.json"
        self.all_object_poses = json.load(open(object_poses_file))
        
        self.grasp_poses_file = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation/grasp_poses_v2.json"
        self.grasp_poses = json.load(open(self.grasp_poses_file))

        self.data_folder = "/home/chris/Chris/placement_ws/src/data/path_simulation/raw_data_v4/"

        # Offset of the grasp from the object center
        # Grasp poses are based on the gripper center, so we need to offset it to transform it to the tool_center, i.e. 
        # the middle of the gripper fingers
        self.grasp_offset = [0, 0, -0.07] 

        self.data_dict = {}
        self.count = 0
        self.episode_count = 0

    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()

        self.ground_plane = self.world.scene.add_default_ground_plane()
        # Load robot and target
        self._articulation, self._target = self.load_assets()
        self.gripper: ParallelGripper = self._articulation.gripper
        self.gripper.set_default_state(self.gripper.joint_opened_positions)
        
        # Add assets to the world scene
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        self.controller = RRTController(
            name = "RRT_controller",
            robot_articulation=self._articulation,
        )
        self.articulation_controller = self._articulation.get_articulation_controller()
        
        self._articulation.set_enabled_self_collisions(True)
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        # Set up the physics scene
        self.world.reset()
        self.controller.add_obstacle(self.ground_plane, static=True)
        # self._target.set_world_pose(self.object_pose_at_grasp["position"], 
        #                             self.object_pose_at_grasp["orientation_quat"])
        
        # Set up the kinematics solver
        # self.setup_kinematics()

    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def load_assets(self):
        """Load the Franka robot and target frame"""
        # Add the Franka robot to the stage
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        articulation = Franka(prim_path=robot_prim_path, usd_path=path_to_robot_usd)
        
        # Add the target frame to the stage
        ycb_prim = add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd", "/World/Ycb_object")
        UsdPhysics.RigidBodyAPI.Apply(ycb_prim)
        UsdPhysics.CollisionAPI.Apply(ycb_prim)
        target = XFormPrim("/World/Ycb_object")
        # target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        return articulation, target

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
    def get_custom_gains(self):
        return (1e15 * np.ones(9), 1e13 * np.ones(9))
    

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

    def check_grasp_success(self):
        """Check if the grasp was successful"""
        # print(f"You are in the grasp success check")
        
        # Check if gripper is fully closed (no object grasped)
        if np.floor(self.gripper.get_joint_positions()[0] * 100) == 0:
            # print("Gripper is fully closed, didn't grasp object")
            return False
            
        return True
    
    def reset(self):
        """Reset the simulation"""
        if self.world:
            self.world.reset()
            self.gripper.open()
            self.controller.reset()
            self._articulation.set_joints_default_state()

    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()
        state = "POSE"
        # If self.grasp_poses is a dict, we need to get the first key-value pair
        first_key = list(self.grasp_poses.keys())[0]
        self.grasp_pose = self.grasp_poses[first_key]
        # Remove the first item from the dict
        self.grasp_poses.pop(first_key)
        self.object_poses = deepcopy(self.all_object_poses)

        # Main simulation loop
        while simulation_app.is_running():
            # Step the simulation
            self.world.step(render=True)
            print(f"The current progress is: {self.episode_count} / {len(self.grasp_poses)}: {len(self.all_object_poses) - len(self.object_poses)}/{len(self.all_object_poses)}")
            if state == "POSE": 
                if self.object_poses:
                    self.current_object_pose = self.object_poses.pop(0)
                    self.current_object_pose['position'][0] = 0.3
                    self.current_object_pose['position'][1] = 0.0
                    self._target.set_world_pose(self.current_object_pose["position"], 
                                                self.current_object_pose["orientation_quat"])
                    failure = False
                    collisions = []
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
                    self.current_object_pose['position'][0] = 0.3
                    self.current_object_pose['position'][1] = 0.0
                    self._target.set_world_pose(self.current_object_pose["position"], 
                                                self.current_object_pose["orientation_quat"])
                    
                state = "GRASP"

                internal_counter = 0
            

            if state == "GRASP":
                # Local grasp pose in gripper frame  
                grasp_pose_local = [self.grasp_pose["position"], self.grasp_pose["orientation_wxyz"]]
                # World grasp pose in gripper frame
                grasp_pose_world = transform_relative_pose(grasp_pose_local, 
                                                        self.current_object_pose["position"], 
                                                        self.current_object_pose["orientation_quat"])
                # World grasp pose in tool center frame
                grasp_pose_center = local_transform(grasp_pose_world, self.grasp_offset)
                # Update the robot's position
                actions = self.controller.forward(
                            target_end_effector_position=np.array(grasp_pose_center[0]),
                            target_end_effector_orientation=np.array(grasp_pose_center[1]),
                        )
                if self.controller.ik_check:
                    kps, kds = self.get_custom_gains()
                    self.articulation_controller.set_gains(kps, kds)
                    # Apply the actions to the robot
                    self.articulation_controller.apply_action(actions)
                    collisions.append(self.check_for_collisions())

                    if self.controller.is_done():
                         print(f"controller is done")
                         state = "GRIP"
                else:
                    print(f"No plan found for the current grasp pose")
                    state = "POSE"

                



            if state == "GRIP":
                self.gripper.close()
                # Wait for the gripper to finish closing
                if not hasattr(self, "grip_timer"):
                    self.grip_timer = 0
                    
                self.grip_timer += 1
                
                # Give enough time for the gripper to close (typically 10-20 frames is sufficient)
                if self.grip_timer >= 30:
                    object_position, object_orientation = self._target.get_world_pose()
                    grasp_position, grasp_orientation = self.gripper.get_world_pose()
                    self.data_dict[self.count] = {
                        "collisions": collisions, 
                        "grasp_position": grasp_position,
                        "grasp_orientation": grasp_orientation,
                        "object_position": object_position,
                        "object_orientation": object_orientation,
                        "feasibility": self.controller.ik_check,
                        "grasp_success": self.check_grasp_success(),
                    }
                    self.count += 1  # Make sure to increment the counter
                    self.grip_timer = 0  # Reset the timer for next grasp
                    self.reset()
                    state = "POSE"

            
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env =  PhysicsCollection()
    env.run()
    
    # Close the simulation application
    simulation_app.close()