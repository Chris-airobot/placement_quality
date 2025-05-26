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
from omni.isaac.core.objects import FixedCuboid, VisualCuboid, DynamicCuboid
from pxr import Sdf, UsdLux
from omni.isaac.motion_generation import ArticulationTrajectory
from omni.isaac.motion_generation.lula import RRT
from omni.isaac.franka import Franka
from omni.isaac.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from omni.isaac.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from omni.isaac.motion_generation.path_planning_interface import PathPlanner
from omni.isaac.core.utils.types import ArticulationAction
# Import the collision detection functionality
from collision_check import GroundCollisionDetector
from pxr import UsdGeom, Gf, Usd
from placement_quality.cube_simulation import helper
from omni.physx import get_physx_scene_query_interface
from omni.isaac.core.articulations import Articulation
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()


class Testing:
    def __init__(self):
        # Core variables for the kinematic solver
        self._kinematics_solver = None
        self._articulation = None
        self._articulation_controller = None
        self._target = None
        self.world = None
        self.collision_detector = None
        self.base_path = "/World/panda"
        self.robot_parts_to_check = []
        self.collision_detected = False

        # Offset of the grasp from the object center
        # Grasp poses are based on the gripper center, so we need to offset it to transform it to the tool_center, i.e. 
        # the middle of the gripper fingers
        self.grasp_offset = [0, 0, -0.065] 
        self.gripper_max_aperture = 0.05
        self._last_solution = None
        self._action_sequence = None
        self._physics_dt = 1.0/60.0
        self.ik_check = None
        self._rrt_interpolation_max_dist = 0.01

        self.grasp_position = [0.4506616621146338, -0.06541458836272551, 0.237598010485652436]
        self.grasp_orientation = [-0.10862850243802431, 0.7199423352353054, -0.11904264190219786, -0.6750642427228336]

    def setup_scene(self):
        """Create a new stage and set up the scene with lighting and camera"""
        create_new_stage()
        self._add_light_to_stage()
        
        # Create a world instance
        self.world: World = World()
        
        # Load robot and target
        self._articulation, self._target = self.load_assets()
        self._articulation_controller = self._articulation.get_articulation_controller()
        
        self.ground_plane = DynamicCuboid(
                            prim_path="/World/CollisionGround",
                            name="collision_ground",
                            position=np.array([0.0, 0.0, -0.0005]),  # Match your visual ground position
                            scale=np.array([20.0, 20.0, 0.001]),     # Match size and thickness
                            color=np.array([0.0, 0.0, 0.0])     # Make it invisible if you want (alpha=0)
                        )

        # Add assets to the world scene
        self.world.scene.add_default_ground_plane()
        self.world.scene.add(self._articulation)
        self.world.scene.add(self._target)
        self.world.scene.add(self.ground_plane)
        
        
        # self._articulation.set_enabled_self_collisions(True)
        # Set up the ground collision detector
        self.setup_collision_detection()
        
        self.setup_controller()

        # Set up the physics scene
        self.world.reset()





    
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
        articulation = Articulation(robot_prim_path)
        
        # Add the target frame to the stage
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd", "/World/Ycb_object")
        target = XFormPrim("/World/Ycb_object")
        # target.set_default_state(np.array([0.35, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))
        
        
        return articulation, target

    def setup_controller(self):
        # Load default RRT config files stored in the omni.isaac.motion_generation extension
        rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        rrt_config["end_effector_frame_name"] = "panda_hand"
        rrt = RRT(**rrt_config)

        # Create a trajectory generator to convert RRT cspace waypoints to trajectories
        self._cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            rrt_config["robot_description_path"], rrt_config["urdf_path"]
        )

        # It is important that the Robot Description File includes optional Jerk and Acceleration limits so that the generated trajectory
        # can be followed closely by the simulated robot Articulation
        for i in range(len(rrt.get_active_joints())):
            assert self._cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limit(i)
            assert self._cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limit(i)
        
        
        # Create a visualizer to visualize the path planner
        self._path_planner_visualizer = PathPlannerVisualizer(self._articulation, rrt)
        self._path_planner: RRT = self._path_planner_visualizer.get_path_planner()
        print(f"At this moment, the ground plane is {self.ground_plane}")
        self._path_planner.add_obstacle(self.ground_plane, static=True)

        self._robot: Franka = self._path_planner_visualizer.get_robot_articulation() 

     

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
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation):
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
        target_end_effector_orientation):
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

    def add_obstacle(self, obstacle, static: bool = False) -> None:
        self._path_planner.add_obstacle(obstacle, static)


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

        # draw_frame(grasp_pose_center[0], grasp_pose_center[1])
        # Track any movements of the robot base
        actions = self.forward(
                target_end_effector_position=np.array(self.grasp_position),
                target_end_effector_orientation=np.array(self.grasp_orientation),
            )
        
        if self.ik_check:
            kps, kds = (1e15 * np.ones(9), 1e13 * np.ones(9))
            self._articulation_controller.set_gains(kps, kds)
            # Apply the actions to the robot
            self._articulation_controller.apply_action(actions)
        else:
            carb.log_warn("IK did not converge to a solution. No action is being taken")
        
        # Check for collisions with the ground
        self.check_for_collisions()

        if self.is_done():
            print("The robot has reached the target 1")
            self.grasp_position = [0.4506616621146338, -0.06541458836272551, -0.037598010485652436]
            self._path_planner.reset()
            self.add_obstacle(self.ground_plane, static=True)
            self._action_sequence = None
            self._last_solution = None

    def is_done(self) -> bool:
        if len(self._action_sequence) <= 1:
            return True
        else:
            return False
        


    def run(self):
        """Main loop to run the simulation"""
        # Set up the scene
        self.setup_scene()
        
        # Main simulation loop
        while simulation_app.is_running():
            # Step the simulation
            self.world.step(render=True)

            # Update the robot's position
            self.update(step=1.0/60.0)
        
        print("Simulation ended.")


if __name__ == "__main__":
    # Create and run the standalone IK example
    env = Testing()
    env.run()
    
    # Close the simulation application
    simulation_app.close()