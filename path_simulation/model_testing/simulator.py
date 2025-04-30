import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import numpy as np
import omni
from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.scenes import Scene
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from scipy.spatial.transform import Rotation as R
from collision_check import GroundCollisionDetector
from pxr import Gf
from pxr import Sdf, UsdLux
from omni.isaac.core.prims import XFormPrim

from RRT_controller import RRTController
from RRT_task import RRTTask


        

class Simulator:
    def __init__(self, use_physics=False):
        # Basic variables
        self.world = None
        self.object = None
        self.controller = None
        self.articulation_controller = None
        self.robot = None
        self.task = None
        self.stage = omni.usd.get_context().get_stage()
        self.use_physics = use_physics
        # Counters and timers
        self.grasps = []
        self.current_grasp = None
        # State tracking
        self.state = "INIT"  # States: INIT, GRASP, PLACE
        self.base_path = "/World/Franka"

        # Test data
        with open("/media/chris/OS2/Users/24330/Desktop/placement_quality/unseen/balanced_samples.json", "r") as f:
            self.test_data: list[dict] = json.load(f)

        self.current_data = None
        self.data_index = 0


    def start(self):
        # Simulation Environment Setup
        self.world: World = World(stage_units_in_meters=1.0)
        self.task = RRTTask("RRT_task", use_physics=self.use_physics)
        self.world.add_task(self.task)
        self.world.reset()
        
        # Robot and planner setup
        self.robot: Franka = self.world.scene.get_object(self.task.get_params()["robot_name"]["value"])
        self.articulation_controller = self.robot.get_articulation_controller()
        self.controller = RRTController(
            name="RRT_controller",
            robot_articulation=self.robot,
        )

        self.state = "GRASP"
        self.current_data = self.test_data[self.data_index]
        self.data_index += 1
        

        self.setup_collision_detection()
        self._add_light_to_stage()

        self.task.set_params(
            object_position=np.array([0.4, 0, self.current_data["initial_object_pose"][0]]),
            object_orientation=np.array(self.current_data["initial_object_pose"][1:]),
        )

    def reset(self):
        """Reset the simulation environment"""
        print("Resetting simulation environment, robot invisible")
        self.world.reset()
        self.controller.reset()
        # self.task.object_init(False)
        
        # Reset state flags
        self.state = "GRASP"
        self.task.set_params(
            object_position=np.array([0.4, 0, self.current_data["initial_object_pose"][0]]),
            object_orientation=np.array(self.current_data["initial_object_pose"][1:]),
        )

        self.start_logging = True
        self.data_recorded = False


    def calculate_final_grasp_pose(self):

        grasp_position_initial             = np.array(self.current_data["grasp_pose"][0:3])
        grasp_orientation_wxyz_initial     = self.current_data["grasp_pose"][3:7]

        # Initial object pose in world: [z, qw, qx, qy, qz]
        object_height_initial              = self.current_data["initial_object_pose"][0]
        object_orientation_wxyz_initial    = self.current_data["initial_object_pose"][1:5]
        # assume x=0.4, y=0.0
        object_position_initial            = np.array([0.4, 0.0, object_height_initial])

        # Final object pose in world: [z, qw, qx, qy, qz]
        object_height_final                = self.current_data["final_object_pose"][0]
        object_orientation_wxyz_final      = self.current_data["final_object_pose"][1:5]
        object_position_final              = np.array([0.4, 0.0, object_height_final])
    


        # scipy Rotation.from_quat expects [x, y, z, w]
        rotation_grasp_initial = R.from_quat([
            grasp_orientation_wxyz_initial[1],
            grasp_orientation_wxyz_initial[2],
            grasp_orientation_wxyz_initial[3],
            grasp_orientation_wxyz_initial[0]
        ]).as_matrix()

        rotation_object_initial = R.from_quat([
            object_orientation_wxyz_initial[1],
            object_orientation_wxyz_initial[2],
            object_orientation_wxyz_initial[3],
            object_orientation_wxyz_initial[0]
        ]).as_matrix()

        rotation_object_final = R.from_quat([
            object_orientation_wxyz_final[1],
            object_orientation_wxyz_final[2],
            object_orientation_wxyz_final[3],
            object_orientation_wxyz_final[0]
        ]).as_matrix()

        # --- 3) Compute the fixed hand‐to‐object transform ---------------

        # R_grasp_in_object = R_object_initial^T * R_grasp_initial
        rotation_grasp_in_object_frame = (
            rotation_object_initial.T @ rotation_grasp_initial
        )

        # p_grasp_in_object = R_object_initial^T * (p_grasp_initial – p_object_initial)
        translation_grasp_in_object_frame = (
            rotation_object_initial.T @ (grasp_position_initial - object_position_initial)
        )

        # --- 4) Re‐apply that to the new object pose --------------------

        # p_grasp_final = R_object_final * p_grasp_in_object + p_object_final
        final_grasp_position = (
            rotation_object_final @ translation_grasp_in_object_frame
            + object_position_final
        )

        # R_grasp_final = R_object_final * R_grasp_in_object
        rotation_final_grasp = (
            rotation_object_final @ rotation_grasp_in_object_frame
        )

        # --- 5) Convert back to quaternion [w, x, y, z] -----------------

        # scipy gives [x, y, z, w]
        quaternion_final_grasp_xyzw = R.from_matrix(
            rotation_final_grasp
        ).as_quat()

        # reorder to [w, x, y, z]
        quaternion_final_grasp_wxyz = np.array([
            quaternion_final_grasp_xyzw[3],
            quaternion_final_grasp_xyzw[0],
            quaternion_final_grasp_xyzw[1],
            quaternion_final_grasp_xyzw[2]
        ])

        # --- 6) Stack into your final grasp pose ------------------------

        # final_grasp_pose = np.concatenate([
        #     final_grasp_position,
        #     quaternion_final_grasp_wxyz
        # ])

        return final_grasp_position, quaternion_final_grasp_wxyz

    def _add_light_to_stage(self):
        """Add a spherical light to the stage"""
        sphereLight = UsdLux.SphereLight.Define(omni.usd.get_context().get_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])


    def setup_collision_detection(self):
        """Set up the ground collision detector"""
        stage = omni.usd.get_context().get_stage()
        self.collision_detector = GroundCollisionDetector(stage)
        
        # Create a virtual ground plane at z=-0.05 (lowered to avoid false positives)
        self.collision_detector.create_virtual_ground(
            size_x=20.0, 
            size_y=20.0, 
            position=Gf.Vec3f(0, 0, -0.001/2)  # Lower the ground to avoid false positives
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
        
    def check_for_collisions(self):
        """Check if any robot parts are colliding with the ground"""
        for part_path in self.robot_parts_to_check:
            if self.collision_detector.is_colliding_with_ground(part_path):
                self.collision_detected = True
                return True
            else:
                # print(f"No collision detected")
                return False
            