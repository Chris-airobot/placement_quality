from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from omni.isaac.core.objects import FixedCuboid, VisualCuboid, DynamicCuboid
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units, add_reference_to_stage
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.franka import Franka
from omni.isaac.nucleus import get_assets_root_path

from pxr import UsdPhysics
class RRTTask(BaseTask):
    def __init__(
        self,
        name: str,
        use_physics: bool=False,
        initial_position: Optional[np.ndarray] = None,
        initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseTask.__init__(self, name=name, offset=offset)
        self._robot = None
        self.use_physics = use_physics
        self._ycb_name = None
        self._ycb = None
        self._ycb_prim_path = None
        self._frame = None
        self._ycb_initial_position = initial_position
        self._ycb_initial_orientation = initial_orientation
        self._ycb_target_position = target_position
        self._ycb_target_orientation = target_orientation
        self._obstacle_walls = OrderedDict()
        self.ground_plane = None
        return

    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        if self.use_physics:
            scene.add_default_ground_plane()

            self.ground_plane = DynamicCuboid(
                prim_path="/World/CollisionGround",
                name="collision_ground",
                position=np.array([0.0, 0.0, -0.0005]),  # Match your visual ground position
                scale=np.array([20.0, 20.0, 0.001]),     # Match size and thickness
                color=np.array([0.0, 0.0, 0.0])     # Make it invisible if you want (alpha=0)
            )
            scene.add(self.ground_plane)

        # Add the object to the scene
        self._ycb = self.set_ycb(name="ycb_object", prim_path="/World/Ycb_object", use_physics=self.use_physics)
        scene.add(self._ycb)

        # Add the target to the scene
        # self._target = self.set_ycb(name="target_object", prim_path="/World/Target_object")
        # scene.add(self._target)

        # Add the robot to the scene
        self._robot = self.set_robot()
        scene.add(self._robot)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._frame = XFormPrim("/World/target", scale=[.04,.04,.04], name="target")
        self._frame.set_default_state(np.array([0.3, 0, 0.5]),
                                np.array([0, 0, 0, 1]))
        
        scene.add(self._frame)

        # Set the object parameters
        self.set_params(
            object_position=self._ycb_initial_position,
            object_orientation=self._ycb_initial_orientation,
        )

        # Add the robot and object to the task objects
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._ycb.name] = self._ycb
        # self._task_objects[self._target.name] = self._target

        # Move the task objects to their frame
        self._move_task_objects_to_their_frame()
        return



    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        
        self._franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=self._franka_prim_path, name=self._franka_robot_name)


    def set_ycb(self, name: str, prim_path: str, use_physics: bool=False) -> XFormPrim:
        """[summary]

        Returns:
            [type]: [description]
        """
        selected_object = "009_gelatin_box.usd"
        ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"

        usd_path = ROOT_PATH + selected_object

        # Create the initial object.
        self._ycb_prim_path = find_unique_string_name(
            initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        self._ycb_name = find_unique_string_name(
            initial_name=name, is_unique_fn=lambda x: not self.scene.object_exists(x)
        )

        object_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        if use_physics:
            import omni
            import omni.physx
            from pxr import UsdGeom, UsdShade, PhysxSchema
            stage = omni.usd.get_context().get_stage()

            # -- REMOVE existing collision/mesh colliders under object_prim
            children = [c for c in stage.GetPrimAtPath(prim_path).GetChildren()]
            for child in children:
                # Remove any child prims named 'collision', 'collider', or that are meshes
                if 'collis' in child.GetName().lower() or child.GetTypeName() in ['Mesh', 'TriangleMesh', 'GeomMesh']:
                    print(f"Removing existing collider: {child.GetPath()}")
                    stage.RemovePrim(child.GetPath())

            # Set RigidBody and Mass
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(object_prim)
            UsdPhysics.CollisionAPI.Apply(object_prim)
            mass_api = UsdPhysics.MassAPI.Apply(object_prim)
            mass_api.CreateMassAttr(0.086)
            
            # ---- PhysX Material ----
            material_path = prim_path + "/physx_material"

            for mat_type in ["PhysxSchema.PhysxMaterial", "PhysicsMaterial"]:
                if not stage.GetPrimAtPath(material_path).IsValid():
                    mat_prim = stage.DefinePrim(material_path, mat_type)
                else:
                    mat_prim = stage.GetPrimAtPath(material_path)
                # Set attributes if they exist
                try:
                    mat_prim.GetAttribute("physxStaticFriction").Set(2.0)
                    mat_prim.GetAttribute("physxDynamicFriction").Set(2.0)
                    mat_prim.GetAttribute("physxRestitution").Set(0.0)
                    break  # Succeed and exit loop
                except Exception:
                    continue

            # Create a cube under /World/Ycb_object
            collider_path = prim_path + "/box_collider"
            collider = UsdGeom.Cube.Define(stage, collider_path)
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

            # Set the cube to be scaled as the gelatin box dimensions (YCB 009: 0.073 x 0.032 x 0.101 meters)
            collider.CreateSizeAttr(1.0)
            collider.AddScaleOp().Set((0.08, 0.065, 0.025))
            collider.AddTranslateOp().Set((0, 0, 0))  # Centered; adjust if necessary
            # collider.GetPrim().GetAttribute('visibility').Set('invisible')

            # ---- Material Binding ----
            def bind_physx_material(prim, mat_prim):
                binding_api = UsdShade.MaterialBindingAPI(prim)
                binding_api.Bind(UsdShade.Material(mat_prim))

            bind_physx_material(object_prim, mat_prim)
            bind_physx_material(collider.GetPrim(), mat_prim)


        return XFormPrim(prim_path=prim_path, name=name, )


    def set_params(
        self,
        object_position: Optional[np.ndarray] = None,
        object_orientation: Optional[np.ndarray] = None,
        # object_target_position: Optional[np.ndarray] = None,
        # object_target_orientation: Optional[np.ndarray] = None,
    ) -> None:
        """Set the object parameters."""
        if object_position is not None or object_orientation is not None:
            self._ycb.set_world_pose(position=object_position, orientation=object_orientation)
        # if object_target_position is not None or object_target_orientation is not None:
        #     self._target.set_world_pose(position=object_target_position, orientation=object_target_orientation)
        return

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["object_initial_position"] = {"value": self._ycb_initial_position, "modifiable": True}
        params_representation["object_initial_orientation"] = {"value": self._ycb_initial_orientation, "modifiable": True}
        params_representation["object_target_position"] = {"value": self._ycb_target_position, "modifiable": True}
        params_representation["object_target_orientation"] = {"value": self._ycb_target_orientation, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def get_task_objects(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return self._task_objects

    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        # ycb_position, ycb_orientation = self._ycb.get_local_pose()
        return {
            self._robot.name: {
                "joint_positions": np.array(joints_state.positions),
                "joint_velocities": np.array(joints_state.velocities),
            },
            # self._ycb.name: {"position": np.array(ycb_position), "orientation": np.array(ycb_orientation)},
        }

    def target_reached(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        end_effector_position, _ = self._robot.end_effector.get_world_pose()
        ycb_position, _ = self._ycb.get_world_pose()
        if np.mean(np.abs(np.array(end_effector_position) - np.array(ycb_position))) < (0.035 / get_stage_units()):
            return True
        else:
            return False


    def add_obstacle(self, position: np.ndarray = None, orientation=None):
        """[summary]

        Args:
            position (np.ndarray, optional): [description]. Defaults to np.array([0.1, 0.1, 1.0]).
        """
        # TODO: move to task frame if there is one
        cube_prim_path = find_unique_string_name(
            initial_name="/World/WallObstacle", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        cube_name = find_unique_string_name(initial_name="wall", is_unique_fn=lambda x: not self.scene.object_exists(x))
        if position is None:
            position = np.array([0.6, 0.1, 0.2]) / get_stage_units()
        if orientation is None:
            orientation = euler_angles_to_quat(np.array([0, 0, np.pi / 3]))
        cube = self.scene.add(
            VisualCuboid(
                name=cube_name,
                position=position + self._offset,
                orientation=orientation,
                prim_path=cube_prim_path,
                size=1.0,
                scale=np.array([0.1, 0.5, 0.6]) / get_stage_units(),
                color=np.array([0, 0, 1.0]),
            )
        )
        self._obstacle_walls[cube.name] = cube
        return cube

    def remove_obstacle(self, name: Optional[str] = None) -> None:
        """[summary]

        Args:
            name (Optional[str], optional): [description]. Defaults to None.
        """
        if name is not None:
            self.scene.remove_object(name)
            del self._obstacle_walls[name]
        else:
            obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]
        return

    def get_obstacles(self) -> List:
        return list(self._obstacle_walls.values())

    def get_obstacle_to_delete(self) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        obstacle_to_delete = list(self._obstacle_walls.keys())[-1]
        return self.scene.get_object(obstacle_to_delete)

    def obstacles_exist(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if len(self._obstacle_walls) > 0:
            return True
        else:
            return False

    def cleanup(self) -> None:
        """[summary]"""
        obstacles_to_delete = list(self._obstacle_walls.keys())
        for obstacle_to_delete in obstacles_to_delete:
            self.scene.remove_object(obstacle_to_delete)
            del self._obstacle_walls[obstacle_to_delete]
        return



    def get_custom_gains(self) -> Tuple[np.array, np.array]:
        return (1e15 * np.ones(9), 1e13 * np.ones(9))


# class FrankaPathPlanningTask(RRTTask):
#     def __init__(
#         self,
#         name: str,
#         ycb_prim_path: Optional[str] = None,
#         ycb_name: Optional[str] = None,
#         ycb_position: Optional[np.ndarray] = None,
#         ycb_orientation: Optional[np.ndarray] = None,
#         offset: Optional[np.ndarray] = None,
#         franka_prim_path: Optional[str] = None,
#         franka_robot_name: Optional[str] = None,
#     ) -> None:
#         RRTTask.__init__(
#             self,
#             name=name,
#             ycb_prim_path=ycb_prim_path,
#             ycb_name=ycb_name,
#             ycb_position=ycb_position,
#             ycb_orientation=ycb_orientation,
#             offset=offset,
#         )
#         self._franka_prim_path = franka_prim_path
#         self._franka_robot_name = franka_robot_name
#         self._franka = None
#         return


    