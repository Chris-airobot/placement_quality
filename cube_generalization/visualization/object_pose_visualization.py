import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from isaacsim import SimulationApp
import json
import time
from scipy.spatial.transform import Rotation as R

DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON   = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height":1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
    "physics_dt": 1.0/60.0,  # Physics timestep (60Hz)
    "rendering_dt": 1.0/30.0,  # Rendering timestep (30Hz)
}

simulation_app = SimulationApp(CONFIG)


import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from transforms3d.euler import euler2quat
from omni.isaac.nucleus import get_assets_root_path
import random
from pxr import UsdPhysics, Gf, UsdGeom
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.objects import VisualCuboid
from utils import sample_fixed_surface_poses_increments, sample_cuboid_dimensions, flip_pose_keep_delta





class PoseVisualization:
    def __init__(self, num_poses=1000, num_simultaneous=1000):
        # Basic variables
        self.world = None
        self.scene = None
        self.objects = []  # Changed to list of objects
        self.num_poses = num_poses
        self.num_simultaneous = min(num_simultaneous, num_poses)  # Number of objects to drop at once

        self.box_dims = sample_cuboid_dimensions()
        

    def start(self, ycb=True):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.scene = self.world.scene
        self.scene.add_default_ground_plane()
        self.world.reset()
        self.create_objects(ycb)


        
    def create_objects(self, ycb=True):
        """Create multiple YCB objects as rigid bodies with physics enabled"""
        self.objects = []
        
        # Create the specified number of objects
        for i in range(self.num_simultaneous):
            prim_path = f"/World/Cube_{i}"
            name = f"cube_{i}"
            
            obj = VisualCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array([0, 0, 0]),
                    scale=self.box_dims,  # [x, y, z]
                    color=np.array([0.8, 0.8, 0.8])  # Default gray
                )
            
            self.scene.add(obj)
            self.objects.append(obj)
            
        return self.objects
    

def main():
    # Create 6 objects instead of just 1
    env = PoseVisualization(num_poses=72, num_simultaneous=12)
    env.start(ycb=False)
    max_steps = 100000000000000000000000000000000000000000
    step = 0

    # Generate base poses and their flipped variants, then combine them
    poses = sample_fixed_surface_poses_increments(env.box_dims)
    flip_poses = [flip_pose_keep_delta(p, env.box_dims, num_turns=3) for p in poses]

    all_poses = poses + flip_poses  # visualise both sets
    total_poses = len(all_poses)

    objects_count = len(env.objects)
    # How many refresh steps until we have shown every pose once
    poses_per_object = (total_poses + objects_count - 1) // objects_count
    
    print("\n=== Surface Label Visualization ===")
    print("Color Legend:")
    print("  Red = Z-up (top face pointing up)")
    print("  Green = Z-down (bottom face pointing up)")
    print("  Blue = X-up (side face pointing up)")
    print("  Yellow = X-down (opposite side pointing up)")
    print("  Magenta = Y-up (front face pointing up)")
    print("  Cyan = Y-down (back face pointing up)")
    print("=====================================\n")
    
    while simulation_app.is_running():
        if step < max_steps:
            # Update each object with its corresponding pose
            for obj_idx, obj in enumerate(env.objects):
                # Determine pair index (0-5) and whether this object shows the
                # flipped variant (odd indices) or the original (even indices).
                pair_idx   = obj_idx // 2
                is_flipped = (obj_idx % 2) == 1

                # Select the pose accordingly
                pose = flip_poses[pair_idx] if is_flipped else poses[pair_idx]

                # Base XY location for the pair (x coordinate grows by 0.5)
                base_x = 0.5 * (pair_idx + 1)

                position = pose[:3]
                position[0] = base_x
                # Put original and flipped at different Y positions
                position[1] = 0.4 if is_flipped else 0.2
                orientation = pose[3:]
                
                
                # Set the object's pose
                obj.set_world_pose(
                    position=position,
                    orientation=orientation
                )
                
    
            
            # Advance simulation
            env.world.step(render=True)
            simulation_app.update()
            
            # Move to next step
            step += 1
            
            # Optional: add a small delay to make visualization clearer
        else:
            # Reset the loop to start over
            step = 0
            print("\n=== Restarting visualization loop ===\n")
    
    print("Pose visualization complete")
    simulation_app.close()
    
if __name__ == "__main__":
    main()