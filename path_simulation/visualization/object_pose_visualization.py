import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from isaacsim import SimulationApp
import json
import time

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
from pxr import UsdPhysics
from omni.isaac.core.prims import XFormPrim

ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"

class PoseCollection:
    def __init__(self, num_poses=1000, num_simultaneous=1000, output_file="/home/chris/Chris/placement_ws/src/object_poses.json"):
        # Basic variables
        self.world = None
        self.scene = None
        self.objects = []  # Changed to list of objects
        self.num_poses = num_poses
        self.num_simultaneous = min(num_simultaneous, num_poses)  # Number of objects to drop at once
        self.output_file = output_file
        self.collected_poses = []
        self.object_name = "009_gelatin_box.usd"  # Fixed to one object
        self.settlement_threshold = 0.001  # Position change threshold in meters
        self.settlement_time = 1.5  # Time object must be still to be considered settled
        self.drop_height = 1.5  # Height above ground to drop objects from (meters)
        # Area dimensions for distributing objects
        self.area_size = max(5.0, (self.num_simultaneous ** 0.5) * 0.3)  # Scale area based on object count
        # Add tracking for previous positions
        self.previous_positions = {}
        self.position_history = {}
        self.position_timestamps = {}
        self.save_buffer = []  # Buffer to collect poses before saving
        self.save_buffer_size = 5000  # Save every 5000 poses (adjust as needed)
        
    def start(self):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.scene = self.world.scene
        self.scene.add_default_ground_plane()
        self.world.reset()
        self.create_objects()
        
    def create_objects(self):
        """Create multiple YCB objects as rigid bodies with physics enabled"""
        self.objects = []
        
        # Create the specified number of objects
        for i in range(self.num_simultaneous):
            prim_path = f"/World/Ycb_object_{i}"
            name = f"ycb_object_{i}"
            
            unique_prim_path = find_unique_string_name(
                initial_name=prim_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            unique_name = find_unique_string_name(
                initial_name=name, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            
            usd_path = ROOT_PATH + self.object_name
            
            # Add reference to stage and get the prim
            object_prim = add_reference_to_stage(usd_path=usd_path, prim_path=unique_prim_path)
            
            # # Apply physics properties directly to the prim
            UsdPhysics.RigidBodyAPI.Apply(object_prim)
            UsdPhysics.CollisionAPI.Apply(object_prim)
            
            # Create the RigidPrim object after applying physics properties
            obj = RigidPrim(
                prim_path=unique_prim_path,
                name=unique_name,
                position=np.array([0, 0, 0]),
                scale=[0.8, 0.8, 0.8]
            )
            
            # Make sure physics is enabled
            obj.enable_rigid_body_physics()
            
            self.scene.add(obj)
            self.objects.append(obj)
            
        return self.objects
    


    def sample_uniform_orientations(self, save_path="/home/chris/Chris/placement_ws/src/object_orientations.json", step_deg=0.5):
        """
        Returns a dict mapping each surface name to a list of quaternions
        sampled every `step_deg` degrees around the "free" axis.
        """
        # baseline Euler angles (deg) for each surface, and which index to vary:
        # roll (0), pitch (1), yaw (2)
        configs = {
            "top_surface":    (np.array([   0.0,   0.0,   0.0]), 2), # 0.01475
            "bottom_surface": (np.array([-180.0,   0.0,   0.0]), 2), # 0.01415
            "left_surface":   (np.array([  90.0,   0.0,   0.0]), 1), # 0.03586
            "right_surface":  (np.array([ -90.0,   0.0,   0.0]), 1), # 0.03644
            "front_surface":  (np.array([  90.0,   0.0,  90.0]), 1), # 0.04427
            "back_surface":   (np.array([  90.0,   0.0, -90.0]), 1), # 0.04447
        }

        # generate 720 samples from -180° to +179.5°
        angles = np.arange(-180, 180, step_deg)

        all_orients = {}
        for name, (base_euler, var_idx) in configs.items():
            quats = []
            for a in angles:
                e = base_euler.copy()
                e[var_idx] = a
                # convert to radians and then quaternion
                q = euler2quat(*np.deg2rad(e), axes='rxyz')
                quats.append(q)
            all_orients[name] = quats
        
        # Save orientations to file
        save_data = {}
        for surface, quaternions in all_orients.items():
            # Convert numpy arrays to lists for JSON serialization
            save_data[surface] = [q.tolist() for q in quaternions]
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"Saved {len(angles)} orientations per surface to {save_path}")

        return all_orients





def main():
    # Create 6 objects instead of just 1
    env = PoseCollection(num_poses=432, num_simultaneous=6)
    env.start()

    # Load the pose file
    poses = json.load(open("/home/chris/Chris/placement_ws/src/object_poses_v2.json"))
    total_poses = len(poses)
    
    # Number of poses per object
    poses_per_object = total_poses // 6  # Should be 72
    
    # Main simulation loop
    step = 0
    max_steps = poses_per_object  # We'll run for 72 steps
    
    while simulation_app.is_running():
        if step < max_steps:
            # Update each object with its corresponding pose
            for obj_idx, obj in enumerate(env.objects):
                # Calculate the pose index for this object at this step
                pose_idx = obj_idx * poses_per_object + step
                
                # Get position and orientation from the pose file
                position = poses[pose_idx]["position"].copy()
                # Offset each object horizontally so they don't overlap
                position[0] = 0.4 + obj_idx * 0.5  # Space objects along x-axis
                position[1] = 0.0  # Keep y position the same
                
                # Set the object's pose
                obj.set_world_pose(
                    position=position,
                    orientation=poses[pose_idx]["orientation_quat"]
                )
            
            print(f"Step: {step}/{max_steps} - displaying poses {step} to {step+6*poses_per_object-1}")
            
            # Advance simulation
            env.world.step(render=True)
            simulation_app.update()
            
            # Move to next step
            step += 1
            
            # Optional: add a small delay to make visualization clearer
            time.sleep(0.05)
        else:
            # Reset the loop to start over
            step = 0
    
    print("Pose visualization complete")
    simulation_app.close()
    
if __name__ == "__main__":
    main()