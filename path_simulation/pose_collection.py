import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from isaacsim import SimulationApp
import json
import time
from datetime import datetime

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

from typing import Optional, List, Dict

import numpy as np
import omni
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.utils import extensions, prims
from omni.isaac.core.utils.types import ArticulationAction
from transforms3d.euler import euler2quat, quat2euler
from omni.isaac.nucleus import get_assets_root_path
import random
from pxr import UsdPhysics
from omni.physx.scripts import physicsUtils

ROOT_PATH = get_assets_root_path() + "/Isaac/Props/YCB/Axis_Aligned/"

class PoseCollection:
    def __init__(self, num_poses=100, num_simultaneous=1000, output_file="ycb_stable_poses.json"):
        # Basic variables
        self.world = None
        self.scene = None
        self.objects = []  # Changed to list of objects
        self.num_poses = num_poses
        self.num_simultaneous = min(num_simultaneous, num_poses)  # Number of objects to drop at once
        self.output_file = output_file
        self.collected_poses = []
        self.object_name = "009_gelatin_box.usd"  # Fixed to one object
        self.settlement_threshold = 0.005  # Velocity threshold for considering object settled (m/s)
        self.settlement_time = 2.0  # Time (in seconds) object must be still to be considered settled
        self.drop_height = 1.5  # Height above ground to drop objects from (meters)
        # Area dimensions for distributing objects
        self.area_size = max(5.0, (self.num_simultaneous ** 0.5) * 0.3)  # Scale area based on object count
        
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
            
            # Apply physics properties directly to the prim
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
    
    def set_drop_poses(self):
        """Position the objects above the ground with random orientations in a grid pattern"""
        initial_poses = []
        
        # Calculate grid dimensions for distributing objects
        grid_size = int(np.ceil(np.sqrt(self.num_simultaneous)))
        spacing = self.area_size / grid_size
        
        # Place objects in a grid with small random offsets to prevent perfect alignment
        for i, obj in enumerate(self.objects):
            row = i // grid_size
            col = i % grid_size
            
            # Calculate position with small random offset
            x = (col - grid_size/2) * spacing + np.random.uniform(-0.05, 0.05)
            y = (row - grid_size/2) * spacing + np.random.uniform(-0.05, 0.05)
            z = self.drop_height
            
            # Random orientation (Euler angles in degrees)
            euler_angles_deg = [random.uniform(0, 360) for _ in range(3)]
            euler_angles_rad = np.deg2rad(euler_angles_deg)
            quat = euler2quat(*euler_angles_rad)
            
            # Set the initial pose
            pos = np.array([x, y, z])
            obj.set_world_pose(position=pos, orientation=quat)
            initial_poses.append((pos, quat))
        
        # Wait a physics step to ensure all poses are set
        self.world.step(render=True)
        return initial_poses
    
    def are_objects_settled(self):
        """Check if all objects have settled"""
        for obj in self.objects:
            linear_velocity = obj.get_linear_velocity()
            angular_velocity = obj.get_angular_velocity()
            
            linear_speed = np.linalg.norm(linear_velocity)
            angular_speed = np.linalg.norm(angular_velocity)
            
            # If any object is still moving significantly, return False
            if linear_speed >= self.settlement_threshold or angular_speed >= self.settlement_threshold * 10:
                return False
                
        return True
    
    def wait_for_settlement(self, max_wait_time=15.0):
        """Wait for all objects to settle on the ground"""
        start_time = time.time()
        
        # Step a few times before starting to check (let objects fall a bit)
        for _ in range(20):
            self.world.step(render=True)
        
        settled_count = 0
        last_report_time = start_time
        
        while time.time() - start_time < max_wait_time:
            self.world.step(render=True)
            
            # Check if all objects have settled
            if self.are_objects_settled():
                return True
                
            # Report progress every second
            current_time = time.time()
            if current_time - last_report_time > 1.0:
                print(f"Waiting for settlement... ({int(current_time - start_time)}s / {max_wait_time}s)")
                last_report_time = current_time
        
        # If we get here, objects didn't all settle within time limit
        print("Not all objects settled within time limit")
        return False
    
    def record_poses(self):
        """Record the current poses of all settled objects"""
        poses = []
        
        for i, obj in enumerate(self.objects):
            position = obj.get_world_pose()[0]
            orientation_quat = obj.get_world_pose()[1]
            
            # Check if object is on the ground (z close to 0)
            if position[2] > 0.5:  # Ignore objects that might be resting on top of others
                continue
                
            # Convert quaternion to Euler angles for easier interpretation
            euler_angles_rad = quat2euler(orientation_quat)
            euler_angles_deg = np.rad2deg(euler_angles_rad)
            
            pose_data = {
                "object_name": self.object_name,
                "object_id": i,
                "position": position.tolist(),
                "orientation_quat": orientation_quat.tolist(),
                "orientation_euler_deg": euler_angles_deg.tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
            poses.append(pose_data)
            self.collected_poses.append(pose_data)
            
        return poses
    
    def save_poses(self):
        """Save all collected poses to file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.collected_poses, f, indent=2)
        print(f"Saved {len(self.collected_poses)} poses to {self.output_file}")

def main():
    # Adjust these parameters as needed
    total_poses_needed = 1000
    simultaneous_objects = 100  # You can increase this based on your system's capabilities
    
    env = PoseCollection(num_poses=total_poses_needed, num_simultaneous=simultaneous_objects)
    env.start()
    
    poses_collected = 0
    batch_count = 0
    
    try:
        while simulation_app.is_running() and poses_collected < env.num_poses:
            batch_count += 1
            print(f"Batch {batch_count}: Dropping {len(env.objects)} objects simultaneously")
            
            # Set drop positions and orientations for all objects
            env.set_drop_poses()
            
            # Wait for objects to settle
            print("Waiting for objects to settle...")
            if env.wait_for_settlement(max_wait_time=15.0):
                # Record all stable poses
                batch_poses = env.record_poses()
                new_poses = len(batch_poses)
                poses_collected += new_poses
                print(f"Recorded {new_poses} stable poses (Total: {poses_collected}/{env.num_poses})")
                
                # If we've collected enough poses, break
                if poses_collected >= env.num_poses:
                    break
            else:
                print("Not all objects settled properly, but continuing...")
                # Still record what we can
                batch_poses = env.record_poses()
                new_poses = len(batch_poses)
                poses_collected += new_poses
                print(f"Recorded {new_poses} stable poses (Total: {poses_collected}/{env.num_poses})")
            
            # Small delay between batches
            time.sleep(0.5)
        
        # Save all collected poses
        env.save_poses()
        
    except KeyboardInterrupt:
        print("Collection interrupted by user")
        if env.collected_poses:
            env.save_poses()
    
    print("Pose collection complete")
    simulation_app.close()
    
if __name__ == "__main__":
    main()