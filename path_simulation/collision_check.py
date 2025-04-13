import omni
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, PhysicsSchemaTools
import carb

class GroundCollisionDetector:
    """
    Detects collisions between objects and a virtual ground using mesh overlap checks.
    """
    
    def __init__(self, stage, ground_height=0.001):
        """
        Initialize the ground collision detector.
        """
        self.stage = stage
        self.ground_height = ground_height
        self.virtual_ground_path = None
        self.colliding_parts = set()  # Track which parts are colliding
        self.non_colliding_part = "/World/panda/panda_link0"
        
    def create_virtual_ground(self, path="/World/VirtualGround", 
                              size_x=10.0, size_y=10.0, 
                              position=Gf.Vec3f(0, 0, 0),
                              color=Gf.Vec3f(0.2, 0.2, 0.2)):
        """
        Create a thin box that represents the ground visually but has no physics properties.
        """
        self.virtual_ground_path = path
        
        # Create a thin box to represent the ground
        box_geom = UsdGeom.Cube.Define(self.stage, path)
        box_geom.CreateSizeAttr(1.0)
        
        # Set the scale to create a thin ground plane
        scale = Gf.Vec3f(size_x, size_y, self.ground_height)
        box_geom.AddScaleOp().Set(scale)
        
        # Set position and appearance
        box_geom.AddTranslateOp().Set(position)
        box_geom.CreateDisplayColorAttr().Set([color])
        
        return self.stage.GetPrimAtPath(path)
    
    def is_colliding_with_ground(self, robot_part_path):
        """
        Check if a robot part mesh is colliding with the virtual ground using mesh overlap.
        
        Args:
            robot_part_path: USD path to the robot part to check
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        if not self.virtual_ground_path:
            print("Virtual ground not created yet. Call create_virtual_ground first.")
            return False
        
        
        # Get scene query interface
        scene_query = get_physx_scene_query_interface()
        
        # Encode the robot part path for the mesh overlap query
        try:
            # Encode the robot part path for PhysX
            path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.virtual_ground_path))
            
            # This will store our collision result
            collision_detected = [False]
            
            # Callback function for when a hit is detected
            def report_hit(hit):         
                # More flexible comparison approach
                hit_path_str = str(hit.rigid_body)
                if hit_path_str != self.non_colliding_part:
                    collision_detected[0] = True
                    self.colliding_parts.add(hit_path_str)
                    print(f"Right now, the colliding parts are: {self.colliding_parts}")
                    
                    # Change the color of the virtual ground instead of the robot parts
                    try:
                        ground_geom = UsdGeom.Cube(self.stage.GetPrimAtPath(self.virtual_ground_path))
                        if ground_geom:
                            ground_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red ground
                    except Exception as e:
                        print(f"Error changing ground color: {e}")
                
                return True  # Continue checking other overlaps
            
            # Perform the mesh overlap check
            num_hits = scene_query.overlap_shape(
                path_tuple[0],  # Encoded USD path identifier
                path_tuple[1],  # Encoded instance identifier
                report_hit,           # Callback when hit is found
                False                 # Don't return immediately after first hit
            )
            

            # Reset ground color when no collisions are detected
            if not collision_detected[0]:
                try:
                    ground_geom = UsdGeom.Cube(self.stage.GetPrimAtPath(self.virtual_ground_path))
                    if ground_geom:
                        ground_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])  # Default gray
                except Exception as e:
                    print(f"Error resetting ground color: {e}")
        
            return num_hits
            
        except Exception as e:
            print(f"Error in mesh overlap detection: {e}")
            return False
    
    def is_any_part_colliding(self, robot_parts_paths):
        """
        Check if any part of the robot is colliding with the virtual ground.
        """
        return any(self.is_colliding_with_ground(part) for part in robot_parts_paths)


# Example usage:
"""
# Setup the stage
stage = omni.usd.get_context().get_stage()

# Create the detector
detector = GroundCollisionDetector(stage)

# Create a virtual ground
detector.create_virtual_ground(size_x=20.0, size_y=20.0, position=Gf.Vec3f(0, 0, -0.0005))

# In your simulation loop
robot_path = "/World/Robot"
if detector.is_colliding_with_ground(robot_path):
    print("Robot is colliding with ground!")

# Check multiple parts
robot_parts = ["/World/Robot/Base", "/World/Robot/Arm1", "/World/Robot/Gripper"]
if detector.is_any_part_colliding(robot_parts):
    print("At least one robot part is colliding with ground!")
"""
