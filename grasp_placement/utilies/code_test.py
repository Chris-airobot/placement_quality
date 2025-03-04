import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def compute_viewpoint(object_center, radius, azimuth_deg, elevation_deg):
    """
    Compute a viewpoint on a sphere given an object center, radius, azimuth, and elevation.
    """
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    x = object_center[0] + radius * math.cos(elevation) * math.cos(azimuth)
    y = object_center[1] + radius * math.cos(elevation) * math.sin(azimuth)
    z = object_center[2] + radius * math.sin(elevation)
    return np.array([x, y, z])

def compute_lookat_orientation(camera_position, target_position, up_vector=np.array([0, 0, 1])):
    """
    Compute a quaternion (in [w, x, y, z] order) so that a camera at camera_position
    will look at target_position. If the forward vector is nearly parallel to up_vector,
    a fallback up vector is used to avoid degenerate cross products.
    """
    forward = target_position - camera_position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return np.array([1, 0, 0, 0])  # No valid orientation if camera == target

    forward = forward / forward_norm
    # If forward is nearly parallel to up_vector, pick a different up.
    if abs(np.dot(forward, up_vector)) > 0.99:
        up_vector = np.array([0, 1, 0])  # fallback up

    right = np.cross(up_vector, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / right_norm

    true_up = np.cross(forward, right)
    R_mat = np.column_stack((right, true_up, forward))
    
    # as_quat() returns [x, y, z, w]
    quat_xyzw = R.from_matrix(R_mat).as_quat()
    # Convert to [w, x, y, z]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz

if __name__ == '__main__':
    # Example: center of a cube at (x,y,z)
    object_center = np.array([-0.21, -0.28, 0.03])
    # Distance from the object (radius of the sampling sphere)
    radius = 0.4

    # Sample multiple angles:
    #   - Azimuth from 0 to 360 in increments
    #   - Elevation: e.g. 30°, 45°, 60° (above the horizontal plane)
    azimuth_samples = 8
    elevation_angles = [30, 45, 60]

    waypoints_positions = []
    waypoints_orientations = []

    for elevation_deg in elevation_angles:
        for i in range(azimuth_samples):
            az = (360.0 / azimuth_samples) * i
            cam_pos = compute_viewpoint(object_center, radius, az, elevation_deg)
            cam_orient = compute_lookat_orientation(cam_pos, object_center)

            waypoints_positions.append(cam_pos)
            waypoints_orientations.append(cam_orient)

    # Print out the results
    for idx, (pos, orient) in enumerate(zip(waypoints_positions, waypoints_orientations)):
        print(f"Waypoint {idx}:")
        print(f"  Position: {pos}")
        print(f"  Orientation (w,x,y,z): {orient}")
