import open3d as o3d
import numpy as np
import os


def obtain_mesh_from_usd():
    """
    This function obtains the mesh from the USD file and saves it as an OBJ file.
    IMPORTANT: The unit of the mesh is in centimeters.
    """
    from pxr import Usd, UsdGeom
    import omni.usd

    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()

    # Specify the USD path to your mesh object
    mesh_path = "/World/_09_gelatin_box/_09_gelatin_box"  # Adjust this to your object's prim path
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim:
        raise ValueError(f"No prim found at path: {mesh_path}")

    usd_mesh = UsdGeom.Mesh(mesh_prim)
    if not usd_mesh:
        raise ValueError("The prim at the specified path is not a mesh.")

    # Retrieve vertex positions, face counts, and face indices
    points = usd_mesh.GetPointsAttr().Get()
    face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
    face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
    min_point = [min(p[i] for p in points) for i in range(3)]
    max_point = [max(p[i] for p in points) for i in range(3)]
    print(f"USD mesh bounds: min={min_point}, max={max_point}")

    # Open a file to write the OBJ data
    output_path = "/home/chris/Desktop/MyObject.obj"  # Adjust the path accordingly
    with open(output_path, "w") as f:
        # Write vertices
        for pt in points:
            f.write("v {} {} {}\n".format(pt[0], pt[1], pt[2]))
        # Write faces
        index_offset = 0
        for count in face_counts:
            # OBJ indices start at 1, so add 1 to each index
            indices = [str(i + 1) for i in face_indices[index_offset:index_offset + count]]
            f.write("f " + " ".join(indices) + "\n")
            index_offset += count

    print(f"Mesh exported to {output_path}")

def sample_points_from_mesh(mesh_path, output_path, scale_factor=0.01):
    """
    This function samples points from the mesh and saves them as a PCD file.
    IMPORTANT: The unit of the mesh is in centimeters.
    """
    # Load the mesh (replace with your mesh file or data extracted from Isaac Sim)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError("The loaded mesh does not have any triangles.")

    # Optionally, compute vertex normals for a better visualization
    mesh.compute_vertex_normals()

    # Sample points uniformly from the mesh surface
    # Adjust number_of_points for your desired density
    number_of_points = 10000
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    
    # Scale down the point cloud
    points = np.asarray(pcd.points)
    points = points * scale_factor  # Scale down by factor (e.g., 0.01 for cm to m)
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add color information to the point cloud - white color for all points
    # This ensures 'rgba' field will be present when loaded in ROS
    colors = np.ones((len(pcd.points), 3))  # RGB values between 0 and 1 (white)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"number of points: {len(pcd.points)}")
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    # Save the point cloud to a file (PCD or PLY format)
    o3d.io.write_point_cloud(output_path, pcd)


########################
# TF conversion helpers
########################

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion given as a numpy array [x, y, z, w].
    """
    return q / np.linalg.norm(q)

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert a normalized quaternion [x, y, z, w] into a 3x3 rotation matrix.
    """
    x, y, z, w = q
    # Pre-compute repeated values.
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([
        [1 - 2*(yy + zz),    2*(xy - wz),    2*(xz + wy)],
        [   2*(xy + wz), 1 - 2*(xx + zz),    2*(yz - wx)],
        [   2*(xz - wy),    2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    return R

def create_object_to_world_tf(translation: list, quaternion: list) -> np.ndarray:
    """
    Creates a 4x4 homogeneous transformation matrix from translation and quaternion.
    
    Parameters:
      translation: [x, y, z]
      quaternion: [x, y, z, w]
      
    Returns:
      A 4x4 numpy array representing the transformation.
    """
    q = normalize_quaternion(np.array(quaternion))
    R = quaternion_to_rotation_matrix(q)
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = R
    tf_matrix[:3, 3] = np.array(translation)
    return tf_matrix

########################
# Point Cloud Utilities
########################

def invert_transform(transform: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix.
    """
    return np.linalg.inv(transform)

def transform_point_cloud(pcd: o3d.geometry.PointCloud, transform: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Applies a 4x4 transformation to a point cloud. Points are converted to homogeneous coordinates.
    """
    points = np.asarray(pcd.points)
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack((points, ones))
    transformed_points = (transform @ homogeneous_points.T).T[:, :3]
    
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    if pcd.has_colors():
        transformed_pcd.colors = pcd.colors
    if pcd.has_normals():
        transformed_pcd.normals = pcd.normals
    return transformed_pcd

def center_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Centers the point cloud so its centroid is at the origin.
    """
    points = np.asarray(pcd.points)
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    centered_pcd = o3d.geometry.PointCloud()
    centered_pcd.points = o3d.utility.Vector3dVector(centered_points)
    if pcd.has_colors():
        centered_pcd.colors = pcd.colors
    if pcd.has_normals():
        centered_pcd.normals = pcd.normals
    return centered_pcd

def compute_scaling_factor(reference_pcd: o3d.geometry.PointCloud, mesh_pcd: o3d.geometry.PointCloud) -> float:
    """
    Compute a uniform scaling factor by comparing the axis-aligned bounding box extents
    of the reference point cloud (correctly scaled, e.g., from cameras in local frame)
    and the mesh-based point cloud (perfect shape but with wrong scale).
    """
    bbox_ref = reference_pcd.get_axis_aligned_bounding_box()
    extent_ref = np.array(bbox_ref.get_extent())
    
    bbox_mesh = mesh_pcd.get_axis_aligned_bounding_box()
    extent_mesh = np.array(bbox_mesh.get_extent())
    
    ratios = extent_ref / extent_mesh
    scale_factor = float(np.mean(ratios))
    return scale_factor

def create_perfect_pcd(mesh_pcd_file: str,
                       camera_pcd_file: str,
                       object_to_world_tf: np.ndarray,
                       center: bool = True) -> o3d.geometry.PointCloud:
    """
    Using three sources:
      - mesh_pcd_file: A perfect-shaped point cloud from the mesh (but with wrong scaling).
      - camera_pcd_file: A point cloud from simulation cameras (correct scale but incomplete).
      - object_to_world_tf: The transformation (4x4) from object frame to world frame.
      
    This function transforms the camera point cloud into the object's local frame,
    computes a uniform scaling factor by comparing bounding boxes, applies that
    factor to the mesh point cloud, and then optionally centers it.
    """
    # Load point clouds
    pcd_mesh = o3d.io.read_point_cloud(mesh_pcd_file)
    pcd_cam = o3d.io.read_point_cloud(camera_pcd_file)
    
    # Add debug print statements to check point clouds and transformations
    print(f"Mesh PCD loaded: {len(pcd_mesh.points)} points")
    print(f"Camera PCD loaded: {len(pcd_cam.points)} points")
    
    # Bring the camera point cloud to the object's local frame.
    world_to_object_tf = invert_transform(object_to_world_tf)
    pcd_cam_local = transform_point_cloud(pcd_cam, world_to_object_tf)
    print(f"Camera PCD after transformation: {len(pcd_cam_local.points)} points")
    
    # Debug: Check bounds before scaling
    bbox_cam = pcd_cam_local.get_axis_aligned_bounding_box()
    bbox_mesh = pcd_mesh.get_axis_aligned_bounding_box()
    print(f"Camera PCD bounds: {bbox_cam.min_bound} to {bbox_cam.max_bound}")
    print(f"Mesh PCD bounds: {bbox_mesh.min_bound} to {bbox_mesh.max_bound}")
    
    # Compute scaling factor by comparing extents of the local camera PCD and the mesh PCD.
    scale_factor = compute_scaling_factor(pcd_cam_local, pcd_mesh)
    print("Computed scaling factor:", scale_factor)
    
    # Add safety check for scaling factor
    if scale_factor < 0.001 or scale_factor > 1000:
        print("WARNING: Extreme scaling factor detected. Using default value of 1.0")
        scale_factor = 1.0
    
    # Apply the scaling factor to the mesh point cloud.
    mesh_points = np.asarray(pcd_mesh.points)
    scaled_mesh_points = mesh_points * scale_factor
    pcd_mesh_scaled = o3d.geometry.PointCloud()
    pcd_mesh_scaled.points = o3d.utility.Vector3dVector(scaled_mesh_points)
    if pcd_mesh.has_colors():
        pcd_mesh_scaled.colors = pcd_mesh.colors
        
    # Center the point cloud if desired.
    if center:
        pcd_mesh_scaled = center_point_cloud(pcd_mesh_scaled)
    
    # Debug: Check bounds after scaling
    bbox_scaled = pcd_mesh_scaled.get_axis_aligned_bounding_box()
    print(f"Scaled mesh PCD bounds: {bbox_scaled.min_bound} to {bbox_scaled.max_bound}")
    
    return pcd_mesh_scaled

def create_coordinate_frame(size=0.1, origin=[0, 0, 0]):
    """
    Create a coordinate frame visualization object with the given size.
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

def main():
    # Set the file paths of your point clouds.
    # 'mesh_pcd_file' is obtained from the mesh (perfect shape but scaling is off).
    # 'camera_pcd_file' is captured from simulation cameras (correct scale but incomplete).
    mesh_pcd_file = "/home/chris/Desktop/MyObject.pcd"
    camera_pcd_file = "/home/chris/Chris/placement_ws/src/data/YCB_data/run_20250408_221137/Pcd_1/pointcloud.pcd"
    
    # Verify files exist before proceeding
    if not os.path.exists(mesh_pcd_file):
        print(f"ERROR: Mesh PCD file not found: {mesh_pcd_file}")
        return
    if not os.path.exists(camera_pcd_file):
        print(f"ERROR: Camera PCD file not found: {camera_pcd_file}")
        return
    
    # Provide the object's TF (pose in world) when the camera PCD was obtained.
    # Replace the following with your actual transformation.
    translation = [0.11550285667181015, 0.31492993235588074, 0.029152002185583115]
    quaternion = [-0.17946921452822076, -0.6818775030728548, 0.6855304330626114, 0.18133366258017514]

    object_to_world_tf = create_object_to_world_tf(translation, quaternion)
    print("Object to World Transformation Matrix:\n", object_to_world_tf)
    
    # Create the perfect point cloud:
    perfect_pcd = create_perfect_pcd(mesh_pcd_file,
                                     camera_pcd_file,
                                     object_to_world_tf,
                                     center=True)
    
    # Load original point clouds for comparison
    pcd_mesh = o3d.io.read_point_cloud(mesh_pcd_file)
    pcd_cam = o3d.io.read_point_cloud(camera_pcd_file)
    
    # Create coordinate frames for each point cloud
    coord_frame_size = 0.05  # Adjust size as needed
    origin_frame = create_coordinate_frame(size=coord_frame_size)
    
    # Set distinct colors for each point cloud
    perfect_pcd_vis1 = o3d.geometry.PointCloud(perfect_pcd)
    perfect_pcd_vis1.paint_uniform_color([0, 0.8, 0])  # Bright green
    
    # Visualization 1: Perfect PCD with coordinate frame (improved colors)
    print("\n=== Visualizing Perfect PCD with coordinate frame ===")
    print("Press 'Q' to close the visualizer and continue.")
    
    # Use custom visualization with better settings for the first view
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="Perfect Point Cloud with Coordinate Frame")
    vis1.add_geometry(perfect_pcd_vis1)
    vis1.add_geometry(origin_frame)
    
    # Improve visualization settings
    opt1 = vis1.get_render_option()
    opt1.point_size = 5.0  # Larger points for better visibility
    opt1.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    vis1.reset_view_point(True)
    vis1.run()
    vis1.destroy_window()
    
    # Visualization 2: All point clouds together with clear color coding
    print("\n=== Visualizing all point clouds together for size comparison ===")
    print("COLOR LEGEND:")
    print("  BRIGHT RED: Original mesh point cloud")
    print("  BRIGHT BLUE: Camera point cloud (local frame)")
    print("  BRIGHT GREEN: Perfect scaled mesh point cloud (centered)")
    print("Press 'Q' to close the visualizer and continue.")
    
    # Create copies for visualization to avoid modifying the originals
    pcd_mesh_vis = o3d.geometry.PointCloud(pcd_mesh)
    
    # FIXED: Use the correct transformation (world to object, not object to world)
    world_to_object_tf = invert_transform(object_to_world_tf)
    pcd_cam_local_vis = transform_point_cloud(pcd_cam, world_to_object_tf)
    
    perfect_pcd_vis2 = o3d.geometry.PointCloud(perfect_pcd)
    
    # Apply very distinct colors
    pcd_mesh_vis.paint_uniform_color([1, 0, 0])       # Bright red
    pcd_cam_local_vis.paint_uniform_color([0, 0, 1])  # Bright blue
    perfect_pcd_vis2.paint_uniform_color([0, 1, 0])   # Bright green
    
    # Set up custom visualizer for the second view
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Point Cloud Comparison")
    
    # Add all geometries
    vis2.add_geometry(perfect_pcd_vis2)
    vis2.add_geometry(pcd_mesh_vis)
    vis2.add_geometry(pcd_cam_local_vis)
    vis2.add_geometry(origin_frame)
    
    # Improve visualization settings
    opt2 = vis2.get_render_option()
    opt2.point_size = 5.0  # Larger points for better visibility
    opt2.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    vis2.reset_view_point(True)
    vis2.run()
    vis2.destroy_window()
    
    # Visualization 3: Side-by-side comparison
    print("\n=== Visualizing side-by-side comparison ===")
    print("COLOR LEGEND:")
    print("  BLUE: Camera point cloud (local frame)")
    print("  GREEN: Perfect scaled mesh point cloud (centered)")
    print("Press 'Q' to close the visualizer.")
    
    # Create new objects for the side-by-side view
    pcd_cam_local_vis3 = o3d.geometry.PointCloud(pcd_cam_local_vis)
    perfect_pcd_right = o3d.geometry.PointCloud(perfect_pcd_vis2)
    
    # Move the perfect PCD to the right for side-by-side comparison
    points = np.asarray(perfect_pcd_right.points)
    
    # FIXED: Calculate a more reasonable separation distance
    # Get the bounding box sizes first
    bbox_cam = pcd_cam_local_vis3.get_axis_aligned_bounding_box()
    bbox_perfect = perfect_pcd_right.get_axis_aligned_bounding_box()
    
    # Use a smaller multiplier (1.2 instead of 1.5) and average the extents
    # This will place them closer together
    cam_extent = bbox_cam.get_extent()
    perfect_extent = bbox_perfect.get_extent()
    move_distance = max(cam_extent[0], perfect_extent[0]) * 1.2
    
    # Print the move distance for debugging
    print(f"Moving perfect point cloud by {move_distance} units")
    
    points[:, 0] += move_distance
    perfect_pcd_right.points = o3d.utility.Vector3dVector(points)
    
    # Add coordinate frame for the moved point cloud
    moved_frame = create_coordinate_frame(size=coord_frame_size, origin=[move_distance, 0, 0])
    
    # Set up custom visualizer for the third view
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name="Side-by-Side Comparison")
    
    # Add geometries
    vis3.add_geometry(pcd_cam_local_vis3)
    vis3.add_geometry(perfect_pcd_right)
    vis3.add_geometry(origin_frame)
    vis3.add_geometry(moved_frame)
    
    # Improve visualization settings
    opt3 = vis3.get_render_option()
    opt3.point_size = 5.0  # Larger points for better visibility
    opt3.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    vis3.reset_view_point(True)
    vis3.run()
    vis3.destroy_window()
    
    # Save the perfect point cloud
    output_file = "/home/chris/Desktop/perfect_local_scaled_object.pcd"
    o3d.io.write_point_cloud(output_file, perfect_pcd)
    print(f"Perfect point cloud saved as '{output_file}'.")
    
    print("\nSummary of point cloud statistics:")
    print(f"Camera PCD: {len(pcd_cam.points)} points")
    print(f"Mesh PCD: {len(pcd_mesh.points)} points")
    print(f"Perfect PCD: {len(perfect_pcd.points)} points")
    
    # Display bounding box information for size comparison
    bbox_cam_local = pcd_cam_local_vis3.get_axis_aligned_bounding_box()
    bbox_perfect = perfect_pcd_right.get_axis_aligned_bounding_box()
    
    print("\nBounding box dimensions:")
    print(f"Camera PCD: {bbox_cam_local.get_extent()}")
    print(f"Perfect PCD: {bbox_perfect.get_extent()}")

if __name__ == '__main__':
    main()
    # sample_points_from_mesh("/home/chris/Desktop/MyObject.obj", "/home/chris/Desktop/MyObject.pcd")

