import open3d as o3d

def visualize_pcd(file_path):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if the PCD file was loaded successfully
    if not pcd.has_points():
        print("Error: No points found in the PCD file.")
        return
    
    # Display the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="PCD Viewer")

# Example usage
file_path = "/home/chris/Chris/placement_ws/src/placement_quality/docker_files/pointcloud.pcd"  # Change this to your PCD file path
visualize_pcd(file_path)
