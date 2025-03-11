#!/usr/bin/env python3

import os
import sys
import argparse
from helper import process_existing_pointcloud, view_pcd, compare_point_clouds

def main():
    parser = argparse.ArgumentParser(description="Process and visualize point clouds")
    parser.add_argument("--input", type=str, default="/home/chris/Chris/placement_ws/src/merged_pointcloud_raw.pcd",
                        help="Path to input point cloud file")
    parser.add_argument("--output", type=str, default="/home/chris/Chris/placement_ws/src/processed_pointcloud.pcd",
                        help="Path to save processed point cloud")
    parser.add_argument("--view", action="store_true", help="View the processed point cloud")
    parser.add_argument("--compare", action="store_true", help="Compare original and processed point clouds")
    parser.add_argument("--show-normals", action="store_true", help="Show normals in visualization")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return 1
    
    # Process the point cloud
    print(f"Processing point cloud: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    processed_pcd = process_existing_pointcloud(args.input, args.output)
    
    if processed_pcd is None:
        print("Error: Failed to process point cloud")
        return 1
    
    print(f"Point cloud processed successfully and saved to {args.output}")
    
    # Visualization options
    if args.compare:
        print("Comparing original and processed point clouds...")
        compare_point_clouds(args.input, args.output, 
                            titles=["Original Point Cloud", "Processed Point Cloud"])
    elif args.view:
        print(f"Viewing processed point cloud: {args.output}")
        view_pcd(args.output, show_normals=args.show_normals)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 