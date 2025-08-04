import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import glob
from scipy.spatial.transform import Rotation as R

def cuboid_corners_local_ordered(dx, dy, dz):
    """
    Ordered corners: Top 4 first (z=+dz/2), then bottom 4 (z=-dz/2)
    Within each level: clockwise when viewed from above (right-hand rule)
    Returns array of shape (8, 3) in consistent local frame ordering
    """
    # Top face (z = +dz/2) - clockwise from (-x,-y) when viewed from above
    top_corners = [
        [-dx/2, -dy/2, +dz/2],  # 0: left-front-top
        [+dx/2, -dy/2, +dz/2],  # 1: right-front-top  
        [+dx/2, +dy/2, +dz/2],  # 2: right-back-top
        [-dx/2, +dy/2, +dz/2],  # 3: left-back-top
    ]
    
    # Bottom face (z = -dz/2) - clockwise from (-x,-y) when viewed from above
    bottom_corners = [
        [-dx/2, -dy/2, -dz/2],  # 4: left-front-bottom
        [+dx/2, -dy/2, -dz/2],  # 5: right-front-bottom
        [+dx/2, +dy/2, -dz/2],  # 6: right-back-bottom  
        [-dx/2, +dy/2, -dz/2],  # 7: left-back-bottom
    ]
    
    return np.array(top_corners + bottom_corners)

def generate_box_pointcloud(dx, dy, dz, num_points=1024):
    """
    Generate num_points uniformly distributed within a box of dimensions dx, dy, dz
    Returns points in local frame centered at origin
    """
    # Generate random points within the box
    points = np.random.uniform(
        low=[-dx/2, -dy/2, -dz/2],
        high=[dx/2, dy/2, dz/2], 
        size=(num_points, 3)
    )
    return points.astype(np.float32)

def transform_points_to_world(points_local, pose_world):
    """Transform points from local frame to world frame using pose [x,y,z,qw,qx,qy,qz]"""
    pos = pose_world[:3]
    quat_wxyz = pose_world[3:]
    # Convert to scipy format [x,y,z,w]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot = R.from_quat(quat_xyzw)
    
    # Transform: world_point = R * local_point + translation
    points_world = rot.apply(points_local) + pos
    return points_world

def get_unique_dimensions_from_files(processed_data_dir):
    """Extract unique object dimensions from processed data filenames"""
    # Get all JSON files in the processed_data directory
    json_files = glob.glob(os.path.join(processed_data_dir, "processed_object_*.json"))
    
    dimensions = []
    for filepath in json_files:
        # Extract filename: processed_object_0.1_0.075_0.11.json
        filename = os.path.basename(filepath)
        # Parse dimensions from filename
        parts = filename.replace("processed_object_", "").replace(".json", "").split("_")
        if len(parts) == 3:
            try:
                dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
                dimensions.append((dx, dy, dz))
            except ValueError:
                continue
    
    print(f"Found {len(dimensions)} unique object dimensions from processed_data files")
    return dimensions

def _zscore_normalize(x, mean=None, std=None, eps=1e-8):
    """
    Z-score normalization with optional pre-computed statistics
    Returns normalized tensor and statistics (mean, std)
    """
    # Ensure x is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Get device of input tensor
    device = x.device
    
    if mean is None:
        mean = x.mean(0, keepdims=True)
    else:
        # Ensure mean is on the same device as x
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device)
        else:
            mean = torch.tensor(mean, dtype=torch.float32, device=device)
    
    if std is None:
        std = x.std(0, keepdims=True)
    else:
        # Ensure std is on the same device as x
        if isinstance(std, torch.Tensor):
            std = std.to(device)
        else:
            std = torch.tensor(std, dtype=torch.float32, device=device)
    
    normalized = (x - mean) / (std + eps)
    return normalized, mean, std

class PlacementDataset(Dataset):
    def __init__(self, data_path, processed_data_dir=None):
        with open(data_path, 'r') as f:
            data = json.load(f)
        N = len(data)
        
        # Extract unique dimensions from processed_data directory if provided
        if processed_data_dir is None:
            # Default path based on data_path structure
            base_dir = os.path.dirname(os.path.dirname(data_path))  # Go up from combined_data/
            processed_data_dir = os.path.join(base_dir, "processed_data")
        
        self.unique_dimensions = get_unique_dimensions_from_files(processed_data_dir)
        self.dim_to_index = {dim: i for i, dim in enumerate(self.unique_dimensions)}
        
        # Pre-allocate tensors
        self.corners = torch.empty((N, 8, 3), dtype=torch.float32)
        self.dimension_indices = torch.empty(N, dtype=torch.long)  # Map each sample to dimension index
        self.grasp = torch.empty((N, 7), dtype=torch.float32)
        self.init_pose = torch.empty((N, 7), dtype=torch.float32)
        self.final_pose = torch.empty((N, 7), dtype=torch.float32)
        self.label = torch.empty((N, 2), dtype=torch.float32)
        
        for i, sample in enumerate(data):
            dx, dy, dz = sample['object_dimensions']
            init_pose = np.array(sample['initial_object_pose'])
            
            # Generate ordered corners in local frame, then transform to world frame
            corners_local_ordered = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)
            corners_world = transform_points_to_world(corners_local_ordered, init_pose)
            self.corners[i] = torch.from_numpy(corners_world)
            
            # Map sample to its dimension index
            dim_key = tuple(sample['object_dimensions'])
            if dim_key in self.dim_to_index:
                self.dimension_indices[i] = self.dim_to_index[dim_key]
            else:
                # Find closest match if exact match not found (for floating point precision issues)
                closest_idx = 0
                min_diff = float('inf')
                for j, (udx, udy, udz) in enumerate(self.unique_dimensions):
                    diff = abs(dx - udx) + abs(dy - udy) + abs(dz - udz)
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = j
                self.dimension_indices[i] = closest_idx
            
            self.grasp[i] = torch.tensor(sample['grasp_pose'], dtype=torch.float32)
            self.init_pose[i] = torch.tensor(sample['initial_object_pose'], dtype=torch.float32)
            self.final_pose[i] = torch.tensor(sample['final_object_pose'], dtype=torch.float32)
            self.label[i, 0] = float(sample['success_label'])
            self.label[i, 1] = float(sample['collision_label'])

    def __len__(self):
        return self.corners.size(0)

    def __getitem__(self, idx):
        return (
            self.corners[idx],
            self.dimension_indices[idx],  # Return dimension index instead of PCD
            self.grasp[idx],
            self.init_pose[idx],
            self.final_pose[idx],
            self.label[idx]
        )

class WorldFrameDataset(Dataset):
    def __init__(self, data_path, embeddings_file, normalization_stats=None, is_training=True):
        """
        Enhanced dataset with feature normalization
        
        Args:
            data_path: Path to JSON data file
            embeddings_file: Path to pre-computed embeddings
            normalization_stats: Dict with pre-computed stats for validation/test sets
            is_training: Whether this is the training set (compute new stats) or not (use provided stats)
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        N = len(data)
        
        # Load embeddings and keep as float16
        self.embeddings = np.load(embeddings_file)  # [N, 1024] float16
        print(f"Loaded embeddings: {self.embeddings.shape}, dtype: {self.embeddings.dtype}")
        
        # Pre-allocate tensors
        self.corners = torch.empty((N, 8, 3), dtype=torch.float32)
        self.grasp = torch.empty((N, 7), dtype=torch.float32)
        self.init_pose = torch.empty((N, 7), dtype=torch.float32)
        self.final_pose = torch.empty((N, 7), dtype=torch.float32)
        self.label = torch.empty((N, 2), dtype=torch.float32)
        
        for i, sample in enumerate(data):
            dx, dy, dz = sample['object_dimensions']
            init_pose = np.array(sample['initial_object_pose'])
            
            # Generate ordered corners in local frame, then transform to world frame
            corners_local_ordered = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)
            corners_world = transform_points_to_world(corners_local_ordered, init_pose)
            self.corners[i] = torch.from_numpy(corners_world)
            
            self.grasp[i] = torch.tensor(sample['grasp_pose'], dtype=torch.float32)
            self.init_pose[i] = torch.tensor(sample['initial_object_pose'], dtype=torch.float32)
            self.final_pose[i] = torch.tensor(sample['final_object_pose'], dtype=torch.float32)
            self.label[i, 0] = float(sample['success_label'])
            self.label[i, 1] = float(sample['collision_label'])

        # ✨ ENHANCED FEATURE NORMALIZATION ✨
        if is_training or normalization_stats is None:
            # Compute normalization statistics (training set only)
            print("Computing enhanced normalization statistics...")
            
            # Normalize corners (reshape to [N, 24] for global normalization)
            corners_flat = self.corners.reshape(N, -1)
            self.corners, corners_mean, corners_std = _zscore_normalize(corners_flat)
            self.corners = self.corners.reshape(N, 8, 3)
            
            # ✨ SEPARATE POSE NORMALIZATION (POSITION + ORIENTATION) ✨
            # Extract position (xyz) and orientation (quaternion) separately
            grasp_pos = self.grasp[:, :3]      # [x, y, z]
            grasp_ori = self.grasp[:, 3:]      # [qw, qx, qy, qz]
            init_pos = self.init_pose[:, :3]   # [x, y, z]
            init_ori = self.init_pose[:, 3:]   # [qw, qx, qy, qz]
            final_pos = self.final_pose[:, :3] # [x, y, z]
            final_ori = self.final_pose[:, 3:] # [qw, qx, qy, qz]
            
            # ✨ UNIFIED POSITION NORMALIZATION ✨
            # Compute global position statistics across all pose types
            all_positions = torch.cat([grasp_pos, init_pos, final_pos], dim=0)  # [3N, 3]
            pos_mean = all_positions.mean(0, keepdims=True)  # [1, 3]
            pos_std = all_positions.std(0, keepdims=True) + 1e-8  # [1, 3]
            
            # Apply unified normalization to each position component
            grasp_pos_norm = (grasp_pos - pos_mean) / pos_std
            init_pos_norm = (init_pos - pos_mean) / pos_std
            final_pos_norm = (final_pos - pos_mean) / pos_std
            
            # Normalize orientations (quaternions) separately
            grasp_ori_norm, grasp_ori_mean, grasp_ori_std = _zscore_normalize(grasp_ori)
            init_ori_norm, init_ori_mean, init_ori_std = _zscore_normalize(init_ori)
            final_ori_norm, final_ori_mean, final_ori_std = _zscore_normalize(final_ori)
            
            # Reconstruct normalized poses
            self.grasp = torch.cat([grasp_pos_norm, grasp_ori_norm], dim=1)
            self.init_pose = torch.cat([init_pos_norm, init_ori_norm], dim=1)
            self.final_pose = torch.cat([final_pos_norm, final_ori_norm], dim=1)
            
            # Store enhanced stats for validation/test sets
            self.normalization_stats = {
                'corners_mean': corners_mean, 'corners_std': corners_std,
                'pos_mean': pos_mean, 'pos_std': pos_std,  # ✨ Unified position stats
                'grasp_ori_mean': grasp_ori_mean, 'grasp_ori_std': grasp_ori_std,
                'init_ori_mean': init_ori_mean, 'init_ori_std': init_ori_std,
                'final_ori_mean': final_ori_mean, 'final_ori_std': final_ori_std
            }
            
            print("✅ Computed enhanced normalization statistics with separate pose components")
            
        else:
            # Use pre-computed statistics (validation/test sets)
            print("Applying enhanced normalization statistics...")
            
            corners_flat = self.corners.reshape(N, -1)
            self.corners, _, _ = _zscore_normalize(
                corners_flat, 
                normalization_stats['corners_mean'], 
                normalization_stats['corners_std']
            )
            self.corners = self.corners.reshape(N, 8, 3)
            
            # ✨ APPLY SEPARATE POSE NORMALIZATION ✨
            # Extract position and orientation components
            grasp_pos = self.grasp[:, :3]      # [x, y, z]
            grasp_ori = self.grasp[:, 3:]      # [qw, qx, qy, qz]
            init_pos = self.init_pose[:, :3]   # [x, y, z]
            init_ori = self.init_pose[:, 3:]   # [qw, qx, qy, qz]
            final_pos = self.final_pose[:, :3] # [x, y, z]
            final_ori = self.final_pose[:, 3:] # [qw, qx, qy, qz]
            
            # ✨ APPLY UNIFIED POSITION NORMALIZATION ✨
            # Use pre-computed unified position statistics
            pos_mean = normalization_stats['pos_mean']
            pos_std = normalization_stats['pos_std']
            
            # Ensure stats are on the same device as tensors
            if isinstance(pos_mean, torch.Tensor):
                pos_mean = pos_mean.to(self.grasp.device)
            else:
                pos_mean = torch.tensor(pos_mean, device=self.grasp.device)
            if isinstance(pos_std, torch.Tensor):
                pos_std = pos_std.to(self.grasp.device)
            else:
                pos_std = torch.tensor(pos_std, device=self.grasp.device)
            
            # Apply unified normalization to each position component
            grasp_pos_norm = (grasp_pos - pos_mean) / pos_std
            init_pos_norm = (init_pos - pos_mean) / pos_std
            final_pos_norm = (final_pos - pos_mean) / pos_std
            
            # Normalize orientations with separate stats
            grasp_ori_norm, _, _ = _zscore_normalize(
                grasp_ori, normalization_stats['grasp_ori_mean'], normalization_stats['grasp_ori_std']
            )
            init_ori_norm, _, _ = _zscore_normalize(
                init_ori, normalization_stats['init_ori_mean'], normalization_stats['init_ori_std']
            )
            final_ori_norm, _, _ = _zscore_normalize(
                final_ori, normalization_stats['final_ori_mean'], normalization_stats['final_ori_std']
            )
            
            # Reconstruct normalized poses
            self.grasp = torch.cat([grasp_pos_norm, grasp_ori_norm], dim=1)
            self.init_pose = torch.cat([init_pos_norm, init_ori_norm], dim=1)
            self.final_pose = torch.cat([final_pos_norm, final_ori_norm], dim=1)
            
            # Store the provided stats
            self.normalization_stats = normalization_stats
            print("✅ Applied enhanced normalization statistics")
        
        # Print enhanced normalization statistics for verification
        print(f"Enhanced feature statistics after normalization:")
        print(f"  Corners: mean={self.corners.mean():.6f}, std={self.corners.std():.6f}")
        print(f"  All positions: mean={torch.cat([self.grasp[:, :3], self.init_pose[:, :3], self.final_pose[:, :3]], dim=0).mean():.6f}, std={torch.cat([self.grasp[:, :3], self.init_pose[:, :3], self.final_pose[:, :3]], dim=0).std():.6f}")
        print(f"  Grasp orientations: mean={self.grasp[:, 3:].mean():.6f}, std={self.grasp[:, 3:].std():.6f}")
        print(f"  Init orientations: mean={self.init_pose[:, 3:].mean():.6f}, std={self.init_pose[:, 3:].std():.6f}")
        print(f"  Final orientations: mean={self.final_pose[:, 3:].mean():.6f}, std={self.final_pose[:, 3:].std():.6f}")

    def __len__(self):
        return self.corners.size(0)

    def __getitem__(self, idx):
        # ✨ ONLY CHANGE: Don't convert to float32 here
        embedding = torch.from_numpy(self.embeddings[idx])  # Keep original dtype
        
        return (
            self.corners[idx],
            embedding,  # Will be float16 if embeddings are float16
            self.grasp[idx],
            self.init_pose[idx],
            self.final_pose[idx],
            self.label[idx]
        )


