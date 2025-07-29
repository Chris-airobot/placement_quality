import torch
from torch.utils.data import Dataset
import numpy as np
import json

def cuboid_corners_local(dx, dy, dz):
    # Returns array of shape (8, 3)
    signs = np.array([[sx, sy, sz] for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]])
    return 0.5 * signs * np.array([dx, dy, dz])

class PlacementDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        N = len(data)
        # Pre-allocate tensors
        self.corners = torch.empty((N, 8, 3), dtype=torch.float32)
        self.grasp = torch.empty((N, 7), dtype=torch.float32)
        self.init_pose = torch.empty((N, 7), dtype=torch.float32)
        self.final_pose = torch.empty((N, 7), dtype=torch.float32)
        self.label = torch.empty((N, 2), dtype=torch.float32)
        for i, sample in enumerate(data):
            dx, dy, dz = sample['object_dimensions']
            corners = cuboid_corners_local(dx, dy, dz).astype(np.float32)
            self.corners[i] = torch.from_numpy(corners)
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
            self.grasp[idx],
            self.init_pose[idx],
            self.final_pose[idx],
            self.label[idx]
        )
