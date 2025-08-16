import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _quat_wxyz_to_R(q):
    # q: (..., 4) as [w,x,y,z]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # broadcast-safe math
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    r00 = 1 - 2*(yy + zz); r01 = 2*(xy - wz);   r02 = 2*(xz + wy)
    r10 = 2*(xy + wz);     r11 = 1 - 2*(xx + zz); r12 = 2*(yz - wx)
    r20 = 2*(xz - wy);     r21 = 2*(yz + wx);     r22 = 1 - 2*(xx + yy)
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (..., 3, 3)
    return R

def _world_to_local_pose(pose_w, R_o_init, t_o_init):
    """
    pose_w: (B,7) world [tx,ty,tz,qw,qx,qy,qz]
    R_o_init: (B,3,3) rotation of initial object in world
    t_o_init: (B,3)   translation of initial object in world
    returns (B,7) local pose in initial object frame
    """
    t_w = pose_w[:, :3]
    q_w = pose_w[:, 3:7]
    R_w = _quat_wxyz_to_R(q_w)
    # local translation/orientation
    t_loc = torch.einsum('bij,bj->bi', R_o_init.transpose(-1, -2), (t_w - t_o_init))
    R_loc = torch.einsum('bij,bjk->bik', R_o_init.transpose(-1, -2), R_w)
    # convert R_loc -> quaternion (wxyz)
    # stable trace-based conversion
    tr = R_loc[..., 0, 0] + R_loc[..., 1, 1] + R_loc[..., 2, 2]
    qw = torch.empty_like(tr)
    qx = torch.empty_like(tr)
    qy = torch.empty_like(tr)
    qz = torch.empty_like(tr)
    mask = tr > 0
    s = torch.sqrt(torch.clamp(tr[mask] + 1.0, min=1e-12)) * 2.0
    qw[mask] = 0.25 * s
    qx[mask] = (R_loc[mask, 2, 1] - R_loc[mask, 1, 2]) / s
    qy[mask] = (R_loc[mask, 0, 2] - R_loc[mask, 2, 0]) / s
    qz[mask] = (R_loc[mask, 1, 0] - R_loc[mask, 0, 1]) / s
    mask0 = (~mask) & (R_loc[..., 0, 0] >= R_loc[..., 1, 1]) & (R_loc[..., 0, 0] >= R_loc[..., 2, 2])
    s0 = torch.sqrt(torch.clamp(1.0 + R_loc[mask0,0,0] - R_loc[mask0,1,1] - R_loc[mask0,2,2], min=1e-12)) * 2.0
    qx[mask0] = 0.25 * s0
    qw[mask0] = (R_loc[mask0,2,1] - R_loc[mask0,1,2]) / s0
    qy[mask0] = (R_loc[mask0,0,1] + R_loc[mask0,1,0]) / s0
    qz[mask0] = (R_loc[mask0,0,2] + R_loc[mask0,2,0]) / s0
    mask1 = (~mask) & (~mask0) & (R_loc[..., 1, 1] >= R_loc[..., 2, 2])
    s1 = torch.sqrt(torch.clamp(1.0 + R_loc[mask1,1,1] - R_loc[mask1,0,0] - R_loc[mask1,2,2], min=1e-12)) * 2.0
    qy[mask1] = 0.25 * s1
    qw[mask1] = (R_loc[mask1,0,2] - R_loc[mask1,2,0]) / s1
    qx[mask1] = (R_loc[mask1,0,1] + R_loc[mask1,1,0]) / s1
    qz[mask1] = (R_loc[mask1,1,2] + R_loc[mask1,2,1]) / s1
    mask2 = (~mask) & (~mask0) & (~mask1)
    s2 = torch.sqrt(torch.clamp(1.0 + R_loc[mask2,2,2] - R_loc[mask2,0,0] - R_loc[mask2,1,1], min=1e-12)) * 2.0
    qz[mask2] = 0.25 * s2
    qw[mask2] = (R_loc[mask2,1,0] - R_loc[mask2,0,1]) / s2
    qx[mask2] = (R_loc[mask2,0,2] + R_loc[mask2,2,0]) / s2
    qy[mask2] = (R_loc[mask2,1,2] + R_loc[mask2,2,1]) / s2
    q_loc = torch.stack([qw, qx, qy, qz], dim=-1)
    return torch.cat([t_loc, q_loc], dim=-1)


# Import PointNet++ only if available
POINTNET_AVAILABLE = True
try:
    import sys
    pointnet_path = '/home/chris/Chris/placement_ws/src/Pointnet_Pointnet2_pytorch'
    sys.path.append(pointnet_path)
    sys.path.append(pointnet_path + '/models')
    from pointnet2_cls_ssg import get_model
except ImportError:
    POINTNET_AVAILABLE = False
    print("Warning: PointNet++ not available. Models that require PointNet++ will be disabled.")

class PointNetEncoder(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        self.pointnet = get_model(num_class=40, normal_channel=False)
        # Load pre-trained weights
        checkpoint_path = '/home/chris/Chris/placement_ws/src/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.pointnet.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze PointNet weights
        for param in self.pointnet.parameters():
            param.requires_grad = False
        
        # Project features to desired dimensionality
        self.projection = nn.Linear(1024, feat_dim)  # PointNet++ outputs 1024 features
        
    def forward(self, x):
        # x: [B, N, 3] or [B, 3, N]
        if x.dim() == 3 and x.size(-1) == 3:
            x = x.transpose(1, 2)  # [B, N, 3] -> [B, 3, N]
        
        with torch.no_grad():
            feat, _ = self.pointnet(x)  # [B, 1024]
        return self.projection(feat)  # [B, feat_dim]

class BoxCornerEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # Enhanced corner processing
        self.mlp = nn.Sequential(
            nn.Linear(8 * 3, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim), nn.ReLU()
        )

    def forward(self, corners):
        # corners: [B, 8, 3]
        return self.mlp(corners.view(corners.size(0), -1))  # [B, out_dim]

class EnhancedCollisionPredictionNet(nn.Module):
    """
    Enhanced model with better fusion architecture and normalization
    """
    def __init__(self, use_pointnet=True, use_corners=True, embed_dim=1024):
        super().__init__()
        self.use_pointnet = use_pointnet and POINTNET_AVAILABLE
        self.use_corners = use_corners
        
        if not self.use_pointnet and not self.use_corners:
            raise ValueError("At least one of use_pointnet or use_corners must be True")
        
        # ✨ SMART EMBEDDING PROCESSING ✨
        if self.use_pointnet:
            self.embed_norm = nn.LayerNorm(embed_dim)
            self.embed_projection = nn.Sequential(
                nn.Linear(embed_dim, 512), 
                nn.LayerNorm(512), 
                nn.GELU(),
                nn.Linear(512, 256), 
                nn.GELU(),
                nn.Dropout(0.1)
            )
        
        if self.use_corners:
            # Enhanced corner processing
            self.corners_net = nn.Sequential(
                nn.Linear(8*3, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 32)
            )
        
        # Enhanced pose processing
        self.pose_net = nn.Sequential(
            nn.Linear(7*3, 64), nn.ReLU(), nn.Dropout(0.1),  # grasp + init + final
            nn.Linear(64, 32), nn.ReLU()
        )
        
        # Calculate fusion input dimension
        fusion_dim = 0
        if self.use_pointnet:
            fusion_dim += 256  # embed_projection output
        if self.use_corners:
            fusion_dim += 32   # corners_net output
        fusion_dim += 32       # pose_net output
        
        # Enhanced fusion network
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)  # Single logit output
        )

    def forward(self, points=None, corners=None, grasp_pose=None, init_pose=None, final_pose=None):
        # Validate inputs
        if self.use_pointnet and points is None:
            raise ValueError("points must be provided when use_pointnet=True")
        if self.use_corners and corners is None:
            raise ValueError("corners must be provided when use_corners=True")
        if grasp_pose is None or init_pose is None or final_pose is None:
            raise ValueError("grasp_pose, init_pose, and final_pose must all be provided")
        
        # Process object features
        features = []
        
        if self.use_pointnet:
            # ✨ SMART DTYPE HANDLING ✨
            # Convert float16 embeddings to float32 ONLY when needed
            if points.dtype == torch.float16:
                points = points.float()  # Convert here, not in dataset
            
            embed_feat = self.embed_norm(points)
            embed_feat = self.embed_projection(embed_feat)
            features.append(embed_feat)
        
        if self.use_corners:
            # Process corners through dedicated network
            corner_feat = self.corners_net(corners.view(corners.size(0), -1))
            features.append(corner_feat)
        
        # Process pose information
        pose_input = torch.cat([grasp_pose, init_pose, final_pose], dim=1)
        pose_feat = self.pose_net(pose_input)
        features.append(pose_feat)
        
        # Fuse all features
        fused = torch.cat(features, dim=1)
        logits = self.fusion_head(fused)
        
        return logits

# Legacy model for backward compatibility
class CollisionPredictionNet(nn.Module):
    """
    Original model - kept for compatibility
    """
    def __init__(self, use_pointnet=True, use_corners=True, pointnet_feat_dim=256, corner_feat_dim=256):
        super().__init__()
        self.use_pointnet = use_pointnet and POINTNET_AVAILABLE
        self.use_corners = use_corners
        
        if not self.use_pointnet and not self.use_corners:
            raise ValueError("At least one of use_pointnet or use_corners must be True")
        
        # Object encoders
        if self.use_pointnet:
            self.pointnet_encoder = PointNetEncoder(pointnet_feat_dim)
        
        if self.use_corners:
            self.corner_encoder = BoxCornerEncoder(corner_feat_dim)
        
        # Calculate total object feature dimension
        obj_feat_dim = 0
        if self.use_pointnet:
            obj_feat_dim += pointnet_feat_dim
        if self.use_corners:
            obj_feat_dim += corner_feat_dim

        # Pose encoder: grasp + initial + final pose
        in_dim = 7 + 7 + 7  # grasp + initial + final pose
        self.pose_encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )

        # Fusion network with dropout for better generalization
        self.fusion_fc = nn.Sequential(
            nn.Linear(obj_feat_dim + 128, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU()
        )

        # Single output head for collision prediction
        self.out_collision = nn.Linear(64, 1)

    def forward(self, points=None, corners=None, grasp_pose=None, init_pose=None, final_pose=None):
        # Validate inputs
        if self.use_pointnet and points is None:
            raise ValueError("points must be provided when use_pointnet=True")
        if self.use_corners and corners is None:
            raise ValueError("corners must be provided when use_corners=True")
        if grasp_pose is None or init_pose is None or final_pose is None:
            raise ValueError("grasp_pose, init_pose, and final_pose must all be provided")
        
        # Extract object features
        obj_features = []
        
        if self.use_pointnet:
            # Check if points are pre-computed embeddings [batch, feat_dim] or raw points [batch, num_points, 3]
            if points.dim() == 2:
                # Pre-computed embeddings - use directly
                pointnet_feat = points
            else:
                # Raw point clouds - process through PointNet encoder
                pointnet_feat = self.pointnet_encoder(points)
            obj_features.append(pointnet_feat)
        
        if self.use_corners:
            corner_feat = self.corner_encoder(corners)
            obj_features.append(corner_feat)
        
        # Concatenate object features
        if len(obj_features) == 1:
            obj_feat = obj_features[0]
        else:
            obj_feat = torch.cat(obj_features, dim=1)

        # Encode pose information
        pose_input = torch.cat([grasp_pose, init_pose, final_pose], dim=1)
        pose_feat = self.pose_encoder(pose_input)

        # Fuse object and pose features
        fused = torch.cat([obj_feat, pose_feat], dim=1)
        hidden = self.fusion_fc(fused)
        
        # Output collision prediction logits
        logit_c = self.out_collision(hidden)
        return logit_c

# Convenience functions for creating specific model configurations
def create_pointnet_only_model():
    """Create a model that only uses PointNet encoder"""
    return CollisionPredictionNet(use_pointnet=True, use_corners=False)

def create_corners_only_model():
    """Create a model that only uses BoxCorner encoder"""
    return CollisionPredictionNet(use_pointnet=False, use_corners=True)

def create_combined_model():
    """Create the ENHANCED model with better fusion"""
    return EnhancedCollisionPredictionNet(use_pointnet=True, use_corners=True, embed_dim=1024)





class CornersOnlyCollisionNet(nn.Module):
    """
    Lightweight corners-only model that uses concatenated init+final corners (16×3)
    and the three poses (grasp/init/final).
    """
    def __init__(self, use_local_pose=False):
        super().__init__()
        # corners: 8*3 = 48 dims
        self.corners_net = nn.Sequential(
            nn.Linear(8*3, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        self.use_local_pose = use_local_pose
        # poses (grasp + init + final)
        self.pose_net = nn.Sequential(
            nn.Linear(7*3, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(32 + 32, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, corners=None, grasp_pose=None, init_pose=None, final_pose=None, **_):
        if corners is None or grasp_pose is None or init_pose is None or final_pose is None:
            raise ValueError("corners, grasp_pose, init_pose, final_pose are required")
        c = self.corners_net(corners.view(corners.size(0), -1))
        p = self.pose_net(torch.cat([grasp_pose, init_pose, final_pose], dim=1))
        return self.fusion_head(torch.cat([c, p], dim=1))


def create_corners_only_fast_model(use_local_pose=False):
    """Create the lightweight corners-only model (16×3 corners)."""
    return CornersOnlyCollisionNet(use_local_pose=use_local_pose)




class FinalCornersAuxModel(nn.Module):
    """
    Inputs:
      corners_24 : [B, 24]  (final corners in world, z-scored)
      aux12      : [B, 12]  (t_loc_z 3 + R_loc_6D 6 + dims_z 3)

    Output:
      logits     : [B, 1]   (collision-at-placement logit)
    """
    def __init__(self,
                 corners_hidden=(128, 64),
                 aux_hidden=(64, 32),
                 head_hidden=128,
                 dropout_p=0.05):
        super().__init__()

        # Corners branch: 24 -> 128 -> 64
        self.corners_net = nn.Sequential(
            nn.Linear(24, corners_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(corners_hidden[0], corners_hidden[1]),
            nn.ReLU(inplace=True),
        )

        # Aux branch (t_loc_z + R6 + dims_z): 12 -> 64 -> 32
        self.aux_net = nn.Sequential(
            nn.Linear(12, aux_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.ReLU(inplace=True),
        )

        # Fusion head: (64 + 32) -> 128 -> 1
        self.head = nn.Sequential(
            nn.Linear(corners_hidden[1] + aux_hidden[1], head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, corners_24: torch.Tensor, aux12: torch.Tensor) -> torch.Tensor:
        # sanity checks help catch shape bugs early
        assert corners_24.dim() == 2 and corners_24.size(-1) == 24, f"corners_24 shape {corners_24.shape}"
        assert aux12.dim() == 2 and aux12.size(-1) == 12, f"aux12 shape {aux12.shape}"

        c_feat = self.corners_net(corners_24)
        a_feat = self.aux_net(aux12)
        fused  = torch.cat([c_feat, a_feat], dim=-1)
        logits = self.head(fused)              # [B, 1]
        return logits