import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import PointNet++ only if available (used by legacy models only)
POINTNET_AVAILABLE = True
try:
    # Prefer the local implementation bundled with this repo
    from pointnet2.pointnet2_cls_ssg import get_model
except Exception:
    try:
        # Fallback: add local folder explicitly to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, 'pointnet2'))
        from pointnet2_cls_ssg import get_model
    except Exception:
        # If still not available, disable PointNet-dependent legacy models
        POINTNET_AVAILABLE = False
        print("Warning: PointNet++ not available. Legacy models that require PointNet++ will be disabled.")

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
        # Enhanced model uses precomputed embeddings; it does NOT require PointNet++ runtime
        self.use_pointnet = use_pointnet
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

def create_legacy_combined_model():
    """Create the original model for comparison"""
    return CollisionPredictionNet(use_pointnet=True, use_corners=True, pointnet_feat_dim=1024)

def create_original_enhanced_model():
    """Create the original enhanced model without LayerNorm for loading old checkpoints"""
    
    class OriginalEnhancedCollisionPredictionNet(nn.Module):
        """
        Original enhanced model without LayerNorm - for checkpoint compatibility
        """
        def __init__(self, use_pointnet=True, use_corners=True, embed_dim=1024):
            super().__init__()
            # Original enhanced model also operates on precomputed embeddings
            self.use_pointnet = use_pointnet
            self.use_corners = use_corners
            
            if not self.use_pointnet and not self.use_corners:
                raise ValueError("At least one of use_pointnet or use_corners must be True")
            
            # ORIGINAL EMBEDDING PROCESSING (matching checkpoint structure)
            if self.use_pointnet:
                self.embed_norm = nn.LayerNorm(embed_dim)
                self.embed_projection = nn.Sequential(
                    nn.Linear(embed_dim, 512),  # index 0
                    nn.GELU(),                  # index 1 (no params)
                    nn.Dropout(0.1),            # index 2 (no params)
                    nn.Linear(512, 256),        # index 3  
                    nn.GELU()                   # index 4 (no params)
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
                # SMART DTYPE HANDLING
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
    
    return OriginalEnhancedCollisionPredictionNet(use_pointnet=True, use_corners=True, embed_dim=1024)
