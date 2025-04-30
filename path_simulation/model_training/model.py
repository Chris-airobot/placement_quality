import torch
import torch.nn as nn
from pointnet2.pointnet2_utils import PointNetSetAbstraction

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat_dim=256):
        super(PointNetEncoder, self).__init__()
        # Use a simplified SetAbstraction module (as per existing repo)
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3, [64, 64, 128], False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128+3, [128, 128, global_feat_dim], True)
    
    def forward(self, xyz):
        # xyz: B x 3 x N
        B, N, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1)  # convert [B, N, 3] → [B, 3, N]
        
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # l2_points: B x global_feat_dim x 1 → B x global_feat_dim
        global_feat = l2_points.view(B, -1)

        return global_feat

class GraspObjectFeasibilityNet(nn.Module):
    def __init__(self, surface_embed_dim=8, use_static_obj=False):
        super().__init__()
        self.use_static_obj = use_static_obj
        if not use_static_obj:
            self.pointnet = PointNetEncoder(256)

        # NEW: surface embeddings (6 surfaces each)
        self.init_surf_emb = nn.Embedding(6, surface_embed_dim)
        self.final_surf_emb= nn.Embedding(6, surface_embed_dim)

        # Pose encoder now sees: 7(grasp)+7(init pose)+7(final pose)+2*surface_embed_dim
        in_dim = 7 + 7 + 7 + 2*surface_embed_dim
        self.pose_encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )

        # Two separate heads if predicting two labels (optional, if needed)
        self.out_success = nn.Linear(64, 1)
        self.out_collision = nn.Linear(64, 1)

    def forward(self, points, grasp_pose, init_pose, final_pose, surfaces):
        # 1) object feature
        if self.use_static_obj:
            # static_obj_feat is [1,256] buffer registered in main()
            obj_feat = self.static_obj_feat.expand(grasp_pose.size(0), -1)
        else:
            obj_feat = self.pointnet(points)

        # 2) surface embeddings
        init_emb  = self.init_surf_emb (surfaces[:,0])  # [B, E]
        final_emb = self.final_surf_emb(surfaces[:,1])

        # 3) pose feature
        x = torch.cat([grasp_pose, init_pose, final_pose, init_emb, final_emb], dim=1)
        pose_feat = self.pose_encoder(x)

        # 4) fuse & heads
        fused = torch.cat([obj_feat, pose_feat], dim=1)
        hidden = self.fusion_fc(fused)
        logit_s = self.out_success   (hidden)
        logit_c = self.out_collision (hidden)

        # return raw logits (BCEWithLogitsLoss expects logits!)
        return logit_s, logit_c

