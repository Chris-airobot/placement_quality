import torch
import torch.nn as nn

class BoxCornerEncoder(nn.Module):
    """
    Encodes 8 box corners (shape [B, 8, 3]) using a small MLP.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),                   # [B, 8, 3] → [B, 24]
            nn.Linear(24, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU()
        )

    def forward(self, corners):
        # corners: [B, 8, 3]
        return self.mlp(corners)            # [B, out_dim]

class GraspObjectFeasibilityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.corner_encoder = BoxCornerEncoder(256)


        in_dim = 7 + 7 + 7  # grasp + initial + final pose
        self.pose_encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )

        self.out_success = nn.Linear(64, 1)
        self.out_collision = nn.Linear(64, 1)

    def forward(self, corners, grasp_pose, init_pose, final_pose):
        # corners: [B, 8, 3]
        obj_feat = self.corner_encoder(corners)

        x = torch.cat([grasp_pose, init_pose, final_pose], dim=1)
        pose_feat = self.pose_encoder(x)

        fused = torch.cat([obj_feat, pose_feat], dim=1)
        hidden = self.fusion_fc(fused)
        logit_s = self.out_success(hidden)
        logit_c = self.out_collision(hidden)

        return logit_s, logit_c
