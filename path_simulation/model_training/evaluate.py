#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import glob
from dataset import KinematicFeasibilityDataset
from model import GraspObjectFeasibilityNet, PointNetEncoder
from train import load_pointcloud

def evaluate_checkpoint(checkpoint, test_loader, device, static_obj_feat):
    # 1) Load the raw checkpoint
    checkpoint_data = torch.load(checkpoint, map_location=device)
    # if you saved via Lightning it may be under 'state_dict'
    raw_state_dict = checkpoint_data.get("state_dict", checkpoint_data)

    # 2) Strip any unwanted prefix (e.g. "_orig_mod.")
    prefix_to_strip = "_orig_mod."
    cleaned_state_dict = {}
    for key, tensor in raw_state_dict.items():
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = tensor
    model = GraspObjectFeasibilityNet(use_static_obj=True).to(device)
    model.register_buffer('static_obj_feat', static_obj_feat)  # now model.static_obj_feat is available
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # accumulate TP/FP/TN/FN for both tasks...
    metrics = {
      'true_pos_success':0,'false_pos_success':0,'true_neg_success':0,'false_neg_success':0,
      'true_pos_collision':0,'false_pos_collision':0,'true_neg_collision':0,'false_neg_collision':0
    }
    with torch.no_grad():
        for grasp, init, final, success_label, collision_label in test_loader:
            grasp, init, final = [t.to(device) for t in (grasp, init, final)]
            success_label, collision_label = success_label.to(device).unsqueeze(1), collision_label.to(device).unsqueeze(1)
            raw_success, raw_collision = model(None, grasp, init, final)
            pred_success = (torch.sigmoid(raw_success)>0.5).long(); pred_collision = (torch.sigmoid(raw_collision)>0.5).long()

            metrics['true_pos_success'] += ((pred_success==1)&(success_label==1)).sum().item()
            metrics['false_pos_success'] += ((pred_success==1)&(success_label==0)).sum().item()
            metrics['true_neg_success'] += ((pred_success==0)&(success_label==0)).sum().item()
            metrics['false_neg_success'] += ((pred_success==0)&(success_label==1)).sum().item()
            metrics['true_pos_collision'] += ((pred_collision==1)&(collision_label==1)).sum().item()
            metrics['false_pos_collision'] += ((pred_collision==1)&(collision_label==0)).sum().item()
            metrics['true_neg_collision'] += ((pred_collision==0)&(collision_label==0)).sum().item()
            metrics['false_neg_collision'] += ((pred_collision==0)&(collision_label==1)).sum().item()
    # compute precision/recall/f1
    precision_success = metrics['true_pos_success'] / (metrics['true_pos_success']+metrics['false_pos_success']+1e-8)
    recall_success  = metrics['true_pos_success'] / (metrics['true_pos_success']+metrics['false_neg_success']+1e-8)
    f1_success   = 2*precision_success*recall_success / (precision_success+recall_success+1e-8)
    precision_collision = metrics['true_pos_collision'] / (metrics['true_pos_collision']+metrics['false_pos_collision']+1e-8)
    recall_collision  = metrics['true_pos_collision'] / (metrics['true_pos_collision']+metrics['false_neg_collision']+1e-8)
    f1_collision   = 2*precision_collision*recall_collision / (precision_collision+recall_collision+1e-8)
    
    # Calculate accuracy for both tasks
    total_samples_success = metrics['true_pos_success'] + metrics['true_neg_success'] + metrics['false_pos_success'] + metrics['false_neg_success']
    accuracy_success = (metrics['true_pos_success'] + metrics['true_neg_success']) / (total_samples_success + 1e-8)
    
    total_samples_collision = metrics['true_pos_collision'] + metrics['true_neg_collision'] + metrics['false_pos_collision'] + metrics['false_neg_collision']
    accuracy_collision = (metrics['true_pos_collision'] + metrics['true_neg_collision']) / (total_samples_collision + 1e-8)
    
    return {
        'checkpoint': os.path.basename(checkpoint),
        'precision_success': precision_success,
        'recall_success': recall_success,
        'f1_success': f1_success,
        'accuracy_success': accuracy_success,
        'precision_collision': precision_collision,
        'recall_collision': recall_collision,
        'f1_collision': f1_collision,
        'accuracy_collision': accuracy_collision
    }



def main(dir_path):
    test_data_json = os.path.join(dir_path, 'combined_data', 'test.json')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # load test set
    test_ds = KinematicFeasibilityDataset(test_data_json)
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    print(f"Loaded test set: {len(test_ds)} samples → {len(test_loader)} batches\n")

    # ─── find all .pth under models/*/*.pth ────────────────────────────────
    # ckpts = []
    # for root, _, files in os.walk(os.path.join(dir_path, 'models')):
    #     for f in files:
    #         if f.endswith('.pth'):
    #             ckpts.append(os.path.join(root, f))
    # ckpts.sort()
    # print(f"Found {len(ckpts)} checkpoints to evaluate\n")
    ckpt = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/models/model_20250626_220307/best_model_0_1045_pth"

    # ——— 5) Static point‐cloud embedding ———————————————————————
    print("Computing static point-cloud embedding …")
    pcd_path      = '/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/perfect_cube.pcd'
    object_pcd_np = load_pointcloud(pcd_path)
    object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
    print(f"Loaded point cloud with {object_pcd.shape[0]} points...")

    # forward once through PointNetEncoder
    with torch.no_grad():
        pn = PointNetEncoder(global_feat_dim=256).to(device)
        static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
    print("Done.\n")




    print("→ Evaluating", os.path.basename(ckpt))
    rec = evaluate_checkpoint(ckpt, test_loader, device, static_obj_feat)

    # print results
    print("\nModel results:")
    print(rec)



if __name__ == '__main__':
    dir_path = '/home/chris/Chris/placement_ws/src/data/box_simulation/v2'
    main(dir_path)