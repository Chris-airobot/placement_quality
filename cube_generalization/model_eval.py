#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import glob
from dataset import PlacementDataset
from model import GraspObjectFeasibilityNet
import json

def evaluate_checkpoint(checkpoint, test_loader, device):
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
    model = GraspObjectFeasibilityNet().to(device)
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # accumulate TP/FP/TN/FN for both tasks...
    metrics = {
      'true_pos_success':0,'false_pos_success':0,'true_neg_success':0,'false_neg_success':0,
      'true_pos_collision':0,'false_pos_collision':0,'true_neg_collision':0,'false_neg_collision':0
    }
    with torch.no_grad():
        for corners, grasp, init, final, label in test_loader:
            corners, grasp, init, final = [t.to(device) for t in (corners, grasp, init, final)]
            success_label = label[:, 0].unsqueeze(1).to(device)
            collision_label = label[:, 1].unsqueeze(1).to(device)
            raw_success, raw_collision = model(corners, grasp, init, final)
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
        # 'precision_success': precision_success,
        # 'recall_success': recall_success,
        # 'f1_success': f1_success,
        # 'accuracy_success': accuracy_success,
        'precision_collision': precision_collision,
        'recall_collision': recall_collision,
        'f1_collision': f1_collision,
        'accuracy_collision': accuracy_collision
    }



def main(dir_path):
    test_data_json = os.path.join(dir_path, 'data_collection/combined_data', 'test.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # load test set
    test_ds = PlacementDataset(test_data_json)
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    print(f"Loaded test set: {len(test_ds)} samples → {len(test_loader)} batches\n")

    ckpt = "/home/chris/Chris/placement_ws/src/data/box_simulation/v3/training/models/model_20250723_042907/best_model_0_4566_pth"
    checkpoint_data = torch.load(ckpt, map_location=device)
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
    model = GraspObjectFeasibilityNet().to(device)
    model.load_state_dict(cleaned_state_dict)
    model.eval()

    # Load raw test data for individual evaluation
    with open(test_data_json, 'r') as f:
        test_data = json.load(f)

    counter = 0
    for data in test_data:
        # Get cuboid dimensions and compute corners
        dx, dy, dz = data['object_dimensions']
        from dataset import cuboid_corners_local
        corners = cuboid_corners_local(dx, dy, dz).astype(np.float32)
        
        # Convert to tensors and add batch dimension
        corners_tensor = torch.tensor(corners, dtype=torch.float32).unsqueeze(0).to(device)
        grasp_tensor = torch.tensor(data["grasp_pose"], dtype=torch.float32).unsqueeze(0).to(device)
        init_tensor = torch.tensor(data["initial_object_pose"], dtype=torch.float32).unsqueeze(0).to(device)
        final_tensor = torch.tensor(data["final_object_pose"], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            raw_success, raw_collision = model(corners_tensor, grasp_tensor, init_tensor, final_tensor)
            
        # Apply sigmoid to convert logits to probabilities
        pred_success = torch.sigmoid(raw_success)
        pred_collision = torch.sigmoid(raw_collision)
        
        # Extract scalar values from tensors
        pred_success_val = pred_success.item()
        pred_collision_val = pred_collision.item()
            
        counter += 1
        # print(f"For {counter}: pred_success: {pred_success_val:.4f}, pred_collision: {1-pred_collision_val:.4f}")

    # Run batch evaluation for summary metrics
    print("→ Evaluating", os.path.basename(ckpt))
    rec = evaluate_checkpoint(ckpt, test_loader, device)

    # print results
    print("\nModel results:")
    print(rec)



if __name__ == '__main__':
    dir_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v3/"
    main(dir_path)