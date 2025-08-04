import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import numpy as np
import time
import json

# Import existing dataset and model
from dataset import PlacementDataset, generate_box_pointcloud, transform_points_to_world
from model import create_combined_model

def save_plots(history, log_dir):
    """Save only collision metrics."""
    epochs = list(range(1, len(history['train_collision_loss']) + 1))
    
    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_collision_loss'], label='Train Collision Loss', linewidth=2)
    plt.plot(epochs, history['val_collision_loss'], label='Val Collision Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Collision Prediction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'loss_curves.png'), dpi=150)
    plt.close()
    
    # Collision metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_collision_accuracy'], label='Train', linewidth=2)
    plt.plot(epochs, history['val_collision_accuracy'], label='Val', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Collision Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_collision_precision'], label='Train', linewidth=2)
    plt.plot(epochs, history['val_collision_precision'], label='Val', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.title('Collision Precision')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_collision_recall'], label='Train', linewidth=2)
    plt.plot(epochs, history['val_collision_recall'], label='Val', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Recall'); plt.title('Collision Recall')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_collision_f1'], label='Train', linewidth=2)
    plt.plot(epochs, history['val_collision_f1'], label='Val', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('F1 Score'); plt.title('Collision F1')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'collision_metrics.png'), dpi=150)
    plt.close()

class FastPlacementDataset(torch.utils.data.Dataset):
    """Optimized dataset that generates world-frame point clouds efficiently"""
    
    def __init__(self, data_path, num_points=512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.num_points = num_points
        
        # Pre-generate point clouds for unique dimensions (cache)
        print("Pre-generating point clouds for unique dimensions...")
        self.dimension_pcds = {}
        unique_dimensions = set()
        
        for sample in self.data:
            dim_key = tuple(sample['object_dimensions'])
            unique_dimensions.add(dim_key)
        
        for dx, dy, dz in unique_dimensions:
            # Generate local point cloud once per unique dimension
            local_pcd = generate_box_pointcloud(dx, dy, dz, num_points=num_points)
            self.dimension_pcds[(dx, dy, dz)] = local_pcd
        
        print(f"Cached {len(self.dimension_pcds)} unique dimension point clouds")
        
        # Pre-allocate tensors
        N = len(self.data)
        self.corners = torch.empty((N, 8, 3), dtype=torch.float32)
        self.grasp = torch.empty((N, 7), dtype=torch.float32)
        self.init_pose = torch.empty((N, 7), dtype=torch.float32)
        self.final_pose = torch.empty((N, 7), dtype=torch.float32)
        self.label = torch.empty((N, 2), dtype=torch.float32)
        self.dimensions = []
        
        for i, sample in enumerate(self.data):
            dx, dy, dz = sample['object_dimensions']
            init_pose = np.array(sample['initial_object_pose'])
            
            # Generate ordered corners in local frame, then transform to world frame
            from dataset import cuboid_corners_local_ordered
            corners_local_ordered = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)
            corners_world = transform_points_to_world(corners_local_ordered, init_pose)
            self.corners[i] = torch.from_numpy(corners_world)
            
            self.grasp[i] = torch.tensor(sample['grasp_pose'], dtype=torch.float32)
            self.init_pose[i] = torch.tensor(sample['initial_object_pose'], dtype=torch.float32)
            self.final_pose[i] = torch.tensor(sample['final_object_pose'], dtype=torch.float32)
            self.label[i, 0] = float(sample['success_label'])
            self.label[i, 1] = float(sample['collision_label'])
            self.dimensions.append((dx, dy, dz))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get cached local point cloud for this dimension
        dim_key = self.dimensions[idx]
        local_pcd = self.dimension_pcds[dim_key]
        
        # Transform to world frame using initial pose
        init_pose = self.init_pose[idx].numpy()
        world_pcd = transform_points_to_world(local_pcd, init_pose)
        
        return (
            torch.from_numpy(world_pcd).float(),  # World-frame point cloud
            self.corners[idx],
            self.grasp[idx],
            self.init_pose[idx],
            self.final_pose[idx],
            self.label[idx]
        )

def main(dir_path):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Fast Training started on {device} ===")
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(dir_path, "training", "logs", f"fast_training_{timestamp}")
    model_dir = os.path.join(dir_path, "training", "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Hyperparameters - optimized for speed
    BATCH_SIZE = 256  # Larger batches for efficiency
    EPOCHS = 150
    INITIAL_LR = 5e-5
    WEIGHT_DECAY = 1e-4
    NUM_POINTS = 512  # Fewer points for speed
    
    # Load datasets
    train_path = os.path.join(dir_path, "data_collection/combined_data/train.json")
    val_path = os.path.join(dir_path, "data_collection/combined_data/val.json")
    
    print("Loading datasets...")
    train_dataset = FastPlacementDataset(train_path, num_points=NUM_POINTS)
    val_dataset = FastPlacementDataset(val_path, num_points=NUM_POINTS)
    
    print(f"  → {len(train_dataset)} train samples")
    print(f"  → {len(val_dataset)} val samples")
    
    # DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,  # More workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Model with optimizations
    model = create_combined_model().to(device)
    model = torch.compile(model)  # Compile for speed
    
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False, min_lr=1e-7)
    
    # History tracking
    history = {k: [] for k in [
        'train_collision_loss', 'train_collision_accuracy', 'train_collision_precision',
        'train_collision_recall', 'train_collision_f1',
        'val_collision_loss', 'val_collision_accuracy', 'val_collision_precision',
        'val_collision_recall', 'val_collision_f1'
    ]}
    
    best_val_loss = float("inf")
    patience = 30
    no_improve = 0
    
    print("Start training...")
    
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_tp, train_fp, train_fn, train_tn = 0, 0, 0, 0
        
        for point_clouds, corners, grasp, init, final, label in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            point_clouds, corners, grasp, init, final, label = [t.to(device) for t in (point_clouds, corners, grasp, init, final, label)]
            
            # Extract only collision label (index 1)
            collision_label = label[:, 1:2]  # Keep as [batch, 1]
            
            optimizer.zero_grad()
            with autocast():
                # Pass world-frame point clouds to model (PointNet gets trained!)
                collision_logits = model(point_clouds, corners, grasp, init, final)
                loss = criterion(collision_logits, collision_label)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            batch_size = collision_label.size(0)
            train_loss += loss.item() * batch_size
            train_total += batch_size
            
            pred = (torch.sigmoid(collision_logits) > 0.5).float()
            train_correct += (pred == collision_label).sum().item()
            train_tp += ((pred == 1) & (collision_label == 1)).sum().item()
            train_fp += ((pred == 1) & (collision_label == 0)).sum().item()
            train_fn += ((pred == 0) & (collision_label == 1)).sum().item()
            train_tn += ((pred == 0) & (collision_label == 0)).sum().item()
        
        # Compute training metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_prec = train_tp / (train_tp + train_fp + 1e-8)
        train_rec = train_tp / (train_tp + train_fn + 1e-8)
        train_f1 = 2 * train_prec * train_rec / (train_prec + train_rec + 1e-8)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_tp, val_fp, val_fn, val_tn = 0, 0, 0, 0
        
        with torch.no_grad():
            for point_clouds, corners, grasp, init, final, label in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                point_clouds, corners, grasp, init, final, label = [t.to(device) for t in (point_clouds, corners, grasp, init, final, label)]
                
                collision_label = label[:, 1:2]
                
                collision_logits = model(point_clouds, corners, grasp, init, final)
                loss = criterion(collision_logits, collision_label)
                
                batch_size = collision_label.size(0)
                val_loss += loss.item() * batch_size
                val_total += batch_size
                
                pred = (torch.sigmoid(collision_logits) > 0.5).float()
                val_correct += (pred == collision_label).sum().item()
                val_tp += ((pred == 1) & (collision_label == 1)).sum().item()
                val_fp += ((pred == 1) & (collision_label == 0)).sum().item()
                val_fn += ((pred == 0) & (collision_label == 1)).sum().item()
                val_tn += ((pred == 0) & (collision_label == 0)).sum().item()
        
        # Compute validation metrics
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_prec = val_tp / (val_tp + val_fp + 1e-8)
        val_rec = val_tp / (val_tp + val_fn + 1e-8)
        val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec + 1e-8)
        
        # Update history
        history['train_collision_loss'].append(train_loss)
        history['train_collision_accuracy'].append(train_acc)
        history['train_collision_precision'].append(train_prec)
        history['train_collision_recall'].append(train_rec)
        history['train_collision_f1'].append(train_f1)
        
        history['val_collision_loss'].append(val_loss)
        history['val_collision_accuracy'].append(val_acc)
        history['val_collision_precision'].append(val_prec)
        history['val_collision_recall'].append(val_rec)
        history['val_collision_f1'].append(val_f1)
        
        # Print progress
        print(f"\nEpoch {epoch}: Val Collision Loss={val_loss:.4f}")
        print(f"  → Collision Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6e}\n")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            model_path = os.path.join(model_dir, f"fast_best_model_{best_val_loss:.4f}.pth".replace('.', '_'))
            torch.save(model.state_dict(), model_path)
            print(f"→ Saved new best model: {os.path.basename(model_path)}\n")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("Training completed!")
    save_plots(history, log_dir)

if __name__ == "__main__":
    data_folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4"
    main(data_folder) 