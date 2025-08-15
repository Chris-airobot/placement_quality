import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import os
import glob
import json
import sys

# Import sklearn for advanced metrics
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

from dataset import WorldFrameDataset, WorldFrameCornersOnlyDataset
from model import create_combined_model, create_corners_only_fast_model  # latest + corners-only

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # keeps it numerically stable

    def forward(self, logits, targets):
        """
        logits : (N, 1)  raw scores
        targets: (N, 1)  0 or 1
        """
        bce_loss = self.bce(logits, targets)
        p_t = torch.exp(-bce_loss)          # = sigmoid(logit) if y==1 else 1-sigmoid
        focal = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal

class Tee:
    """Duplicate stdout/stderr to console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

def save_enhanced_plots(history, log_dir):
    """Enhanced plotting with ROC-AUC and PR-AUC curves"""
    
    # Create subplots for all metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Enhanced Training Metrics', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(history['train_collision_loss'], label='Train', color='blue', alpha=0.7)
    axes[0, 0].plot(history['val_collision_loss'], label='Val', color='orange', alpha=0.7)
    axes[0, 0].set_title('Collision Prediction Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('BCE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_collision_accuracy'], label='Train', color='blue', alpha=0.7)
    axes[0, 1].plot(history['val_collision_accuracy'], label='Val', color='orange', alpha=0.7)
    axes[0, 1].set_title('Collision Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ✨ ROC-AUC curves (NEW)
    axes[1, 0].plot(history['train_collision_roc_auc'], label='Train', color='blue', alpha=0.7)
    axes[1, 0].plot(history['val_collision_roc_auc'], label='Val', color='orange', alpha=0.7)
    axes[1, 0].set_title('ROC-AUC Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ROC-AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ✨ PR-AUC curves (NEW)
    axes[1, 1].plot(history['train_collision_pr_auc'], label='Train', color='blue', alpha=0.7)
    axes[1, 1].plot(history['val_collision_pr_auc'], label='Val', color='orange', alpha=0.7)
    axes[1, 1].set_title('Precision-Recall AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PR-AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Precision curves
    axes[2, 0].plot(history['train_collision_precision'], label='Train', color='blue', alpha=0.7)
    axes[2, 0].plot(history['val_collision_precision'], label='Val', color='orange', alpha=0.7)
    axes[2, 0].set_title('Collision Precision')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Precision')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # F1 curves
    axes[2, 1].plot(history['train_collision_f1'], label='Train', color='blue', alpha=0.7)
    axes[2, 1].plot(history['val_collision_f1'], label='Val', color='orange', alpha=0.7)
    axes[2, 1].set_title('Collision F1 Score')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('F1 Score')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'enhanced_training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compute_advanced_metrics(y_true, y_pred_probs, threshold=0.5):
    """Compute advanced classification metrics"""
    y_pred = (y_pred_probs > threshold).astype(int)
    
    # Basic metrics
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Advanced metrics
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except:
        roc_auc = 0.5  # Default if all same class
    
    try:
        pr_auc = average_precision_score(y_true, y_pred_probs)
    except:
        pr_auc = y_true.mean()  # Default to class balance
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def main(dir_path, emb_path=None, corners_only: bool = False):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Enable TF32 matmul on Ampere+ for faster training
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True
    print(f"=== Enhanced Training started on {device} ===")
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(dir_path, "training", "logs", f"training_{timestamp}")
    model_dir = os.path.join(dir_path, "training", "models", f"model_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # ✨ ADD THESE 4 LINES FOR LOGGING ✨
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(os.path.join(log_dir, 'training.log'), 'a')
    sys.stdout = Tee(orig_stdout, log_file)
    
    # ✨ ENHANCED HYPERPARAMETERS ✨
    BATCH_SIZE = 16384
    EPOCHS = 150
    INITIAL_LR = 1e-4  # 🚨 INCREASED from 5e-5 
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = max(8, min(32, (os.cpu_count() or 8) - 2))
    # Speed controls (do not limit data per epoch)
    COMPUTE_TRAIN_ADVANCED = False  # Skip train ROC/PR to save time
    CLIP_MAX_NORM = None            # Disable grad clipping by default
    
    # Load datasets with feature normalization
    train_path = os.path.join(dir_path, "data_collection/combined_data/train.json")
    val_path = os.path.join(dir_path, "data_collection/combined_data/val.json")
    if not corners_only:
        # Use shard directories directly; dataset will stream all *_chunk_*.npy inside
        train_embeddings_file = os.path.join(emb_path, "train")
        val_embeddings_file = os.path.join(emb_path, "val")
    
    print("Loading datasets with feature normalization...")
    
    # Load training dataset (computes normalization stats)
    # Optional cache dir if embeddings path is provided; otherwise skip
    if emb_path:
        fast_cache = os.path.join(emb_path, "_mm_cache")
        os.makedirs(fast_cache, exist_ok=True)
    if corners_only:
        train_dataset = WorldFrameCornersOnlyDataset(train_path, normalization_stats=None, is_training=True)
        val_dataset = WorldFrameCornersOnlyDataset(val_path, normalization_stats=train_dataset.normalization_stats, is_training=False)
    else:
        train_dataset = WorldFrameDataset(train_path, train_embeddings_file,
                                          normalization_stats=None, is_training=True)
        val_dataset = WorldFrameDataset(val_path, val_embeddings_file,
                                        normalization_stats=train_dataset.normalization_stats,
                                        is_training=False)
    
    print(f"  → {len(train_dataset)} train samples")
    print(f"  → {len(val_dataset)} val samples")
    
    # ✨ ENHANCED MODEL ✨
    model = (create_corners_only_fast_model() if corners_only else create_combined_model()).to(device)
    # Optional graph capture / compile for small MLP speedup
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("✅ torch.compile enabled")
    except Exception as e:
        print(f"⚠️ torch.compile unavailable/failed: {e}")
    scaler = GradScaler()
    
    # ✨ CLASS IMBALANCE ANALYSIS (moved here for WeightedRandomSampler) ✨
    collision_counts = torch.tensor([
        (train_dataset.label[:,1]==0).sum(),  # negatives
        (train_dataset.label[:,1]==1).sum()   # positives
    ], dtype=torch.float32)
    if collision_counts[1] == 0:
        raise ValueError("No positive samples found in training data")
    pos_weight = collision_counts[0] / collision_counts[1]
    
    print(f"Class distribution - Negative: {collision_counts[0]:.0f}, Positive: {collision_counts[1]:.0f}")
    print(f"Using pos_weight: {pos_weight:.3f}")
    
    # DataLoaders with optional WeightedRandomSampler
    print("Setting up DataLoaders...")
    
    # ✨ WEIGHTED RANDOM SAMPLER FOR CLASS BALANCE ✨
    # Torch's multinomial has a 2^24 category limit; skip weighted sampler for huge datasets
    USE_WEIGHTED = len(train_dataset) <= (1 << 24)
    if USE_WEIGHTED:
        collision_labels = train_dataset.label[:, 1].cpu()
        sample_weights = torch.zeros_like(collision_labels)
        sample_weights[collision_labels == 0] = 1.0 / collision_counts[0]
        sample_weights[collision_labels == 1] = 1.0 / collision_counts[1]
        weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader_weighted = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=weighted_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=16,
        )
    else:
        print("Dataset > 2^24; disabling WeightedRandomSampler to avoid multinomial limit.")
        train_loader_weighted = None
    
    train_loader_normal = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=16
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=16
    )
    
    print(f"✅ Created weighted sampler for balanced training")
    print(f"   Negative class weight: {(1.0 / collision_counts[0]).item():.2e}")
    print(f"   Positive class weight: {(1.0 / collision_counts[1]).item():.2e}")
    
    # ✨ LOSS FUNCTION WITH FOCAL LOSS FOR CLASS IMBALANCE ✨
    # Use focal loss instead of BCE for better handling of class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    print(f"✅ Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.3f}")
    
    # ✨ ENHANCED OPTIMIZER & SCHEDULER ✨
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    
    # Warm-up scheduler + ReduceLROnPlateau
    warmup_epochs = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - warmup_epochs), eta_min=1e-6)
        
    # ✨ ENHANCED HISTORY TRACKING ✨
    history = {k: [] for k in [
        'train_collision_loss', 'train_collision_accuracy', 'train_collision_precision',
        'train_collision_recall', 'train_collision_f1', 'train_collision_roc_auc', 'train_collision_pr_auc',
        'val_collision_loss', 'val_collision_accuracy', 'val_collision_precision',
        'val_collision_recall', 'val_collision_f1', 'val_collision_roc_auc', 'val_collision_pr_auc'
    ]}
    
    best_val_loss = float("inf")
    best_val_roc_auc = 0.0
    best_val_pr_auc = 0.0
    patience = 30
    patience_counter = 0

    
    print(f"Starting training for {EPOCHS} epochs...")
    print(f"Initial learning rate: {INITIAL_LR}")
    
    for epoch in range(EPOCHS):
        # Always use weighted sampler when available
        use_weighted_sampler = (train_loader_weighted is not None)
        current_train_loader = train_loader_weighted if use_weighted_sampler else train_loader_normal
        
        sampler_type = "Weighted" if use_weighted_sampler else "Normal"
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Using {sampler_type} Sampling")
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_predictions, train_labels = ([] if COMPUTE_TRAIN_ADVANCED else None), ([] if COMPUTE_TRAIN_ADVANCED else None)
        
        for batch in tqdm(current_train_loader, desc=f"Epoch {epoch+1} [Train]", file=orig_stderr):
            if corners_only:
                corners, grasp, init, final, label = batch
                embeddings = None
            else:
                corners, embeddings, grasp, init, final, label = batch
            # Non-blocking H2D copies (requires pin_memory=True in DataLoader)
            corners = corners.to(device, non_blocking=True)
            grasp   = grasp.to(device, non_blocking=True)
            init    = init.to(device, non_blocking=True)
            final   = final.to(device, non_blocking=True)
            label   = label.to(device, non_blocking=True)
            if embeddings is not None:
                embeddings = embeddings.to(device, non_blocking=True)
            collision_label = label[:, 1:2]
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                if corners_only:
                    collision_logits = model(corners=corners, grasp_pose=grasp, init_pose=init, final_pose=final)
                else:
                    collision_logits = model(embeddings, corners, grasp, init, final)
                loss = criterion(collision_logits, collision_label)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CLIP_MAX_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate metrics
            batch_size = collision_label.size(0)
            train_loss += loss.item() * batch_size
            train_total += batch_size
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(collision_logits)
                preds = (probs > 0.5).to(collision_label.dtype)
                train_correct += (preds == collision_label).sum().item()
                if COMPUTE_TRAIN_ADVANCED:
                    train_predictions.extend(probs.detach().cpu().numpy().flatten())
                    train_labels.extend(collision_label.detach().cpu().numpy().flatten())
        
        # Apply warmup for first 2 epochs
        if epoch < 2:
            warmup_scheduler.step()
        
        # Compute training metrics
        train_loss /= max(1, train_total)
        if COMPUTE_TRAIN_ADVANCED:
            train_metrics = compute_advanced_metrics(np.array(train_labels), np.array(train_predictions))
        else:
            train_metrics = {
                'accuracy': (train_correct / max(1, train_total)),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1': float('nan'),
                'roc_auc': float('nan'),
                'pr_auc': float('nan')
            }
        
        # Validation phase
        model.eval()
        val_loss, val_total = 0.0, 0
        val_predictions, val_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", file=orig_stderr):
                if corners_only:
                    corners, grasp, init, final, label = batch
                    embeddings = None
                else:
                    corners, embeddings, grasp, init, final, label = batch
                corners = corners.to(device, non_blocking=True)
                grasp   = grasp.to(device, non_blocking=True)
                init    = init.to(device, non_blocking=True)
                final   = final.to(device, non_blocking=True)
                label   = label.to(device, non_blocking=True)
                if embeddings is not None:
                    embeddings = embeddings.to(device, non_blocking=True)
                
                collision_label = label[:, 1:2]
                
                with autocast(device_type='cuda'):
                    if corners_only:
                        collision_logits = model(corners=corners, grasp_pose=grasp, init_pose=init, final_pose=final)
                    else:
                        collision_logits = model(embeddings, corners, grasp, init, final)
                    loss = criterion(collision_logits, collision_label)
                    
                    batch_size = collision_label.size(0)
                    val_loss += loss.item() * batch_size
                    val_total += batch_size
                    
                    # Store predictions for advanced metrics
                    probs = torch.sigmoid(collision_logits).cpu().numpy().flatten()
                    labels = collision_label.cpu().numpy().flatten()
                    val_predictions.extend(probs)
                    val_labels.extend(labels)
        
        # Compute validation metrics
        val_loss /= max(1, val_total)
        val_metrics = compute_advanced_metrics(np.array(val_labels), np.array(val_predictions))

        # Optional: quick threshold sweep on val for F1
        probs = np.array(val_predictions)
        labels = np.array(val_labels)
        ths = np.linspace(0.05, 0.95, 19)
        f1s = [compute_advanced_metrics(labels, probs, threshold=t)['f1'] for t in ths]
        best_t = float(ths[int(np.argmax(f1s))])
        print(f"  Val best F1 threshold ≈ {best_t:.2f}")
        
        # Apply main scheduler after warmup
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        # Store metrics in history
        history['train_collision_loss'].append(train_loss)
        history['train_collision_accuracy'].append(train_metrics['accuracy'])
        history['train_collision_precision'].append(train_metrics['precision'])
        history['train_collision_recall'].append(train_metrics['recall'])
        history['train_collision_f1'].append(train_metrics['f1'])
        history['train_collision_roc_auc'].append(train_metrics['roc_auc'])
        history['train_collision_pr_auc'].append(train_metrics['pr_auc'])
        
        history['val_collision_loss'].append(val_loss)
        history['val_collision_accuracy'].append(val_metrics['accuracy'])
        history['val_collision_precision'].append(val_metrics['precision'])
        history['val_collision_recall'].append(val_metrics['recall'])
        history['val_collision_f1'].append(val_metrics['f1'])
        history['val_collision_roc_auc'].append(val_metrics['roc_auc'])
        history['val_collision_pr_auc'].append(val_metrics['pr_auc'])
        
        # ✨ ENHANCED LOGGING WITH SAMPLING STRATEGY ✨
        current_lr = optimizer.param_groups[0]['lr']
        sampling_status = "🎯 Weighted" if use_weighted_sampler else "📊 Normal"
        
        print(f"Epoch {epoch+1}/{EPOCHS} | {sampling_status} | LR: {current_lr:.2e}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
        
        # ✨ MONITOR TARGET METRICS (as suggested) ✨
        target_loss_reached = val_loss < 0.25
        target_roc_reached = val_metrics['roc_auc'] > 0.905
        
        if target_loss_reached:
            print(f"  🎯 TARGET REACHED: Validation loss below 0.25 ({val_loss:.4f})")
        if target_roc_reached:
            print(f"  🏆 TARGET REACHED: ROC-AUC above 0.905 ({val_metrics['roc_auc']:.4f})")
        
        if use_weighted_sampler:
            print(f"  📈 Weighted sampling epoch {epoch+1}/10 - monitoring class balance improvements")
        
        # Save best model (dual criteria: loss and ROC-AUC)
        is_best_loss = val_loss < best_val_loss
        is_best_roc = val_metrics['roc_auc'] > best_val_roc_auc
        
        if is_best_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_roc_auc': val_metrics['roc_auc'],
                'normalization_stats': train_dataset.normalization_stats,
                'focal_loss_config': {'alpha': 0.25, 'gamma': 2.0}  # Save focal loss config
            }, os.path.join(model_dir, f'best_model_loss_{timestamp}.pth'))
            print(f"  💾 Saved best loss model (loss: {val_loss:.4f})")
        
        if is_best_roc:
            best_val_roc_auc = val_metrics['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_roc_auc': val_metrics['roc_auc'],
                'normalization_stats': train_dataset.normalization_stats,
                'focal_loss_config': {'alpha': 0.25, 'gamma': 2.0}  # Save focal loss config
            }, os.path.join(model_dir, f'best_model_roc_{timestamp}.pth'))
            print(f"  🎯 Saved best ROC-AUC model (ROC-AUC: {val_metrics['roc_auc']:.4f})")

        # Save best by PR-AUC (new)
        if val_metrics['pr_auc'] > best_val_pr_auc:
            best_val_pr_auc = val_metrics['pr_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_roc_auc': val_metrics['roc_auc'],
                'val_pr_auc': val_metrics['pr_auc'],
                'normalization_stats': train_dataset.normalization_stats,
            }, os.path.join(model_dir, f'best_model_prauc_{timestamp}.pth'))
        
        if not is_best_loss:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
        # Save plots every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_enhanced_plots(history, log_dir)
        
        print("-" * 80)
    
    # Final saves
    save_enhanced_plots(history, log_dir)
    
    # Save final model and history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_roc_auc': val_metrics['roc_auc'],
        'normalization_stats': train_dataset.normalization_stats,
        'focal_loss_config': {'alpha': 0.25, 'gamma': 2.0}  # Save focal loss config
    }, os.path.join(model_dir, f'final_model_{timestamp}.pth'))
    
    with open(os.path.join(log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("✅ Training completed!")
    # ✨ Close log file and restore stdout/stderr
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file.close()
    print(f"📊 Logs saved to: {log_dir}")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")
    print(f"🎯 Best validation ROC-AUC: {best_val_roc_auc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/chris/Chris/placement_ws/src/data/box_simulation/v5",
                        help="Root data dir containing data_collection/combined_data")
    parser.add_argument("--embeddings", type=str, default="/home/chris/Chris/placement_ws/src/data/box_simulation/v5/embeddings")
    parser.add_argument("--corners-only", action="store_true", help="Train without embeddings using corners-only model")
    args = parser.parse_args()
    main(args.data, None if args.corners_only else args.embeddings, corners_only=args.corners_only)
