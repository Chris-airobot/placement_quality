#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import glob
from dataset import WorldFrameDataset
from model import create_combined_model, create_original_enhanced_model, create_corners_only_fast_model
import json
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


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


class ShardEmbeddings:
    """Lightweight reader for experiment embeddings supporting:
    - Single .npy array (memmap)
    - Directory of shard .npy files (e.g., *_chunk_*.npy)
    - Shard index file: <stem>_shards.txt listing absolute or relative shard paths
    """
    def __init__(self, embeddings_path: str):
        self._init_backend(embeddings_path)

    def _init_backend(self, embeddings_path: str):
        ep = os.path.abspath(embeddings_path)
        base_dir = os.path.dirname(ep)
        base_name = os.path.basename(ep)
        base_stem = base_name[:-4] if base_name.endswith('.npy') else base_name
        shards_txt = os.path.join(base_dir, base_stem + "_shards.txt")

        def _open_single(path: str):
            self.is_sharded = False
            self.arr = np.load(path, mmap_mode='r')
            self.total = int(self.arr.shape[0])
            print(f"Loaded experiment embeddings (single): {self.arr.shape}, dtype: {self.arr.dtype}")

        def _prepare_shards(paths: list):
            paths = sorted(paths)
            if not paths:
                raise FileNotFoundError("Shard list is empty")
            self.is_sharded = True
            self.shards = paths
            self.shapes = []
            self.starts = []
            acc = 0
            for p in paths:
                a = np.load(p, mmap_mode='r')
                self.shapes.append(int(a.shape[0]))
                self.starts.append(acc)
                acc += int(a.shape[0])
                del a
            self.total = acc
            self.cache = {}
            print(f"Loaded experiment embeddings (sharded): {len(paths)} shards, total rows={self.total}")

        if os.path.isdir(ep):
            shard_paths = sorted(glob.glob(os.path.join(ep, "*_chunk_*.npy")))
            if not shard_paths:
                shard_paths = sorted(glob.glob(os.path.join(ep, "*.npy")))
            _prepare_shards(shard_paths)
            return

        if os.path.isfile(ep) and ep.endswith('.npy'):
            try:
                _open_single(ep)
                return
            except Exception:
                pass

        if os.path.isfile(shards_txt):
            with open(shards_txt, 'r') as f:
                shard_paths = [line.strip() for line in f if line.strip()]
            _prepare_shards(shard_paths)
            return

        # Auto-detect sibling chunk files
        pattern = os.path.join(base_dir, f"{base_stem}_chunk_*.npy")
        shard_paths = sorted(glob.glob(pattern))
        if shard_paths:
            _prepare_shards(shard_paths)
            return

        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    def __len__(self):
        return self.total

    def get_row(self, idx: int):
        if not getattr(self, 'is_sharded', False):
            return self.arr[idx]
        import bisect
        si = bisect.bisect_right(self.starts, idx) - 1
        if si < 0:
            si = 0
        local_idx = idx - self.starts[si]
        mm = self.cache.get(si)
        if mm is None:
            mm = np.load(self.shards[si], mmap_mode='r')
            self.cache[si] = mm
        return mm[local_idx]


def load_checkpoint_with_original_architecture(checkpoint_path, device, original_architecture=True):
    """Load checkpoint using the original model architecture"""
    
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    # Get state dict
    if 'model_state_dict' in checkpoint_data:
        state_dict = checkpoint_data['model_state_dict']
    else:
        state_dict = checkpoint_data
    
    # Strip any unwanted prefix
    prefix_to_strip = "_orig_mod."
    cleaned_state_dict = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = tensor
    
    # Create model with ORIGINAL architecture (no LayerNorm)
    if original_architecture:
        model = create_original_enhanced_model().to(device)
    else:
        print("Using combined model")
        model = create_combined_model().to(device)
    
    # Load the state dict - should work perfectly now
    try:
        model.load_state_dict(cleaned_state_dict)
        print("✅ Loaded checkpoint with original model architecture")
        return model, checkpoint_data
    except RuntimeError as e:
        print(f"❌ Error loading with original architecture: {str(e)}")
        raise e


def load_corners_only_checkpoint(checkpoint_path, device):
    """Load a corners-only checkpoint into the corners-only model."""
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    raw_sd = checkpoint_data['model_state_dict'] if 'model_state_dict' in checkpoint_data else checkpoint_data
    # Strip torch.compile prefix if present
    prefix = "_orig_mod."
    state_dict = {}
    for k, v in raw_sd.items():
        state_dict[k[len(prefix):]] = v if k.startswith(prefix) else v
    model = create_corners_only_fast_model().to(device)
    try:
        model.load_state_dict(state_dict)
        print("✅ Loaded corners-only checkpoint")
        return model, checkpoint_data
    except RuntimeError as e:
        print(f"❌ Error loading corners-only model: {str(e)}")
        raise e


def evaluate_checkpoint(checkpoint_path, test_loader, device, original_architecture=True):
    """Evaluate a checkpoint with enhanced metrics"""
    
    # Load checkpoint with original architecture
    model, checkpoint_data = load_checkpoint_with_original_architecture(checkpoint_path, device, original_architecture)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for corners, embeddings, grasp, init, final, label in tqdm(test_loader, desc="Evaluating"):
            # Move to device and handle embeddings
            corners, grasp, init, final, label = [t.to(device) for t in (corners, grasp, init, final, label)]
            embeddings = embeddings.to(device).float() if embeddings.dtype == torch.float16 else embeddings.to(device)
            
            # Extract collision label
            collision_label = label[:, 1:2]  # [batch, 1]

            # Forward pass
            collision_logits = model(embeddings, corners, grasp, init, final)

            # Compute loss
            loss = criterion(collision_logits, collision_label)
            total_loss += loss.item() * collision_label.size(0)
            total_samples += collision_label.size(0)
            
            # Store predictions for advanced metrics
            probs = torch.sigmoid(collision_logits).cpu().numpy().flatten()
            labels = collision_label.cpu().numpy().flatten()
            all_predictions.extend(probs)
            all_labels.extend(labels)

    # Compute metrics
    avg_loss = total_loss / total_samples
    metrics = compute_advanced_metrics(np.array(all_labels), np.array(all_predictions))
    
    return {
        'loss': avg_loss,
        **metrics
    }


def main(dir_path):
    # Setup paths
    test_data_json = os.path.join(dir_path, 'data_collection/combined_data/test.json')
    test_embeddings_file = os.path.join(dir_path, 'embeddings/test_embeddings.npy')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # Find latest checkpoints
    model_dir = os.path.join(dir_path, 'training/models/model_20250804_175834')
    if not os.path.exists(model_dir):
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    # Find all checkpoints
    checkpoints = []
    for pattern in ['best_model_loss_*.pth', 'best_model_roc_*.pth', 'final_model_*.pth']:
        checkpoints.extend(glob.glob(os.path.join(model_dir, pattern)))
    
    if not checkpoints:
        print("❌ No checkpoints found in model directory")
        return
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    print(f"Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        print(f"  → {os.path.basename(ckpt)}")
    
    # Load test dataset (we'll reuse this for all models)
    print("\nLoading test dataset...")
    
    # Get normalization stats from the first checkpoint
    first_checkpoint = checkpoints[0]
    checkpoint_data = torch.load(first_checkpoint, map_location=device)
    normalization_stats = checkpoint_data.get('normalization_stats', None)
    
    if normalization_stats is None:
        print("❌ No normalization stats found in checkpoint. Loading training dataset...")
        # Fallback: load training dataset to get stats
        train_data_json = os.path.join(dir_path, 'data_collection/combined_data/train.json')
        train_embeddings_file = os.path.join(dir_path, 'embeddings/train_embeddings.npy')
        train_dataset = WorldFrameDataset(train_data_json, train_embeddings_file, 
                                        normalization_stats=None, is_training=True)
        normalization_stats = train_dataset.normalization_stats
        print("✅ Got normalization stats from training dataset")
    else:
        print("✅ Found normalization stats in checkpoint")
    
    # Load test dataset with normalization stats
    test_dataset = WorldFrameDataset(test_data_json, test_embeddings_file,
                                   normalization_stats=normalization_stats,
                                   is_training=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    print(f"Loaded test set: {len(test_dataset)} samples → {len(test_loader)} batches\n")
    
    # Evaluate all checkpoints
    all_results = {}
    
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n{'='*60}")
        print(f"→ Evaluating {checkpoint_name}")
        print(f"{'='*60}")
        
        try:
            results = evaluate_checkpoint(checkpoint_path, test_loader, device, original_architecture=False)
            all_results[checkpoint_name] = results
            
            print(f"\n📊 Results for {checkpoint_name}:")
            print(f"  Loss: {results['loss']:.4f}")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1 Score: {results['f1']:.4f}")
            print(f"  ROC-AUC: {results['roc_auc']:.4f}")
            print(f"  PR-AUC: {results['pr_auc']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_name}: {str(e)}")
            all_results[checkpoint_name] = {'error': str(e)}
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("📊 MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Find the best model for each metric
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    best_models = {}
    
    for metric in metrics:
        best_value = None
        best_model = None
        
        for model_name, results in all_results.items():
            if 'error' in results:
                continue
            value = results[metric]
            
            if best_value is None:
                best_value = value
                best_model = model_name
            elif metric == 'loss':  # Lower is better
                if value < best_value:
                    best_value = value
                    best_model = model_name
            else:  # Higher is better
                if value > best_value:
                    best_value = value
                    best_model = model_name
        
        best_models[metric] = best_model
    
    # Print results table
    print(f"{'Model':<30} {'Loss':<8} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'ROC':<6} {'PR':<6}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<30} {'ERROR':<8}")
            continue
            
        print(f"{model_name:<30} "
              f"{results['loss']:<8.4f} "
              f"{results['accuracy']:<6.4f} "
              f"{results['precision']:<6.4f} "
              f"{results['recall']:<6.4f} "
              f"{results['f1']:<6.4f} "
              f"{results['roc_auc']:<6.4f} "
              f"{results['pr_auc']:<6.4f}")
    
    print("-" * 80)
    print("🏆 BEST MODELS:")
    for metric, best_model in best_models.items():
        if best_model:
            print(f"  {metric.upper()}: {best_model}")
    
    # Save comprehensive results
    results_file = os.path.join(dir_path, 'evaluation_results_comprehensive.json')
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'best_models': best_models,
            'test_samples': len(test_dataset),
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    print(f"\n💾 Comprehensive results saved to: {results_file}")


def analyze_experiment_predictions():
    """Read and display experiment generation results and model predictions"""
    
    # Paths
    experiment_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_generation.json"
    predictions_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions.json"
    
    print("🔍 ANALYZING EXPERIMENT PREDICTIONS")
    print("=" * 60)
    
    # Load experiment data
    try:
        with open(experiment_file, 'r') as f:
            experiments = json.load(f)
        print(f"✅ Loaded {len(experiments)} experiments")
    except FileNotFoundError:
        print(f"❌ Experiment file not found: {experiment_file}")
        return
    except Exception as e:
        print(f"❌ Error loading experiment file: {e}")
        return
    
    # Load predictions
    try:
        predictions = []
        with open(predictions_file, 'r') as f:
            for line in f:
                predictions.append(json.loads(line.strip()))
        print(f"✅ Loaded {len(predictions)} predictions")
    except FileNotFoundError:
        print(f"❌ Predictions file not found: {predictions_file}")
        print("💡 Run experiment_generation.py first to generate predictions")
        return
    except Exception as e:
        print(f"❌ Error loading predictions file: {e}")
        return
    
    # Analyze by object dimensions
    dimension_stats = {}
    
    for i, (exp, pred) in enumerate(zip(experiments, predictions)):
        dims = tuple(exp['object_dimensions'])
        pred_value = pred['pred_no_collision']
        
        if dims not in dimension_stats:
            dimension_stats[dims] = {
                'count': 0,
                'predictions': [],
                'avg_prediction': 0.0,
                'min_prediction': float('inf'),
                'max_prediction': float('-inf')
            }
        
        dimension_stats[dims]['count'] += 1
        dimension_stats[dims]['predictions'].append(pred_value)
        dimension_stats[dims]['min_prediction'] = min(dimension_stats[dims]['min_prediction'], pred_value)
        dimension_stats[dims]['max_prediction'] = max(dimension_stats[dims]['max_prediction'], pred_value)
    
    # Calculate averages
    for dims in dimension_stats:
        preds = dimension_stats[dims]['predictions']
        dimension_stats[dims]['avg_prediction'] = sum(preds) / len(preds)
    
    # Display results
    print(f"\n📊 PREDICTION ANALYSIS BY OBJECT DIMENSIONS")
    print("=" * 60)
    
    for dims, stats in sorted(dimension_stats.items()):
        print(f"\n Object Dimensions: {dims}")
        print(f"   Count: {stats['count']} experiments")
        print(f"   Average Prediction: {stats['avg_prediction']:.4f}")
        print(f"   Min Prediction: {stats['min_prediction']:.4f}")
        print(f"   Max Prediction: {stats['max_prediction']:.4f}")
        
        # Categorize predictions
        high_conf = sum(1 for p in stats['predictions'] if p > 0.8)
        med_conf = sum(1 for p in stats['predictions'] if 0.3 <= p <= 0.8)
        low_conf = sum(1 for p in stats['predictions'] if p < 0.3)
        
        print(f"   High Confidence (>0.8): {high_conf} ({high_conf/stats['count']*100:.1f}%)")
        print(f"   Medium Confidence (0.3-0.8): {med_conf} ({med_conf/stats['count']*100:.1f}%)")
        print(f"   Low Confidence (<0.3): {low_conf} ({low_conf/stats['count']*100:.1f}%)")
    
    # Show some sample predictions
    print(f"\n SAMPLE PREDICTIONS (first 10)")
    print("=" * 60)
    for i in range(min(10, len(experiments))):
        exp = experiments[i]
        pred = predictions[i]
        print(f"Experiment {i+1}:")
        print(f"  Dimensions: {exp['object_dimensions']}")
        print(f"  Prediction: {pred['pred_no_collision']:.4f}")
        print(f"  Initial Pose: {exp['initial_object_pose'][:3]}")  # Just position
        print(f"  Final Pose: {exp['final_object_pose'][:3]}")      # Just position
        print()


def generate_experiment_predictions(corners_only: bool = True, zero_embeddings: bool = False):
    """Generate model predictions for experiments and write predictions JSON.

    - corners_only: use corners-only model/checkpoint
    - zero_embeddings: for combined model, feed a zero vector for embeddings to test usefulness
    """
    
    # Paths
    experiment_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiments.json"
    # Can be a single .npy file, a directory containing *_chunk_*.npy, or a shards index <stem>_shards.txt
    embeddings_path = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_embeddings.npy"
    if corners_only:
        predictions_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions_corners_only.json"
    elif zero_embeddings:
        predictions_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions_zero_emb.json"
    else:
        predictions_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/test_data_predictions.json"
    
    print("🚀 GENERATING EXPERIMENT PREDICTIONS")
    print("=" * 60)
    
    # Load experiment data
    try:
        with open(experiment_file, 'r') as f:
            experiments = json.load(f)
        print(f"✅ Loaded {len(experiments)} experiments from {experiment_file}")
    except FileNotFoundError:
        print(f"❌ Experiment file not found: {experiment_file}")
        return
    except Exception as e:
        print(f"❌ Error loading experiment file: {e}")
        return
    
    # Load embeddings only if not corners-only and not zero-embedding mode
    if not corners_only and not zero_embeddings:
        try:
            emb = ShardEmbeddings(embeddings_path)
            print(f"✅ Loaded experiment embeddings backend, total rows: {len(emb):,}")
        except Exception as e:
            print(f"❌ Error initializing embeddings backend from {embeddings_path}: {e}")
            return
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the best model checkpoint (update path as needed)
    checkpoint_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/training/models/model_20250814_210656/best_model_roc_20250814_210656.pth"
    
    try:
        if corners_only:
            model, checkpoint_data = load_corners_only_checkpoint(checkpoint_path, device)
        else:
            model, checkpoint_data = load_checkpoint_with_original_architecture(checkpoint_path, device, False)
        model.eval()
        print(f"✅ Loaded model from {checkpoint_path}")
        
        # Extract normalization stats from checkpoint data
        normalization_stats = checkpoint_data.get('normalization_stats', None)

        
        if normalization_stats is None:
            print("⚠️ No normalization stats found in checkpoint. Computing from training dataset...")
            # Load training dataset to compute normalization stats
            train_data_json = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/combined_data/train.json"
            train_embeddings_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/embeddings/train_embeddings.npy"
            
            try:
                from dataset import WorldFrameDataset
                train_dataset = WorldFrameDataset(train_data_json, train_embeddings_file, 
                                                normalization_stats=None, is_training=True)
                normalization_stats = train_dataset.normalization_stats
                print("✅ Computed normalization stats from training dataset")
                print(f"Available normalization keys: {list(normalization_stats.keys())}")
            except Exception as e:
                print(f"❌ Error computing normalization stats: {e}")
                normalization_stats = None
        else:
            print("✅ Found normalization stats in checkpoint")
            print(f"Available normalization keys: {list(normalization_stats.keys())}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Import helpers
    from dataset import cuboid_corners_local_ordered, _zscore_normalize, transform_points_to_world
    
    # Generate predictions
    print(f"\n🔮 Generating predictions for {len(experiments)} experiments...")
    result_records = []
    
    for i, data in enumerate(tqdm(experiments, desc="Predicting")):
        # Extract dimensions and poses
        dx, dy, dz = data['object_dimensions']
        init_pose = np.array(data['initial_object_pose'], dtype=np.float32)
        final_pose = np.array(data['final_object_pose'], dtype=np.float32)
        grasp_pose = np.array(data['grasp_pose'], dtype=np.float32)

        # Compute world-frame corners for init and final
        corners_local = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)
        init_world = transform_points_to_world(corners_local, init_pose).reshape(1, -1)  # [1,24]
        final_world = transform_points_to_world(corners_local, final_pose).reshape(1, -1)  # [1,24]
        
        # Apply normalization if stats are available and have the right keys
        if normalization_stats is not None and 'corners_mean' in normalization_stats:
            # Normalize corners separately for init and final using same stats
            c_mean = normalization_stats['corners_mean']
            c_std  = normalization_stats['corners_std']
            init_norm, _, _ = _zscore_normalize(init_world, c_mean, c_std)
            # Use single-pose corners (init) to match training (8x3 input)
            corners = init_norm.numpy().reshape(1, 8, 3)
            
            # Split position and orientation
            grasp_pos = grasp_pose[:3]
            grasp_ori = grasp_pose[3:]
            init_pos = init_pose[:3]
            init_ori = init_pose[3:]
            final_pos = final_pose[:3]
            final_ori = final_pose[3:]
            
            # Normalize positions using unified stats
            pos_mean = normalization_stats['pos_mean']
            pos_std = normalization_stats['pos_std']
            
            # Convert to numpy arrays if they're tensors
            if isinstance(pos_mean, torch.Tensor):
                pos_mean = pos_mean.cpu().numpy()
            if isinstance(pos_std, torch.Tensor):
                pos_std = pos_std.cpu().numpy()
            
            grasp_pos_norm = (grasp_pos - pos_mean) / pos_std
            init_pos_norm = (init_pos - pos_mean) / pos_std
            final_pos_norm = (final_pos - pos_mean) / pos_std
            
            # Normalize orientations
            grasp_ori_norm, _, _ = _zscore_normalize(
                grasp_ori.reshape(1, -1),
                normalization_stats['grasp_ori_mean'],
                normalization_stats['grasp_ori_std']
            )
            init_ori_norm, _, _ = _zscore_normalize(
                init_ori.reshape(1, -1),
                normalization_stats['init_ori_mean'],
                normalization_stats['init_ori_std']
            )
            final_ori_norm, _, _ = _zscore_normalize(
                final_ori.reshape(1, -1),
                normalization_stats['final_ori_mean'],
                normalization_stats['final_ori_std']
            )
            
            # Reconstruct normalized poses
            grasp_normalized = np.concatenate([grasp_pos_norm.flatten(), grasp_ori_norm.flatten()])
            init_normalized = np.concatenate([init_pos_norm.flatten(), init_ori_norm.flatten()])
            final_normalized = np.concatenate([final_pos_norm.flatten(), final_ori_norm.flatten()])
        else:
            # Raw features (use init-only corners to match training)
            corners = init_world.reshape(1, 8, 3)
            grasp_normalized = grasp_pose
            init_normalized = init_pose
            final_normalized = final_pose
        
        # Convert to tensors with batch dimension
        corners_tensor = torch.tensor(corners, dtype=torch.float32).to(device)
        grasp_tensor = torch.tensor(grasp_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        init_tensor = torch.tensor(init_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        final_tensor = torch.tensor(final_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            if corners_only:
                collision_logits = model(corners=corners_tensor, grasp_pose=grasp_tensor, init_pose=init_tensor, final_pose=final_tensor)
            else:
                # Build embedding tensor
                if zero_embeddings:
                    # Use a zero vector matching embed_dim (1024 in combined model)
                    embedding_tensor = torch.zeros((1, 1024), dtype=torch.float32, device=device)
                else:
                    real_embedding = emb.get_row(i)
                    embedding_tensor = torch.tensor(real_embedding, dtype=torch.float32).unsqueeze(0).to(device)
                collision_logits = model(embedding_tensor, corners_tensor, grasp_tensor, init_tensor, final_tensor)
            pred_collision = torch.sigmoid(collision_logits)
            pred_no_collision = 1 - pred_collision.item()  # Probability of no collision
            pred_no_collision_label = 1 if pred_no_collision > 0.9 else 0
        
        # Store result in the same format as the original file
        result_record = {
            "object_dimensions": data["object_dimensions"],
            "pred_no_collision": float(pred_no_collision),
            "pred_no_collision_label": int(pred_no_collision_label)
        }
        result_records.append(result_record)
        
        # Show progress every 100 experiments
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(experiments)} experiments")
    
    # Write results to file in the same format
    print(f"\n💾 Writing {len(result_records)} predictions to {predictions_file}")
    with open(predictions_file, 'w') as f_out:
        for rec in result_records:
            f_out.write(json.dumps(rec) + "\n")
    
    print(f"✅ Successfully wrote predictions to {predictions_file}")
    
    # Show some statistics
    predictions = [rec['pred_no_collision'] for rec in result_records]
    print(f"\n📊 PREDICTION STATISTICS:")
    print(f"  Total experiments: {len(predictions)}")
    print(f"  Average prediction: {sum(predictions)/len(predictions):.4f}")
    print(f"  Min prediction: {min(predictions):.4f}")
    print(f"  Max prediction: {max(predictions):.4f}")
    
    # Show first few predictions
    print(f"\n🔍 FIRST 5 PREDICTIONS:")
    for i in range(min(5, len(result_records))):
        rec = result_records[i]
        print(f"  {i+1}. Dimensions: {rec['object_dimensions']}, No collision prob: {rec['pred_no_collision']:.4f}")


if __name__ == '__main__':
    experiment = True
    if experiment:
        generate_experiment_predictions(corners_only=False, zero_embeddings=True)
    else:
        dir_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/"
        main(dir_path)