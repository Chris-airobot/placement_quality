import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from dataset import PlacementDataset, generate_box_pointcloud, transform_points_to_world

# Add PointNet++ path
pointnet_path = '/home/chris/Chris/placement_ws/src/Pointnet_Pointnet2_pytorch'
sys.path.append(pointnet_path)
sys.path.append(os.path.join(pointnet_path, 'models'))

# Import after adding paths
from pointnet2.pointnet2_cls_ssg import get_model

def precompute_embeddings(data_path, output_file):
    """Precompute world-frame embeddings and save in one efficient file"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = PlacementDataset(data_path)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Load pre-trained PointNet++ model
    pretrained_model_path = '/home/riot/Chris/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    
    # Create PointNet++ model
    pointnet = get_model(num_class=40, normal_channel=False).to(device)  # 40 classes for ModelNet40
    
    # Load pre-trained weights
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    pointnet.load_state_dict(checkpoint['model_state_dict'])
    pointnet.eval()
    
    print(f"Loaded pre-trained PointNet++ from {pretrained_model_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # ✅ NEW: Incremental saving setup
    batch_size = 384
    chunk_size = 5000  # Save every 5000 samples
    
    # Create temporary files for incremental saving
    temp_dir = os.path.dirname(output_file)
    temp_prefix = os.path.basename(output_file).replace('.npy', '_chunk_')
    
    chunk_files = []
    
    with torch.no_grad():
        for chunk_idx, chunk_start in enumerate(tqdm(range(0, len(dataset), chunk_size), desc="Processing chunks")):
            chunk_end = min(chunk_start + chunk_size, len(dataset))
            
            # Small chunk array (not giant array)
            chunk_embeddings = []
            
            for batch_start in range(chunk_start, chunk_end, batch_size):
                batch_end = min(batch_start + batch_size, chunk_end)
                current_batch_size = batch_end - batch_start
                
                # Prepare batch tensors
                batch_pcds = torch.zeros(current_batch_size, 1024, 3, device=device) # Changed num_points to 1024
                
                # Process batch
                for i, idx in enumerate(range(batch_start, batch_end)):
                    corners, dim_idx, grasp, init, final, label = dataset[idx]
                    
                    # Get dimensions for this sample
                    dx, dy, dz = dataset.unique_dimensions[dim_idx]
                    
                    # Generate local point cloud
                    local_pcd = generate_box_pointcloud(dx, dy, dz, num_points=1024) # Changed num_points to 1024
                    
                    # Transform to world frame using initial pose
                    world_pcd = transform_points_to_world(local_pcd, init.numpy())
                    
                    # Add to batch (PointNet++ expects [B, N, 3] format)
                    batch_pcds[i] = torch.from_numpy(world_pcd).float()
                
                # PointNet++ expects input in [B, N, 3] format
                batch_pcds = batch_pcds.transpose(1, 2)  # [B, 3, N] -> [B, N, 3]
                
                # Process entire batch at once
                batch_output = pointnet(batch_pcds)  # Returns (logits, trans, trans_feat)
                batch_embeddings = batch_output[1].squeeze(-1) # Changed to .squeeze(-1)
                chunk_embeddings.append(batch_embeddings.cpu().numpy())
            
            # ✅ NEW: Save this chunk immediately
            chunk_array = np.concatenate(chunk_embeddings, axis=0).astype(np.float16)
            chunk_file = os.path.join(temp_dir, f"{temp_prefix}{chunk_idx:04d}.npy")
            np.save(chunk_file, chunk_array)
            chunk_files.append(chunk_file)
            
            print(f"Saved chunk {chunk_idx+1}/{(len(dataset) + chunk_size - 1) // chunk_size}: {chunk_file}")
    
    # ✅ NEW: Combine all chunk files into final file
    print("Combining all chunks into final file...")
    all_embeddings = []
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file)
        all_embeddings.append(chunk_data)
    
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_file, final_embeddings)
    
    # ✅ NEW: Clean up temporary files
    print("Cleaning up temporary files...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"Saved {len(dataset)} embeddings to {output_file}")
    print(f"File size: {file_size_gb:.2f} GB")

def precompute_experiment_embeddings(experiment_file, output_file):
    """Precompute embeddings for experiment generation data"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load experiment data
    with open(experiment_file, 'r') as f:
        experiments = json.load(f)
    print(f"Loaded {len(experiments)} experiments from {experiment_file}")
    
    # Load pre-trained PointNet++ model
    pretrained_model_path = '/home/riot/Chris/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    # Create PointNet++ model
    pointnet = get_model(num_class=40, normal_channel=False).to(device)
    
    # Load pre-trained weights
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    pointnet.load_state_dict(checkpoint['model_state_dict'])
    pointnet.eval()
    
    print(f"Loaded pre-trained PointNet++ from {pretrained_model_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Batch processing setup
    batch_size = 384
    chunk_size = 5000
    
    # Create temporary files for incremental saving
    temp_dir = os.path.dirname(output_file)
    temp_prefix = os.path.basename(output_file).replace('.npy', '_chunk_')
    
    chunk_files = []
    
    with torch.no_grad():
        for chunk_idx, chunk_start in enumerate(tqdm(range(0, len(experiments), chunk_size), desc="Processing chunks")):
            chunk_end = min(chunk_start + chunk_size, len(experiments))
            
            # Small chunk array
            chunk_embeddings = []
            
            for batch_start in range(chunk_start, chunk_end, batch_size):
                batch_end = min(batch_start + batch_size, chunk_end)
                current_batch_size = batch_end - batch_start
                
                # Prepare batch tensors
                batch_pcds = torch.zeros(current_batch_size, 1024, 3, device=device)
                
                # Process batch
                for i, idx in enumerate(range(batch_start, batch_end)):
                    exp_data = experiments[idx]
                    
                    # Extract object dimensions and initial pose from experiment format
                    dx, dy, dz = exp_data['object_dimensions']
                    initial_pose = exp_data['initial_object_pose']  # [x, y, z, qw, qx, qy, qz]
                    
                    # Generate local point cloud for this object
                    local_pcd = generate_box_pointcloud(dx, dy, dz, num_points=1024)
                    
                    # Transform to world frame using initial pose
                    world_pcd = transform_points_to_world(local_pcd, np.array(initial_pose))
                    
                    # Add to batch (PointNet++ expects [B, N, 3] format)
                    batch_pcds[i] = torch.from_numpy(world_pcd).float()
                
                # PointNet++ expects input in [B, 3, N] format, so transpose
                batch_pcds = batch_pcds.transpose(1, 2)  # [B, N, 3] -> [B, 3, N]
                
                # Process entire batch at once
                batch_output = pointnet(batch_pcds)  # Returns (logits, features)
                batch_embeddings = batch_output[1].squeeze(-1)  # Extract 1024-D features
                chunk_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Save this chunk immediately
            chunk_array = np.concatenate(chunk_embeddings, axis=0).astype(np.float16)
            chunk_file = os.path.join(temp_dir, f"{temp_prefix}{chunk_idx:04d}.npy")
            np.save(chunk_file, chunk_array)
            chunk_files.append(chunk_file)
            
            print(f"Saved chunk {chunk_idx+1}/{(len(experiments) + chunk_size - 1) // chunk_size}: {chunk_file}")
    
    # Combine all chunk files into final file
    print("Combining all chunks into final file...")
    all_embeddings = []
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file)
        all_embeddings.append(chunk_data)
    
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_file, final_embeddings)
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"✅ Saved {len(experiments)} embeddings to {output_file}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Embeddings shape: {final_embeddings.shape}")
    
    return output_file

def test_small_subset():
    """Test with just 100 samples to validate everything works"""
    print("=== TESTING WITH 100 SAMPLES ===")
    
    # Load small subset
    with open("/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/combined_data/val.json", 'r') as f:
        full_data = json.load(f)
    
    small_data = full_data[:100]  # Just 100 samples
    test_path = "test_100.json"
    with open(test_path, 'w') as f:
        json.dump(small_data, f)
    
    # Copy processed_data directory for test
    import shutil
    if not os.path.exists("processed_data"):
        shutil.copytree("/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/processed_data", "processed_data")
    
    # Test preprocessing
    test_output = "./test_embeddings.npy"
    precompute_embeddings(test_path, test_output)
    
    # Check file size
    file_size_mb = os.path.getsize(test_output) / (1024**2)
    expected_size_mb = 100 * 1024 * 2 / (1024**2)  # float16 = 2 bytes
    print(f"File size: {file_size_mb:.2f} MB (expected: {expected_size_mb:.2f} MB)")
    
    # Test loading
    embeddings = np.load(test_output)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Validate file size
    file_size_mb = os.path.getsize(test_output) / (1024**2)
    expected_size_mb = 100 * 1024 * 2 / (1024**2)  # float16 = 2 bytes
    print(f"File size: {file_size_mb:.2f} MB (expected: {expected_size_mb:.2f} MB)")
    
    # if abs(file_size_mb - expected_size_mb) > 0.1:
    #     print(f"❌ File size wrong! Expected ~{expected_size_mb:.2f}MB, got {file_size_mb:.2f}MB")
    #     return False
    
    # Validate embedding content
    print("Validating embedding content...")
    if embeddings.shape != (100, 1024):
        print(f"❌ Wrong shape! Expected (100, 1024), got {embeddings.shape}")
        return False
    
    if not np.isfinite(embeddings).all():
        print("❌ Embeddings contain NaN or inf values!")
        return False
    
    # Check embedding values are reasonable (not all zeros, reasonable range)
    if np.allclose(embeddings, 0):
        print("❌ All embeddings are zero!")
        return False
    
    embedding_std = np.std(embeddings)
    if embedding_std < 0.01:
        print(f"❌ Embeddings too uniform! std={embedding_std:.6f}")
        return False
    
    print(f"✅ Embeddings look good! std={embedding_std:.6f}")
    
    # Test dataset loading
    from dataset import WorldFrameDataset
    test_dataset = WorldFrameDataset(test_path, test_output)
    print(f"Dataset length: {len(test_dataset)}")
    
    # Test one sample
    sample = test_dataset[0]
    print(f"Sample shapes: corners={sample[0].shape}, embedding={sample[1].shape}")
    
    # Test model forward pass
    print("Testing model forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import create_combined_model
    model = create_combined_model().to(device)
    model.eval()
    
    with torch.no_grad():
        corners, embedding, grasp, init, final, label = sample
        corners, embedding, grasp, init, final = [t.to(device) for t in (corners, embedding, grasp, init, final)]
        
        logits = model(embedding.unsqueeze(0), corners.unsqueeze(0), grasp.unsqueeze(0), init.unsqueeze(0), final.unsqueeze(0))
        print(f"✅ Model forward pass works! Output shape: {logits.shape}")
    
    # Cleanup
    os.remove(test_path)
    # os.remove(test_output)  # Commented out to keep test embeddings for inspection
    shutil.rmtree("processed_data")
    print("✅ All tests passed! Ready for full dataset.")
    print(f"Test embeddings saved to: {test_output}")
    return True

def validate_full_dataset():
    """Validate the full validation dataset embeddings"""
    print("=== VALIDATING FULL VALIDATION DATASET ===")
    
    val_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/combined_data/val.json"
    embeddings_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/embeddings/val_embeddings.npy"
    
    # Check file exists
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return False
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Check file size
    file_size_gb = os.path.getsize(embeddings_file) / (1024**3)
    expected_size_gb = embeddings.shape[0] * 1024 * 4 / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB (expected: {expected_size_gb:.2f} GB)")
    
    # if abs(file_size_gb - expected_size_gb) > 0.1:
    #     print(f"❌ File size wrong!")
    #     return False
    
    # Validate content
    print("Validating embedding content...")
    if not np.isfinite(embeddings).all():
        print("❌ Embeddings contain NaN or inf values!")
        return False
    
    if np.allclose(embeddings, 0):
        print("❌ All embeddings are zero!")
        return False
    
    embedding_std = np.std(embeddings)
    print(f"Embedding statistics:")
    print(f"  - Mean: {np.mean(embeddings):.6f}")
    print(f"  - Std: {embedding_std:.6f}")
    print(f"  - Min: {np.min(embeddings):.6f}")
    print(f"  - Max: {np.max(embeddings):.6f}")
    
    if embedding_std < 0.01:
        print(f"❌ Embeddings too uniform!")
        return False
    
    # Test dataset loading
    print("Testing dataset loading...")
    from dataset import WorldFrameDataset
    dataset = WorldFrameDataset(val_path, embeddings_file)
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    print("Testing sample loading...")
    for i in range(3):
        sample = dataset[i]
        corners, embedding, grasp, init, final, label = sample
        print(f"  Sample {i}: corners={corners.shape}, embedding={embedding.shape}, label={label}")
    
    # Test model forward pass
    print("Testing model forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import create_combined_model
    model = create_combined_model().to(device)
    model.eval()
    
    with torch.no_grad():
        sample = dataset[0]
        corners, embedding, grasp, init, final, label = sample
        corners, embedding, grasp, init, final = [t.to(device) for t in (corners, embedding, grasp, init, final)]
        
        logits = model(embedding.unsqueeze(0), corners.unsqueeze(0), grasp.unsqueeze(0), init.unsqueeze(0), final.unsqueeze(0))
        print(f"✅ Model forward pass works! Output shape: {logits.shape}")
    
    print("✅ Full validation dataset is correct!")
    return True

if __name__ == "__main__":
    import sys
    test = False
    validate = False
    experiment = True  # New flag for experiment data
    
    if test:
        test_small_subset()
    elif validate:
        validate_full_dataset()
    elif experiment:
        # Process experiment generation data
        experiment_file = "/home/riot/Chris/placement_quality/cube_generalization/experiments.json"
        output_file = "/home/riot/Chris/placement_quality/cube_generalization/experiment_embeddings.npy"
        
        precompute_experiment_embeddings(experiment_file, output_file)
    else:
        # Original functionality
        data_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/combined_data/test.json"
        output_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/embeddings/test_embeddings.npy"
        
        precompute_embeddings(data_path, output_file) 