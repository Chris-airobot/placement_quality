import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import glob
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def cuboid_corners_local_ordered(dx, dy, dz):
    """
    Ordered corners: Top 4 first (z=+dz/2), then bottom 4 (z=-dz/2)
    Within each level: clockwise when viewed from above (right-hand rule)
    Returns array of shape (8, 3) in consistent local frame ordering
    """
    # Top face (z = +dz/2) - clockwise from (-x,-y) when viewed from above
    top_corners = [
        [-dx/2, -dy/2, +dz/2],  # 0: left-front-top
        [+dx/2, -dy/2, +dz/2],  # 1: right-front-top  
        [+dx/2, +dy/2, +dz/2],  # 2: right-back-top
        [-dx/2, +dy/2, +dz/2],  # 3: left-back-top
    ]
    
    # Bottom face (z = -dz/2) - clockwise from (-x,-y) when viewed from above
    bottom_corners = [
        [-dx/2, -dy/2, -dz/2],  # 4: left-front-bottom
        [+dx/2, -dy/2, -dz/2],  # 5: right-front-bottom
        [+dx/2, +dy/2, -dz/2],  # 6: right-back-bottom  
        [-dx/2, +dy/2, -dz/2],  # 7: left-back-bottom
    ]
    
    return np.array(top_corners + bottom_corners)

def generate_box_pointcloud(dx, dy, dz, num_points=1024):
    """
    Generate num_points uniformly distributed within a box of dimensions dx, dy, dz
    Returns points in local frame centered at origin
    """
    # Generate random points within the box
    points = np.random.uniform(
        low=[-dx/2, -dy/2, -dz/2],
        high=[dx/2, dy/2, dz/2], 
        size=(num_points, 3)
    )
    return points.astype(np.float32)

def transform_points_to_world(points_local, pose_world):
    """Transform Nx3 points from local to world with pose [x,y,z,qw,qx,qy,qz] using fast numpy math."""
    # Ensure float32 for speed
    p = np.asarray(points_local, dtype=np.float32)
    t = np.asarray(pose_world[:3], dtype=np.float32)
    qw, qx, qy, qz = [np.float32(pose_world[3]), np.float32(pose_world[4]), np.float32(pose_world[5]), np.float32(pose_world[6])]
    # Normalize quaternion (safety)
    norm = np.float32(np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    # Rotation matrix (world R from object to world)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R00 = 1 - 2*(yy + zz); R01 = 2*(xy - wz);     R02 = 2*(xz + wy)
    R10 = 2*(xy + wz);     R11 = 1 - 2*(xx + zz); R12 = 2*(yz - wx)
    R20 = 2*(xz - wy);     R21 = 2*(yz + wx);     R22 = 1 - 2*(xx + yy)
    Rm = np.array([[R00, R01, R02],
                   [R10, R11, R12],
                   [R20, R21, R22]], dtype=np.float32)
    # world = p @ R^T + t
    return (p @ Rm.T) + t

def get_unique_dimensions_from_files(processed_data_dir):
    """Extract unique object dimensions from processed data filenames"""
    # Get all JSON files in the processed_data directory
    json_files = glob.glob(os.path.join(processed_data_dir, "processed_object_*.json"))
    
    dimensions = []
    for filepath in json_files:
        # Extract filename: processed_object_0.1_0.075_0.11.json
        filename = os.path.basename(filepath)
        # Parse dimensions from filename
        parts = filename.replace("processed_object_", "").replace(".json", "").split("_")
        if len(parts) == 3:
            try:
                dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
                dimensions.append((dx, dy, dz))
            except ValueError:
                continue
    
    print(f"Found {len(dimensions)} unique object dimensions from processed_data files")
    return dimensions

def _zscore_normalize(x, mean=None, std=None, eps=1e-8):
    """
    Z-score normalization with optional pre-computed statistics
    Returns normalized tensor and statistics (mean, std)
    """
    # Ensure x is a tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Get device of input tensor
    device = x.device
    
    if mean is None:
        mean = x.mean(0, keepdims=True)
    else:
        # Ensure mean is on the same device as x
        if isinstance(mean, torch.Tensor):
            mean = mean.to(device)
        else:
            mean = torch.tensor(mean, dtype=torch.float32, device=device)
    
    if std is None:
        std = x.std(0, keepdims=True)
    else:
        # Ensure std is on the same device as x
        if isinstance(std, torch.Tensor):
            std = std.to(device)
        else:
            std = torch.tensor(std, dtype=torch.float32, device=device)
    
    normalized = (x - mean) / (std + eps)
    return normalized, mean, std

class WorldFrameDataset(Dataset):
    def __init__(self, data_path, embeddings_file, normalization_stats=None, is_training=True, cache_dir: str = None):
        """
        Memory-efficient dataset with feature normalization and large-file support.
        - Builds on-disk memmaps from a huge JSON using streaming, then reads rows lazily.
        - Embeddings are memory-mapped.
        - Corners are computed on-the-fly per item and normalized using cached stats.
        """
        self.data_path = data_path
        self._setup_storage(data_path, cache_dir)
        # Keep original embeddings handle for lazy re-init in worker processes
        self._emb_path = embeddings_file
        self.N = int(self._meta['num_items'])

        # Initialize embeddings backend (single .npy or many shards)
        self._init_embeddings_backend(embeddings_file)

        # Safety: ensure embeddings cover all items
        if getattr(self, 'emb_total', None) is not None and self.emb_total != self.N:
            raise ValueError(
                f"Embeddings count mismatch: dataset has {self.N} items but embeddings provide {self.emb_total} rows. "
                f"Check that your embeddings directory/file matches the split and is complete."
            )

        # Load memmaps for fast row access
        self.mm_dims  = np.memmap(self._meta['dims_file'],  dtype=np.float32, mode='r', shape=(self.N, 3))
        self.mm_grasp = np.memmap(self._meta['grasp_file'], dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_init  = np.memmap(self._meta['init_file'],  dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_final = np.memmap(self._meta['final_file'], dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_label = np.memmap(self._meta['label_file'], dtype=np.float32, mode='r', shape=(self.N, 2))

        # Compute or apply normalization statistics
        if is_training or normalization_stats is None:
            print("Computing normalization statistics (sampled, no precomputed corners)...")
            stats = self._compute_stats_streamed(sample_cap=min(200000, self.N))
            self.normalization_stats = stats
            print("✅ Computed enhanced normalization statistics with separate pose components (sampled)")
        else:
            self.normalization_stats = normalization_stats
            print("✅ Applied provided normalization statistics")

    def _init_embeddings_backend(self, embeddings_path: str):
        """Support single .npy, shards index, or auto-detected chunk files."""
        ep = os.path.abspath(embeddings_path)
        base_dir = os.path.dirname(ep)
        base_name = os.path.basename(ep)
        base_stem = base_name[:-4] if base_name.endswith('.npy') else base_name

        shards_txt = os.path.join(base_dir, base_stem + "_shards.txt")

        def _open_single(path: str):
            print(f"you are here with path: {path}")
            
            arr = np.load(path, mmap_mode='r')
            self.emb_is_sharded = False
            self.embeddings = arr
            
            self.emb_total = arr.shape[0]
            print(self.embeddings)
            print()
            print()
            print(f"Loaded embeddings (single): {arr.shape}, dtype: {arr.dtype}")

        def _prepare_shards(paths: list):
            paths = sorted(paths)
            self.emb_is_sharded = True
            self.emb_shard_paths = paths
            # Read shapes without holding open fds
            self.emb_shard_sizes = []
            for p in paths:
                a = np.load(p, mmap_mode='r')
                self.emb_shard_sizes.append(a.shape[0])
                del a
            # cumulative starts
            starts = []
            acc = 0
            for sz in self.emb_shard_sizes:
                starts.append(acc)
                acc += sz
            self._shard_starts = starts
            self.emb_total = acc
            self._shard_cache = {}
            self._shard_lru = []
            self._shard_cache_max = 64
            print(f"Loaded embeddings (sharded): {len(paths)} shards, total rows={self.emb_total}")

        # First: if a directory is provided, treat it as a shard folder
        if os.path.isdir(ep):
            shard_paths = sorted(glob.glob(os.path.join(ep, "*_chunk_*.npy")))
            if not shard_paths:
                shard_paths = sorted(glob.glob(os.path.join(ep, "*.npy")))
            if not shard_paths:
                raise FileNotFoundError(f"No .npy shards found in directory: {ep}")
            _prepare_shards(shard_paths)
            return

        # Priority: exact file (if valid), shards index, auto-detect chunks
        if os.path.isfile(ep) and ep.endswith('.npy'):
            try:
                print(f"Loading embeddings (single): {ep}")
                _open_single(ep)
                return
            except Exception:
                # Not a valid single-array .npy → fall through to shards handling
                pass
        elif os.path.isfile(shards_txt):
            with open(shards_txt, 'r') as f:
                shard_paths = [line.strip() for line in f if line.strip()]
            if not shard_paths:
                raise ValueError(f"Shard index is empty: {shards_txt}")
            _prepare_shards(shard_paths)
        else:
            # Auto-detect chunk files next to target name
            pattern = os.path.join(base_dir, f"{base_stem}_chunk_*.npy")
            shard_paths = sorted(glob.glob(pattern))
            if not shard_paths:
                # Fallback: any npy files in the directory (excluding the handle itself)
                shard_paths = sorted(
                    p for p in glob.glob(os.path.join(base_dir, '*.npy'))
                    if os.path.abspath(p) != os.path.abspath(ep)
                )
            if not shard_paths:
                raise FileNotFoundError(f"Embeddings not found: {ep}. Tried shards index {shards_txt}, pattern {pattern}, and *.npy fallback")
            _prepare_shards(shard_paths)


    def _get_embedding_row(self, idx: int):
        # Re-initialize backend lazily if worker lost attributes
        if not getattr(self, 'emb_is_sharded', False) and not hasattr(self, 'embeddings'):
            self._init_embeddings_backend(self._emb_path)
        if getattr(self, 'emb_total', None) is not None and idx >= self.emb_total:
            raise IndexError(
                f"Embedding index {idx} out of range for total {self.emb_total}. "
                f"This indicates incomplete or mismatched embeddings."
            )
        if not getattr(self, 'emb_is_sharded', False):
            return torch.from_numpy(self.embeddings[idx])
        # find shard via binary search
        import bisect
        si = bisect.bisect_right(self._shard_starts, idx) - 1
        if si < 0:
            si = 0
        local_idx = idx - self._shard_starts[si]
        path = self.emb_shard_paths[si]
        mm = self._shard_cache.get(si)
        if mm is None:
            # open and cache with LRU eviction
            mm = np.load(path, mmap_mode='r')
            self._shard_cache[si] = mm
            self._shard_lru.append(si)
            if len(self._shard_lru) > self._shard_cache_max:
                old = self._shard_lru.pop(0)
                try:
                    del self._shard_cache[old]
                except Exception:
                    pass
        return torch.from_numpy(mm[local_idx])

    def _setup_storage(self, json_path: str, cache_dir: str = None):
        """Ensure memmaps exist for the large JSON; build them if missing or stale.

        Rebuilds the cache if the source JSON path, size, or mtime differs from what was recorded.
        """
        json_path = os.path.abspath(json_path)
        base = os.path.splitext(os.path.basename(json_path))[0]
        # Allow overriding memmap cache location to a fast SSD
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(json_path), f".{base}_mm")
        os.makedirs(cache_dir, exist_ok=True)
        meta_path = os.path.join(cache_dir, "meta.json")

        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self._meta = json.load(f)
            # quick check files exist
            needed = ['dims_file','grasp_file','init_file','final_file','label_file','num_items']
            files_ok = all(k in self._meta for k in needed) and all(os.path.exists(self._meta[k]) for k in needed if k.endswith('_file'))
            # detect if source changed
            current_size = os.path.getsize(json_path) if os.path.exists(json_path) else None
            current_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else None
            recorded_path = self._meta.get('json_path')
            recorded_size = self._meta.get('json_size')
            recorded_mtime = self._meta.get('json_mtime')
            source_unchanged = (recorded_path == json_path and recorded_size == current_size and recorded_mtime == current_mtime)
            if files_ok and source_unchanged:
                return
            # Otherwise, rebuild cache
            try:
                for k in ['dims_file','grasp_file','init_file','final_file','label_file']:
                    p = self._meta.get(k)
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                os.remove(meta_path)
            except Exception:
                pass

        # Build memmaps via streaming
        try:
            import ijson  # type: ignore
        except Exception as e:
            raise RuntimeError("ijson is required to stream large JSON files. Please install it: pip install ijson") from e

        print(f"Indexing large JSON to memmaps: {json_path}")
        # First pass: count
        count = 0
        with open(json_path, 'rb') as f:
            for _ in ijson.items(f, 'item'):
                count += 1
        print(f"  Found {count} items")

        # Create memmaps
        dims_file  = os.path.join(cache_dir, 'dims.f32.npy')
        grasp_file = os.path.join(cache_dir, 'grasp.f32.npy')
        init_file  = os.path.join(cache_dir, 'init.f32.npy')
        final_file = os.path.join(cache_dir, 'final.f32.npy')
        label_file = os.path.join(cache_dir, 'label.f32.npy')

        mm_dims  = np.memmap(dims_file,  dtype=np.float32, mode='w+', shape=(count, 3))
        mm_grasp = np.memmap(grasp_file, dtype=np.float32, mode='w+', shape=(count, 7))
        mm_init  = np.memmap(init_file,  dtype=np.float32, mode='w+', shape=(count, 7))
        mm_final = np.memmap(final_file, dtype=np.float32, mode='w+', shape=(count, 7))
        mm_label = np.memmap(label_file, dtype=np.float32, mode='w+', shape=(count, 2))

        # Second pass: fill
        i = 0
        with open(json_path, 'rb') as f:
            for obj in tqdm(ijson.items(f, 'item'), total=count, desc='Indexing to memmaps'):
                dx, dy, dz = obj['object_dimensions']
                mm_dims[i] = (dx, dy, dz)
                mm_grasp[i] = np.asarray(obj['grasp_pose'], dtype=np.float32)
                mm_init[i]  = np.asarray(obj['initial_object_pose'], dtype=np.float32)
                mm_final[i] = np.asarray(obj['final_object_pose'], dtype=np.float32)
                mm_label[i, 0] = float(obj['success_label'])
                mm_label[i, 1] = float(obj['collision_label'])
                i += 1
        # Flush
        del mm_dims, mm_grasp, mm_init, mm_final, mm_label

        self._meta = {
            'num_items': count,
            'dims_file': dims_file,
            'grasp_file': grasp_file,
            'init_file': init_file,
            'final_file': final_file,
            'label_file': label_file,
            'json_path': json_path,
            'json_size': os.path.getsize(json_path) if os.path.exists(json_path) else None,
            'json_mtime': os.path.getmtime(json_path) if os.path.exists(json_path) else None,
        }
        with open(meta_path, 'w') as f:
            json.dump(self._meta, f)
        print("  Memmaps built and indexed")

    def _compute_stats_streamed(self, sample_cap: int = 200000):
        """Compute normalization stats from a random subset (seeded) to avoid bias and limit IO.

        - Select up to sample_cap unique indices uniformly at random (seeded for reproducibility)
        - Sort indices to improve memmap read locality
        """
        N = self.N
        take = int(min(sample_cap, N))
        rng = np.random.RandomState(42)
        idxs = rng.choice(N, size=take, replace=False).astype(np.int64)
        idxs.sort()

        # Positions across grasp, init, final
        gp = self.mm_grasp[idxs, :3]
        ip = self.mm_init[idxs, :3]
        fp = self.mm_final[idxs, :3]
        all_pos = np.concatenate([gp, ip, fp], axis=0)
        pos_mean = torch.from_numpy(all_pos.mean(axis=0, keepdims=True).astype(np.float32))
        pos_std  = torch.from_numpy(all_pos.std(axis=0, keepdims=True).astype(np.float32)) + 1e-8

        # Orientations
        go = torch.from_numpy(self.mm_grasp[idxs, 3:].astype(np.float32))
        io = torch.from_numpy(self.mm_init[idxs, 3:].astype(np.float32))
        fo = torch.from_numpy(self.mm_final[idxs, 3:].astype(np.float32))
        grasp_ori_norm, grasp_ori_mean, grasp_ori_std = _zscore_normalize(go)
        init_ori_norm, init_ori_mean, init_ori_std   = _zscore_normalize(io)
        final_ori_norm, final_ori_mean, final_ori_std = _zscore_normalize(fo)

        # Corners stats: compute on sampled subset
        acc_sum = torch.zeros((1, 24), dtype=torch.float32)
        acc_sq  = torch.zeros((1, 24), dtype=torch.float32)
        for i in idxs:
            dx, dy, dz = self.mm_dims[i]
            init_pose = self.mm_init[i]
            corners_local = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)
            corners_world = transform_points_to_world(corners_local, np.array(init_pose))
            cf = torch.from_numpy(corners_world.reshape(1, -1))
            acc_sum += cf
            acc_sq  += cf * cf
        m = float(len(idxs))
        corners_mean = acc_sum / m
        corners_std  = torch.sqrt(torch.clamp(acc_sq / m - corners_mean * corners_mean, min=1e-12)) + 1e-8

        return {
                'corners_mean': corners_mean, 'corners_std': corners_std,
                'pos_mean': pos_mean, 'pos_std': pos_std,
                'grasp_ori_mean': grasp_ori_mean, 'grasp_ori_std': grasp_ori_std,
                'init_ori_mean': init_ori_mean, 'init_ori_std': init_ori_std,
                'final_ori_mean': final_ori_mean, 'final_ori_std': final_ori_std
            }

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Embedding stays float16 until cast on GPU
        embedding = self._get_embedding_row(idx)

        # Load memmapped rows
        dims  = self.mm_dims[idx]
        grasp = torch.from_numpy(self.mm_grasp[idx]).to(torch.float32)
        init  = torch.from_numpy(self.mm_init[idx]).to(torch.float32)
        final = torch.from_numpy(self.mm_final[idx]).to(torch.float32)
        label = torch.from_numpy(self.mm_label[idx]).to(torch.float32)

        # Compute corners on-the-fly (cheap for batch sizes)
        cx, cy, cz = float(dims[0]), float(dims[1]), float(dims[2])
        corners_local = cuboid_corners_local_ordered(cx, cy, cz).astype(np.float32)
        corners_world = transform_points_to_world(corners_local, init.numpy()).astype(np.float32)
        corners = torch.from_numpy(corners_world.reshape(1, -1)).to(torch.float32)

        # Apply normalization
        ns = self.normalization_stats
        # Corners
        corners = (corners - ns['corners_mean']) / ns['corners_std']
        corners = corners.reshape(8, 3)
        # Positions (unified)
        pos_mean = ns['pos_mean']; pos_std = ns['pos_std']
        grasp_pos = (grasp[:3] - pos_mean.squeeze(0)) / pos_std.squeeze(0)
        init_pos  = (init[:3]  - pos_mean.squeeze(0)) / pos_std.squeeze(0)
        final_pos = (final[:3] - pos_mean.squeeze(0)) / pos_std.squeeze(0)
        # Orientations (z-score per field)
        grasp_ori = _zscore_normalize(grasp[3:].unsqueeze(0), ns['grasp_ori_mean'], ns['grasp_ori_std'])[0].squeeze(0)
        init_ori  = _zscore_normalize(init[3:].unsqueeze(0),  ns['init_ori_mean'],  ns['init_ori_std'])[0].squeeze(0)
        final_ori = _zscore_normalize(final[3:].unsqueeze(0), ns['final_ori_mean'], ns['final_ori_std'])[0].squeeze(0)

        grasp_n = torch.cat([grasp_pos, grasp_ori], dim=0)
        init_n  = torch.cat([init_pos,  init_ori ], dim=0)
        final_n = torch.cat([final_pos, final_ori], dim=0)
        
        return (
            corners,
            embedding,
            grasp_n,
            init_n,
            final_n,
            label
        )

    @property
    def label(self):
        """Expose labels as a torch Tensor for code expecting train_dataset.label.
        Backed by the memmap; zero-copy wrapper.
        """
        return torch.from_numpy(self.mm_label)



class WorldFrameCornersOnlyDataset(Dataset):
    def __init__(self, data_path, normalization_stats=None, is_training=True, cache_dir: str = None):
        """
        Corners-only, memory-efficient dataset:
        - Builds on-disk memmaps from a huge JSON using streaming
        - No embeddings; returns only normalized corners and poses
        """
        self.data_path = data_path
        self._setup_storage(data_path, cache_dir)
        self.N = int(self._meta['num_items'])

        # Load memmaps for fast row access
        self.mm_dims  = np.memmap(self._meta['dims_file'],  dtype=np.float32, mode='r', shape=(self.N, 3))
        self.mm_grasp = np.memmap(self._meta['grasp_file'], dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_init  = np.memmap(self._meta['init_file'],  dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_final = np.memmap(self._meta['final_file'], dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_label = np.memmap(self._meta['label_file'], dtype=np.float32, mode='r', shape=(self.N, 2))

        # Compute or apply normalization statistics
        if is_training or normalization_stats is None:
            print("Computing normalization statistics (corners-only dataset)...")
            stats = self._compute_stats_streamed(sample_cap=min(200000, self.N))
            self.normalization_stats = stats
            print("✅ Computed normalization statistics (corners + poses)")
        else:
            self.normalization_stats = normalization_stats
            print("✅ Applied provided normalization statistics")

    def _setup_storage(self, json_path: str, cache_dir: str = None):
        # Reuse implementation from WorldFrameDataset
        return WorldFrameDataset._setup_storage(self, json_path, cache_dir)

    def _compute_stats_streamed(self, sample_cap: int = 200000):
        """Compute normalization stats:
        - corners: init-local (flattened 1×24)
        - positions/orientations: world (same as WorldFrameDataset)
        """
        N = self.N
        take = int(min(sample_cap, N))
        rng = np.random.RandomState(42)
        idxs = rng.choice(N, size=take, replace=False).astype(np.int64)
        idxs.sort()

        # Positions across grasp, init, final (world)
        gp = self.mm_grasp[idxs, :3]
        ip = self.mm_init[idxs, :3]
        fp = self.mm_final[idxs, :3]
        all_pos = np.concatenate([gp, ip, fp], axis=0)
        pos_mean = torch.from_numpy(all_pos.mean(axis=0, keepdims=True).astype(np.float32))
        pos_std  = torch.from_numpy(all_pos.std(axis=0, keepdims=True).astype(np.float32)) + 1e-8

        # Orientations (world) — z-score per component
        go = torch.from_numpy(self.mm_grasp[idxs, 3:].astype(np.float32))
        io = torch.from_numpy(self.mm_init[idxs, 3:].astype(np.float32))
        fo = torch.from_numpy(self.mm_final[idxs, 3:].astype(np.float32))
        _, grasp_ori_mean, grasp_ori_std = _zscore_normalize(go)
        _, init_ori_mean,  init_ori_std  = _zscore_normalize(io)
        _, final_ori_mean, final_ori_std = _zscore_normalize(fo)

        # Corners (init-local), accumulate mean/std over flattened (1×24)
        acc_sum = torch.zeros((1, 24), dtype=torch.float32)
        acc_sq  = torch.zeros((1, 24), dtype=torch.float32)
        for i in idxs:
            dx, dy, dz = self.mm_dims[i]
            corners_local = cuboid_corners_local_ordered(float(dx), float(dy), float(dz)).astype(np.float32)
            cf = torch.from_numpy(corners_local.reshape(1, -1))  # (1,24)
            acc_sum += cf
            acc_sq  += cf * cf
        m = float(len(idxs))
        corners_mean = acc_sum / m
        corners_std  = torch.sqrt(torch.clamp(acc_sq / m - corners_mean * corners_mean, min=1e-12)) + 1e-8

        return {
            'corners_mean': corners_mean, 'corners_std': corners_std,
            'pos_mean': pos_mean, 'pos_std': pos_std,
            'grasp_ori_mean': grasp_ori_mean, 'grasp_ori_std': grasp_ori_std,
            'init_ori_mean': init_ori_mean,   'init_ori_std': init_ori_std,
            'final_ori_mean': final_ori_mean, 'final_ori_std': final_ori_std,
        }

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Load memmapped rows
        dims  = self.mm_dims[idx]
        grasp = torch.from_numpy(self.mm_grasp[idx]).to(torch.float32)
        init  = torch.from_numpy(self.mm_init[idx]).to(torch.float32)
        final = torch.from_numpy(self.mm_final[idx]).to(torch.float32)
        label = torch.from_numpy(self.mm_label[idx]).to(torch.float32)

        # Compute corners on-the-fly for BOTH poses and concatenate (init || final)
        cx, cy, cz = float(dims[0]), float(dims[1]), float(dims[2])
        corners_local = cuboid_corners_local_ordered(cx, cy, cz).astype(np.float32)
        corners = torch.from_numpy(corners_local).to(torch.float32)

        # Apply normalization (reuse stats keys from WorldFrameDataset)
        ns = self.normalization_stats
        # Z-score corners as a single 8×3 set using (1×24) stats
        c_mean = ns['corners_mean']  # shape (1,24)
        c_std  = ns['corners_std']   # shape (1,24)
        corners = corners.reshape(1, 24)
        corners = (corners - c_mean) / (c_std + 1e-8)
        corners = corners.reshape(8, 3)

        # Positions (unified)
        pos_mean = ns['pos_mean']; pos_std = ns['pos_std']
        grasp_pos = (grasp[:3] - pos_mean.squeeze(0)) / pos_std.squeeze(0)
        init_pos  = (init[:3]  - pos_mean.squeeze(0)) / pos_std.squeeze(0)
        final_pos = (final[:3] - pos_mean.squeeze(0)) / pos_std.squeeze(0)
        # Orientations (z-score per field)
        grasp_ori = _zscore_normalize(grasp[3:].unsqueeze(0), ns['grasp_ori_mean'], ns['grasp_ori_std'])[0].squeeze(0)
        init_ori  = _zscore_normalize(init[3:].unsqueeze(0),  ns['init_ori_mean'],  ns['init_ori_std'])[0].squeeze(0)
        final_ori = _zscore_normalize(final[3:].unsqueeze(0), ns['final_ori_mean'], ns['final_ori_std'])[0].squeeze(0)

        grasp_n = torch.cat([grasp_pos, grasp_ori], dim=0)
        init_n  = torch.cat([init_pos,  init_ori ], dim=0)
        final_n = torch.cat([final_pos, final_ori], dim=0)

        # Return: corners (init||final), poses, label
        return (
            corners,  # two faces of 8 corners each flattened then re-shaped is not semantic; keep as (16,3)
            grasp_n,
            init_n,
            final_n,
            label,
        )

    @property
    def label(self):
        return torch.from_numpy(self.mm_label)



# --- ADD THIS SMALL CLASS TO dataset.py ---------------------------------------
class FinalCornersHandDataset(Dataset):
    """
    Minimal dataset for Option-B:
      X_corners: final pose corners in WORLD frame, flattened (24) and z-scored
      X_hand:    [ t_loc (3, z-scored) , R_loc 6D (6, raw) ]  => 9 dims
      y:         collision label at placement (float scalar)

    Shapes returned:
      corners_24   : torch.float32, (24,)
      hand_9       : torch.float32, (9,)
      label        : torch.float32, ()  # scalar 0/1

    Normalization keys it saves/loads:
            'final_corners_mean', 'final_corners_std',
                'tloc_mean', 'tloc_std',
                'dims_mean', 'dims_std'
    """
    def __init__(self, data_path, normalization_stats=None, is_training=True, cache_dir: str = None,
             stats_sample_cap: int = 200_000,
             # NEW: PHM config
             include_phm: bool = True,
             hand_down_axis: str = "z",
             hand_down_sign: int = -1,
             h_palm_down: float = 0.018,
             z_ped: float = 0.1):
        self.data_path = data_path
        # Reuse your existing memmap cache builder
        WorldFrameDataset._setup_storage(self, data_path, cache_dir)
        self.N = int(self._meta['num_items'])

        # Memmaps (same as your other datasets)
        self.mm_dims  = np.memmap(self._meta['dims_file'],  dtype=np.float32, mode='r', shape=(self.N, 3))
        self.mm_grasp = np.memmap(self._meta['grasp_file'], dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_init  = np.memmap(self._meta['init_file'],  dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_final = np.memmap(self._meta['final_file'], dtype=np.float32, mode='r', shape=(self.N, 7))
        self.mm_label = np.memmap(self._meta['label_file'], dtype=np.float32, mode='r', shape=(self.N, 2))

        # Stats: compute once (training) or use provided (evaluation)
        if is_training:
            self.normalization_stats = self._compute_stats(stats_sample_cap)
            print("✅ FinalCornersHandDataset: computed stats for final corners + t_loc")
        else:
            assert normalization_stats is not None, (
                "Pass train normalization_stats to val/test to avoid drift."
            )
            self.normalization_stats = normalization_stats
            print("✅ FinalCornersHandDataset: using provided stats")

        # NEW: store PHM config
        self.include_phm = bool(include_phm)
        self._hd_axis_idx = {"x":0, "y":1, "z":2}[hand_down_axis.lower()]
        self._hd_sign = float(hand_down_sign)  # +1 or -1
        self._h_palm_down = float(h_palm_down)
        self._z_ped = float(z_ped)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # --- load rows (numpy -> torch) ---------------------------------------
        dims  = self.mm_dims[idx]            # (3,)
        grasp = self.mm_grasp[idx]           # (7,) [tx,ty,tz,qw,qx,qy,qz]
        init  = self.mm_init[idx]            # (7,)
        final = self.mm_final[idx]           # (7,)
        y     = self.mm_label[idx, 1]        # collision label at placement

        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])

        # --- final corners in WORLD frame (24) --------------------------------
        corners_local = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)  # (8,3)
        final_corners_world = transform_points_to_world(corners_local, final).reshape(-1)  # (24,)
        # z-score with dataset stats
        ns = self.normalization_stats
        final_corners_24 = (torch.from_numpy(final_corners_world) - ns['final_corners_mean']) / (ns['final_corners_std'] + 1e-8)

        # --- tiny hand-relative block wrt INIT pose ---------------------------
        t_h = np.asarray(grasp[:3], np.float32)
        t_o = np.asarray(init[:3],  np.float32)

        # scipy uses [x,y,z,w]
        R_h = R.from_quat([grasp[4], grasp[5], grasp[6], grasp[3]]).as_matrix().astype(np.float32)
        R_o = R.from_quat([init[4],  init[5],  init[6],  init[3]]).as_matrix().astype(np.float32)

        R_loc = (R_o.T @ R_h).astype(np.float32)            # (3,3)
        t_loc = (R_o.T @ (t_h - t_o)).astype(np.float32)    # (3,)

        # 6D rotation representation = first two columns of R_loc
        r6 = R_loc[:, :2].reshape(-1)                       # (6,)

        # z-score t_loc, leave r6 raw (already bounded)
        t_loc_z = (torch.from_numpy(t_loc) - ns['tloc_mean']) / (ns['tloc_std'] + 1e-8)
        r6_t    = torch.from_numpy(r6)

        hand_9  = torch.cat([t_loc_z.to(torch.float32), r6_t.to(torch.float32)], dim=0)  # (9,)

        # dims z-score (if stats exist), then append → aux (,)
        ns = self.normalization_stats
        d_mean = ns.get('dims_mean', None)
        d_std  = ns.get('dims_std',  None)
        dims_f32 = torch.from_numpy(dims.astype(np.float32))
        if d_mean is not None and d_std is not None:
            dims_z = (dims_f32 - d_mean) / (d_std + 1e-8)
        else:
            dims_z = dims_f32

        aux = torch.cat([hand_9, dims_z.to(torch.float32)], dim=0)  # (12,)

        # --- NEW: PHM (Palm Height Margin) at place ---
        if self.include_phm:
            # Final orientation (Isaac [qw,qx,qy,qz] -> SciPy [x,y,z,w])
            R_f = R.from_quat([final[4], final[5], final[6], final[3]]).as_matrix().astype(np.float32)
            # Hand at place
            R_place = (R_f @ R_loc).astype(np.float32)
            t_place = (final[:3] + (R_f @ t_loc)).astype(np.float32)  # palm/wrist origin proxy
            # Hand "down" in world
            e = np.zeros(3, dtype=np.float32); e[self._hd_axis_idx] = self._hd_sign
            a_world = R_place @ e
            a_world = a_world / (np.linalg.norm(a_world) + 1e-9)
            # Palm bottom point; PHM = z(bottom) - z_ped
            palm_bottom_z = float(t_place[2] + a_world[2] * self._h_palm_down)
            phm = palm_bottom_z - self._z_ped
            aux = torch.cat([aux, torch.tensor([phm], dtype=torch.float32)], dim=0)  # (13,)

        return final_corners_24.to(torch.float32), aux, torch.tensor(float(y), dtype=torch.float32)

    # ---- helpers --------------------------------------------------------------
    def _compute_stats(self, sample_cap: int = 200_000):
        N = self.N
        take = int(min(sample_cap, N))
        rng = np.random.RandomState(42)
        idxs = rng.choice(N, size=take, replace=False).astype(np.int64)
        idxs.sort()

        # Accumulate mean/std for final-corners (24) and t_loc (3)
        sum_c = np.zeros(24, dtype=np.float64)
        sq_c  = np.zeros(24, dtype=np.float64)
        sum_t = np.zeros(3,  dtype=np.float64)
        sq_t  = np.zeros(3,  dtype=np.float64)
        sum_d = np.zeros(3, dtype=np.float64)
        sq_d  = np.zeros(3, dtype=np.float64)

        for i in idxs:
            dims  = self.mm_dims[i]
            final = self.mm_final[i]
            grasp = self.mm_grasp[i]
            init  = self.mm_init[i]

            dims = self.mm_dims[i]             # (3,)
            sum_d += dims
            sq_d  += dims * dims

            dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])

            # corners(final, world)
            cl = cuboid_corners_local_ordered(dx, dy, dz).astype(np.float32)
            cw = transform_points_to_world(cl, final).reshape(-1)  # (24,)
            sum_c += cw
            sq_c  += cw * cw

            # t_loc wrt init
            t_h = np.asarray(grasp[:3], np.float32)
            t_o = np.asarray(init[:3],  np.float32)
            R_o = R.from_quat([init[4], init[5], init[6], init[3]]).as_matrix().astype(np.float32)
            t_loc = (R_o.T @ (t_h - t_o)).astype(np.float32)  # (3,)
            sum_t += t_loc
            sq_t  += t_loc * t_loc

        m = float(len(idxs))
        c_mean = torch.from_numpy((sum_c / m).astype(np.float32))
        c_std  = torch.from_numpy(np.sqrt(np.maximum(sq_c / m - (sum_c / m)**2, 1e-12)).astype(np.float32))
        t_mean = torch.from_numpy((sum_t / m).astype(np.float32))
        t_std  = torch.from_numpy(np.sqrt(np.maximum(sq_t / m - (sum_t / m)**2, 1e-12)).astype(np.float32))
        d_mean = torch.from_numpy((sum_d / m).astype(np.float32))
        d_std  = torch.from_numpy(np.sqrt(np.maximum(sq_d / m - (sum_d / m)**2, 1e-12)).astype(np.float32))

        return {
            'final_corners_mean': c_mean, 'final_corners_std': c_std,
            'tloc_mean': t_mean, 'tloc_std': t_std,
            'dims_mean': d_mean, 'dims_std': d_std,
        }
# --- END ADDITION -----