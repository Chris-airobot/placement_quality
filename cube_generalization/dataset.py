# collision_dataset_v6.py
# - Streams combined_data_v6/all_data.json (no fail-safes).
# - Builds memmaps: corners(24), t_loc(3), R_loc6(6), dims(3), final_pose(7), label(1=collision).
# - Computes & saves train-only stats (corners, t_loc, dims).
# - Dataset returns: corners_24[z], aux_12 = [t_loc_z(3), R_loc6(6 raw), dims_z(3)], label_collision.
# - Includes a small sanity check (shapes, base rate, and a geometric check).

import os, json, math, numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

# ----------------------------- Paths -----------------------------

DATA_ROOT   = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection"
COMBINED    = os.path.join(DATA_ROOT, "combined_data", "train.json")
MEMMAP_DIR  = os.path.join(DATA_ROOT, "memmaps_train")  # you can change this

# --------------------------- Utilities ---------------------------

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def _stream_array(path: str):
    """Fast JSON array streamer using buffered raw_decode.

    Assumes file is a single JSON array of objects: [ {...}, {...}, ... ].
    """
    decoder = json.JSONDecoder()
    buf = ""
    idx = 0
    in_array = False
    # 1–4 MB chunks balance I/O and parser cost
    chunk_size = 4 * 1024 * 1024
    with open(path, "r") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                # process any remaining buffer tail
                # attempt a final decode; then exit
                while True:
                    # skip whitespace/commas
                    while idx < len(buf) and buf[idx] in " \r\n\t,":
                        idx += 1
                    if idx >= len(buf):
                        return
                    if not in_array:
                        # seek opening '['
                        lb = buf.find('[', idx)
                        if lb == -1:
                            return
                        in_array = True
                        idx = lb + 1
                        continue
                    if idx < len(buf) and buf[idx] == ']':
                        return
                    try:
                        obj, end = decoder.raw_decode(buf, idx)
                    except json.JSONDecodeError:
                        return
                    yield obj
                    idx = end
            else:
                buf += chunk
                # keep buffer from growing unbounded; retain tail
                if len(buf) > 8 * chunk_size:
                    # drop consumed prefix
                    buf = buf[idx:]
                    idx = 0
                # try to decode as many objects as possible from buffer
                while True:
                    # skip whitespace/commas
                    while idx < len(buf) and buf[idx] in " \r\n\t,":
                        idx += 1
                    if idx >= len(buf):
                        # need more data
                        break
                    if not in_array:
                        lb = buf.find('[', idx)
                        if lb == -1:
                            # need more data for '['
                            break
                        in_array = True
                        idx = lb + 1
                        continue
                    if buf[idx] == ']':
                        return
                    try:
                        obj, end = decoder.raw_decode(buf, idx)
                    except json.JSONDecodeError:
                        # need more data
                        break
                    yield obj
                    idx = end

def quat_wxyz_to_R_batched(q: np.ndarray) -> np.ndarray:
    # q: (B,4) [w,x,y,z], returns R: (B,3,3)
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w /= n; x /= n; y /= n; z /= n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:,0,0] = 1 - 2*(yy + zz); R[:,0,1] = 2*(xy - wz);   R[:,0,2] = 2*(xz + wy)
    R[:,1,0] = 2*(xy + wz);     R[:,1,1] = 1 - 2*(xx+zz); R[:,1,2] = 2*(yz - wx)
    R[:,2,0] = 2*(xz - wy);     R[:,2,1] = 2*(yz + wx);   R[:,2,2] = 1 - 2*(xx+yy)
    return R

SIGNS_8x3 = np.array([
    [-1,-1,-1],
    [-1,-1, 1],
    [-1, 1,-1],
    [-1, 1, 1],
    [ 1,-1,-1],
    [ 1,-1, 1],
    [ 1, 1,-1],
    [ 1, 1, 1],
], dtype=np.float32)  # consistent vertex ordering

def corners_world_from_dims_final_batch(dims_b: np.ndarray, final_b: np.ndarray) -> np.ndarray:
    """
    dims_b : (B,3) [dx,dy,dz]
    final_b: (B,7) [tx,ty,tz,qw,qx,qy,qz]
    return : (B,24) flattened world corners (x1,y1,z1,...,x8,y8,z8)
    """
    B = dims_b.shape[0]
    T = final_b[:, :3].astype(np.float32)                 # (B,3)
    q = final_b[:, 3:].astype(np.float32)                 # (B,4) wxyz
    R = quat_wxyz_to_R_batched(q.astype(np.float32))      # (B,3,3)

    half = 0.5 * dims_b.astype(np.float32)                # (B,3)
    # C_obj[b,8,3] = SIGNS_8x3 * half[b]
    C_obj = SIGNS_8x3[None, :, :] * half[:, None, :]      # (B,8,3)

    # C_w = C_obj @ R^T + T
    C_w = np.einsum("bij,bkj->bki", R, C_obj)             # (B,8,3)  (R * C_obj^T)^T
    C_w = C_w + T[:, None, :]                             # (B,8,3)
    return C_w.reshape(B, 24)                             # (B,24)

# ----------------------- Memmap builder & meta -----------------------

def _count_total(in_path: str) -> int:
    c = 0
    for _ in tqdm(_stream_array(in_path), desc=f"Counting {os.path.basename(in_path)}", unit="obj", dynamic_ncols=True, smoothing=0.1):
        c += 1
    return c

def build_memmaps(in_path: str = COMBINED, out_dir: str = MEMMAP_DIR, batch: int = 200_000):
    ensure_dir(out_dir)
    N = _count_total(in_path)

    # Preallocate memmaps
    mm_dims   = np.memmap(os.path.join(out_dir, "dims3.mm"),   dtype=np.float32, mode="w+", shape=(N,3))
    mm_tloc   = np.memmap(os.path.join(out_dir, "tloc3.mm"),   dtype=np.float32, mode="w+", shape=(N,3))
    mm_rloc6  = np.memmap(os.path.join(out_dir, "rloc6.mm"),   dtype=np.float32, mode="w+", shape=(N,6))
    mm_final7 = np.memmap(os.path.join(out_dir, "final7.mm"),  dtype=np.float32, mode="w+", shape=(N,7))
    mm_c24    = np.memmap(os.path.join(out_dir, "corners24.mm"), dtype=np.float32, mode="w+", shape=(N,24))
    mm_y      = np.memmap(os.path.join(out_dir, "label1.mm"), dtype=np.float32, mode="w+", shape=(N,1))

    # Second pass: fill
    w = 0
    buf_dims, buf_final, buf_tloc, buf_rloc6, buf_y = [], [], [], [], []
    pbar = tqdm(total=N, desc="Building memmaps", unit="obj", dynamic_ncols=True, smoothing=0.1)

    for obj in _stream_array(in_path):
        buf_dims.append(obj["object_dimensions"])
        buf_final.append(obj["final_object_pose"])
        buf_tloc.append(obj["t_loc"])
        buf_rloc6.append(obj["R_loc6"])
        # training target = collision only
        y = 1.0 if float(obj["collision_label"]) > 0.5 else 0.0
        buf_y.append([y])
        pbar.update(1)

        if len(buf_dims) == batch:
            dims_b  = np.asarray(buf_dims,  dtype=np.float32)
            final_b = np.asarray(buf_final, dtype=np.float32)
            tloc_b  = np.asarray(buf_tloc,  dtype=np.float32)
            rloc6_b = np.asarray(buf_rloc6, dtype=np.float32)
            y_b     = np.asarray(buf_y,     dtype=np.float32)

            c24_b = corners_world_from_dims_final_batch(dims_b, final_b)

            mm_dims[w:w+batch]   = dims_b
            mm_final7[w:w+batch] = final_b
            mm_tloc[w:w+batch]   = tloc_b
            mm_rloc6[w:w+batch]  = rloc6_b
            mm_c24[w:w+batch]    = c24_b
            mm_y[w:w+batch]      = y_b
            w += batch
            buf_dims.clear(); buf_final.clear(); buf_tloc.clear(); buf_rloc6.clear(); buf_y.clear()

    # tail
    if len(buf_dims) > 0:
        dims_b  = np.asarray(buf_dims,  dtype=np.float32)
        final_b = np.asarray(buf_final, dtype=np.float32)
        tloc_b  = np.asarray(buf_tloc,  dtype=np.float32)
        rloc6_b = np.asarray(buf_rloc6, dtype=np.float32)
        y_b     = np.asarray(buf_y,     dtype=np.float32)
        c24_b   = corners_world_from_dims_final_batch(dims_b, final_b)
        mm_dims[w:w+len(dims_b)]   = dims_b
        mm_final7[w:w+len(dims_b)] = final_b
        mm_tloc[w:w+len(dims_b)]   = tloc_b
        mm_rloc6[w:w+len(dims_b)]  = rloc6_b
        mm_c24[w:w+len(dims_b)]    = c24_b
        mm_y[w:w+len(dims_b)]      = y_b
    pbar.close()

    # flush
    mm_dims.flush(); mm_final7.flush(); mm_tloc.flush(); mm_rloc6.flush(); mm_c24.flush(); mm_y.flush()

    # meta
    meta = {
        "N": N,
        "dims_file":   os.path.join(out_dir, "dims3.mm"),
        "tloc_file":   os.path.join(out_dir, "tloc3.mm"),
        "rloc6_file":  os.path.join(out_dir, "rloc6.mm"),
        "final7_file": os.path.join(out_dir, "final7.mm"),
        "corners24_file": os.path.join(out_dir, "corners24.mm"),
        "label_file":  os.path.join(out_dir, "label1.mm"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[memmap] wrote {N} rows to {out_dir}")

# ----------------------------- Stats -----------------------------

def compute_train_stats(mem_dir: str = MEMMAP_DIR, sample_cap: int = 200_000) -> Dict[str, List[float]]:
    meta = json.load(open(os.path.join(mem_dir, "meta.json"), "r"))
    N = int(meta["N"])
    take = min(sample_cap, N)
    rng = np.random.RandomState(42)
    idxs = rng.choice(N, size=take, replace=False)

    mm_dims  = np.memmap(meta["dims_file"],    dtype=np.float32, mode="r", shape=(N,3))
    mm_tloc  = np.memmap(meta["tloc_file"],    dtype=np.float32, mode="r", shape=(N,3))
    mm_c24   = np.memmap(meta["corners24_file"], dtype=np.float32, mode="r", shape=(N,24))

    d = mm_dims[idxs]   # (take,3)
    t = mm_tloc[idxs]   # (take,3)
    c = mm_c24[idxs]    # (take,24)

    stats = {
        "corners_mean": c.mean(axis=0).astype(np.float32).tolist(),
        "corners_std":  (c.std(axis=0) + 1e-12).astype(np.float32).tolist(),
        "tloc_mean":    t.mean(axis=0).astype(np.float32).tolist(),
        "tloc_std":     (t.std(axis=0) + 1e-12).astype(np.float32).tolist(),
        "dims_mean":    d.mean(axis=0).astype(np.float32).tolist(),
        "dims_std":     (d.std(axis=0) + 1e-12).astype(np.float32).tolist(),
    }
    with open(os.path.join(mem_dir, "stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"[stats] saved train stats to {mem_dir}/stats.json (take={take})")
    return stats

def _quat_wxyz_to_r6(qw: float, qx: float, qy: float, qz: float) -> torch.Tensor:
    """Return 6D rotation (first two columns) from a unit quaternion (w,x,y,z)."""
    # normalize
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw / (n + 1e-12), qx / (n + 1e-12), qy / (n + 1e-12), qz / (n + 1e-12)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    # rotation matrix columns (R[:,0], R[:,1])
    c0 = (1 - 2*(yy+zz),     2*(xy+wz),     2*(xz-wy))
    c1 = (    2*(xy-wz), 1 - 2*(xx+zz),     2*(yz+wx))
    return torch.tensor([c0[0], c0[1], c0[2], c1[0], c1[1], c1[2]], dtype=torch.float32)

# ----------------------------- Dataset -----------------------------

class FinalCornersHandDataset(Dataset):
    """
    Returns
      corners_24 : (24,) float32, z-scored with train stats
      aux_18     : (18,) float32 = [ t_loc_z(3), R_loc6(6 raw), dims_z(3), R_final6(6 raw) ]
      label      : ()    float32  = collision (1 if collision, else 0)
    """
    def __init__(self,
                 mem_dir: str,
                 normalization_stats: Optional[Dict[str, List[float]]] = None,
                 is_training: bool = True):
        meta = json.load(open(os.path.join(mem_dir, "meta.json"), "r"))
        self.N = int(meta["N"])
        self.mm_dims   = np.memmap(meta["dims_file"],    dtype=np.float32, mode="r", shape=(self.N,3))
        self.mm_tloc   = np.memmap(meta["tloc_file"],    dtype=np.float32, mode="r", shape=(self.N,3))
        self.mm_rloc6  = np.memmap(meta["rloc6_file"],   dtype=np.float32, mode="r", shape=(self.N,6))
        self.mm_c24    = np.memmap(meta["corners24_file"], dtype=np.float32, mode="r", shape=(self.N,24))
        self.mm_final  = np.memmap(meta["final7_file"],     dtype=np.float32, mode="r", shape=(self.N,7))  # <-- use this
        self.mm_y      = np.memmap(meta["label_file"],   dtype=np.float32, mode="r", shape=(self.N,1))

        if normalization_stats is None:
            if is_training:
                normalization_stats = json.load(open(os.path.join(mem_dir, "stats.json"), "r"))
            else:
                raise RuntimeError("Provide train normalization_stats for eval.")
        self.ns = {
            "c_mu": torch.tensor(normalization_stats["corners_mean"], dtype=torch.float32),
            "c_sd": torch.tensor(normalization_stats["corners_std"],  dtype=torch.float32),
            "t_mu": torch.tensor(normalization_stats["tloc_mean"],    dtype=torch.float32),
            "t_sd": torch.tensor(normalization_stats["tloc_std"],     dtype=torch.float32),
            "d_mu": torch.tensor(normalization_stats["dims_mean"],    dtype=torch.float32),
            "d_sd": torch.tensor(normalization_stats["dims_std"],     dtype=torch.float32),
        }

    def __len__(self): return self.N

    def __getitem__(self, i: int):
        c24 = torch.from_numpy(self.mm_c24[i].copy())  # (24,)
        t   = torch.from_numpy(self.mm_tloc[i].copy()) # (3,)
        r6  = torch.from_numpy(self.mm_rloc6[i].copy())# (6,)
        d3  = torch.from_numpy(self.mm_dims[i].copy()) # (3,)
        f7  = self.mm_final[i]                            # (7,) [tx,ty,tz,qw,qx,qy,qz]
        y   = torch.tensor(float(self.mm_y[i,0]), dtype=torch.float32)

        c24 = (c24 - self.ns["c_mu"]) / (self.ns["c_sd"] + 1e-8)
        t_z = (t   - self.ns["t_mu"]) / (self.ns["t_sd"] + 1e-8)
        d_z = (d3  - self.ns["d_mu"]) / (self.ns["d_sd"] + 1e-8)

        qw, qx, qy, qz = float(f7[3]), float(f7[4]), float(f7[5]), float(f7[6])
        rf6 = _quat_wxyz_to_r6(qw, qx, qy, qz)            # (6,)
        aux18 = torch.cat([t_z, r6, d_z, rf6], dim=0)  # (12,)
        return c24, aux18, y

# -------------------------- Sanity Check --------------------------

def _R_from_q(q: np.ndarray) -> np.ndarray:
    # q: (4,) wxyz
    return quat_wxyz_to_R_batched(q[None,:]).reshape(3,3)

def _object_frame_from_world_corners(cw: np.ndarray, final7: np.ndarray) -> np.ndarray:
    # cw: (8,3), final7: [tx,ty,tz,qw,qx,qy,qz]
    T = final7[:3].astype(np.float32)
    q = final7[3:].astype(np.float32)
    R = _R_from_q(q)                                    # (3,3)
    return (cw - T[None,:]) @ R                         # (8,3)  (since cw = co @ R^T + T)

def quick_sanity(mem_dir: str = MEMMAP_DIR, sample_n: int = 5000):
    meta = json.load(open(os.path.join(mem_dir, "meta.json"), "r"))
    N = int(meta["N"])
    mm_dims  = np.memmap(meta["dims_file"],    dtype=np.float32, mode="r", shape=(N,3))
    mm_c24   = np.memmap(meta["corners24_file"], dtype=np.float32, mode="r", shape=(N,24))
    mm_final = np.memmap(meta["final7_file"],  dtype=np.float32, mode="r", shape=(N,7))
    mm_y     = np.memmap(meta["label_file"],   dtype=np.float32, mode="r", shape=(N,1))

    # Base rate
    take = min(sample_n, N)
    idxs = np.linspace(0, N-1, num=take, dtype=np.int64)
    base = float((mm_y[idxs,0] > 0.5).mean())
    print(f"[sanity] collision base rate over {take}: {base:.3f}")

    # Geometric check on a few
    for j in idxs[:10]:
        dims = mm_dims[j]                  # (3,)
        final = mm_final[j]                # (7,)
        cw = mm_c24[j].reshape(8,3)        # (8,3)
        co = _object_frame_from_world_corners(cw, final)  # (8,3)
        ext = co.max(axis=0) - co.min(axis=0)
        if not np.allclose(ext, dims, atol=1e-4):
            raise AssertionError("Recovered dims from corners mismatch.")
    print("[sanity] corners ↔ dims round-trip OK on 10 samples")

# ------------------------------ CLI ------------------------------

if __name__ == "__main__":
    ensure_dir(MEMMAP_DIR)
    build_memmaps(COMBINED, MEMMAP_DIR, batch=200_000)
    compute_train_stats(MEMMAP_DIR, sample_cap=200_000)
    quick_sanity(MEMMAP_DIR, sample_n=5000)
