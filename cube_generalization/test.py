# add_rfinal6_memmap.py
import os, json, numpy as np

def quat_wxyz_to_r6(qw, qx, qy, qz):
    # vectorized quaternion -> first two columns of rotation matrix
    xx = qx*qx; yy = qy*qy; zz = qz*qz
    xy = qx*qy; xz = qx*qz; yz = qy*qz
    wx = qw*qx; wy = qw*qy; wz = qw*qz

    r00 = 1.0 - 2.0*(yy + zz)
    r10 = 2.0*(xy + wz)
    r20 = 2.0*(xz - wy)

    r01 = 2.0*(xy - wz)
    r11 = 1.0 - 2.0*(xx + zz)
    r21 = 2.0*(yz + wx)

    # stack columns [R[:,0], R[:,1]] -> shape (N,6)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=1).astype(np.float32)

def main(mem_dir):
    meta_path = os.path.join(mem_dir, "meta.json")
    meta = json.load(open(meta_path, "r"))
    N = int(meta["N"])

    final_path = meta["final_file"]      # existing (N,7) memmap: [tx,ty,tz,qw,qx,qy,qz]
    fin = np.memmap(final_path, dtype=np.float32, mode="r", shape=(N,7))

    qw = fin[:,3].astype(np.float32)
    qx = fin[:,4].astype(np.float32)
    qy = fin[:,5].astype(np.float32)
    qz = fin[:,6].astype(np.float32)

    # normalize quats once (cheap, avoids drift)
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n

    r6 = quat_wxyz_to_r6(qw, qx, qy, qz)  # (N,6)

    out_path = os.path.join(mem_dir, "rfinal6.mmap")
    out = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(N,6))
    out[:] = r6
    out.flush()

    meta["rfinal6_file"] = out_path
    json.dump(meta, open(meta_path, "w"))
    print(f"✅ wrote rfinal6 memmap: {out_path}  shape=({N},6) and updated meta.json")

if __name__ == "__main__":
    # hardcode your memmaps folder here:
    MEM_DIR = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps_train"
    main(MEM_DIR)