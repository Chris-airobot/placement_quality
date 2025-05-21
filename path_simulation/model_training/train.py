import torch
from torch.utils.data import DataLoader
from dataset import KinematicFeasibilityDataset
from model import GraspObjectFeasibilityNet
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pointnet2.pointnet2_utils import farthest_point_sample, index_points
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import PointNetEncoder
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Sampler
torch.backends.cudnn.benchmark = True
import torch

# Enable TF32 tensor‐core acceleration on supported hardware:
torch.set_float32_matmul_precision('high')

class NumpyWeightedSampler(Sampler):
    def __init__(self, weights, num_samples=None, seed=None):
        self.weights = np.array(weights, dtype=np.float64)
        self.probs = self.weights / self.weights.sum()
        self.num_samples = num_samples or len(self.probs)
        self.rs = np.random.RandomState(seed)

    def __iter__(self):
        idx = self.rs.choice(
            len(self.probs),
            size=self.num_samples,
            replace=True,
            p=self.probs
        )
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples
    

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
        # Pretend we are never a TTY, so Dynamo skips color logic
        return False


# Load point cloud
def load_pointcloud(pcd_path, target_points=1024):
    if type(pcd_path) == str:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
    else:
        points = pcd_path
    
    # Convert to torch tensor for FPS
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # [1, N, 3]
    
    # Apply farthest point sampling to downsample
    if len(points) > target_points:
        fps_idx = farthest_point_sample(points_tensor, target_points)
        points_downsampled = index_points(points_tensor, fps_idx).squeeze(0).numpy()
        print(f"Downsampled point cloud from {len(points)} to {len(points_downsampled)} points")
        return points_downsampled
    
    return points

def save_plots(history, log_dir):
    epochs = list(range(1, len(history['train_success_loss']) + 1))
    # Loss plot
    plt.figure()
    plt.plot(epochs, history['train_success_loss'], label='Train Success Loss')
    plt.plot(epochs, history['val_success_loss'],   label='Val Success Loss')
    plt.plot(epochs, history['train_collision_loss'], label='Train Collision Loss')
    plt.plot(epochs, history['val_collision_loss'],   label='Val Collision Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'loss_curves.png'))
    plt.close()
    # Accuracy / Precision / Recall / F1 plot for success
    plt.figure()
    plt.plot(epochs, history['train_success_accuracy'], label='Train Success Acc')
    plt.plot(epochs, history['val_success_accuracy'],   label='Val Success Acc')
    plt.plot(epochs, history['train_success_precision'], label='Train Success Prec')
    plt.plot(epochs, history['val_success_precision'],   label='Val Success Prec')
    plt.plot(epochs, history['train_success_recall'],    label='Train Success Recall')
    plt.plot(epochs, history['val_success_recall'],      label='Val Success Recall')
    plt.plot(epochs, history['train_success_f1'],        label='Train Success F1')
    plt.plot(epochs, history['val_success_f1'],          label='Val Success F1')
    plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'success_metrics.png'))
    plt.close()
    # Accuracy / Precision / Recall / F1 plot for collision
    plt.figure()
    plt.plot(epochs, history['train_collision_accuracy'], label='Train Collision Acc')
    plt.plot(epochs, history['val_collision_accuracy'],   label='Val Collision Acc')
    plt.plot(epochs, history['train_collision_precision'], label='Train Collision Prec')
    plt.plot(epochs, history['val_collision_precision'],   label='Val Collision Prec')
    plt.plot(epochs, history['train_collision_recall'],    label='Train Collision Recall')
    plt.plot(epochs, history['val_collision_recall'],      label='Val Collision Recall')
    plt.plot(epochs, history['train_collision_f1'],        label='Train Collision F1')
    plt.plot(epochs, history['val_collision_f1'],          label='Val Collision F1')
    plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'collision_metrics.png'))
    plt.close()

def main(dir_path):

    # ——— 1) Logging setup —————————————————————————————————————
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(dir_path, 'logs', f'training_{timestamp}')
    model_dir = os.path.join(dir_path, 'models', f'model_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, 'training.log'), 'a')

    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    # ——— 2) Device & data splits —————————————————————————————
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Training started on {device} ===")
    train_data_json = os.path.join(dir_path, 'train.json')
    val_data_json = os.path.join(dir_path, 'val.json')
    print(f"Loading train dataset from {train_data_json}...")
    print(f"Loading val dataset from {val_data_json}...")

    train_dataset = KinematicFeasibilityDataset(train_data_json)
    val_dataset = KinematicFeasibilityDataset(val_data_json)
    N_train = len(train_dataset); print(f"  → {N_train} train samples")
    N_val = len(val_dataset); print(f"  → {N_val}   val samples\n")

   # ——— 3) Weighted sampler (train only) ————————————————————————
    print("Preparing sampler weights...")
    # Prepare or load sampler weights for success & collision
    weights_path = os.path.join(dir_path, 'sample_weights.npy')
    if os.path.exists(weights_path):
        sample_weights = np.load(weights_path)
        print(f"Loaded sample_weights from {weights_path}")
    else:
        # Compute once and save
        labels_s = np.array([int(train_dataset[i][3].item()) for i in range(N_train)])
        labels_c = np.array([int(train_dataset[i][4].item()) for i in range(N_train)])
        counts_s = np.bincount(labels_s, minlength=2)
        counts_c = np.bincount(labels_c, minlength=2)
        w_s = {c: N_train/(2*counts_s[c]) for c in (0,1)}
        w_c = {c: N_train/(2*counts_c[c]) for c in (0,1)}
        sample_weights = np.array([w_s[labels_s[i]] + w_c[labels_c[i]] for i in range(N_train)], dtype=np.float32)
        np.save(weights_path, sample_weights)
        print(f"Saved sample_weights to {weights_path}")

    sampler = NumpyWeightedSampler(sample_weights, num_samples=N_train, seed=42)
    print("Done preparing sampler weights...")
    epochs = 300

    # ——— 4) DataLoaders —————————————————————————————————————rs
    train_loader = DataLoader(
        train_dataset, batch_size=512, sampler=sampler,
        num_workers=24, pin_memory=False,        # ← flip this to False
        persistent_workers=False, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=24, pin_memory=False,
        persistent_workers=False, prefetch_factor=4
    )

    # ——— 5) Static point‐cloud embedding ———————————————————————
    print("Computing static point-cloud embedding …")
    pcd_path      = '/home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/pointcloud_no_plane.pcd'
    object_pcd_np = load_pointcloud(pcd_path)
    object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
    print(f"Loaded point cloud with {object_pcd.shape[0]} points...")

    # forward once through PointNetEncoder
    with torch.no_grad():
        pn = PointNetEncoder(global_feat_dim=256).to(device)
        static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
    print("Done.\n")

    # ——— 6) Model, optimizer, scaler, scheduler —————————————————————
    model = GraspObjectFeasibilityNet(use_static_obj=True).to(device)
    model.register_buffer('static_obj_feat', static_obj_feat)  # now model.static_obj_feat is available
    model = torch.compile(model)
    scaler = GradScaler(init_scale=2**16, growth_interval=2000, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=epochs * len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10000
    )

    # ——— 7) History buckets & training loop —————————————————————
    history = {k: [] for k in [
        'train_success_loss','train_collision_loss',
        'train_success_accuracy','train_collision_accuracy',
        'train_success_precision','train_collision_precision',
        'train_success_recall','train_collision_recall',
        'train_success_f1','train_collision_f1',
        'val_success_loss','val_collision_loss',
        'val_success_accuracy','val_collision_accuracy',
        'val_success_precision','val_collision_precision',
        'val_success_recall','val_collision_recall',
        'val_success_f1','val_collision_f1'
    ]}

    best_val_loss = float("inf")

    print("Start training...")

    for epoch in range(1, epochs+1):
        # Training
        model.train()
        sums = { 'loss_s':0., 'loss_c':0., 'n':0 }
        conf = { 'tp_s':0,'fp_s':0,'fn_s':0,'tn_s':0,
                 'tp_c':0,'fp_c':0,'fn_c':0,'tn_c':0 }
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", 
                         file=orig_stderr, leave=False)
        for grasp, init, final, sl, cl in train_bar:
            grasp, init, final = [t.to(device, non_blocking=True)
                                  for t in (grasp, init, final)]
            sl = sl.to(device).unsqueeze(1)
            cl = cl.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                log_s, log_c = model(None, grasp, init, final)
                loss_s = criterion(log_s, sl)
                loss_c = criterion(log_c, cl)
                loss   = loss_s + loss_c
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            # — step the LR scheduler on every batch —  
            scheduler.step()

            # accumulate loss
            bs = sl.size(0)
            sums['loss_s'] += loss_s.item() * bs
            sums['loss_c'] += loss_c.item() * bs
            sums['n']      += bs

            # predictions
            pred_s = (torch.sigmoid(log_s) > 0.5).long()
            pred_c = (torch.sigmoid(log_c) > 0.5).long()
            # update confusion for success
            conf['tp_s'] += ((pred_s==1)&(sl==1)).sum().item()
            conf['fp_s'] += ((pred_s==1)&(sl==0)).sum().item()
            conf['fn_s'] += ((pred_s==0)&(sl==1)).sum().item()
            conf['tn_s'] += ((pred_s==0)&(sl==0)).sum().item()
            conf['tp_c'] += ((pred_c==1)&(cl==1)).sum().item()
            conf['fp_c'] += ((pred_c==1)&(cl==0)).sum().item()
            conf['fn_c'] += ((pred_c==0)&(cl==1)).sum().item()
            conf['tn_c'] += ((pred_c==0)&(cl==0)).sum().item()

        # compute train metrics
        n = sums['n']
        train_loss_s = sums['loss_s']/n
        train_loss_c = sums['loss_c']/n
        train_acc_s  = (conf['tp_s']+conf['tn_s'])/n
        train_acc_c  = (conf['tp_c']+conf['tn_c'])/n
        train_prec_s = conf['tp_s']/(conf['tp_s']+conf['fp_s']+1e-8)
        train_prec_c = conf['tp_c']/(conf['tp_c']+conf['fp_c']+1e-8)
        train_rec_s  = conf['tp_s']/(conf['tp_s']+conf['fn_s']+1e-8)
        train_rec_c  = conf['tp_c']/(conf['tp_c']+conf['fn_c']+1e-8)
        train_f1_s   = 2*train_prec_s*train_rec_s/(train_prec_s+train_rec_s+1e-8)
        train_f1_c   = 2*train_prec_c*train_rec_c/(train_prec_c+train_rec_c+1e-8)

        # record train
        history['train_success_loss'].append(train_loss_s)
        history['train_collision_loss'].append(train_loss_c)
        history['train_success_accuracy'].append(train_acc_s)
        history['train_collision_accuracy'].append(train_acc_c)
        history['train_success_precision'].append(train_prec_s)
        history['train_collision_precision'].append(train_prec_c)
        history['train_success_recall'].append(train_rec_s)
        history['train_collision_recall'].append(train_rec_c)
        history['train_success_f1'].append(train_f1_s)
        history['train_collision_f1'].append(train_f1_c)



        # Validation
        model.eval()
        sums = {'loss_s':0., 'loss_c':0., 'n':0}
        conf = {'tp_s':0,'fp_s':0,'fn_s':0,'tn_s':0,
                'tp_c':0,'fp_c':0,'fn_c':0,'tn_c':0}

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", 
                       file=orig_stderr, leave=False)
        
        with torch.no_grad():
            for grasp, init, final, sl, cl in val_bar:
                grasp, init, final = [t.to(device, non_blocking=True)
                                      for t in (grasp, init, final)]
                sl    = sl.to(device).unsqueeze(1)
                cl    = cl.to(device).unsqueeze(1)

                log_s, log_c = model(None, grasp, init, final)
                loss_s = criterion(log_s, sl)
                loss_c = criterion(log_c, cl)

                bs = sl.size(0)
                sums['loss_s'] += loss_s.item() * bs
                sums['loss_c'] += loss_c.item() * bs
                sums['n']      += bs

                pred_s = (torch.sigmoid(log_s) > 0.5).long()
                pred_c = (torch.sigmoid(log_c) > 0.5).long()
                conf['tp_s'] += ((pred_s==1)&(sl==1)).sum().item()
                conf['fp_s'] += ((pred_s==1)&(sl==0)).sum().item()
                conf['fn_s'] += ((pred_s==0)&(sl==1)).sum().item()
                conf['tn_s'] += ((pred_s==0)&(sl==0)).sum().item()
                conf['tp_c'] += ((pred_c==1)&(cl==1)).sum().item()
                conf['fp_c'] += ((pred_c==1)&(cl==0)).sum().item()
                conf['fn_c'] += ((pred_c==0)&(cl==1)).sum().item()
                conf['tn_c'] += ((pred_c==0)&(cl==0)).sum().item()


        # compute val metrics
        n = sums['n']
        val_loss_s = sums['loss_s']/n
        val_loss_c = sums['loss_c']/n
        val_acc_s  = (conf['tp_s']+conf['tn_s'])/n
        val_acc_c  = (conf['tp_c']+conf['tn_c'])/n
        val_prec_s = conf['tp_s']/(conf['tp_s']+conf['fp_s']+1e-8)
        val_prec_c = conf['tp_c']/(conf['tp_c']+conf['fp_c']+1e-8)
        val_rec_s  = conf['tp_s']/(conf['tp_s']+conf['fn_s']+1e-8)
        val_rec_c  = conf['tp_c']/(conf['tp_c']+conf['fn_c']+1e-8)
        val_f1_s   = 2*val_prec_s*val_rec_s/(val_prec_s+val_rec_s+1e-8)
        val_f1_c   = 2*val_prec_c*val_rec_c/(val_prec_c+val_rec_c+1e-8)


        # record val
        history['val_success_loss'].append(val_loss_s)
        history['val_collision_loss'].append(val_loss_c)
        history['val_success_accuracy'].append(val_acc_s)
        history['val_collision_accuracy'].append(val_acc_c)
        history['val_success_precision'].append(val_prec_s)
        history['val_collision_precision'].append(val_prec_c)
        history['val_success_recall'].append(val_rec_s)
        history['val_collision_recall'].append(val_rec_c)
        history['val_success_f1'].append(val_f1_s)
        history['val_collision_f1'].append(val_f1_c)

        # — Checkpoint & logging ————————————————————————
        val_total_loss = val_loss_s + val_loss_c
        print(f"\nEpoch {epoch}: Val Total Loss={val_total_loss:.4f}\n")
        
        # Reduce LR on plateau
        # log current LR (optional)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate now: {current_lr:.6e}\n")

        # save this epoch's metrics
        csv_path = os.path.join(log_dir, 'epoch_metrics.csv')
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a') as csvf:
            if write_header:
                csvf.write(','.join(history.keys()) + '\n')
            row = ','.join(f"{history[k][-1]:.4f}" for k in history) + '\n'
            csvf.write(row)

        # save best-model only
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            ckpt = os.path.join(model_dir, f"best_model_{best_val_loss:.4f}.pth".replace('.', '_'))
            torch.save(model.state_dict(), ckpt)
            print(f"→ Saved new best model: {os.path.basename(ckpt)}\n")
        

    print("Training completed!")
    log_file.close()

     # ——— 8) Post-training plots —————————————————————————————
    save_plots(history, log_dir)




if __name__ == "__main__":
    # Paths
    my_dir = "/home/chris/Chris/placement_ws/src/data/path_simulation"
    main(my_dir)