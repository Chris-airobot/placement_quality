import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb  # <--- Import W&B

# Add the parent folder to sys.path so `learning_models` is found
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from learning_models.dataset import MyStabilityDataset

# Base directory for saving models and images
DIR = "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/models/"

###############################################################################
# 1) MODEL DEFINITION (Two-head network: classification and regression)
###############################################################################
class StabilityNet(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 1)  # Binary classification (logits)
        self.reg_head = nn.Linear(hidden_dim, 1)     # Regression (stability score)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        cls_out = self.class_head(x)
        reg_out = self.reg_head(x)
        return cls_out, reg_out

###############################################################################
# 2) LOSS FUNCTIONS
###############################################################################
def combined_loss(pred_cls, label_cls, pred_reg, label_reg):
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    
    # Flatten predictions and labels to 1D
    pred_cls = pred_cls.view(-1)
    label_cls = label_cls.view(-1)
    loss_cls = criterion_cls(pred_cls, label_cls)
    
    pred_reg = pred_reg.view(-1)
    label_reg = label_reg.view(-1)
    loss_reg = criterion_reg(pred_reg, label_reg)
    
    return loss_cls + loss_reg, loss_cls, loss_reg

###############################################################################
# 3) HELPER FUNCTIONS FOR TRAINING AND EVALUATION
###############################################################################
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    for x, (cls_lbl, reg_lbl) in loader:
        x = x.float().to(device)
        cls_lbl = cls_lbl.float().to(device)
        reg_lbl = reg_lbl.float().to(device)

        optimizer.zero_grad()
        pred_cls, pred_reg = model(x)
        loss, loss_cls, loss_reg = combined_loss(pred_cls, cls_lbl, pred_reg, reg_lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_reg_loss += loss_reg.item()

    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    return avg_loss, avg_cls_loss, avg_reg_loss

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    with torch.no_grad():
        for x, (cls_lbl, reg_lbl) in loader:
            x = x.float().to(device)
            cls_lbl = cls_lbl.float().to(device)
            reg_lbl = reg_lbl.float().to(device)
            pred_cls, pred_reg = model(x)
            loss, loss_cls, loss_reg = combined_loss(pred_cls, cls_lbl, pred_reg, reg_lbl)
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()

    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    return avg_loss, avg_cls_loss, avg_reg_loss

def save_model(model, path='stability_net.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def sample_hyperparams():
    # hidden_dim = 2 ** random.randint(6, 9)  # e.g., 64 to 512
    batch_size = random.choice([16, 32, 64])
    log_lr = random.uniform(-5, -3)
    learning_rate = 10 ** log_lr
    seed = random.randint(0, 9999)
    scheduler_factor = random.choice([0.5, 0.6, 0.7])
    scheduler_patience = random.randint(2, 6)
    num_epochs = random.choice([30, 50, 70])
    params = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'optimizer': 'adam',
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'seed': seed,
        # 'hidden_dim': hidden_dim
    }
    return params

###############################################################################
# 4) MAIN TRAINING FUNCTION WITH W&B LOGGING AND SAVING HYPERPARAMETERS & PLOTS
###############################################################################
def main(params):
    # Initialize W&B run
    run = wandb.init(project="stability-regression",
                     config=params,
                     name=f"run_seed_{params['seed']}")
    run_dir = os.path.join(DIR, run.name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save hyperparameters to JSON in run_dir
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")
    with open(hyperparams_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_path}")

    # Set random seed
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, seed={seed}")

    # Load data & split
    file_path = "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json"
    with open(file_path, "r") as file:
        all_entries = json.load(file)
    random.shuffle(all_entries)

    N = len(all_entries)
    train_size = int(0.8 * N)
    valid_size = int(0.1 * N)
    test_size  = N - train_size - valid_size

    train_entries = all_entries[:train_size]
    valid_entries = all_entries[train_size:train_size + valid_size]
    test_entries  = all_entries[train_size + valid_size:]

    train_dataset = MyStabilityDataset(train_entries)
    valid_dataset = MyStabilityDataset(valid_entries)
    test_dataset  = MyStabilityDataset(test_entries)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=params['batch_size'], shuffle=False)

    # Build model & optimizer
    model = StabilityNet(input_dim=21).to(device)
    if params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['scheduler_factor'], 
        patience=params['scheduler_patience'], 
        verbose=True
    )

    # Training loop
    train_losses, valid_losses = [], []
    for epoch in range(params['num_epochs']):
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_cls_loss, val_reg_loss = eval_model(model, valid_loader, device)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{params['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Reg: {val_reg_loss:.4f})")

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_cls_loss": train_cls_loss,
            "train_reg_loss": train_reg_loss,
            "val_loss": val_loss,
            "val_cls_loss": val_cls_loss,
            "val_reg_loss": val_reg_loss,
        })

        scheduler.step(val_loss)

    # Final evaluation & Save
    test_loss, test_cls_loss, test_reg_loss = eval_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} (Cls: {test_cls_loss:.4f}, Reg: {test_reg_loss:.4f})")
    wandb.log({"test_loss": test_loss, "test_cls_loss": test_cls_loss, "test_reg_loss": test_reg_loss})

    model_path = os.path.join(run_dir, "stability_net.pth")
    save_model(model, model_path)

    # Plot training vs validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.grid(True)
    
    # Save plot inside run_dir
    fig_path = os.path.join(run_dir, "train_val_loss.png")
    plt.savefig(fig_path)
    print(f"Train/Val loss figure saved to {fig_path}")
    
    # Also save plot in a central images folder
    images_dir = os.path.join(DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    image_file_path = os.path.join(images_dir, f"seed_{params['seed']}.png")
    plt.savefig(image_file_path)
    print(f"Plot saved to {image_file_path}")

    wandb.finish()

###############################################################################
# 5) ENTRY POINT
###############################################################################
if __name__ == "__main__":
    total_experiment = 100
    for i in range(total_experiment):
        params = sample_hyperparams()
        print(f"\n=== Experiment {i+1}/{total_experiment} with {params} ===\n")
        main(params)
