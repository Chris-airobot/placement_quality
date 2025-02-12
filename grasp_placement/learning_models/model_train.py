import torch.nn as nn
import torch.optim as optim
import torch
from dataset import MyStabilityDataset
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import time
import os
import numpy as np

class StabilityNet(nn.Module):
    """
    input_dim depends on how many floats in your input.
    Here: 3+4 (grasp) + 3+4 (init) + 3+4 (target) = 3+4 + 3+4 + 3+4 = 21 total.
    Adjust accordingly.
    """
    def __init__(self, input_dim=21, hidden_dim=128):
        super().__init__()
        # Shared backbone
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Classification head (2 outputs => for CrossEntropy)
        self.class_head = nn.Linear(hidden_dim, 2)

        # Regression head (5 outputs => for your 5 float targets)
        self.reg_head = nn.Linear(hidden_dim, 1)
        # self.reg_head = nn.Linear(hidden_dim, 5)


        self.relu = nn.ReLU()

    def forward(self, x):

        # Shared backbone
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))

        # Classification => shape [B, 2]
        out_class = self.class_head(h)

        # Regression => shape [B, 5]
        out_reg = self.reg_head(h)

        return out_class, out_reg
    
############################################
# 3) LOSS FUNCTION
############################################
def two_head_loss(pred_class, pred_stability, label_class, label_stability):
    """
    pred_class: [B, 2] (logits)
    pred_stability: [B, 1] (regressed value)
    label_class: [B] (0 or 1, int)
    label_stability: [B] (float in [0,1])
    """
    # Classification => cross entropy
    criterion_ce = nn.CrossEntropyLoss()
    loss_class = criterion_ce(pred_class, label_class)

    # Regression => MSE
    criterion_mse = nn.MSELoss()
    pred_stability = pred_stability.view(-1)  # shape [B]
    loss_reg = criterion_mse(pred_stability, label_stability)

    # Combine
    total_loss = loss_class + loss_reg
    return total_loss, loss_class.item(), loss_reg.item()


############################################
# 4) TRAINING LOOP (Skeleton)
############################################
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    for x, (feas_lbl, reg_lbl) in loader:
        x = x.float().to(device)  # [B, input_dim]
        feas_lbl = feas_lbl.to(device)  # [B], long
        reg_lbl = reg_lbl.float().to(device)  # [B, 5]

        optimizer.zero_grad()
        out_class, out_reg = model(x)
        loss, l_c, l_r = two_head_loss(out_class, out_reg, feas_lbl, reg_lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += l_c
        total_reg_loss += l_r

    n_batches = len(loader)
    return (total_loss / n_batches,
            total_cls_loss / n_batches,
            total_reg_loss / n_batches)

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    correct_cls = 0
    total_samples = 0

    with torch.no_grad():
        for x, (feas_lbl, reg_lbl) in loader:
            x = x.float().to(device)
            feas_lbl = feas_lbl.to(device)
            reg_lbl = reg_lbl.float().to(device)

            out_class, out_reg = model(x)
            loss, lc, lr = two_head_loss(out_class, out_reg, feas_lbl, reg_lbl)


            total_loss += loss.item()
            total_cls_loss += lc
            total_reg_loss += lr

            # Classification accuracy
            preds = torch.argmax(out_class, dim=1)  # [B]
            correct_cls += (preds == feas_lbl).sum().item()
            total_samples += feas_lbl.shape[0]


    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    accuracy = correct_cls / total_samples if total_samples > 0 else 0.0

    return avg_loss, avg_cls_loss, avg_reg_loss, accuracy




def main():
    # Device setup: use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    file_path = "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/processed_data.json"
    with open(file_path, "r") as file:
        all_entries = json.load(file)

    # Shuffle data in-place
    random.shuffle(all_entries)

    N = len(all_entries)
    train_size = int(0.8 * N)   # 80%
    valid_size = int(0.1 * N)   # 10%
    test_size  = N - train_size - valid_size  # 10% remainder

    train_entries = all_entries[:train_size]
    valid_entries = all_entries[train_size:train_size + valid_size]
    test_entries  = all_entries[train_size + valid_size:]


    # Build Datasets
    train_dataset = MyStabilityDataset(train_entries)
    valid_dataset = MyStabilityDataset(valid_entries)
    test_dataset  = MyStabilityDataset(test_entries)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    stability_scores = [sample[-1] for sample in train_dataset.samples]  # sample[2] is your reg_lbl
    print("Stability score stats:")
    print("Min:", np.min(stability_scores))
    print("Max:", np.max(stability_scores))
    print("Mean:", np.mean(stability_scores))
    print("Std:", np.std(stability_scores))
    
    plt.figure(figsize=(8, 6))
    plt.hist(stability_scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Stability Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Stability Scores in Train Dataset')
    plt.grid(True)
    plt.show()

    # Build model
    model = StabilityNet(input_dim=21, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Define a learning rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3, verbose=True)
    # We'll store metrics for plotting
    num_epochs = 50
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Dummy training loop
    for epoch in range(num_epochs):
        # --- TRAIN ---
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(model, train_loader, optimizer, device)
        # --- VALIDATION ---
        val_loss, val_cls_loss, val_reg_loss, val_acc = eval_model(model, valid_loader, device)

        # We can also compute train accuracy if we want:
        # (We'll do a quick pass with eval_model on the train_loader again,
        # but it's a second pass. That might be slightly slower but simpler.)
        # Or we can track an approximate train accuracy from the forward pass
        # in 'train_one_epoch'. For brevity, let's do a quick pass:
        _, _, _, train_acc = eval_model(model, train_loader, device)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
        print(f"  Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.3f}")
        scheduler.step(val_loss)

    print("Training done.")

    # Plot train/valid losses
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Plot train/valid accuracy
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(valid_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # FINAL EVALUATION ON TEST SET
    test_loss, test_cls_loss, test_reg_loss, test_acc = eval_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.3f}")




if __name__ == "__main__":
    main()
