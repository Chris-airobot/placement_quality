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

class StabilityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # single scalar output
        )

    def forward(self, x):
        return self.net(x)
    

def train_regression_model(train_entries, valid_entries, 
                           num_epochs=50, batch_size=32, lr=1e-3,
                           save_model_path=None):
    """
    all_entries: list of your JSON-based data items
    """
    # Decide on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = MyStabilityDataset(train_entries)
    valid_dataset = MyStabilityDataset(valid_entries)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model
    input_dim = len(train_dataset[0][0])  # length of the input vector
    model = StabilityNet(input_dim).to(device)

     # Loss & optimizer
    criterion = nn.MSELoss()  # we want to predict a single float in [0,1]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # For tracking losses
    train_losses = []
    valid_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, 
                                     desc=f"Epoch [{epoch+1}/{num_epochs}] (Train)", 
                                     leave=False):
            # x_batch: shape [B, input_dim]
            # y_batch: shape [B, 1]

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)         # shape [B, 1]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_batch.size(0)

        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_valid_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in tqdm(valid_loader, 
                                     desc=f"Epoch [{epoch+1}/{num_epochs}] (Val)", 
                                     leave=False):
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                pred_val = model(x_val)
                loss_val = criterion(pred_val, y_val)
                total_valid_loss += loss_val.item() * x_val.size(0)

        avg_valid_loss = total_valid_loss / len(valid_dataset)
        valid_losses.append(avg_valid_loss)


        print(f"Epoch [{epoch+1}/{num_epochs}]: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_valid_loss:.4f}")

    # ---- Plot Train & Valid Losses Side by Side ----
    skip_epochs = 10

    plt.figure(figsize=(8, 5))
    plt.plot(range(skip_epochs+1, num_epochs + 1), train_losses[skip_epochs:], label='Train Loss', marker='o')
    plt.plot(range(skip_epochs+1, num_epochs + 1), valid_losses[skip_epochs:], label='Validation Loss', marker='s')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} sec")

    if save_model_path is not None:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model.state_dict(), save_model_path)
            print(f"Model weights saved to: {save_model_path}")

    return model, train_losses, valid_losses


if __name__ == "__main__":
    file_path = "/home/chris/Downloads/python/processed_data.json"
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
    model, train_losses, valid_losses = train_regression_model(train_entries, valid_entries, 
                                                            num_epochs=500, batch_size=16, lr=1e-3,
                                                            save_model_path="/home/chris/Downloads/python/stability_model_back.pth")
