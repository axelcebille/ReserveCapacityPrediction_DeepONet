import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from dataConfig import Config
from data import SimulationData
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  
from torch.utils.data import random_split

from torch_geometric.data import Data, Batch

from dataset import MultiColDataset
from models import DeepONet

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

seed = 42
set_seed(seed)


config = Config()
data = SimulationData(config)
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')

# --- Create multi-column dataset ---
dataset = MultiColDataset(data)

# Separate by flags
test_set = "in_flag" # "out_flag"
train_val_samples, test_samples = [], []
for i, sample in enumerate(dataset.samples):
    if sample[test_set]:
        test_samples.append(i)
    else:
        train_val_samples.append(i)

train_size = int(0.8 * len(train_val_samples))
val_size = len(train_val_samples) - train_size

# Split dataset
train_samples, val_samples = random_split(train_val_samples, [train_size, val_size])
# Create subset datasets
from torch.utils.data import Subset
train_dataset = Subset(dataset, train_samples)
val_dataset   = Subset(dataset, val_samples)
test_dataset   = Subset(dataset, test_samples)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader   = DataLoader(test_dataset, batch_size=8, shuffle=False)

# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepONet(branch_in=7, trunk_in=4, out_dim=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

# --- Training loop ---
n_epochs = 1000
# early stop params
best_val_loss = float('inf')
patience = 50
patience_counter = 0
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0

    for graphs, trunk_inputs, targets, _, _ in train_loader:
        graphs = graphs.to(device)
        trunk_inputs = trunk_inputs.to(device)
        targets = targets.to(device)

        preds = model(graphs, trunk_inputs)

        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * graphs.num_graphs

    avg_train_loss = total_loss / len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    ood_loss = 0.0
    with torch.no_grad():
        for graphs, trunk_inputs, targets, _, _ in val_loader:
            graphs = graphs.to(device)
            trunk_inputs = trunk_inputs.to(device)
            targets = targets.to(device)

            preds = model(graphs, trunk_inputs)
            val_loss += criterion(preds, targets).item() * graphs.num_graphs

        for graphs, trunk_inputs, targets, _, _ in test_loader:
            graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
            preds = model(graphs, trunk_inputs)
            ood_loss += criterion(preds, targets).item() * graphs.num_graphs

    avg_val_loss = val_loss / len(val_dataset)
    avg_test_loss = ood_loss / len(test_dataset)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | Test: {avg_test_loss:.6f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_deeponet_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# --- Final OOD evaluation ---
model.load_state_dict(torch.load('best_deeponet_model.pth'))
model.eval()
print("Final Test evaluation:")
from utils import reserve_capacity_mape
test_loss = 0.0
y_pred_all = []
y_true_all = []
with torch.no_grad():
    for graphs, trunk_inputs, targets, _, _ in test_loader:
                graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
                preds = model(graphs, trunk_inputs)
                test_loss += criterion(preds, targets).item() * graphs.num_graphs
                y_pred_all.append(preds.cpu().numpy())
                y_true_all.append(targets.cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    mape_loss = reserve_capacity_mape(y_true_all, y_pred_all)
    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test: avg_ood_loss {avg_test_loss} | mape {mape_loss}")

