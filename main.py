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
data.load_features('SemesterProjectAxelCebille/data/features.pkl')

# --- Create multi-column dataset ---
dataset = MultiColDataset(data)

# Separate by flags
train_samples, val_samples, ood_samples = [], [], []
for i, sample in enumerate(dataset.samples):
    if sample["out_flag"]:
        ood_samples.append(i)
    elif sample["in_flag"]:
        val_samples.append(i)
    else:
        train_samples.append(i)

# Create subset datasets
from torch.utils.data import Subset
train_dataset = Subset(dataset, train_samples)
val_dataset   = Subset(dataset, val_samples)
ood_dataset   = Subset(dataset, ood_samples)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
ood_loader   = DataLoader(ood_dataset, batch_size=8, shuffle=False)

# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepONet(branch_in=7, trunk_in=4, out_dim=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

# --- Training loop ---
n_epochs = 50
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

        for graphs, trunk_inputs, targets, _, _ in ood_loader:
            graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
            preds = model(graphs, trunk_inputs)
            ood_loss += criterion(preds, targets).item() * graphs.num_graphs

    avg_val_loss = val_loss / len(val_dataset)
    avg_ood_loss = ood_loss / len(ood_dataset)

    print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | OOD: {avg_ood_loss:.6f}")

# --- Final OOD evaluation ---
print("Final OOD evaluation:")
ood_loss = 0.0
with torch.no_grad():
    for graphs, trunk_inputs, targets, _, _ in ood_loader:
                graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
                preds = model(graphs, trunk_inputs)
                ood_loss += criterion(preds, targets).item() * graphs.num_graphs
    print(f"OOD: preds {preds} | targets {targets}")


# --- Save model ---
torch.save(model.state_dict(), 'deeponet_model.pth')