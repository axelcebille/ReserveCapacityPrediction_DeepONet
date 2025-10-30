import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from dataConfig import Config
from data import SimulationData
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
#from torch_geometric.loader import DataLoader  
from torch.utils.data import random_split

from torch_geometric.data import Data, Batch

from dataset import MultiColDataset
from models import DeepONet

from torch_geometric.utils import unbatch
from torch_geometric.data import Batch
from scipy.spatial.transform import Rotation as Rot

def random_Rt():
    R = torch.tensor(Rot.random().as_matrix(), dtype=torch.float32)
    t = torch.randn(3) * 0.1  # small random translation
    return R, t

def transform_graph(graph, R, t):
    coords = graph.x[:, :3]
    dev = graph.x[:, 3:6]
    curv = graph.x[:, 6:]

    device = graph.x.device  # get current graph device
    R = R.to(device)
    t = t.to(device)

    coords_trans = coords @ R.T + t
    dev_trans = dev @ R.T
    x_trans = torch.cat([coords_trans, dev_trans, curv], dim=1)

    src, dst = graph.edge_index
    edge_vecs = coords_trans[dst] - coords_trans[src]
    edge_norms = torch.norm(edge_vecs, dim=1, keepdim=True)

    graph.x = x_trans
    graph.edge_attr = edge_norms
    return graph

def transform_batch_independent(graphs):
    """Apply independent random rotations/translations to each graph in a DataLoader batch."""
    graphs_list = graphs.to_data_list()
    new_graphs = []

    for g in graphs_list:
        R, t = random_Rt()
        g_trans = transform_graph(g, R, t)
        new_graphs.append(g_trans)

    graphs_trans = type(graphs).from_data_list(new_graphs)
    return graphs_trans

def monotonic_penalty(y_pred):
    # if shape is (T, 2), add a batch dimension
    if y_pred.dim() == 2:
        y_pred = y_pred.unsqueeze(0)  # → (1, T, 2)
    diffs = y_pred[:, 1:, :] - y_pred[:, :-1, :]   # (batch, T-1, 2)
    penalty = torch.relu(diffs).mean()  # penalize increases
    return penalty

def loss_fn(y_pred, y_true, criterion, lambda_mon=0.001):
    base = criterion(y_pred, y_true)
    pen  = monotonic_penalty(y_pred)
    return base + lambda_mon * pen

from torch.utils.data import DataLoader
def collate_sequences(batch):
    """
    Custom collate function for MultiColSequenceDataset.
    Keeps variable-length sequences (no stacking on time dimension).
    """
    graphs_list, trunk_inputs, targets, in_flags, out_flags = zip(*batch)

    # 'graphs_list' is a list of lists of PyG Data objects (one per sequence)
    # We leave it as is (list of sequences).
    trunk_inputs = torch.stack(trunk_inputs)  # same size across samples

    return list(graphs_list), trunk_inputs, list(targets), list(in_flags), list(out_flags)


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
from dataset import MultiColSequenceDataset
#dataset = MultiColDataset(data)
dataset = MultiColSequenceDataset(data)

# Separate by flags
test_set = "in_flag" # "out_flag"
train_val_samples, test_samples = [], []
#for i, sample in enumerate(dataset.samples):
#    if sample[test_set]:
#        test_samples.append(i)
#    else:
#        train_val_samples.append(i)
for i, col in enumerate(dataset.column_types):
    feats = dataset.data.features[col]
    if test_set == "in_flag":
        flag = feats["In-distribution test"]
    else:
        flag = feats["Out-of-distribution test"]

    if flag:
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

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepONet(branch_in=7, trunk_in=4, out_dim=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=300,      # number of epochs until the LR reaches its minimum
    eta_min=1e-5    # minimum learning rate
)
criterion = torch.nn.MSELoss()

# --- Training loop ---
n_epochs = 1000
transfo_flag = False  # whether to use transformed graphs for training
# early stop params
best_val_loss = float('inf')
patience = 50
patience_counter = 0
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0

    for graphs, trunk_inputs, targets, _, _ in train_loader:
        for i in range(len(graphs)):
            # Each graphs_list[i] is a list of Data objects (one sequence)
            seq_graphs = graphs[i]
            # ✅ Combine all timesteps into a single Batch
            batched_graph = Batch.from_data_list([g.to(device) for g in seq_graphs])
            trunk_input = trunk_inputs[i].to(device)
            target = targets[i].to(device)
            if transfo_flag:
                graphs_trans = transform_batch_independent(batched_graph)
                graphs_used = graphs_trans
            else:
                graphs_used = batched_graph

            preds = model(graphs_used, trunk_input)
            loss = loss_fn(preds, target, criterion)
            total_loss += loss.item() * graphs_used.num_graphs
            loss.backward()
        
        optimizer.zero_grad()
        optimizer.step()


    avg_train_loss = total_loss / len(train_dataset)
    scheduler.step()  # update learning rate
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    ood_loss = 0.0
    with torch.no_grad():
        for graphs, trunk_inputs, targets, _, _ in val_loader:
            for i in range(len(graphs)):
                # Each graphs_list[i] is a list of Data objects (one sequence)
                seq_graphs = graphs[i]
                # ✅ Combine all timesteps into a single Batch
                batched_graph = Batch.from_data_list([g.to(device) for g in seq_graphs])
                trunk_input, target = trunk_inputs[i].to(device), targets[i].to(device)
                preds = model(batched_graph, trunk_input)
                val_loss += loss_fn(preds, target, criterion).item() * batched_graph.num_graphs

        for graphs, trunk_inputs, targets, _, _ in test_loader:
            for i in range(len(graphs)):
                # Each graphs_list[i] is a list of Data objects (one sequence)
                seq_graphs = graphs[i]
                # ✅ Combine all timesteps into a single Batch
                batched_graph = Batch.from_data_list([g.to(device) for g in seq_graphs])
                trunk_input, target = trunk_inputs[i].to(device), targets[i].to(device)
                preds = model(batched_graph, trunk_input)
                ood_loss += loss_fn(preds, target, criterion).item() * batched_graph.num_graphs

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
y_pred_transfo_all = []
y_true_all = []
with torch.no_grad():
    for graphs, trunk_inputs, targets, _, _ in test_loader:
                for i in range(len(graphs)):
                    # Each graphs_list[i] is a list of Data objects (one sequence)
                    seq_graphs = graphs[i]
                    # ✅ Combine all timesteps into a single Batch
                    batched_graph = Batch.from_data_list([g.to(device) for g in seq_graphs])
                    trunk_input, target = trunk_inputs[i].to(device), targets[i].to(device)
                    graphs_trans = transform_batch_independent(batched_graph)
                    preds = model(batched_graph, trunk_input)
                    preds_transfo = model(graphs_trans, trunk_input)
                    test_loss += loss_fn(preds, target, criterion).item() * batched_graph.num_graphs

                    y_pred_all.append(preds.cpu().numpy())
                    y_pred_transfo_all.append(preds_transfo.cpu().numpy())
                    y_true_all.append(target.cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_pred_transfo_all = np.concatenate(y_pred_transfo_all, axis=0)

    mape_loss = reserve_capacity_mape(y_true_all, y_pred_all)
    mape_loss_transfo = reserve_capacity_mape(y_true_all, y_pred_transfo_all)
    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test: avg_test_loss {avg_test_loss} | mape {mape_loss} | mape transfo. {mape_loss_transfo}")

