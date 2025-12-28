import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from ReserveCapacityPrediction_DeepONet.config.dataConfig import Config
from ReserveCapacityPrediction_DeepONet.src.data import SimulationData
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  
from torch.utils.data import random_split

from torch_geometric.data import Data, Batch

from ReserveCapacityPrediction_DeepONet.src.dataset import MultiColDataset
from ReserveCapacityPrediction_DeepONet.src.models import DeepONet

from scipy.spatial.transform import Rotation as Rot

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

state_dict = torch.load("best_deeponet_model.pth", map_location=device)
model.load_state_dict(state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.MSELoss()

# --- Validation ---
model.eval()
val_loss = 0.0
val_loss_trans = 0.0
ood_loss = 0.0
ood_loss_trans = 0.0
with torch.no_grad():
    for graphs, trunk_inputs, targets, _, _ in val_loader:
        graphs = graphs.to(device)
        graphs_trans = transform_batch_independent(graphs).to(device)
        trunk_inputs = trunk_inputs.to(device)
        targets = targets.to(device)

        preds = model(graphs, trunk_inputs)
        val_loss += criterion(preds, targets).item() * graphs.num_graphs

        preds_trans = model(graphs_trans, trunk_inputs)
        val_loss_trans += criterion(preds_trans, targets).item() * graphs_trans.num_graphs

    for graphs, trunk_inputs, targets, _, _ in test_loader:
        graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
        graphs_trans = transform_batch_independent(graphs).to(device)

        preds = model(graphs, trunk_inputs)
        ood_loss += criterion(preds, targets).item() * graphs.num_graphs

        preds_trans = model(graphs_trans, trunk_inputs)
        ood_loss_trans += criterion(preds_trans, targets).item() * graphs_trans.num_graphs

avg_val_loss = val_loss / len(val_dataset)
avg_test_loss = ood_loss / len(test_dataset)

avg_val_loss_t = val_loss_trans / len(val_dataset)
avg_test_loss_t = ood_loss_trans / len(test_dataset)
    
print(f"Val: {avg_val_loss:.6f} | Val transfo: {avg_val_loss_t:.6f} | Test: {avg_test_loss:.6f} | Test transfo: {avg_test_loss_t:.6f}")


