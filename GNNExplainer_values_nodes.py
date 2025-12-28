import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from src.utils import compute_local_slenderness_ratio

from config.dataConfig import Config
from src.data import SimulationData
from src.dataset import MultiColDataset
from src.dataset import compute_normalization_stats, compute_graph_norm_stats
from src.utils import compute_mean_target, compute_std_target
from src.models import DeepONet
from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer

def get_coords_timestep_i(data:SimulationData , column_type:str, i:int):
    data_col = data.features[column_type] # 214 timesteps
    node_coords = torch.tensor(data_col["coordinate_features"][i].reshape(-1, 3), dtype=torch.float32) # (70,3) timestep i

    return node_coords

import os
import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np

def make_importance_gif_dynamic_graph(data, importance_list, col_type,
                                      gif_path="ReserveCapacityPrediction_DeepONet/figures/gifs/node_importance.gif",
                                      elev=20, azim=45):

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # Normalize global importance values for stable color scale
    all_imps = torch.stack(importance_list).cpu().numpy()
    importances = all_imps[-100:]
    global_min = importances.min()
    global_max = importances.max()

    # Get node coordinates for all timesteps
    all_pos = [get_coords_timestep_i(data, col_type, t) for t in range(len(importance_list))]
    all_pos = np.array(all_pos)
    
    posistions = all_pos[-100:]
    # Compute global axis limits
    x_min, y_min, z_min = posistions.min(axis=(0,1))
    x_max, y_max, z_max = posistions.max(axis=(0,1))

    frames = []

    for t, imp in enumerate(importances):
        pos = posistions[t]
        #imp = imp.detach().cpu().numpy()

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with **fixed vmin/vmax** for colorbar
        sc = ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=imp,
            cmap="viridis",
            s=40,
            vmin=global_min,
            vmax=global_max
        )

        ax.set_title(f"Node Importance – t = {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Fixed camera view
        ax.view_init(elev=elev, azim=azim)

        # Set fixed axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Colorbar with fixed scale
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('Node Importance')

        # Optionally 5 evenly spaced ticks
        tick_labels = np.linspace(global_min, global_max, 5)
        cbar.set_ticks(tick_labels)
        cbar.set_ticklabels([f"{v:.4f}" for v in tick_labels])

        # Save frame
        frame_path = f"_tmp_frame_{t}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close()

        frames.append(imageio.imread(frame_path))

    # Build GIF
    imageio.mimsave(gif_path, frames, fps=5)

    # Cleanup
    for t in range(len(frames)):
        os.remove(f"_tmp_frame_{t}.png")

    print(f"GIF saved to: {gif_path}")


from collections import defaultdict
from torch.utils.data import Subset
import shap  # Make sure shap is installed

# === 1. Load trained model ===
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = DeepONet(branch_in=7, trunk_in=4, hidden_dim=264, hidden_dim_trunk=264,
                 latent_dim=128, out_dim=2, dropout=0.2).to(device)
state_dict = torch.load("best_deeponet_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded and set to evaluation mode.")

# === 2. Prepare DataLoader ===
config = Config()
data = SimulationData(config)
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')

trunk_mean, trunk_std = compute_normalization_stats(data)
graph_mean, graph_std = compute_graph_norm_stats(data)
dataset = MultiColDataset(data, trunk_mean, trunk_std, graph_mean, graph_std)

trunk_mean, trunk_std = trunk_mean.to(device), trunk_std.to(device)
graph_mean, graph_std = graph_mean.to(device), graph_std.to(device)

# Separate by flags
test_set = "in_flag"
train_val_samples, test_samples = [], []
for i, sample in enumerate(dataset.samples):
    if sample[test_set]:
        test_samples.append(i)
    else:
        train_val_samples.append(i)

train_size = int(1 * len(train_val_samples))
train_samples = range(train_size)  # list/range of indices
train_dataset = Subset(dataset, train_samples)

loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# === 3. Use SHAP to compute node importance ===

# === 1. Wrap the branch model for the Explainer ===
class BranchWrapper(nn.Module):
    def __init__(self, branch_model):
        super().__init__()
        self.branch = branch_model

    def forward(self, x, edge_index, edge_attr=None):
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return self.branch(graph)

# === 2. Initialize the Explainer ===
wrapper = BranchWrapper(model.branch).to(device)

explainer = Explainer(
    model=wrapper,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',           # model explanation
    node_mask_type='attributes',        # explain node features
    edge_mask_type='object',            # explain edges
    model_config=dict(
        mode='regression',              # your DeepONet outputs continuous values
        task_level='node',              # node-level explanation
        return_type='raw',              # regression returns raw values
    )
)

# === 3. Dictionary to store node importance per column type ===
importance_by_col = defaultdict(list)
shap_values_by_col = defaultdict(list)
model.eval()


col_type = "W16X100"
# === 4. Loop over your DataLoader ===
for graph, trunk_input, target, in_flag, out_flag, col in loader:
    if col[0] == col_type:
        graph = graph.to(device)

        explanation = explainer(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
        node_scores = explanation.node_mask.abs().sum(dim=1)  # [num_nodes]

        importance_by_col[col[0]].append(node_scores.cpu())
        shap_values_by_col[col[0]].append(explanation.node_mask.cpu())

print("importance calculation done.")

importance_list = importance_by_col[col_type]
make_importance_gif_dynamic_graph(data, importance_list, col_type,
                                            gif_path=f"ReserveCapacityPrediction_DeepONet/figures/gifs_GNNExplainer_importance/GNNExplainer_importance_col{col_type}.gif",
                                            elev=30, azim=60)


print("code runned successfully")