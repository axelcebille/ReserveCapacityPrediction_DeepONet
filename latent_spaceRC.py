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
from src.data import SimulationData, SimulationDataLoader
from src.dataset import MultiColDataset
from src.dataset import compute_normalization_stats, compute_graph_norm_stats
from src.utils import compute_mean_target, compute_std_target
from src.models import DeepONet

def get_latent(out, out_dim, latent_dim):                        # (batch, out_dim * latent_dim)
    latent = out.view(-1, out_dim, latent_dim)      # (batch, out_dim, latent_dim)
    return latent                                          # or choose one output: b[:, i, :]

# === 1. Load trained model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepONet(branch_in=7, trunk_in=4, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=2, dropout=0.2).to(device)
# 2. Load the weights
state_dict = torch.load("model_weights/RC_best_deeponet_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded and set to evaluation mode.")
# === 2. Prepare DataLoader ===
config = Config()
#data = SimulationData(config)
data = SimulationDataLoader()
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')

###################################
# INPUTS FOR T-SNE ANALYSIS HERE #

# Color mapping variable
var = "local_slenderness_ratio" #"residual_capacity", "local_slenderness_ratio"
# Trunk or Branch Analysis
network_type = "trunk"  #"branch" or "trunk"

print(f"Performing Residual Capacity t-sne analysis on {network_type} network colored-mapping for {var}.")
####################################

# Compute normalization stats from training data
trunk_mean, trunk_std = compute_normalization_stats(data)
graph_mean, graph_std = compute_graph_norm_stats(data)

dataset = MultiColDataset(data,trunk_mean, trunk_std, graph_mean, graph_std)
trunk_mean, trunk_std = trunk_mean.to(device), trunk_std.to(device)
graph_mean, graph_std = graph_mean.to(device), graph_std.to(device)

# Separate by flags
test_set = "in_flag" # "out_flag"
train_val_samples, test_samples = [], []
for i, sample in enumerate(dataset.samples):
    if sample[test_set]:
        test_samples.append(i)
    else:
        train_val_samples.append(i)

train_size = int(0.99 * len(train_val_samples))
val_size = len(train_val_samples) - train_size

# Split dataset
train_samples = torch.utils.data.Subset(train_val_samples, range(train_size))
val_samples = torch.utils.data.Subset(train_val_samples, range(train_size, train_size + val_size))

from torch.utils.data import Subset
train_dataset = Subset(dataset, train_samples)

target_mean = compute_mean_target(train_dataset)
target_std = compute_std_target(train_dataset,target_mean)
target_mean, target_std = target_mean.to(device), target_std.to(device)

loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# === 3. Extract branch latent features ===
branch_features = []
trunk_features = []
col_names = []
rc_list = []
lsr_list = []
L_k = 8000

with torch.no_grad():
    for graph, trunk_input, target, in_flag, out_flag, col in loader:
        graph = graph.to(device)
        trunk_input = trunk_input.to(device)

        branch_out = model.branch(graph)  # latent representation
        branch_latent = branch_out.reshape(branch_out.shape[0], -1)  # shape: (batch, out_dim*latent_dim)
        branch_features.append(branch_latent.squeeze(0).cpu())

        trunk_out = model.trunk(trunk_input)
        trunk_latent = trunk_out.reshape(trunk_out.shape[0], -1)  # shape: (batch, out_dim*latent_dim)
        trunk_features.append(trunk_latent.squeeze(0).cpu())

        col_names.append(col)  # or out_flag if you prefer
        rc = target[0, 0].item()
        rc_list.append(rc)

        trunk_input = trunk_input * trunk_std + trunk_mean  # Denormalize
        d, t_w, b_f, t_f = trunk_input[0,0].item(), trunk_input[0,1].item(), trunk_input[0,2].item(), trunk_input[0,3].item()
        lsr = compute_local_slenderness_ratio(L_k, d, t_w, b_f, t_f)
        lsr_list.append(lsr)

branch_features = torch.stack(branch_features)
trunk_features = torch.stack(trunk_features)
rc_list = np.array(rc_list)
lsr_list = np.array(lsr_list)

unique_cols = sorted(list(set(col_names)))
col_to_idx = {col: i for i, col in enumerate(unique_cols)}
color_indices = np.array([col_to_idx[c] for c in col_names])

# === Create figures folder ===
os.makedirs("figures", exist_ok=True)

# === Compute t-SNE for branch network ===
if network_type == "branch":
    features = branch_features
elif network_type == "trunk":
    features = trunk_features

if var == "residual_capacity":
    var_list = rc_list
elif var == "local_slenderness_ratio":
    var_list = lsr_list


tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
emb_tsne = tsne.fit_transform(features)

plt.figure()
sc = plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], c=var_list, cmap="viridis", alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title(f"{network_type} Network Latent Space (t-SNE)")

# Add color scale
cbar = plt.colorbar(sc)
cbar.set_label(f"{var}", rotation=270, labelpad=15)

plt.savefig(f"ReserveCapacityPrediction_DeepONet/figures/tsne_analysis/residual_capacity/tsne_latent_space_{network_type}_{var}.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Figures saved in ./figures/tsne_analysis/ folder.")
