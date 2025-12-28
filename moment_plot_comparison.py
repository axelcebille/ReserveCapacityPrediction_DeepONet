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
from src.utils import compute_mean_target, compute_std_target, elastic_moments_batch, compute_moment_by_column
from src.models import DeepONet, DeepONetFNN

def get_latent(out, out_dim, latent_dim):                        # (batch, out_dim * latent_dim)
    latent = out.view(-1, out_dim, latent_dim)      # (batch, out_dim, latent_dim)
    return latent                                          # or choose one output: b[:, i, :]

# === 1. Load trained model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = DeepONet(branch_in=7, trunk_in=13, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=1, dropout=0.2).to(device)
model = DeepONetFNN(branch_in=7, trunk_in=13, hidden_dim=264,hidden_dim_trunk=264, latent_dim=264, out_dim=1, dropout=0.2).to(device)
# 2. Load the weights
state_dict = torch.load("model_weights/RM_best_deeponet_modelFNN.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded and set to evaluation mode.")
# === 2. Prepare DataLoader ===
#config = Config()
#data = SimulationData(config)
data = SimulationDataLoader()
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')


# Compute normalization stats from training data
trunk_mean, trunk_std = compute_normalization_stats(data, model="rm")
graph_mean, graph_std = compute_graph_norm_stats(data, model="rm")

dataset = MultiColDataset(data,trunk_mean, trunk_std, graph_mean, graph_std, model="rm")
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

target_mean = compute_mean_target(train_dataset, model="rm")
target_std = compute_std_target(train_dataset, target_mean, model="rm")
target_mean,target_std = target_mean.to(device),target_std.to(device)

elastic_moments = compute_moment_by_column(data)

loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


col_obj = "W16X100"
pred_col = []
target_col = []
deformations = []
with torch.no_grad():
    for graph, trunk_input, target, in_flag, out_flag, col in loader:
        if col[0] == col_obj:
            deformations.append(trunk_input[0, 7]*trunk_std[7] + trunk_mean[7])
            graph = graph.to(device)
            trunk_input = trunk_input.to(device)
            target = target.to(device)
            preds = model(graph, trunk_input)
            preds = preds * target_std + target_mean
            target = target.reshape(preds.shape)

            pred_col.append(preds[0])
            target_col.append(target[0])

deformations = [t.detach().cpu().numpy() for t in deformations]
deformations = np.array(deformations)
pred_col = [t.detach().cpu().numpy() for t in pred_col]
pred_col = np.array(pred_col)
target_col = [t.detach().cpu().numpy() for t in target_col]
target_col = np.array(target_col)


plt.figure(figsize=(20,8))
plt.plot(deformations, pred_col, 'b-', linewidth=0.5,label="pred")
plt.plot(deformations, target_col, 'r-', linewidth=0.5,label="true")
plt.xlabel("Deflection [units]")      # or "Rotation", "Deformation", etc.
plt.ylabel("Moment [units]")
plt.title("Moment–Deflection Curve")
plt.legend()
plt.grid(True)
plt.show()


plt.savefig(f"ReserveCapacityPrediction_DeepONet/figures/moment_def_comparisonFNN.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Moment–Deflection plot saved.")