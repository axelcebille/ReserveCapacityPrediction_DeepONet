import os
import torch
import numpy as np
import pandas as pd
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
from src.models import DeepONet, DeepONetFNN

# === 1. Load trained model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model for residual capacity prediction
model = DeepONetFNN(branch_in=7, trunk_in=13, hidden_dim=264,hidden_dim_trunk=264, latent_dim=264, out_dim=1, dropout=0.2).to(device)
# Load the weights

# FNN branch model, with affine transfo invariance, resisting moment
#state_dict = torch.load("model_weights/RM_best_deeponet_modelFNN.pth", map_location="cpu")
#file_path = "ReserveCapacityPrediction_DeepONet/figures/pca_analysis/resisting_moment"

# FNN branch model, with affine transfo invariance, resisting moment
state_dict = torch.load("model_weights/RM_(translation_only)best_deeponet_modelFNN.pth", map_location="cpu")
file_path = "ReserveCapacityPrediction_DeepONet/figures/pca_analysis/resisting_moment(translation_only)"

model.load_state_dict(state_dict)
model.eval()
print("✅ Model loaded and set to evaluation mode.")
# === 2. Prepare DataLoader ===
config = Config()
#data = SimulationData(config)
data = SimulationDataLoader()
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')

def pca_analysisRM(data, model, device, corr_type, network_type, file_path):
    print(f"Performing Resisting Moment PCA analysis on {network_type} network correlated with {corr_type}.")
    # Compute normalization stats from training data
    trunk_mean, trunk_std = compute_normalization_stats(data,model="rm")
    graph_mean, graph_std = compute_graph_norm_stats(data,model="rm")

    dataset = MultiColDataset(data,trunk_mean, trunk_std, graph_mean, graph_std,model="rm")
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

    target_mean = compute_mean_target(train_dataset,model="rm")
    target_std = compute_std_target(train_dataset,target_mean,model="rm")
    target_mean, target_std = target_mean.to(device), target_std.to(device)

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # === 3. Extract branch latent features ===
    branch_features = []
    trunk_features = []
    col_names = []
    rm_list = []
    lsr_list = []
    max_curv = []
    L_k = 8000

    cummulative_disp = []
    prev_col = None
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

            graph.x = graph.x * graph_std + graph_mean  # Denormalize graph features
            max_curv.append(graph.x[:, 6].max().item())

            d, t_w, b_f, t_f, disp = trunk_input[0,0].item(), trunk_input[0,1].item(), trunk_input[0,2].item(), trunk_input[0,3].item(), trunk_input[0,4].item()
            if col == prev_col:
                cummulative_disp.append(cummulative_disp[-1] + abs(disp))
            else:
                prev_col = col
                cummulative_disp.append(abs(disp))

            lsr = compute_local_slenderness_ratio(L_k, d, t_w, b_f, t_f)
            lsr_list.append(lsr)

    branch_features = torch.stack(branch_features)
    trunk_features = torch.stack(trunk_features)
    cummulative_disp = np.array(cummulative_disp)
    max_curv = np.array(max_curv)/1e-5

    if network_type == "branch":
        features = branch_features
    elif network_type == "trunk":
        features = trunk_features


    if corr_type == "residual_capacity":
        dataset2 = MultiColDataset(data,model="rc")

        # Separate by flags
        test_set = "in_flag" # "out_flag"
        train_val_samples, test_samples = [], []
        for i, sample in enumerate(dataset2.samples):
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
        train_dataset = Subset(dataset2, train_samples)
        loader2 = DataLoader(train_dataset, batch_size=1, shuffle=False)

        rc_list = []
        prev_col = None
        for graph, trunk_input, target, in_flag, out_flag, col in loader2:
            rc = target[0, 0].item()
            rc_list.append(rc)

        corr_metric = np.array(rc_list)

    elif corr_type == "max_curvature":
        corr_metric = max_curv

    elif corr_type == "cummulative_displacement":
        corr_metric = cummulative_disp


    unique_cols = sorted(list(set(col_names)))
    col_to_idx = {col: i for i, col in enumerate(unique_cols)}
    color_indices = np.array([col_to_idx[c] for c in col_names])

    # === Create figures folder ===
    #os.makedirs("figures/pca_analysis", exist_ok=True)

    # === Compute PCA (branch or trunk) ===
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(features)

    plt.figure()
    sc = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=corr_metric, cmap="viridis", alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{network_type} Network Latent Space (PCA)")

    # Add color scale
    cbar = plt.colorbar(sc)
    cbar.set_label(f"{corr_type}", rotation=270, labelpad=15)

    plt.savefig(f"{file_path}/RM_pca_latent_space_{network_type}_{corr_type}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- Compute correlations with target ---
    corrs = [np.corrcoef(emb_pca[:, i], corr_metric)[0, 1] for i in range(emb_pca.shape[1])]

    # --- Make a readable summary ---
    # Build dataframe
    corr_df = pd.DataFrame({
        'PC': np.arange(1, len(corrs) + 1),
        'corr': corrs,
        'var': pca.explained_variance_ratio_
    })

    # Sort by absolute correlation
    corr_df = corr_df.reindex(corr_df['corr'].abs().sort_values(ascending=False).index)

    # Plot
    plt.figure(figsize=(9, 4))

    colors = np.where(corr_df['corr'] >= 0, 'steelblue', 'indianred')

    plt.bar(
        corr_df['PC'],
        corr_df['corr'],
        color=colors,
        width=0.8
    )

    # Fixed scale for comparison
    plt.ylim(-1, 1)

    plt.axhline(0, linewidth=0.8)
    plt.grid(axis='y', alpha=0.3)

    plt.xlabel("Principal Component")
    plt.ylabel(f"Correlation with {corr_type}")
    plt.title(f"Correlation of {corr_type} with PCA Components for {network_type} model")

    plt.tight_layout()
    plt.show()

    plt.savefig(f"{file_path}/RM_pca_correlation_{network_type}_{corr_type}.png", dpi=300, bbox_inches="tight")
    plt.close()


    print("✅ Figures saved in ./figures/pca_analysis/ folder.")

# === RUN PCA ANALYSIS FOR RESISTING MOMENT MODEL ===
corr_types = ["cummulative_displacement", "residual_capacity", "max_curvature"]
network_types = ["branch", "trunk"]

for corr_type in corr_types:
    for network_type in network_types:
        pca_analysisRM(data, model, device, corr_type, network_type, file_path)