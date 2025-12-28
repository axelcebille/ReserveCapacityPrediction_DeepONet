import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, GCN2Conv, PNAConv, GATv2Conv,
    GlobalAttention, global_mean_pool
)

class GCN_max_pool(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        self.out_dim = out_dim
        self.latent_dim = latent_dim

        h2 = max(hidden_dim // 2, 8)
        h3 = max(hidden_dim // 4, 8)
        self.mlp_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU()
        )
        self.output_layer = nn.Linear(h3, out_dim*latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)

        # insteal of global_mean_pool try flattening outputs and pass to mlp
        x = global_mean_pool(x, batch)
        # Feed through MLP
        out = self.mlp_layers(x)
        out = self.output_layer(out)
        out = out.view(-1, self.out_dim, self.latent_dim)
 
        return out
    
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim=2, dropout=0.5):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        self.out_dim = out_dim
        self.latent_dim = latent_dim

        # MLP sizes
        h2 = max(hidden_dim // 2, 8)
        h3 = max(hidden_dim // 4, 8)

        # MLP input size = 70 nodes × hidden_dim
        self.mlp_layers = nn.Sequential(
            nn.Linear(70 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
        )

        self.output_layer = nn.Linear(h3, out_dim * latent_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1)

        # --- GCN layers ---
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)

        batch_size = batch.max().item() + 1
        x = x.view(batch_size, 70 * x.size(-1))

        # MLP
        out = self.mlp_layers(x)
        out = self.output_layer(out)

        return out.view(-1, self.out_dim, self.latent_dim)

class FNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim=2, dropout=0.5):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # --- Node-wise feature expansion (replaces GCN layers) ---
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # MLP sizes (same as your GNN)
        h2 = max(hidden_dim // 2, 8)
        h3 = max(hidden_dim // 4, 8)

        # MLP input size = 70 nodes × hidden_dim
        self.mlp_layers = nn.Sequential(
            nn.Linear(70 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(h3, out_dim * latent_dim)

    def forward(self, data):
        """
        data: PyG Batch
        data.x shape = (batch_size * 70, in_dim)
        """

        x = data.x
        batch = data.batch

        batch_size = batch.max().item() + 1
        num_nodes = 70
        in_dim = x.size(-1)

        # reshape like a graph batch
        x = x.view(batch_size, num_nodes, in_dim)

        # --- Node-wise MLP ---
        x = x.view(batch_size * num_nodes, in_dim)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # --- Graph-level ---
        x = x.view(batch_size, num_nodes * self.hidden_dim)

        out = self.mlp_layers(x)
        out = self.output_layer(out)

        return out.view(-1, self.out_dim, self.latent_dim)

class TrunkNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, out_dim=2):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.l3 = nn.Linear(hidden_dim*2, out_dim*latent_dim)

        self.out_dim = out_dim
        self.latent_dim = latent_dim

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        out = F.relu(x)
        out = self.l3(out)
        out = out.view(-1, self.out_dim, self.latent_dim)

        return out

class DeepONet(nn.Module):
    """
    Simple DeepONet: y = <B(u), T(x)> + b
    - branch_in: dimension of the discretized input function (number of sensor points)
    - trunk_in : dimension of the evaluation location x
    - latent_dim: size of feature space where branch & trunk meet
    """
    def __init__(self, branch_in, trunk_in, hidden_dim=64, hidden_dim_trunk=128, latent_dim=128, out_dim=2, dropout=0.5):
        super().__init__()

        # Model dimensions
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        # Branch network
        self.branch = GCN(in_dim=branch_in, hidden_dim=hidden_dim, latent_dim=latent_dim, out_dim=out_dim, dropout=dropout)

        # Trunk network
        self.trunk = TrunkNet(in_dim=trunk_in, hidden_dim=hidden_dim_trunk, latent_dim=latent_dim, out_dim=out_dim)

        # Bias term (scalar per sample, but we learn a single scalar here)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, u_samples, x_loc):
        """
        u_samples : (batch, branch_in_dim)  - discretized function
        x_loc     : (batch, trunk_in_dim)   - evaluation coordinates
        returns   : (batch, output_dim)
        """
        b = self.branch(u_samples)   # (batch, out_dim * latent_dim)
        t = self.trunk(x_loc)    # (batch, out_dim * latent_dim)

        # Reshape to (batch, out_dim, latent_dim)
        #b = b.view(-1, self.out_dim, self.latent_dim)
        #t = t.view(-1, self.out_dim, self.latent_dim)

        # -----------------------------
        # Dot product (key line)
        # -----------------------------
        out = torch.sum(b * t, dim=-1) + self.bias  # (batch, out_dim)
        return out
    
class DeepONetFNN(nn.Module):
    """
    Simple DeepONet: y = <B(u), T(x)> + b
    - branch_in: dimension of the discretized input function (number of sensor points)
    - trunk_in : dimension of the evaluation location x
    - latent_dim: size of feature space where branch & trunk meet
    """
    def __init__(self, branch_in, trunk_in, hidden_dim=64, hidden_dim_trunk=128, latent_dim=128, out_dim=2, dropout=0.5):
        super().__init__()

        # Model dimensions
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        # Branch network
        self.branch = FNN(in_dim=branch_in, hidden_dim=hidden_dim, latent_dim=latent_dim, out_dim=out_dim, dropout=dropout)

        # Trunk network
        self.trunk = TrunkNet(in_dim=trunk_in, hidden_dim=hidden_dim_trunk, latent_dim=latent_dim, out_dim=out_dim)

        # Bias term (scalar per sample, but we learn a single scalar here)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, u_samples, x_loc):
        """
        u_samples : (batch, branch_in_dim)  - discretized function
        x_loc     : (batch, trunk_in_dim)   - evaluation coordinates
        returns   : (batch, output_dim)
        """
        b = self.branch(u_samples)   # (batch, out_dim * latent_dim)
        t = self.trunk(x_loc)    # (batch, out_dim * latent_dim)

        # Reshape to (batch, out_dim, latent_dim)
        #b = b.view(-1, self.out_dim, self.latent_dim)
        #t = t.view(-1, self.out_dim, self.latent_dim)

        # -----------------------------
        # Dot product (key line)
        # -----------------------------
        out = torch.sum(b * t, dim=-1) + self.bias  # (batch, out_dim)
        return out