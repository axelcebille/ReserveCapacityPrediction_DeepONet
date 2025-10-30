import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, GCN2Conv, PNAConv, GATv2Conv,
    GlobalAttention, global_mean_pool
)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        h2 = max(hidden_dim // 2, 8)
        h3 = max(hidden_dim // 4, 8)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, out_dim)
        )

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

        x = global_mean_pool(x, batch)
        return self.mlp(x)

class DeepONet(nn.Module):
    """
    Simple DeepONet: y = <B(u), T(x)> + b
    - branch_in: dimension of the discretized input function (number of sensor points)
    - trunk_in : dimension of the evaluation location x
    - latent_dim: size of feature space where branch & trunk meet
    """
    def __init__(self, branch_in, trunk_in, hidden_dim=64, out_dim=2):
        super().__init__()
        # Branch network
        self.branch = GCN(in_dim=branch_in, hidden_dim=hidden_dim, out_dim=out_dim)

        # Trunk network
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        # Bias term (scalar per sample, but we learn a single scalar here)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, u_samples, x_loc):
        """
        u_samples : (batch, branch_in)  - discretized function
        x_loc     : (batch, trunk_in)   - evaluation coordinates
        returns   : (batch, 1)
        """
        b_feat = self.branch(u_samples)         # (batch, latent_dim)
        t_feat = self.trunk(x_loc)              # (batch, latent_dim)
        out = b_feat * t_feat + self.bias
        return out