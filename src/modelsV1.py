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
        self.mlp_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, h2), nn.ReLU(),
            nn.Linear(h2, h3), nn.ReLU()
        )
        self.output_layer = nn.Linear(h3, out_dim)

    def forward(self, data, return_latent=False):
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
        # Feed through MLP
        latent = self.mlp_layers(x)
        out = self.output_layer(latent)

        if return_latent:
            return latent  
        return out
    
class TrunkNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=2):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.l3 = nn.Linear(hidden_dim*2, out_dim)

    def forward(self, x, return_latent=False):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        latent = F.relu(x)
        out = self.l3(latent)

        if return_latent:
            return latent 
        return out

class DeepONet(nn.Module):
    """
    Simple DeepONet: y = <B(u), T(x)> + b
    - branch_in: dimension of the discretized input function (number of sensor points)
    - trunk_in : dimension of the evaluation location x
    - latent_dim: size of feature space where branch & trunk meet
    """
    def __init__(self, branch_in, trunk_in, hidden_dim=64, hidden_dim_trunk=128, out_dim=2):
        super().__init__()
        # Branch network
        self.branch = GCN(in_dim=branch_in, hidden_dim=hidden_dim, out_dim=out_dim)

        # Trunk network
        self.trunk = TrunkNet(in_dim=trunk_in, hidden_dim=hidden_dim_trunk, out_dim=out_dim)

        # Bias term (scalar per sample, but we learn a single scalar here)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, u_samples, x_loc):
        """
        u_samples : (batch, branch_in)  - discretized function
        x_loc     : (batch, trunk_in)   - evaluation coordinates
        returns   : (batch, 1)
        """
        if x_loc.dim() == 1:
            x_loc = x_loc.unsqueeze(-1)
        b_feat = self.branch(u_samples)         # (batch, latent_dim)
        t_feat = self.trunk(x_loc)              # (batch, latent_dim)

        # Ensure both have same shape for elementwise multiplication
        if b_feat.shape[1] != t_feat.shape[1]:
            raise ValueError(
                f"Shape mismatch: branch={b_feat.shape}, trunk={t_feat.shape}. "
                f"Both must have same out_dim for elementwise product."
            )
        out = b_feat * t_feat + self.bias
        return out