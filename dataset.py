import numpy as np
import torch
from data import SimulationData
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data, Batch



def get_graph_timestep_i(data:SimulationData , column_type:str, i:int):
    data_col = data.features[column_type] # 214 timesteps
    edge_index = data_col["edge_index"] # torch.Size([2, 246])
    edge_norms = torch.tensor(data_col["edge_norms"][i],dtype=torch.float32) # (246, 1) timestep i

    node_coords = torch.tensor(data_col["coordinate_features"][i].reshape(-1, 3), dtype=torch.float32) # (70,3) timestep i
    node_dev = torch.tensor(data_col["deviation_features"][i].reshape(-1, 3), dtype=torch.float32) # (70,3) timestep i
    node_curv = torch.tensor(data_col["curvature_features"][i].reshape(-1,1), dtype=torch.float32) # (70,1) timestep i

    node_feat = torch.cat([node_coords, node_dev, node_curv], dim=1) # (70,7) timestep i

    graph = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_norms)
    
    return graph

def get_trunk_input_col(data:SimulationData , column_type:str):
    data_col = data.features[column_type] # 214 timesteps

    d = data_col["d"]
    t_w = data_col["t_w"]
    b_f = data_col["b_f"]
    t_f = data_col["t_f"]

    return torch.tensor(np.array((d, t_w, b_f, t_f)),dtype=torch.float32)

def get_reserve_capacity_i(data:SimulationData , column_type:str, i:int):
    data_col = data.features[column_type] # 214 timesteps
    reserve_capacity = data_col["reserve_capacities"] # (214,2)

    return torch.tensor(reserve_capacity[i], dtype=torch.float32)

## for single column type ##
class ColDataset(Dataset):
    def __init__(self, data: SimulationData, column_type: str):
        super().__init__()
        self.data = data
        self.column_type = column_type

        # total number of timesteps for this column type
        self.num_timesteps = len(data.features[column_type]["reserve_capacities"])

    def __len__(self):
        return self.num_timesteps

    def __getitem__(self, idx):
        # Build one graph at timestep idx
        graph = get_graph_timestep_i(self.data, self.column_type, idx)

        # Trunk input (same for all timesteps of this column)
        trunk_input = get_trunk_input_col(self.data, self.column_type)

        # Target (reserve capacity)
        reserve_capacity = get_reserve_capacity_i(self.data, self.column_type, idx)

        return graph, trunk_input, reserve_capacity
    

## for multiple column types ##

class MultiColDataset(Dataset):
    def __init__(self, data: SimulationData):
        self.samples = []
        column_types = list(data.features.keys())
        for col in column_types:
            num_t = len(data.features[col]["reserve_capacities"])
            in_flags = data.features[col]["In-distribution test"]
            out_flags = data.features[col]["Out-of-distribution test"]
            for t in range(num_t):
                self.samples.append({
                    "col": col,
                    "t": t,
                    "in_flag": in_flags,
                    "out_flag": out_flags
                })
        self.data = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        col, t = sample["col"], sample["t"]

        graph = get_graph_timestep_i(self.data, col, t)
        trunk_input = get_trunk_input_col(self.data, col)
        target = get_reserve_capacity_i(self.data, col, t)

        return graph, trunk_input, target, sample["in_flag"], sample["out_flag"]
    
    
class MultiColSequenceDataset(Dataset):
    def __init__(self, data: SimulationData):
        self.data = data
        self.column_types = list(data.features.keys())

    def __len__(self):
        return len(self.column_types)

    def __getitem__(self, idx):
        col = self.column_types[idx]
        num_t = len(self.data.features[col]["reserve_capacities"])
        graphs = [get_graph_timestep_i(self.data, col, t) for t in range(num_t)]
        trunk_input = get_trunk_input_col(self.data, col)
        targets = torch.stack([
            get_reserve_capacity_i(self.data, col, t)
            for t in range(num_t)
        ])  # shape (T, 2)

        in_flag = self.data.features[col]["In-distribution test"]
        out_flag = self.data.features[col]["Out-of-distribution test"]

        return graphs, trunk_input, targets, in_flag, out_flag    
