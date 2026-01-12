import numpy as np
import torch
from src.data import SimulationData
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data, Batch
from src.utils import compute_elastic_moment, compute_area_section, compute_moment_of_inertia, compute_local_slenderness_ratio

def compute_normalization_stats(data: SimulationData,model="rc"):
    all_trunk_inputs = []
    for col in data.features.keys():
        if model == "rc":
            trunk_input = get_trunk_input_rc(data, col)
            all_trunk_inputs.append(torch.tensor(trunk_input, dtype=torch.float32))
        elif model == "rm":
            for i in range(len(data.features[col]["displacements"])):
                trunk_input = get_trunk_input_rm(data, col, i)
                #trunk_input = trunk_input.view(-1, 1)  # or unsqueeze(-1)
                all_trunk_inputs.append(trunk_input)

    all_trunk_inputs = torch.stack(all_trunk_inputs)
    mean = all_trunk_inputs.mean(dim=0)
    std = all_trunk_inputs.std(dim=0)
    std[std == 0] = 1.0  # avoid division by zero
    return mean, std

def compute_graph_norm_stats(data: SimulationData,model="rc"):
    xs = []
    for col in data.features.keys():
        num_t = len(data.features[col]["reserve_capacities"])
        for t in range(num_t):
            g = get_graph_timestep_i(data, col, t)
            if hasattr(g, "x"):
                xs.append(g.x)
    xs = torch.cat(xs, dim=0)  # concatenate all node features
    mean = xs.mean(dim=0)
    std = xs.std(dim=0)
    std[std == 0] = 1.0  # prevent division by zero
    return mean, std

## getter functions ##

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

def get_trunk_input_rc(data:SimulationData , column_type:str):
    data_col = data.features[column_type] # 214 timesteps

    d = data_col["d"]
    t_w = data_col["t_w"]
    b_f = data_col["b_f"]
    t_f = data_col["t_f"]

    return torch.tensor(np.array((d, t_w, b_f, t_f)),dtype=torch.float32)

def get_trunk_input_rmV1(data:SimulationData , column_type:str, i:int):
    data_col = data.features[column_type] # 214 timesteps

    d = data_col["d"]
    t_w = data_col["t_w"]
    b_f = data_col["b_f"]
    t_f = data_col["t_f"]
    horizontal_disp = data_col["horizontal_displacements"][i] 
    if i < len(data_col["horizontal_displacements"])-1:
        horizontal_disp_next = data_col["horizontal_displacements"][i+1] 
    else:
        horizontal_disp_next = horizontal_disp

    return torch.tensor(np.array((d, t_w, b_f, t_f, horizontal_disp, horizontal_disp_next)),dtype=torch.float32)

def get_time_window(sequence, i, n_time_step):
    window = []
    first = sequence[0]
    for k in range(n_time_step):
        idx = i - (n_time_step - 1 - k)
        if idx < 0:
            window.append(first)
        else:
            window.append(sequence[idx])
    return np.stack(window, axis=0)

def get_trunk_input_rm(data: SimulationData, column_type: str, i: int, n_time_step=1):
    data_col = data.features[column_type]  # 214 time steps

    # Static geometric features
    d  = data_col["d"]
    t_w = data_col["t_w"]
    b_f = data_col["b_f"]
    t_f = data_col["t_f"]

    I = compute_moment_of_inertia(d, t_w, b_f, t_f)
    A = compute_area_section(d, t_w, b_f, t_f)
    lsr = compute_local_slenderness_ratio(8000, d, t_w, b_f, t_f)

    # All displacements as array/list
    disp_all = data_col["displacements"]

    # Current and next displacement
    disp = disp_all[i]
    if i < len(disp_all) - 1:
        disp_next = disp_all[i + 1]
    else:
        disp_next = disp  # end of sequence
    delta_disp = disp_next - disp
    # --------------------------------------------------------------
    # Build sequence of n_time_step previous displacements
    # --------------------------------------------------------------
    disp_seq = []
    for k in range(n_time_step):
        idx = (i + 1) - (n_time_step - k)   # indexes from (i+1 - n) ... (i)
        if idx < 0:
            # Not enough history -> use displacement at i
            disp_seq.append(disp)
        else:
            disp_seq.append(disp_all[idx])

    #disp_seq.append(horizontal_disp_next)
    disp_seq = [disp, disp_next, delta_disp]
    #disp_seq.append(delta_disp)
    # Convert to numpy array
    disp_seq = np.array(disp_seq, dtype=np.float32)  # shape: (n_time_step,)

    # --------------------------------------------------------------
    # Build final feature vector
    # --------------------------------------------------------------
    # Static geometry + current disp + next disp + sequence history
    features = np.concatenate([
        np.array([d, t_w, b_f, t_f], dtype=np.float32),
        disp_seq.flatten()
    ], axis=0)
    return torch.tensor(features, dtype=torch.float32)

def get_reserve_capacity_i(data:SimulationData , column_type:str, i:int):
    data_col = data.features[column_type] # 214 timesteps
    reserve_capacity = data_col["reserve_capacities"] # (214,2)

    return torch.tensor(reserve_capacity[i], dtype=torch.float32)

def get_resisting_moment_i(data:SimulationData , column_type:str, i:int):
    data_col = data.features[column_type] # 214 timesteps
    if i < len(data_col["resisting_moments"])-1:
        resisting_moment = data_col["resisting_moments"][i+1]
    else:
        resisting_moment = data_col["resisting_moments"][i]

    return torch.tensor(resisting_moment, dtype=torch.float32)


## for multiple column types ##

class MultiColDataset(Dataset):
    def __init__(self, data: SimulationData, trunk_mean=None, trunk_std=None,
                 graph_mean=None, graph_std=None, model="rc", mom_norm=True):
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
        self.model = model
        self.trunk_mean = trunk_mean
        self.trunk_std = trunk_std
        self.graph_mean = graph_mean
        self.graph_std = graph_std

        self.mom_norm = mom_norm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        col, t = sample["col"], sample["t"]

        graph = get_graph_timestep_i(self.data, col, t)
        trunk_input_rc = get_trunk_input_rc(self.data, col)
        target_capacity = get_reserve_capacity_i(self.data, col, t)

        trunk_input_rm = get_trunk_input_rm(self.data, col, t)
        target_moment = get_resisting_moment_i(self.data, col, t)

        if self.model == "rc":
            target = target_capacity  # reserve capacity
            trunk_input = trunk_input_rc
        elif self.model == "rm":
            trunk_input = trunk_input_rm
            d, t_w, b_f, t_f = trunk_input[0].item(), trunk_input[1].item(), trunk_input[2].item(), trunk_input[3].item()
            M_el = compute_elastic_moment(d, t_w, b_f, t_f,fy=355)
            if self.mom_norm:
                target = target_moment/M_el
            else:
                target = target_moment
        else:
            raise ValueError(f"Unknown model type: {self.model}")

        # Normalize trunk input
        if self.trunk_mean is not None and self.trunk_std is not None:
            trunk_input = (trunk_input - self.trunk_mean) / self.trunk_std

        # Normalize graph node features
        if hasattr(graph, "x") and self.graph_mean is not None and self.graph_std is not None:
            graph.x = (graph.x - self.graph_mean) / self.graph_std

        return graph, trunk_input, target, sample["in_flag"], sample["out_flag"], col
    