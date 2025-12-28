import numpy as np
import random
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from config.dataConfig import Config
from src.data import SimulationData, SimulationDataLoader
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  
from torch.utils.data import random_split
from src.utils import compute_mean_target, compute_std_target, compute_stats_by_column_moment,means_and_stds, compute_moment_by_column, elastic_moments_batch, moments_scaler

from torch_geometric.data import Data, Batch

from src.dataset import MultiColDataset
from src.dataset import compute_normalization_stats, compute_graph_norm_stats
from src.models import DeepONet, DeepONetFNN

from torch_geometric.utils import unbatch
from torch_geometric.data import Batch
from scipy.spatial.transform import Rotation as Rot

from src.utils import reserve_capacity_mape
from torch.utils.data import Subset

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
       
def test(model,test_loader,criterion,target_mean,target_std,device,scaler_moment,target_normalize=True):
    # --- Test ---
    model.eval()
    val_loss = 0.0
    test_loss = 0.0

    y_pred_all, y_pred_transfo_all, y_true_all = [], [], []
    with torch.no_grad():
        for graphs, trunk_inputs, targets, _, _, col in test_loader:
                graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
                graphs_trans = transform_batch_independent(graphs).to(device)
                targets = targets.reshape(-1,1)

                if target_normalize:
                    #targets = scaler_moment.transform(targets.cpu().numpy())
                    #targets = torch.from_numpy(targets).float().to(device)
                    targets = (targets - target_mean) / (target_std + 1e-12) 

                preds = model(graphs, trunk_inputs)
                preds_transfo = model(graphs_trans, trunk_inputs)
                targets = targets.reshape(preds.shape)

                y_pred_all.append(preds.detach().squeeze().cpu().numpy())
                y_pred_transfo_all.append(preds_transfo.detach().squeeze().cpu().numpy())
                y_true_all.append(targets.detach().squeeze().cpu().numpy())

                test_loss += criterion(preds, targets).item()

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_pred_transfo_all = np.concatenate(y_pred_transfo_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    mape_loss = reserve_capacity_mape(y_true_all, y_pred_all,model="rm")
    mape_loss_transfo = reserve_capacity_mape(y_true_all, y_pred_transfo_all,model="rm")
    avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss, mape_loss, mape_loss_transfo

seed = 42
set_seed(seed)

#config = Config()
#data = SimulationData(config)
data = SimulationDataLoader()
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')


def test_model_rm(data, model, device, transfo_flag, target_normalize=False, features_normalize=True, weights_name="best_deeponet_model_RM"):
    
    # Compute normalization stats from training data
    trunk_mean, trunk_std = compute_normalization_stats(data, model="rm")
    graph_mean, graph_std = compute_graph_norm_stats(data, model="rm")

    scaler_moment = moments_scaler(data)

    # --- Create multi-column dataset ---
    if features_normalize:
        dataset = MultiColDataset(data, trunk_mean=trunk_mean, trunk_std=trunk_std, graph_mean=graph_mean, graph_std=graph_std, model="rm")
    else:
        dataset = MultiColDataset(data, model="rm")

    # Separate by flags
    test_set = "in_flag" # "out_flag"
    train_val_samples, test_samples = [], []
    for i, sample in enumerate(dataset.samples):
        if sample[test_set]:
            test_samples.append(i)
        else:
            train_val_samples.append(i)

    train_size = int(0.9 * len(train_val_samples))
    val_size = len(train_val_samples) - train_size

    # Split dataset
    train_samples, val_samples = random_split(train_val_samples, [train_size, val_size])
    # Create subset datasets
    train_dataset = Subset(dataset, train_samples)
    val_dataset = Subset(dataset, val_samples)
    test_dataset = Subset(dataset, test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # computes the std and mean of the targets (resisting moment)   
    #stats_target = compute_stats_by_column_moment(dataset)
    target_mean = compute_mean_target(train_dataset, model="rm")
    target_std = compute_std_target(train_dataset, target_mean, model="rm")
    target_mean,target_std = target_mean.to(device),target_std.to(device)
    # computes the elastic moment of each column type
    elastic_moments = compute_moment_by_column(data)
        
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.SmoothL1Loss()   # Huber loss

    # --- Final OOD evaluation ---
    model.load_state_dict(torch.load(f'model_weights/{weights_name}.pth'))
    model.eval()
    print("Final Test evaluation:")
    avg_test_loss, mape_loss, mape_transfo_loss = test(model,test_loader,criterion,target_mean,target_std,device,scaler_moment,target_normalize=target_normalize)
    print(f"Final Test Loss: {avg_test_loss:.4f} | Final MAPE: {mape_loss:.4f} | MAPE transfo: {mape_transfo_loss:.4f}")
    return mape_loss, mape_transfo_loss


# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = DeepONetFNN(branch_in=7, trunk_in=13, hidden_dim=264,hidden_dim_trunk=264, latent_dim=264, out_dim=1, dropout=0.2).to(device) # best model
model2 = DeepONet(branch_in=7, trunk_in=13, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=1, dropout=0.2).to(device) # best model
#model2 = DeepONet(branch_in=7, trunk_in=4, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=2, dropout=0.2).to(device)
model3 = DeepONetFNN(branch_in=7, trunk_in=4, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=2, dropout=0.2).to(device)
#model3 = DeepONetFNN(branch_in=7, trunk_in=13, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=1, dropout=0.2).to(device)
model_list = [model1, model2]

batch_size = 64
transfo_flag = True  # whether to use transformed graphs for training
target_normalize = True
features_normalize = True

weights_name1 = "RM_best_deeponet_modelFNN"
weights_name2 = "RM_best_deeponet_model"
weights_name3 = "RC_best_deeponet_modelFNN"
weight_names = [weights_name1, weights_name2]

mape_losses = []
mape_losses_transfo = []

for i, name in enumerate(weight_names):
    print(f"Testing {name} Model...")

    mape_loss, mape_transfo_loss = test_model_rm(data, model_list[i], device, transfo_flag, target_normalize, 
                               features_normalize, weights_name=name)
    
    mape_losses.append(mape_loss)
    mape_losses_transfo.append(mape_transfo_loss)

print("mape losses of all models tested", mape_losses)
print("mape losses of all models tested (with affine transfo.)", mape_losses_transfo)