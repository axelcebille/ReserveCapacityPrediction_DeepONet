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
from src.utils import compute_mean_target, compute_std_target
from src.utils import random_Rt, transform_graph, transform_batch_independent, reserve_capacity_mape

from torch_geometric.data import Data, Batch

from src.dataset import MultiColDataset
from src.dataset import compute_normalization_stats, compute_graph_norm_stats
from src.models import DeepONet, DeepONetFNN

from torch_geometric.utils import unbatch
from torch_geometric.data import Batch
from scipy.spatial.transform import Rotation as Rot

def custom_loss(preds, targets, criterion, col, target_mean, target_std, lambda_reg=0.1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    loss = criterion(preds, targets) # loss of normlaized values

    preds = preds * (target_std + 1e-12) + target_mean  # denormalize predictions
    same_type_mask = torch.tensor(
        [a == b for a, b in zip(col[:-1], col[1:])],
        dtype=torch.float32
    ).unsqueeze(1)
    delta = preds[1:] - preds[:-1]  # shape (63, 2)
    positive_mask = (delta > 0).float()      # shape (N-1, 2)
    positive_mask, same_type_mask = positive_mask.to(device), same_type_mask.to(device)
    delta_masked = delta * same_type_mask * positive_mask

    loss_reg = delta_masked.mean()

    return loss + lambda_reg * loss_reg

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

seed = 42
set_seed(seed)

def train(model,train_loader,optimizer,criterion,scheduler,target_mean,target_std,device,transfo_flag,lambda_reg=0.1,target_normalize=True):
        model.train()
        total_loss = 0.0

        for graphs, trunk_inputs, targets, _, _, col in train_loader:
            graphs = graphs.to(device)
            graphs_trans = transform_batch_independent(graphs).to(device)
            if transfo_flag:
                graphs_used = graphs_trans
            else:
                graphs_used = graphs
            trunk_inputs = trunk_inputs.to(device)
            targets = targets.to(device)

            if target_normalize:
                targets = (targets - target_mean) / (target_std + 1e-12) 

            preds = model(graphs_used, trunk_inputs)
            #loss = custom_loss(preds, targets, criterion, col, target_mean, target_std, lambda_reg=lambda_reg)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        scheduler.step()  # update learning rate
    
        return total_loss/len(train_loader)

def validation(model,val_loader,scheduler,criterion,target_mean,target_std,device,lambda_reg=0.1,target_normalize=True):
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for graphs, trunk_inputs, targets, _, _, col in val_loader:
            graphs = graphs.to(device)
            trunk_inputs = trunk_inputs.to(device)
            targets = targets.to(device)

            if target_normalize:
                targets = (targets - target_mean) / (target_std + 1e-12) 

            preds = model(graphs, trunk_inputs)
            #val_loss += custom_loss(preds, targets, criterion, col, target_mean, target_std, lambda_reg=lambda_reg).item()
            val_loss += criterion(preds, targets).item()

    avg_val_loss = val_loss / len(val_loader)
    #scheduler.step(avg_val_loss)

    return avg_val_loss

def test(model,test_loader,criterion,target_mean,target_std,device,lambda_reg=0.1,target_normalize=True):
    # --- Test ---
    model.eval()
    val_loss = 0.0
    test_loss = 0.0

    y_pred_all, y_pred_transfo_all, y_true_all = [], [], []
    with torch.no_grad():
        for graphs, trunk_inputs, targets, _, _, col in test_loader:
                graphs, trunk_inputs, targets = graphs.to(device), trunk_inputs.to(device), targets.to(device)
                graphs_trans = transform_batch_independent(graphs).to(device)

                #if target_normalize:
                #    targets = (targets - target_mean) / (target_std + 1e-12) 

                preds = model(graphs, trunk_inputs)
                preds_transfo = model(graphs_trans, trunk_inputs)

                test_loss += criterion(preds, targets).item()
                #test_loss += custom_loss(preds, targets, criterion, col, target_mean, target_std, lambda_reg=lambda_reg).item()


                if target_normalize:
                    preds = preds * (target_std + 1e-12) + target_mean
                    preds_transfo = preds_transfo * (target_std + 1e-12) + target_mean

                y_pred_all.append(preds.detach().squeeze().cpu().numpy())
                y_pred_transfo_all.append(preds_transfo.detach().squeeze().cpu().numpy())
                y_true_all.append(targets.detach().squeeze().cpu().numpy())

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_pred_transfo_all = np.concatenate(y_pred_transfo_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    mape_loss = reserve_capacity_mape(y_true_all, y_pred_all,model="rc")
    mape_loss_transfo = reserve_capacity_mape(y_true_all, y_pred_transfo_all,model="rc")
    avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss, mape_loss, mape_loss_transfo

def train_model_rc(data, model, device, batch_size, n_epochs, transfo_flag, patience, lambda_reg=0.1, target_normalize=False, features_normalize=True, weights_name="best_deeponet_model_RM"):
    
    # Compute normalization stats from training data
    trunk_mean, trunk_std = compute_normalization_stats(data, model="rc")
    graph_mean, graph_std = compute_graph_norm_stats(data, model="rc")

    # --- Create multi-column dataset ---
    if features_normalize:
        dataset = MultiColDataset(data, trunk_mean=trunk_mean, trunk_std=trunk_std, graph_mean=graph_mean, graph_std=graph_std, model="rc")
    else:
        dataset = MultiColDataset(data, model="rc")

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
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_samples)
    val_dataset   = Subset(dataset, val_samples)
    test_dataset   = Subset(dataset, test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # computes the std and mean of the targets (resisting moment)   
    #stats_target = compute_stats_by_column_moment(dataset)
    target_mean = compute_mean_target(train_dataset, model="rc")
    target_std = compute_std_target(train_dataset, target_mean, model="rc")
    target_mean,target_std = target_mean.to(device),target_std.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=300,      # number of epochs until the LR reaches its minimum
        eta_min=1e-7    # minimum learning rate
    )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #optimizer,
    #mode='min',        # Reduce LR when monitored quantity decreases
    #factor=0.5,        # Multiply LR by 0.5
    #patience=5,        # Wait 3 epochs before reducing
    #threshold=1e-4,    # Minimum change to qualify as improvement
    #min_lr=1e-9)       # No LR below this valu)

    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.SmoothL1Loss()   # Huber loss


    # early stop params
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):

        avg_train_loss = train(model,train_loader,optimizer,criterion,scheduler,target_mean,target_std,device,transfo_flag,lambda_reg=lambda_reg,target_normalize=target_normalize)

        avg_val_loss = validation(model,val_loader,scheduler,criterion,target_mean,target_std,device,lambda_reg=lambda_reg,target_normalize=target_normalize)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_test_loss, mape_loss, mape_transfo_loss = test(model,test_loader,criterion,target_mean,target_std,device,target_normalize=target_normalize)
            print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Test: {avg_test_loss:.4f} | MAPE: {mape_loss:.4f} | MAPE transfo: {mape_transfo_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'model_weights/{weights_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Final OOD evaluation ---
    model.load_state_dict(torch.load(f'model_weights/{weights_name}.pth'))
    model.eval()
    print("Final Test evaluation:")
    avg_test_loss, mape_loss, mape_transfo_loss = test(model,test_loader,criterion,target_mean,target_std,device,lambda_reg=lambda_reg,target_normalize=target_normalize)
    print(f"Final Test Loss: {avg_test_loss:.4f} | Final MAPE: {mape_loss:.4f} | MAPE transfo: {mape_transfo_loss:.4f}")
    print(f"Trained DeepONet with: batch_size:{batch_size}, n_epochs:{n_epochs}, affine transfo.:{transfo_flag}, early stopping patience:{patience}, target normalized:{target_normalize}, features normalized:{features_normalize}")
        
    return mape_loss, mape_transfo_loss



#config = Config()
#data = SimulationData(config)
data = SimulationDataLoader()
data.load_features('ReserveCapacityPrediction_DeepONet/data/features.pkl')


# --- Model, optimizer, loss ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = DeepONet(branch_in=7, trunk_in=4, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=2, dropout=0.2).to(device)
model2 = DeepONetFNN(branch_in=7, trunk_in=4, hidden_dim=264,hidden_dim_trunk=264, latent_dim=128, out_dim=2, dropout=0.2).to(device)
model3 = DeepONet(branch_in=7, trunk_in=13, hidden_dim=1000,hidden_dim_trunk=500, latent_dim=264, out_dim=1, dropout=0.4).to(device)
model_list = [model2]

batch_sizes = [300,400,500]
# --- Training loop ---
batch_size = 64
n_epochs = 1000
transfo_flag = True  # whether to use transformed graphs for training
# early stop params
patience = 40
target_normalize = True
features_normalize = True
lambda_reg = 0.0001 # for now not used 

print("Predicting Reserve Capacity...")
mape_losses = []
mape_losses_transfo = []
i = 1
for model in model_list:
    if transfo_flag:
        weights_name = f"RC_best_deeponet_modelFNN"
    else: 
        weights_name = f"RC(no_transfo)_best_deeponet_modelFNN"

    mape_loss, mape_transfo_loss = train_model_rc(data, model, device, batch_size, n_epochs, transfo_flag, patience, lambda_reg, target_normalize,
                                                   features_normalize, weights_name=weights_name)

    mape_losses_transfo.append(mape_transfo_loss)
    mape_losses.append(mape_loss)
    i+=1

print("mape losses of all models tested", mape_losses)
print("mape losses of all models tested (with affine transfo.)", mape_losses_transfo)
