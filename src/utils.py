import numpy as np
from scipy.interpolate import interp1d
import random
import torch
import os
from scipy.spatial.transform import Rotation as Rot
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.signal import argrelextrema


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def get_translation_indices(n_w: int, nk_eff: int, n_h: int) -> list[int]:
    # -------------------------- 7 base offsets ---------------------------
    corner1   = 0                                          # C1
    corner2   = n_w*2 + nk_eff*2 + 1                       # C2
    part1     = n_w*2 + nk_eff*2 + 2                       # P1
    part2     = n_w*2 + nk_eff*3 + 2 + int(n_h/2)          # P2
    part3     = n_w*2 + nk_eff*4 + n_h + 2                 # P3
    corner3   = n_w*2 + nk_eff*4 + n_h + 3                 # C3
    corner4   = n_w*4 + nk_eff*6 + n_h + 4                 # C4

    base = [corner1, part1, corner2 , part2, corner3, part3, corner4]

    # --------------------- 6 explicit mid-offsets ------------------------
    # (rounded to the nearest integer -- use //2 if you prefer floor)
    #mid_C1_P1 = round(corner1 + (corner2-corner1)/ 4)
    #mid_C2_P1 = round(corner1 + (corner2-corner1)*3/ 4)
    #mid_P1_P2 = round((part1   + part2)  / 2)
    #mid_P2_P3 = round((part2   + part3)  / 2)
    #mid_P3_C3 = round(corner3 + (corner4-corner3)/ 4)
    #mid_P3_C4 = round(corner3 + (corner4-corner3)*3/ 4)

    mids = []

    # ---------------- combine, deduplicate, sort ------------------------
    offsets = sorted(set(base + mids))   # 13 unique integers max
    return offsets

def interpolate_3d_line(array, eval_points):
    interp_x = interp1d(array[:, 2], array[:, 0], kind='linear', fill_value='extrapolate')
    interp_y = interp1d(array[:, 2], array[:, 1], kind='linear', fill_value='extrapolate')
    x_interpolated = interp_x(eval_points)
    y_interpolated = interp_y(eval_points)
    z_interpolated = eval_points  # This is the z axis, we take it directly since it's what we interpolate on
    return np.stack((x_interpolated, y_interpolated, z_interpolated), axis=1)

def rotation_matrix_rodrigues(a, theta):
    """a — единичный вектор оси (3,), theta — угол (рад). Возвращает R (3,3)."""
    ax, ay, az = a
    K = np.array([[    0, -az,  ay],
                  [   az,   0, -ax],
                  [  -ay,  ax,   0]])
    I = np.eye(3)
    R = I + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    return R

def align_line_to_z(points, A0, A1, eps=1e-9):
    """
    points: (N,3) массив исходных точек
    A0, A1: две точки (3,) на прямой, задающей целевую ось (из A0 к A1)
    Возвращает: points_aligned (N,3), R (3,3), t (3,)
      где p' = R @ (p - t),  t = A0
    """
    points = np.asarray(points, float)
    A0 = np.asarray(A0, float)
    A1 = np.asarray(A1, float)

    # Шаг 1: перевод так, чтобы A0 стал в нуле
    t = A0.copy()
    P = points - t
    v = A1 - A0
    nv = np.linalg.norm(v)
    if nv < eps:
        raise ValueError("A0 и A1 совпадают — нельзя определить направление прямой.")
    v = v / nv

    k = np.array([0.0, 0.0, 1.0])

    # Шаг 2: матрица поворота R, чтобы v -> k
    dot = float(np.clip(np.dot(v, k), -1.0, 1.0))

    if 1.0 - dot < 1e-12:
        # v уже направлен как k
        R = np.eye(3)
    elif dot + 1.0 < 1e-12:
        # v противоположен k: поворот на 180° вокруг любой оси ⟂ v
        # выбираем ось, ортогональную v
        tmp = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(tmp, v)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        a = tmp - np.dot(tmp, v)*v
        a /= np.linalg.norm(a)
        R = rotation_matrix_rodrigues(a, np.pi)
    else:
        a = np.cross(v, k)
        na = np.linalg.norm(a)
        a /= na
        theta = np.arccos(dot)
        R = rotation_matrix_rodrigues(a, theta)

    P_aligned = (R @ P.T).T
    return P_aligned, R, t

def curvature_3d(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    points: (N, 3) array of 3D coordinates along a single curve (ordered).
    returns: (N,) curvature at each point.
    """
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    if N < 3:
        return np.zeros(N)

    # Arc-length parameter s
    diffs = np.diff(points, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.zeros(N)
    s[1:] = np.cumsum(seglen)

    # First derivatives wrt arc length
    dx_ds = np.gradient(points[:, 0], s)
    dy_ds = np.gradient(points[:, 1], s)
    dz_ds = np.gradient(points[:, 2], s)
    r1 = np.column_stack([dx_ds, dy_ds, dz_ds])

    # Second derivatives wrt arc length
    d2x_ds2 = np.gradient(dx_ds, s)
    d2y_ds2 = np.gradient(dy_ds, s)
    d2z_ds2 = np.gradient(dz_ds, s)
    r2 = np.column_stack([d2x_ds2, d2y_ds2, d2z_ds2])

    # Curvature
    cross = np.cross(r1, r2)
    num = np.linalg.norm(cross, axis=1)
    den = np.linalg.norm(r1, axis=1)**3 + eps
    kappa = num / den
    return kappa

def rm_mape(y_true_all, y_pred_all):
    max_index = argrelextrema(y_true_all, np.greater)
    y_true = y_true_all[max_index]
    y_pred = y_pred_all[max_index]
    return np.mean(np.abs((y_true - y_pred) / y_true))

def reserve_capacity_mape(y_true_all, y_pred_all,model="rc"):
    if model=="rc":
        y_true_all = np.squeeze(y_true_all)
        y_pred_all = np.squeeze(y_pred_all)
        mape_pos = np.mean(np.abs((y_true_all[:, 0] - y_pred_all[:, 0]) / y_true_all[:, 0]))
        mape_neg = np.mean(np.abs((y_true_all[:, 1] - y_pred_all[:, 1]) / y_true_all[:, 1]))
        mape = (mape_pos + mape_neg) / 2
    elif model=="rm":
        y_true_all = y_true_all.squeeze()
        y_pred_all = y_pred_all.squeeze()
        mape = rm_mape(y_true_all, y_pred_all)
        #mape = np.mean(np.abs((y_true_all - y_pred_all) / (np.abs(y_true_all) + 1e-8)))
    return mape

def build_adj2d_from_pairs(num_cols: int, neighbor_pairs, undirected: bool = True) -> np.ndarray:
    A = np.zeros((num_cols, num_cols), dtype=np.int64)
    for i, j in neighbor_pairs:
        A[i, j] = 1
        if undirected:
            A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A

def build_beam_edge_index(num_points: int,
                          num_cols: int = 7,
                          neighbor_pairs=None,
                          undirected: bool = True) -> torch.Tensor:
    if neighbor_pairs is None:
        neighbor_pairs = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)]

    adj2d = build_adj2d_from_pairs(num_cols, neighbor_pairs, undirected=undirected)

    edges = []
    for z in range(num_points):
        for i in range(num_cols):
            idx = z * num_cols + i
            for j in range(num_cols):
                if adj2d[i, j]:
                    nbr = z * num_cols + j
                    if idx != nbr:
                        edges.append([idx, nbr])

        if z + 1 < num_points:
            for i in range(num_cols):
                a = z * num_cols + i
                b = (z + 1) * num_cols + i
                edges.append([a, b])
                if undirected:
                    edges.append([b, a])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def edge_vectors_and_lengths(sample_pos: np.ndarray, edge_index: torch.Tensor):
    if sample_pos.shape[0] in (7,):
        sample_pos = np.transpose(sample_pos, (1, 0, 2))

    num_points, num_cols, _ = sample_pos.shape
    pos = torch.from_numpy(sample_pos).permute(0, 1, 2).reshape(-1, 3)

    row, col = edge_index
    diffs = pos[row] - pos[col]
    dists = diffs.norm(dim=1, keepdim=True)
    return dists.numpy()

## New functions ##

## mean/std calculations ##

def compute_mean_target(train_dataset,model="rc"):
    if model=="rc":
        sum = torch.tensor([0,0],dtype=torch.float32)
    elif model=="rm":
        sum = torch.tensor([0],dtype=torch.float32)
    else:
        raise ValueError("Model type not recognized. Use 'rc' or 'rm'.")
    for data in train_dataset:
        target = data[2]
        sum += target

    mean = sum/len(train_dataset)
    return mean

def compute_std_target(train_dataset,mean,model="rc"):
    if model=="rc":
        sum = torch.tensor([0,0],dtype=torch.float32)
    elif model=="rm":
        sum = torch.tensor([0],dtype=torch.float32)
    else:
        raise ValueError("Model type not recognized. Use 'rc' or 'rm'.")
    for data in train_dataset:
        target = data[2]
        sum += (target - mean)**2

    std = torch.sqrt(sum/len(train_dataset))
    return std

def moments_scaler(data): 
    moments = []
    for key in data.features:
        moments.append(data.features[key]["resisting_moments"])

    moments = np.concatenate(moments, axis=0)
    moments = np.array(moments).reshape(-1,1)

    scaler = StandardScaler()
    moments_scaled = scaler.fit_transform(moments)
    
    return scaler


def compute_stats_by_column_moment(dataset):
    # store all targets grouped by column value
    groups = defaultdict(list)

    for branch_input, trunk_input, target, _, _, col in dataset:
        groups[col].append(target)

    # compute mean and std for each col group
    stats = {}

    for col_value, targets in groups.items():
        arr = np.vstack(targets)  # shape (N_samples, target_dim)
        stats[col_value] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0)
        }
    return stats

def means_and_stds(stats_target, col):
    target_means = []
    target_stds = []

    for c in col:
        target_mean = stats_target[c]["mean"]
        target_std = stats_target[c]["std"]
        target_means.append(target_mean)
        target_stds.append(target_std)

    target_means = np.array(target_means)
    target_stds = np.array(target_stds)

    return torch.tensor(target_means), torch.tensor(target_stds)

def compute_moment_by_column(data):
    groups = defaultdict(list)

    cols = data.features.keys()
    for col in cols:

        d, t_w, b_f, t_f = data.features[col]["d"], data.features[col]["t_w"], data.features[col]["b_f"], data.features[col]["t_f"]

        # compute elastic moment
        M_el = compute_elastic_moment(d, t_w, b_f, t_f, fy=355)

        # store BOTH the original target and the elastic moment
        groups[col].append(float(M_el))

    return groups

def elastic_moments_batch(moments, col):
    elastic_moments = []

    for c in col:
        elastic_moment = moments[c]
        elastic_moments.append(elastic_moment)

    elastic_moments = np.array(elastic_moments)

    return torch.tensor(elastic_moments)
## affine graph transformation functions ## 

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

## Section properties computations ## 

def compute_area_section(d, t_w, b_f, t_f):
    A_w = (d - 2 * t_f) * t_w
    A_f = 2 * b_f * t_f
    A_total = A_w + A_f
    return A_total

def compute_moment_of_inertia(d, t_w, b_f, t_f):
    I_w = (t_w * (d - 2 * t_f)**3) / 12
    I_f = 2 * ((b_f * t_f**3) / 12 + b_f * t_f * ((d / 2 - t_f / 2)**2))
    I_total = I_w + I_f
    return I_total

def compute_local_slenderness_ratio(L_k, d, t_w, b_f, t_f):
    I = compute_moment_of_inertia(d, t_w, b_f, t_f)
    A = compute_area_section(d, t_w, b_f, t_f)
    r = np.sqrt(I / A)
    lsr = L_k / r
    return lsr

def compute_elastic_moment(d, t_w, b_f, t_f, fy):
    I = compute_moment_of_inertia(d, t_w, b_f, t_f)
    Z = I / (d / 2)
    M_elastic = Z * fy
    return M_elastic

def compute_ultimate_moment(d, t_w, b_f, t_f, fy):
    W = b_f*t_f*(d-t_f) + (d-2*t_f)**2/4*t_w
    M_ult = W*fy
    return M_ult