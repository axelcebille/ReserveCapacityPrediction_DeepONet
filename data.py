from utils import *
import h5py
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class SimulationData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = h5py.File(cfg.raw_data_path, 'r')
        self.section_groups = [key for key in self.data.keys()]
        self.features = self.initialize_features_container()

    def initialize_features_container(self):
        features_container = {}
        for section_group in self.section_groups:
            features_container[section_group] = {
                'reserve_capacities': [],
                'loading_protocols': [],
                'coordinate_features': [],
                'deviation_features': [],
                'curvature_features': [],
                'edge_index': None,
                'edge_features': None,
                'd': None,
                't_w': None,
                'b_f': None,
                't_f': None,
            }
        return features_container

    def extract_section_parameters(self):
        for section_group in self.section_groups:
            group = self.data[f'{section_group}/{self.cfg.length_group}/Collapse_consistent/{self.cfg.ratio_limit_group}']
            self.features[section_group]['d'] = group.attrs['d']
            self.features[section_group]['t_w'] = group.attrs['t_w']
            self.features[section_group]['b_f'] = group.attrs['b_f']
            self.features[section_group]['t_f'] = group.attrs['t_f']
        print('Section parameters extracted.')
        return None

    def generate_coordinate_features(self):
        print('Generating coordinate features.')
        for section_group in self.section_groups:
            for loading_protocol_number, loading_protocol in enumerate(self.cfg.loading_protocols):
                solid_nodes = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/Solid_node/center_node_coord']
                indices = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/indices']
                reserve_capacity = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/Reaction/reserve_capacity']
                group = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}']
                n_w = group.attrs['n_w']
                n_h = group.attrs['n_h']
                n_k = group.attrs['n_k']
                n_l = group.attrs['n_l']
                nk_eff = self.cfg.n_k_dict[n_k]

                translation_indices = get_translation_indices(n_w, nk_eff, n_h)
                
                for i in range(1, len(indices)):
                    stripes = []
                    for o in translation_indices:
                        stripe = solid_nodes[i, 0, (n_l+1)*o : (n_l+1)*(o+1), :]
                        z_min, z_max = stripe[:,2].min(), group.attrs['d']
                        z_interpolate = np.linspace(z_min, z_max, num=self.cfg.num_points_in_stripe)
                        stripes.append(interpolate_3d_line(stripe, z_interpolate))
                    self.features[section_group]['coordinate_features'].append(stripes)
                    self.features[section_group]['reserve_capacities'].append(reserve_capacity[indices[i]])
                    self.features[section_group]['loading_protocols'].append(loading_protocol_number)
        print('Coordinate features were generated.')
        return None

    def generate_deviation_features(self):
        print('Generating deviation features.')
        for section_group in self.section_groups:
            for loading_protocol_number, loading_protocol in enumerate(self.cfg.loading_protocols):
                solid_nodes = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/Solid_node/center_node_coord']
                indices = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/indices']
                group = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}']
                n_w = group.attrs['n_w']
                n_h = group.attrs['n_h']
                n_k = group.attrs['n_k']
                n_l = group.attrs['n_l']
                nk_eff = self.cfg.n_k_dict[n_k]

                translation_indices = get_translation_indices(n_w, nk_eff, n_h)

                for i in range(1, len(indices)):
                    stripes = []
                    for o in translation_indices:
                        stripe = solid_nodes[i, 0, (n_l+1)*o : (n_l+1)*(o+1), :]
                        z_min, z_max = stripe[:,2].min(), group.attrs['d']
                        z_interpolate = np.linspace(z_min, z_max, num=self.cfg.num_points_in_stripe)
                        cut_stripe = interpolate_3d_line(stripe, z_interpolate)
                        p0, p1 = cut_stripe[0], cut_stripe[-1]
                        stripe_aligned, R, t = align_line_to_z(cut_stripe, p0, p1)
                        p0[2] = 0
                        stripe_aligned += p0
                        stripes.append(stripe_aligned)
                    self.features[section_group]['deviation_features'].append(stripes)
        print('Deviation features generated.')
        return None

    def generate_curvature_features(self):
        print('Generating curvature features.')
        for section_group in self.section_groups:
            for loading_protocol_number, loading_protocol in enumerate(self.cfg.loading_protocols):
                solid_nodes = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/Solid_node/center_node_coord']
                indices = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}/indices']
                group = self.data[f'{section_group}/{self.cfg.length_group}/{loading_protocol}/{self.cfg.ratio_limit_group}']
                n_w = group.attrs['n_w']
                n_h = group.attrs['n_h']
                n_k = group.attrs['n_k']
                n_l = group.attrs['n_l']
                nk_eff = self.cfg.n_k_dict[n_k]

                translation_indices = get_translation_indices(n_w, nk_eff, n_h)

                for i in range(1, len(indices)):
                    stripes = []
                    for o in translation_indices:
                        stripe = solid_nodes[i, 0, (n_l+1)*o : (n_l+1)*(o+1), :]
                        z_min, z_max = stripe[:,2].min(), group.attrs['d']
                        z_interpolate = np.linspace(z_min, z_max, num=self.cfg.num_points_in_stripe)
                        cut_stripe = interpolate_3d_line(stripe, z_interpolate)
                        kappas = curvature_3d(cut_stripe)
                        stripes.append(kappas)
                    self.features[section_group]['curvature_features'].append(stripes)
        print('Curvature features generated.')
        return None

    def features_to_numpy(self):
        for section_group in self.section_groups:
            self.features[section_group]['reserve_capacities'] = np.array(self.features[section_group]['reserve_capacities'])
            self.features[section_group]['loading_protocols'] = np.array(self.features[section_group]['loading_protocols'])
            self.features[section_group]['coordinate_features'] = np.array(self.features[section_group]['coordinate_features'])
            self.features[section_group]['deviation_features'] = np.array(self.features[section_group]['deviation_features'])
            self.features[section_group]['curvature_features'] = np.array(self.features[section_group]['curvature_features'])
        return None
        
    def filter_reserve_capacities(self, min_abs_rc):
        for section_group in self.section_groups:
            reserve_capacities = self.features[section_group]['reserve_capacities']
            reserve_capacities = np.array(reserve_capacities)
            
            coordinate_features = self.features[section_group]['coordinate_features']
            coordinate_features = np.array(coordinate_features)
            deviation_features = self.features[section_group]['deviation_features']
            deviation_features = np.array(deviation_features)
            curvature_features = self.features[section_group]['curvature_features']
            curvature_features = np.array(curvature_features)

            condition1 = reserve_capacities[:, 0] > min_abs_rc
            condition2 = reserve_capacities[:, 1] > min_abs_rc
            combined_condition = np.logical_and(condition1, condition2)

            self.features[section_group]['reserve_capacities'] = self.features[section_group]['reserve_capacities'][combined_condition]
            self.features[section_group]['loading_protocols'] = self.features[section_group]['loading_protocols'][combined_condition]
            self.features[section_group]['coordinate_features'] = self.features[section_group]['coordinate_features'][combined_condition]
            self.features[section_group]['deviation_features'] = self.features[section_group]['deviation_features'][combined_condition]
            self.features[section_group]['curvature_features'] = self.features[section_group]['curvature_features'][combined_condition]            
        return None

    def make_id_split(self, n_clusters: int = 8, seed: int = 42):
        names = list(self.features.keys())
        rows = []
        for sg in names:
            d   = self.features[sg]['d']
            b_f = self.features[sg]['b_f']
            t_f = self.features[sg]['t_f']
            t_w = self.features[sg]['t_w']
            if None in (d, b_f, t_f, t_w):
                raise ValueError(f"Parameters not set for section {sg}. Run extract_section_parameters() first.")
            rows.append([d / t_w, b_f / t_f, d / b_f])
        ratios = np.asarray(rows, dtype=float)

        if len(names) < n_clusters:
            raise ValueError(f"Requested n_clusters={n_clusters}, but only {len(names)} beams available.")

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(ratios)
        except TypeError:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(ratios)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        for sg in names:
            self.features[sg]['In-distribution test'] = False

        for k in range(n_clusters):
            idxs = np.where(labels == k)[0]
            if idxs.size == 0:
                continue
            pts = ratios[idxs]
            d2c = np.linalg.norm(pts - centroids[k], axis=1)
            order = idxs[np.argsort(d2c)]
            sg_test = names[order[0]]
            self.features[sg_test]['In-distribution test'] = True

        print("In-distribution split flags have been set.")

    def make_ood_split_by_dt_w(self, n_test: int = 8):
        names = list(self.features.keys())
        rows = []
        for sg in names:
            d   = self.features[sg]['d']
            b_f = self.features[sg]['b_f']
            t_f = self.features[sg]['t_f']
            t_w = self.features[sg]['t_w']
            if None in (d, b_f, t_f, t_w):
                raise ValueError(f"Parameters not set for section {sg}. Run extract_section_parameters() first.")
            rows.append([d / t_w, b_f / t_f, d / b_f])
        ratios = np.asarray(rows, dtype=float)

        if len(names) < n_test:
            raise ValueError(f"Requested n_test={n_test}, but only {len(names)} beams available.")

        for sg in names:
            self.features[sg]['Out-of-distribution test'] = False

        order_desc = np.argsort(-ratios[:, 0])  # сортировка по убыванию d/t_w
        test_idx = order_desc[:n_test]
        for i in test_idx:
            sg_test = names[i]
            self.features[sg_test]['Out-of-distribution test'] = True

        print("Out-of-distribution split flags have been set.")

    def get_train_test(self, feature_key: str, test_flag_key: str = "In-distribution test"):
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        for sg in self.section_groups:
            if feature_key not in self.features[sg]:
                continue
            X = self.features[sg][feature_key]
            y = self.features[sg]['reserve_capacities']

            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)

            if X.size == 0 or y.size == 0:
                continue

            is_test = bool(self.features[sg].get(test_flag_key, False))
            if is_test:
                X_test_list.append(X)
                y_test_list.append(y)
            else:
                X_train_list.append(X)
                y_train_list.append(y)

        if len(X_train_list) == 0 or len(X_test_list) == 0:
            raise ValueError(
                f"No samples collected for feature '{feature_key}' with flag '{test_flag_key}'. "
                f"Make sure you ran the split setter and generated features."
            )

        X_train = np.concatenate(X_train_list, axis=0)
        X_test  = np.concatenate(X_test_list,  axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test  = np.concatenate(y_test_list,  axis=0)
        return X_train, X_test, y_train, y_test

    def plot_splits_3d(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        ratios = []
        for sg in self.section_groups:
            d   = self.features[sg]['d']
            b_f = self.features[sg]['b_f']
            t_f = self.features[sg]['t_f']
            t_w = self.features[sg]['t_w']
            if None in (d, b_f, t_f, t_w):
                raise ValueError(
                    f"Parameters not set for section '{sg}'. "
                    "Run extract_section_parameters() first."
                )
            ratios.append([float(d) / float(t_w), float(b_f) / float(t_f), float(d) / float(b_f)])
        ratios = np.asarray(ratios, dtype=float)

        def _plot_one(flag_key: str, fname: str):
            test_mask = np.array([bool(self.features[sg].get(flag_key, False)) for sg in self.section_groups], dtype=bool)
            train_mask = ~test_mask

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(ratios[train_mask, 0], ratios[train_mask, 1], ratios[train_mask, 2], c="blue", s=60, edgecolor="k", label="Train+Val")

            ax.scatter(ratios[test_mask, 0], ratios[test_mask, 1], ratios[test_mask, 2], c="red", s=60, edgecolor="k", label="Test")

            ax.set_xlabel(r"$d/t_w$")
            ax.set_ylabel(r"$b_f/t_f$")
            ax.set_zlabel(r"$d/b_f$")
            ax.legend(loc='best')
            plt.tight_layout()
            outpath = os.path.join(save_dir, fname)
            plt.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)

        _plot_one(flag_key="In-distribution test", fname="in_distribution.pdf")
        _plot_one(flag_key="Out-of-distribution test", fname="out_of_distribution.pdf")

    def plot_splits_2d(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        ratios = []
        for sg in self.section_groups:
            d   = self.features[sg]['d']
            b_f = self.features[sg]['b_f']
            t_f = self.features[sg]['t_f']
            t_w = self.features[sg]['t_w']
            if None in (d, b_f, t_f, t_w):
                raise ValueError(
                    f"Parameters not set for section '{sg}'. "
                    "Run extract_section_parameters() first."
                )
            # ratios.append([float(d) / float(t_w), float(b_f) / float(t_f)])
            ratios.append([float(d) / float(t_w), float(b_f) / (2.0 * float(t_f))])
        ratios = np.asarray(ratios, dtype=float)

        def _plot_one(flag_key: str, fname: str):
            test_mask = np.array([bool(self.features[sg].get(flag_key, False)) for sg in self.section_groups], dtype=bool)
            train_mask = ~test_mask

            fig, ax = plt.subplots(figsize=(5, 4))

            ax.scatter(ratios[train_mask, 0], ratios[train_mask, 1],
                       marker='o', c="blue", s=60, label="Train+Val")

            ax.scatter(ratios[test_mask, 0], ratios[test_mask, 1],
                       marker='x', c="red", s=60, label="Test")

            ax.set_xlabel(r"$d/t_w$")
            ax.set_ylabel(r"$b_f/ (2 t_f$)")
            ax.legend(loc='best')
            plt.tight_layout()

            outpath = os.path.join(save_dir, fname)
            plt.savefig(outpath, format="pdf", bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)

        _plot_one(flag_key="In-distribution test", fname="in_distribution_2d.pdf")
        _plot_one(flag_key="Out-of-distribution test", fname="out_of_distribution_2d.pdf")

    def build_edge_features(self):
        edge_index = build_beam_edge_index(num_points=self.cfg.num_points_in_stripe,
                                           num_cols=self.cfg.num_stripes,
                                           neighbor_pairs=self.cfg.neighbor_stripes,
                                           undirected=True)
        
        for section_group in self.section_groups:
            coords = self.features[section_group].get('coordinate_features', None)
            
            coords = np.asarray(coords, dtype=float)
            n_samples = coords.shape[0]

            dist_list = []
            for k in range(n_samples):
                sample_pos = coords[k]
    
                # sanity checks
                if sample_pos.shape[0] == self.cfg.num_points_in_stripe and sample_pos.shape[1] == self.cfg.num_stripes:
                    sample_pos = np.transpose(sample_pos, (1, 0, 2))
                if sample_pos.shape[0] != self.cfg.num_stripes or sample_pos.shape[1] != self.cfg.num_points_in_stripe or sample_pos.shape[2] != 3:
                    raise ValueError(
                        f"Unexpected sample_pos shape for section '{section_group}': {sample_pos.shape}. "
                        f"Expected ({self.cfg.num_stripes}, {self.cfg.num_points_in_stripe}, 3) or its transpose."
                    )

                edge_dists = edge_vectors_and_lengths(sample_pos, edge_index)
                dist_list.append(edge_dists)

            lengths = np.stack(dist_list, axis=0)
            self.features[section_group]['edge_index'] = edge_index
            self.features[section_group]['edge_norms'] = lengths
        return None

    def save_features(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.features, f)

    def load_features(self, path):
        with open(path, "rb") as f:
            self.features = pickle.load(f)