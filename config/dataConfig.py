collapseConsistentDataPath = 'beam_rc085.hdf5'
cyclicMonotonicDataPath = 'beam_rc85_cyclic_monotonic_loads.hdf5'

class Config:
    raw_data_path = '/scratch/izar/cebille/beam_rc085.hdf5'
    length_group = '8000'
    loading_protocols = ['Collapse_consistent', 'Monotonic', 'Symmetric']
    ratio_limit_group = '50'
    n_k_dict = {2: 2, 4: 3, 6: 4}
    num_points_in_stripe = 10
    num_stripes = 7
    neighbor_stripes = [(0,2), (1,2), (2,3), (3,4), (4,5), (4,6)]# [(0,1), (1,2), (1,3), (3,5), (5,4), (5,6)]
    min_abs_reserve_capacity = 0.8
    n_test = 8
    device = 'cuda:3'