import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState

from Main.data.utils import preprocess_features, preprocess_adj, adj_to_edge_index

# Flag to control whether to use PyG cached loaders (much faster after first load)
USE_PYG_CACHE = True

# ============================================================================
# Dataset Name Mappings
# ============================================================================

# Canonical (friendly) names -> path names (for pkl files)
DATASET_PATH_NAMES = {
    'BA-Shapes': 'syn1',
    'BA-House': 'syn1',
    'BA-Community': 'syn2',
    'Tree-Cycles': 'syn3',
    'Tree-Cycles-Noise': 'syn3_noise',
    'Tree-Cycles-Noise0.5': 'syn3_noise0p5',
    'Tree-Cycles-Noise0.1': 'syn3_noise0p1',
    'Tree-Cycles-Noise0.05': 'syn3_noise0p05',
    'Tree-Cycles-Uniform0.5': 'syn3_uniform0p5',
    'Tree-Grid': 'syn4',
    'BA-2Motifs': 'ba2',
    'BA2Grid-Longa': 'ba2grid_longa',
    'BA-HouseGrid': 'ba_house_grid',
    'BA-HouseGrid-wb60-m3': 'ba_house_grid_wb60_m3',
    'BA-HouseANDGrid': 'ba_house_and_grid',
    'BA-HouseANDGrid-wb15-m1': 'ba_house_and_grid_wb15_m1',
    'BA-HouseORGrid': 'ba_house_or_grid',
    'MUTAG': 'MUTAG',
    'PubMed': 'pubmed',
    'Cora': 'cora',
    'PROTEINS': 'PROTEINS',
    'Infection4': 'infection4',
    'Infection4-one-hot': 'infection4_one_hot',
    'Infection3': 'infection3',
    'Infection3-one-hot': 'infection3_one_hot',
    'Infection3-one-hot-binary': 'infection3_one_hot_binary',
    'Infection3-one-hot-binary-Noise': 'infection3_one_hot_binary_noise',
    'Infection3-one-hot-binary-Bernoulli4': 'infection3_one_hot_binary_bernoulli4',
    'PyG-Infection-ER-L3': 'pyg_infection_er_l3',
    'PyG-Infection-ER-L3-Split43': 'pyg_infection_er_l3_split43',
    'PyG-Infection-ER-L3-Binary-Split43': 'pyg_infection_er_l3_binary_split43',
    'PyG-Infection-BA-L3': 'pyg_infection_ba_l3',
    'PyG-Infection-BA-L3-Split43': 'pyg_infection_ba_l3_split43',
    'PyG-Infection-BA-L3-Binary-Split43': 'pyg_infection_ba_l3_binary_split43',
    'MarkerPath-Exact3': 'marker_path_exact3',
    'InfectedSteiner-3': 'infected_steiner_3',
    'InfectedPair-Unique3': 'infected_pair_unique3',
    'InfectedPair-Dist3': 'infected_pair_dist3',
    'InfectedPair-ER-Dist3-Split43': 'infected_pair_er_dist3_split43',
    'InfectedPair-BA-Dist3-Split43': 'infected_pair_ba_dist3_split43',
    'InfectedPair-ER-L3-Multiclass-Split43': 'infected_pair_er_l3_multiclass_split43',
    'InfectedPair-BA-L3-Multiclass-Split43': 'infected_pair_ba_l3_multiclass_split43',
    'InfectedPair-ER-Dist5-Split43': 'infected_pair_er_dist5_split43',
    'InfectedPair-BA-Dist5-Split43': 'infected_pair_ba_dist5_split43',
    'InfectedPair-Dist3-Noise': 'infected_pair_dist3_noise',
    'InfectedPair-Dist3-Noise0.05': 'infected_pair_dist3_noise0p05',
    'InfectedPair-Dist6': 'infected_pair_dist6',
    'InfectedPair-MultiPath3': 'infected_pair_multipath3',
    'InfectedHouse': 'infected_house',
    'InfectedPentagon': 'infected_pentagon',
    'RandomInfectedTriCenter-3': 'random_infected_tricenter_3',
}

# Legacy/internal names -> canonical names (for backward compatibility)
LEGACY_TO_CANONICAL = {
    'syn1': 'BA-Shapes',
    'ba-house': 'BA-House',
    'ba_house': 'BA-House',
    'syn2': 'BA-Community',
    'syn3': 'Tree-Cycles',
    'syn3_noise': 'Tree-Cycles-Noise',
    'syn3-noise': 'Tree-Cycles-Noise',
    'tree_cycles_noise': 'Tree-Cycles-Noise',
    'tree-cycles-noise': 'Tree-Cycles-Noise',
    'Tree-Cycles-Noise': 'Tree-Cycles-Noise',
    'syn3_noise0p5': 'Tree-Cycles-Noise0.5',
    'syn3_noise_0p5': 'Tree-Cycles-Noise0.5',
    'syn3-noise0p5': 'Tree-Cycles-Noise0.5',
    'syn3-noise0.5': 'Tree-Cycles-Noise0.5',
    'tree_cycles_noise0p5': 'Tree-Cycles-Noise0.5',
    'tree-cycles-noise0p5': 'Tree-Cycles-Noise0.5',
    'Tree-Cycles-Noise0p5': 'Tree-Cycles-Noise0.5',
    'Tree-Cycles-Noise0.5': 'Tree-Cycles-Noise0.5',
    'syn3_noise0p1': 'Tree-Cycles-Noise0.1',
    'syn3_noise_0p1': 'Tree-Cycles-Noise0.1',
    'syn3-noise0p1': 'Tree-Cycles-Noise0.1',
    'syn3-noise0.1': 'Tree-Cycles-Noise0.1',
    'tree_cycles_noise0p1': 'Tree-Cycles-Noise0.1',
    'tree-cycles-noise0p1': 'Tree-Cycles-Noise0.1',
    'Tree-Cycles-Noise0p1': 'Tree-Cycles-Noise0.1',
    'Tree-Cycles-Noise0.1': 'Tree-Cycles-Noise0.1',
    'syn3_noise0p05': 'Tree-Cycles-Noise0.05',
    'syn3_noise_0p05': 'Tree-Cycles-Noise0.05',
    'syn3-noise0p05': 'Tree-Cycles-Noise0.05',
    'syn3-noise0.05': 'Tree-Cycles-Noise0.05',
    'tree_cycles_noise0p05': 'Tree-Cycles-Noise0.05',
    'tree-cycles-noise0p05': 'Tree-Cycles-Noise0.05',
    'Tree-Cycles-Noise0p05': 'Tree-Cycles-Noise0.05',
    'Tree-Cycles-Noise0.05': 'Tree-Cycles-Noise0.05',
    'syn3_uniform0p5': 'Tree-Cycles-Uniform0.5',
    'syn3_uniform_0p5': 'Tree-Cycles-Uniform0.5',
    'syn3-uniform0p5': 'Tree-Cycles-Uniform0.5',
    'syn3-uniform0.5': 'Tree-Cycles-Uniform0.5',
    'tree_cycles_uniform0p5': 'Tree-Cycles-Uniform0.5',
    'tree-cycles-uniform0p5': 'Tree-Cycles-Uniform0.5',
    'Tree-Cycles-Uniform0p5': 'Tree-Cycles-Uniform0.5',
    'Tree-Cycles-Uniform0.5': 'Tree-Cycles-Uniform0.5',
    'Tree-Cycles-Unifrom0p5': 'Tree-Cycles-Uniform0.5',
    'Tree-Cycles-Unifrom0.5': 'Tree-Cycles-Uniform0.5',
    'syn4': 'Tree-Grid',
    'ba2': 'BA-2Motifs',
    'BA2Grid': 'BA2Grid-Longa',
    'BA2grid': 'BA2Grid-Longa',
    'BA-2Grid': 'BA2Grid-Longa',
    'BA-2grid': 'BA2Grid-Longa',
    'ba2grid': 'BA2Grid-Longa',
    'ba2grid_longa': 'BA2Grid-Longa',
    'ba_house_grid': 'BA-HouseGrid',
    'ba-house-grid': 'BA-HouseGrid',
    'BA-House-Grid': 'BA-HouseGrid',
    'ba_house_grid_wb60_m3': 'BA-HouseGrid-wb60-m3',
    'ba-house-grid-wb60-m3': 'BA-HouseGrid-wb60-m3',
    'BA-HouseGrid-wb60-m3': 'BA-HouseGrid-wb60-m3',
    'BA-House-Grid-wb60-m3': 'BA-HouseGrid-wb60-m3',
    'BA-HouseAndGrid': 'BA-HouseANDGrid',
    'BA-HouseAndGRID': 'BA-HouseANDGrid',
    'BA-House-AND-Grid': 'BA-HouseANDGrid',
    'BA-House-And-Grid': 'BA-HouseANDGrid',
    'ba_house_and_grid': 'BA-HouseANDGrid',
    'ba-house-and-grid': 'BA-HouseANDGrid',
    'ba_house_and_grid_wb15_m1': 'BA-HouseANDGrid-wb15-m1',
    'ba-house-and-grid-wb15-m1': 'BA-HouseANDGrid-wb15-m1',
    'BA-HouseAndGrid-wb15-m1': 'BA-HouseANDGrid-wb15-m1',
    'BA-HouseANDGrid-wb15-m1': 'BA-HouseANDGrid-wb15-m1',
    'BA-House-AND-Grid-wb15-m1': 'BA-HouseANDGrid-wb15-m1',
    'BA-House-And-Grid-wb15-m1': 'BA-HouseANDGrid-wb15-m1',
    'BA-HouseOrGrid': 'BA-HouseORGrid',
    'BA-HouseORGRID': 'BA-HouseORGrid',
    'BA-House-OR-Grid': 'BA-HouseORGrid',
    'BA-House-Or-Grid': 'BA-HouseORGrid',
    'ba_house_or_grid': 'BA-HouseORGrid',
    'ba-house-or-grid': 'BA-HouseORGrid',
    'pubmed': 'PubMed',
    'cora': 'Cora',
    'proteins': 'PROTEINS',
    'infection': 'Infection4',
    'infection_one_hot': 'Infection4-one-hot',
    'infection-one-hot': 'Infection4-one-hot',
    'infection4': 'Infection4',
    'infection4_one_hot': 'Infection4-one-hot',
    'infection4-one-hot': 'Infection4-one-hot',
    'infection3': 'Infection3',
    'infection3_one_hot': 'Infection3-one-hot',
    'infection3-one-hot': 'Infection3-one-hot',
    'infection3_one_hot_binary': 'Infection3-one-hot-binary',
    'infection3-one-hot-binary': 'Infection3-one-hot-binary',
    'infection3_one_hot_binary_noise': 'Infection3-one-hot-binary-Noise',
    'infection3-one-hot-binary-noise': 'Infection3-one-hot-binary-Noise',
    'Infection3-one-hot-binary-Noise': 'Infection3-one-hot-binary-Noise',
    'infection3_one_hot_binary_bernoulli4': 'Infection3-one-hot-binary-Bernoulli4',
    'infection3-one-hot-binary-bernoulli4': 'Infection3-one-hot-binary-Bernoulli4',
    'Infection3-one-hot-binary-Bernoulli4': 'Infection3-one-hot-binary-Bernoulli4',
    'pyg_infection_er_l3': 'PyG-Infection-ER-L3',
    'pyg-infection-er-l3': 'PyG-Infection-ER-L3',
    'pyg_infection_er_l3_split43': 'PyG-Infection-ER-L3-Split43',
    'pyg-infection-er-l3-split43': 'PyG-Infection-ER-L3-Split43',
    'pyg_infection_er_l3_binary_split43': 'PyG-Infection-ER-L3-Binary-Split43',
    'pyg-infection-er-l3-binary-split43': 'PyG-Infection-ER-L3-Binary-Split43',
    'pyg_infection_ba_l3': 'PyG-Infection-BA-L3',
    'pyg-infection-ba-l3': 'PyG-Infection-BA-L3',
    'pyg_infection_ba_l3_split43': 'PyG-Infection-BA-L3-Split43',
    'pyg-infection-ba-l3-split43': 'PyG-Infection-BA-L3-Split43',
    'pyg_infection_ba_l3_binary_split43': 'PyG-Infection-BA-L3-Binary-Split43',
    'pyg-infection-ba-l3-binary-split43': 'PyG-Infection-BA-L3-Binary-Split43',
    'marker_path_exact3': 'MarkerPath-Exact3',
    'marker-path-exact3': 'MarkerPath-Exact3',
    'infected_steiner_3': 'InfectedSteiner-3',
    'infected-steiner-3': 'InfectedSteiner-3',
    'infected_pair_unique3': 'InfectedPair-Unique3',
    'infected-pair-unique3': 'InfectedPair-Unique3',
    'infected_pair_dist3': 'InfectedPair-Dist3',
    'infected-pair-dist3': 'InfectedPair-Dist3',
    'infected_pair_er_dist3_split43': 'InfectedPair-ER-Dist3-Split43',
    'infected-pair-er-dist3-split43': 'InfectedPair-ER-Dist3-Split43',
    'infected_pair_ba_dist3_split43': 'InfectedPair-BA-Dist3-Split43',
    'infected-pair-ba-dist3-split43': 'InfectedPair-BA-Dist3-Split43',
    'infected_pair_er_l3_multiclass_split43': 'InfectedPair-ER-L3-Multiclass-Split43',
    'infected-pair-er-l3-multiclass-split43': 'InfectedPair-ER-L3-Multiclass-Split43',
    'infected_pair_ba_l3_multiclass_split43': 'InfectedPair-BA-L3-Multiclass-Split43',
    'infected-pair-ba-l3-multiclass-split43': 'InfectedPair-BA-L3-Multiclass-Split43',
    'infected_pair_er_dist5_split43': 'InfectedPair-ER-Dist5-Split43',
    'infected-pair-er-dist5-split43': 'InfectedPair-ER-Dist5-Split43',
    'infected_pair_ba_dist5_split43': 'InfectedPair-BA-Dist5-Split43',
    'infected-pair-ba-dist5-split43': 'InfectedPair-BA-Dist5-Split43',
    'infected_pair_dist3_noise': 'InfectedPair-Dist3-Noise',
    'infected-pair-dist3-noise': 'InfectedPair-Dist3-Noise',
    'InfectedPair-Dist3-Noise': 'InfectedPair-Dist3-Noise',
    'infected_pair_dist3_noise0p05': 'InfectedPair-Dist3-Noise0.05',
    'infected_pair_dist3_noise_0p05': 'InfectedPair-Dist3-Noise0.05',
    'infected-pair-dist3-noise0p05': 'InfectedPair-Dist3-Noise0.05',
    'infected-pair-dist3-noise0.05': 'InfectedPair-Dist3-Noise0.05',
    'InfectedPair-Dist3-Noise0p05': 'InfectedPair-Dist3-Noise0.05',
    'InfectedPair-Dist3-Noise0.05': 'InfectedPair-Dist3-Noise0.05',
    'infected_pair_dist6': 'InfectedPair-Dist6',
    'infected-pair-dist6': 'InfectedPair-Dist6',
    'infected_pair_multipath3': 'InfectedPair-MultiPath3',
    'infected-pair-multipath3': 'InfectedPair-MultiPath3',
    'infected_pair_multi_path3': 'InfectedPair-MultiPath3',
    'infected-pair-multi-path3': 'InfectedPair-MultiPath3',
    'infected_house': 'InfectedHouse',
    'infected-house': 'InfectedHouse',
    'infected_pentagon': 'InfectedPentagon',
    'infected-pentagon': 'InfectedPentagon',
    'random_infected_tricenter_3': 'RandomInfectedTriCenter-3',
    'random-infected-tricenter-3': 'RandomInfectedTriCenter-3',
    'Infection': 'Infection4',
    'Infection-one-hot': 'Infection4-one-hot',
}

# All supported datasets
NODE_DATASETS = [
    'BA-Shapes',
    'BA-House',
    'BA-Community',
    'Tree-Cycles',
    'Tree-Cycles-Noise',
    'Tree-Cycles-Noise0.5',
    'Tree-Cycles-Noise0.1',
    'Tree-Cycles-Noise0.05',
    'Tree-Cycles-Uniform0.5',
    'Tree-Grid',
    'PubMed',
    'Cora',
    'Infection4',
    'Infection4-one-hot',
    'Infection3',
    'Infection3-one-hot',
    'Infection3-one-hot-binary',
    'Infection3-one-hot-binary-Noise',
    'Infection3-one-hot-binary-Bernoulli4',
    'PyG-Infection-ER-L3',
    'PyG-Infection-ER-L3-Split43',
    'PyG-Infection-ER-L3-Binary-Split43',
    'PyG-Infection-BA-L3',
    'PyG-Infection-BA-L3-Split43',
    'PyG-Infection-BA-L3-Binary-Split43',
]
GRAPH_DATASETS = [
    'BA-2Motifs',
    'BA2Grid-Longa',
    'BA-HouseGrid',
    'BA-HouseGrid-wb60-m3',
    'BA-HouseANDGrid',
    'BA-HouseANDGrid-wb15-m1',
    'BA-HouseORGrid',
    'MarkerPath-Exact3',
    'InfectedSteiner-3',
    'InfectedPair-Unique3',
    'InfectedPair-Dist3',
    'InfectedPair-ER-Dist3-Split43',
    'InfectedPair-BA-Dist3-Split43',
    'InfectedPair-ER-L3-Multiclass-Split43',
    'InfectedPair-BA-L3-Multiclass-Split43',
    'InfectedPair-ER-Dist5-Split43',
    'InfectedPair-BA-Dist5-Split43',
    'InfectedPair-Dist3-Noise',
    'InfectedPair-Dist3-Noise0.05',
    'InfectedPair-Dist6',
    'InfectedPair-MultiPath3',
    'InfectedHouse',
    'InfectedPentagon',
    'RandomInfectedTriCenter-3',
    'MUTAG',
    'PROTEINS',
]
ALL_DATASETS = NODE_DATASETS + GRAPH_DATASETS

from Main.data.pyg_infection_seed_registry import seeded_infection_names
for _canonical, _path, _, _ in seeded_infection_names():
    DATASET_PATH_NAMES[_canonical] = _path
    LEGACY_TO_CANONICAL[_path] = _canonical
    NODE_DATASETS.append(_canonical)
    ALL_DATASETS.append(_canonical)

SYNTHETIC_NODE_BASE_NAMES = {'syn1', 'syn2', 'syn3', 'syn4'}
SYNTHETIC_NODE_VARIANTS = {
    'syn3_noise': {
        'base_path_name': 'syn3',
        'feature_noise_std': 1.0,
        'feature_noise_seed': 1000003,
    },
    'syn3_noise0p5': {
        'base_path_name': 'syn3',
        'feature_noise_std': 0.5,
        'feature_noise_seed': 1000003,
    },
    'syn3_noise0p1': {
        'base_path_name': 'syn3',
        'feature_noise_std': 0.1,
        'feature_noise_seed': 1000003,
    },
    'syn3_noise0p05': {
        'base_path_name': 'syn3',
        'feature_noise_std': 0.05,
        'feature_noise_seed': 1000003,
    },
    'syn3_uniform0p5': {
        'base_path_name': 'syn3',
        'feature_noise_std': 0.5,
        'feature_noise_seed': 1000003,
        'feature_noise_distribution': 'uniform',
    },
}


def _collapse_ba_house_labels(labels) -> np.ndarray:
    """Merge BA-Shapes motif labels {1,2,3} into a single positive class."""
    labels_np = np.array(labels)
    if labels_np.ndim > 1:
        labels_np = np.argmax(labels_np, axis=1)
    labels_np = labels_np.astype(np.int64, copy=False).reshape(-1)
    return (labels_np != 0).astype(np.int64, copy=False)


def _synthetic_node_variant(path_name: str):
    if path_name in SYNTHETIC_NODE_VARIANTS:
        return dict(SYNTHETIC_NODE_VARIANTS[path_name])
    if path_name in SYNTHETIC_NODE_BASE_NAMES:
        return {
            'base_path_name': path_name,
            'feature_noise_std': 0.0,
            'feature_noise_seed': 1000003,
            'feature_noise_distribution': 'gaussian',
        }
    raise KeyError(path_name)


def _apply_synthetic_feature_noise(
    features: np.ndarray,
    std: float,
    seed: int,
    distribution: str = 'gaussian',
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if float(std) <= 0.0:
        return features
    rng = np.random.RandomState(int(seed))
    distribution = str(distribution).lower()
    if distribution == 'gaussian':
        noise = rng.normal(loc=0.0, scale=float(std), size=features.shape).astype(np.float32)
    elif distribution == 'uniform':
        noise = rng.uniform(low=-float(std), high=float(std), size=features.shape).astype(np.float32)
    else:
        raise ValueError(f"Unsupported feature noise distribution: {distribution}")
    return (features + noise).astype(np.float32, copy=False)


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to canonical (friendly) format."""
    if name in DATASET_PATH_NAMES:
        return name
    if name in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[name]
    return name  # Return as-is if not found


def get_path_name(dataset: str) -> str:
    """Get file path name for a dataset (for pkl files)."""
    canonical = normalize_dataset_name(dataset)
    return DATASET_PATH_NAMES.get(canonical, canonical)


def load_graph_dataset(_dataset, shuffle=True):
    """Load a graph dataset and optionally shuffle it.

    :param _dataset: Which dataset to load. Only 'BA-2Motifs'/'ba2' supported here.
                     For MUTAG/PROTEINS, use PyG loader (load_dataset with use_pyg_cache=True).
    :param shuffle: Boolean. Whether to shuffle the loaded dataset.
    :returns: np.array
    """
    # Normalize to path name for pkl loading
    path_name = get_path_name(_dataset)
    
    # Load the chosen dataset from the pickle file.
    if not True:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
        with open(path, 'rb') as fin:
            adjs, features, labels = pkl.load(fin)
    else:
        raise ValueError(f"Unknown graph dataset: {_dataset}. Available: BA-2Motifs. For MUTAG/PROTEINS, use PyG loader.")

    n_graphs = adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
        indices = prng.permutation(indices)

    # Create shuffled data
    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]

    # Create masks
    train_indices = np.arange(0, int(n_graphs*0.8))
    val_indices = np.arange(int(n_graphs*0.8), int(n_graphs*0.9))
    test_indices = np.arange(int(n_graphs*0.9), n_graphs)
    train_mask = np.full((n_graphs), False, dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.full((n_graphs), False, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.full((n_graphs), False, dtype=bool)
    test_mask[test_indices] = True

    # Transform to edge index
    edge_index = adj_to_edge_index(adjs)

    return edge_index, features, labels, train_mask, val_mask, test_mask


def _load_node_dataset(_dataset):
    """Load a node dataset.

    :param _dataset: Which dataset to load. Path name (syn1, syn2, syn3, syn4).
    :returns: np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
    labels = y_train
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]

    return adj, features, labels, train_mask, val_mask, test_mask


def load_dataset(
    _dataset,
    skip_preproccessing=False,
    shuffle=True,
    use_pyg_cache=None,
    normalize_features: bool = True,
):
    """High level function which loads the dataset.

    Supported datasets:
    - Node classification: 'BA-Shapes', 'BA-House', 'BA-Community', 'Tree-Cycles', 'Tree-Cycles-Noise', 'Tree-Cycles-Noise0.5', 'Tree-Cycles-Noise0.1', 'Tree-Cycles-Noise0.05', 'Tree-Cycles-Uniform0.5', 'Tree-Grid', 'PubMed', 'Cora',
      'Infection4', 'Infection4-one-hot', 'Infection3', 'Infection3-one-hot', 'Infection3-one-hot-binary', 'Infection3-one-hot-binary-Noise', 'Infection3-one-hot-binary-Bernoulli4'
    - Graph classification: 'BA-2Motifs', 'BA2Grid-Longa', 'MarkerPath-Exact3', 'InfectedSteiner-3', 'InfectedPair-Unique3', 'InfectedPair-Dist3', 'InfectedPair-Dist3-Noise', 'InfectedPair-Dist3-Noise0.05', 'InfectedPair-Dist6', 'InfectedPair-MultiPath3', 'InfectedHouse', 'InfectedPentagon', 'RandomInfectedTriCenter-3', 'BA-HouseGrid', 'BA-HouseGrid-wb60-m3',
      'BA-HouseANDGrid', 'BA-HouseANDGrid-wb15-m1', 'MUTAG', 'PROTEINS'
    
    Note: MUTAG, PROTEINS, PubMed, Cora, and Infection3/Infection4 variants are loaded through PyG-compatible loaders.

    :param _dataset: Dataset name (supports both friendly and internal names)
    :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
    :param shuffle: Should the returned dataset be shuffled or not.
    :param use_pyg_cache: Whether to use PyG cached loaders (faster after first load). 
                          If None, uses global USE_PYG_CACHE setting.
    :returns: multiple np.arrays
    """
    # Determine whether to use PyG cache
    if use_pyg_cache is None:
        use_pyg_cache = USE_PYG_CACHE
    
    # Try PyG cached loader first (much faster for repeated loads)
    if use_pyg_cache:
        try:
            from Main.data.pyg_datasets import load_dataset_pyg
            print(f"Loading {_dataset} dataset (PyG cached)")
            return load_dataset_pyg(_dataset, shuffle=shuffle, normalize_features=normalize_features)
        except Exception as e:
            print(f"PyG cache failed for {_dataset}, falling back to original loader: {e}")
    
    # Normalize to canonical name then get path name for legacy loader
    canonical_name = normalize_dataset_name(_dataset)
    path_name = get_path_name(_dataset)
    
    # Fall back to original loader
    print(f"Loading {canonical_name} dataset")
    if path_name in SYNTHETIC_NODE_BASE_NAMES or path_name in SYNTHETIC_NODE_VARIANTS:  # Load node_dataset
        variant = _synthetic_node_variant(path_name)
        adj, features, labels, train_mask, val_mask, test_mask = _load_node_dataset(str(variant['base_path_name']))
        if normalize_features:
            preprocessed_features = preprocess_features(features).astype('float32')
        else:
            preprocessed_features = np.asarray(features, dtype=np.float32)
        preprocessed_features = _apply_synthetic_feature_noise(
            preprocessed_features,
            float(variant.get('feature_noise_std', 0.0)),
            int(variant.get('feature_noise_seed', 1000003)),
            str(variant.get('feature_noise_distribution', 'gaussian')),
        )
        if skip_preproccessing:
            graph = adj
        else:
            graph = preprocess_adj(adj)[0].astype('int64').T
        labels = np.argmax(labels, axis=1)
        if canonical_name == 'BA-House':
            labels = _collapse_ba_house_labels(labels)
        return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
    elif path_name == 'ba2':
        return load_graph_dataset(path_name, shuffle)
    elif path_name in (
        'pubmed',
        'cora',
        'MUTAG',
        'PROTEINS',
        'ba2grid_longa',
        'ba_house_grid',
        'ba_house_grid_wb60_m3',
        'ba_house_and_grid',
        'ba_house_and_grid_wb15_m1',
        'ba_house_or_grid',
        'marker_path_exact3',
        'infected_steiner_3',
        'infected_pair_unique3',
        'infected_pair_dist3',
        'infected_pair_er_dist3_split43',
        'infected_pair_ba_dist3_split43',
        'infected_pair_er_dist5_split43',
        'infected_pair_ba_dist5_split43',
        'infected_pair_dist3_noise',
        'infected_pair_dist3_noise0p05',
        'infected_pair_dist6',
        'infected_pair_multipath3',
        'infected_house',
        'infected_pentagon',
        'random_infected_tricenter_3',
        'infection4',
        'infection4_one_hot',
        'infection3',
        'infection3_one_hot',
        'infection3_one_hot_binary',
        'infection3_one_hot_binary_noise',
        'infection3_one_hot_binary_bernoulli4',
        'pyg_infection_er_l3',
        'pyg_infection_ba_l3',
        'pyg_infection_ba_l3_split43',
    ) or path_name in {path for _, path, _, _ in seeded_infection_names()}:
        from Main.data.pyg_datasets import load_dataset_pyg
        return load_dataset_pyg(canonical_name, shuffle=shuffle, normalize_features=normalize_features)
    else:
        raise ValueError(
            f"Unknown dataset: {_dataset}. "
            "MUTAG/PROTEINS/MarkerPath-Exact3/InfectedSteiner-3/InfectedPair-Unique3/InfectedPair-Dist3/InfectedPair-Dist3-Noise/InfectedPair-Dist3-Noise0.05/InfectedPair-Dist6/InfectedPair-MultiPath3/InfectedHouse/InfectedPentagon/RandomInfectedTriCenter-3/BA-HouseGrid variants/BA-HouseANDGrid variants/PubMed/Cora/Infection4/Infection4-one-hot/Infection3/Infection3-one-hot/Infection3-one-hot-binary/Infection3-one-hot-binary-Noise/Infection3-one-hot-binary-Bernoulli4 require PyG loader (use_pyg_cache=True)."
        )


def load_dataset_legacy(_dataset, skip_preproccessing=False, shuffle=True):
    """Original dataset loader without PyG caching. Use this if you need the exact original behavior."""
    print(f"Loading {_dataset} dataset (legacy)")
    if _dataset[:3] == "syn": # Load node_dataset
        adj, features, labels, train_mask, val_mask, test_mask = _load_node_dataset(_dataset)
        preprocessed_features = preprocess_features(features).astype('float32')
        if skip_preproccessing:
            graph = adj
        else:
            graph = preprocess_adj(adj)[0].astype('int64').T
        labels = np.argmax(labels, axis=1)
        return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
    else: # Load graph dataset
        return load_graph_dataset(_dataset, shuffle)
