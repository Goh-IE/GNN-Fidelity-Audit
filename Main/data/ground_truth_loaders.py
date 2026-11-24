import os

from numpy.random.mtrand import RandomState

from Main.data.utils import preprocess_adj, adj_to_edge_index

import pickle as pkl
import numpy as np

# Flag to control whether to use PyG cached loaders
USE_PYG_CACHE = True

# ============================================================================
# Dataset Name Mappings (using canonical names)
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

# Datasets with ground truth explanations available
DATASETS_WITH_GROUND_TRUTH = [
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
    'BA-2Motifs',
    'BA2Grid-Longa',
    'BA-HouseGrid',
    'BA-HouseGrid-wb60-m3',
    'BA-HouseANDGrid',
    'BA-HouseANDGrid-wb15-m1',
    'BA-HouseORGrid',
    'MUTAG',
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
]

from Main.data.pyg_infection_seed_registry import seeded_infection_names
for _canonical, _path, _, _ in seeded_infection_names():
    DATASET_PATH_NAMES[_canonical] = _path
    LEGACY_TO_CANONICAL[_path] = _canonical
    DATASETS_WITH_GROUND_TRUTH.append(_canonical)

# Datasets WITHOUT ground truth (real-world datasets)
DATASETS_WITHOUT_GROUND_TRUTH = ['PubMed', 'Cora', 'PROTEINS']


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to canonical (friendly) format."""
    if name in DATASET_PATH_NAMES:
        return name
    if name in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[name]
    return name


def get_path_name(dataset: str) -> str:
    """Get file path name for a dataset (for pkl files)."""
    canonical = normalize_dataset_name(dataset)
    return DATASET_PATH_NAMES.get(canonical, canonical)


def has_ground_truth(dataset: str) -> bool:
    """Check if dataset has ground truth explanations available."""
    canonical = normalize_dataset_name(dataset)
    return canonical in DATASETS_WITH_GROUND_TRUTH


def load_ba2_ground_truth(shuffle=True):
    """Load a the ground truth from the ba2motif dataset.

    :param shuffle: Wheter the data should be shuffled.
    :returns: np.array, np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
    with open(path, 'rb') as fin:
        adjs, features, labels = pkl.load(fin)

    n_graphs = adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42)  # Make sure that the permutation is always the same, even if we set the seed different
        shuffled_indices = prng.permutation(indices)
    else:
        shuffled_indices = indices

    # Create shuffled data
    shuffled_adjs = adjs[shuffled_indices]
    shuffled_edge_index = adj_to_edge_index(shuffled_adjs)

    np_edge_labels = []

    # Obtain the edge labels.
    insert = 20
    skip = 5
    for edge_index in shuffled_edge_index:
        labels = []
        for pair in edge_index.T:
            r = pair[0]
            c = pair[1]
            # In line with the original PGExplainer code we determine the ground truth based on the location in the index
            if r >= insert and r < insert + skip and c >= insert and c < insert + skip:
                labels.append(1)
            else:
                labels.append(0)
        np_edge_labels.append(np.array(labels))

    return shuffled_edge_index, np_edge_labels


def _load_node_dataset_ground_truth(_dataset):
    """Load a the ground truth from a synthetic node dataset.
    Mutag is a large dataset and can thus take a while to load into memory.
    
    :param shuffle: Whether the data should be shuffled.
    :returns: np.array, np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, _, _, _, _, _, _, _, edge_label_matrix  = pkl.load(fin)
    graph = preprocess_adj(adj)[0].astype('int64').T
    labels = []
    for pair in graph.T:
        labels.append(edge_label_matrix[pair[0], pair[1]])
    labels = np.array(labels)
    return graph, labels


def _load_node_dataset_test_mask(_dataset):
    """Load the original test mask from a synthetic node dataset pkl."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        _, _, _, _, _, _, _, test_mask, _ = pkl.load(fin)
    return np.asarray(test_mask, dtype=bool)


def load_dataset_ground_truth(_dataset, test_indices=None, use_pyg_cache=None):
    """Load a the ground truth from a dataset.
    Optionally we can only request the indices needed for testing.
    
    :param _dataset: Dataset name (canonical or legacy)
    :param test_indices: Only return the indices used by the PGExplainer paper.
    :param use_pyg_cache: Whether to use PyG cached loaders (faster after first load).
                          If None, uses global USE_PYG_CACHE setting.
    :returns: (np.array, np.array), np.array
    :raises ValueError: If dataset doesn't have ground truth available.
    """
    # Normalize to canonical name
    canonical_name = normalize_dataset_name(_dataset)
    
    # Check if ground truth is available
    if not has_ground_truth(canonical_name):
        raise ValueError(f"Dataset '{canonical_name}' does not have ground truth explanations available. "
                        f"Available datasets with ground truth: {DATASETS_WITH_GROUND_TRUTH}")
    
    # Determine whether to use PyG cache
    if use_pyg_cache is None:
        use_pyg_cache = USE_PYG_CACHE
    
    # Try PyG cached loader first (much faster for repeated loads)
    if use_pyg_cache:
        try:
            from Main.data.pyg_datasets import load_dataset_ground_truth_pyg
            return load_dataset_ground_truth_pyg(_dataset, test_indices)
        except Exception as e:
            print(f"PyG cache failed for ground truth {canonical_name}, falling back to original: {e}")
    
    # Get path name for pkl loading
    path_name = get_path_name(canonical_name)
    
    # Original implementation using canonical names
    if canonical_name in (
        'MUTAG',
        'BA-HouseGrid',
        'BA-HouseGrid-wb60-m3',
        'BA-HouseANDGrid',
        'BA-HouseANDGrid-wb15-m1',
        'BA-HouseORGrid',
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
        'MarkerPath-Exact3',
        'InfectedSteiner-3',
        'InfectedPair-Unique3',
        'InfectedPair-Dist3',
        'InfectedPair-ER-Dist3-Split43',
        'InfectedPair-BA-Dist3-Split43',
        'InfectedPair-ER-Dist5-Split43',
        'InfectedPair-BA-Dist5-Split43',
        'InfectedPair-Dist3-Noise',
        'InfectedPair-Dist3-Noise0.05',
        'InfectedPair-Dist6',
        'InfectedPair-MultiPath3',
        'InfectedHouse',
        'InfectedPentagon',
        'RandomInfectedTriCenter-3',
    ) or canonical_name in {name for name, _, _, _ in seeded_infection_names()}:
        from Main.data.pyg_datasets import load_dataset_ground_truth_pyg
        return load_dataset_ground_truth_pyg(canonical_name, test_indices)
    if canonical_name in ('BA-Shapes', 'BA-House', 'BA-Community'):
        graph, labels = _load_node_dataset_ground_truth(path_name)
        test_mask = _load_node_dataset_test_mask(path_name)
        gt_nodes = set()
        for k in range(int(graph.shape[1])):
            if labels[k] <= 0:
                continue
            gt_nodes.add(int(graph[0, k]))
            gt_nodes.add(int(graph[1, k]))

        default_indices = [
            int(i) for i in np.where(test_mask)[0]
            if int(i) in gt_nodes
        ]
        if test_indices is None:
            return (graph, labels), default_indices
        test_index_set = {int(i) for i in test_indices}
        filtered = [int(i) for i in default_indices if int(i) in test_index_set]
        return (graph, labels), filtered
    if canonical_name in ('Tree-Cycles', 'Tree-Cycles-Noise', 'Tree-Cycles-Noise0.5', 'Tree-Cycles-Noise0.1', 'Tree-Cycles-Noise0.05', 'Tree-Cycles-Uniform0.5'):
        graph, labels = _load_node_dataset_ground_truth('syn3')
        if test_indices is None:
            return (graph, labels), range(511,871,6)
        else:
            all = range(511, 871, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    if canonical_name == 'Tree-Grid':
        graph, labels = _load_node_dataset_ground_truth(path_name)
        if test_indices is None:
            return (graph, labels), range(511,800,1)
        else:
            all = range(511, 800, 1)
            filtered = [i for i in all if i in test_indices]
            return (graph, labels), filtered
    elif canonical_name in ('BA-2Motifs', 'BA2Grid-Longa', 'MarkerPath-Exact3', 'InfectedSteiner-3', 'InfectedPair-Unique3', 'InfectedPair-Dist3', 'InfectedPair-ER-Dist3-Split43', 'InfectedPair-BA-Dist3-Split43', 'InfectedPair-ER-L3-Multiclass-Split43', 'InfectedPair-BA-L3-Multiclass-Split43', 'InfectedPair-ER-Dist5-Split43', 'InfectedPair-BA-Dist5-Split43', 'InfectedPair-Dist3-Noise', 'InfectedPair-Dist3-Noise0.05', 'InfectedPair-Dist6', 'InfectedPair-MultiPath3', 'InfectedHouse', 'InfectedPentagon', 'RandomInfectedTriCenter-3', 'BA-HouseGrid', 'BA-HouseGrid-wb60-m3', 'BA-HouseANDGrid', 'BA-HouseANDGrid-wb15-m1', 'BA-HouseORGrid'):
        from Main.data.pyg_datasets import load_dataset_ground_truth_pyg
        return load_dataset_ground_truth_pyg(canonical_name, test_indices)
    else:
        raise ValueError(f"Unknown dataset: {canonical_name}. Available: {DATASETS_WITH_GROUND_TRUTH}")
