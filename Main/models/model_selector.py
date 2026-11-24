import torch
import os
from typing import Optional

from Main.models.Vanilla import NodeGNN as Vanilla_NodeGNN
from Main.models.Vanilla import GraphGNN as Vanilla_GraphGNN
from Main.models.GCN import NodeGCN
from Main.models.GCN import GraphGCN


# ============================================================================
# Dataset Name Mappings
# ============================================================================
# Canonical names (friendly) -> file path names (for pretrained models, pkl files)
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

# Dataset configurations: (num_features, num_classes, task_type)
# Keyed by canonical names
DATASET_CONFIGS = {
    'BA-Shapes': (10, 4, 'node'),
    'BA-House': (10, 2, 'node'),
    'BA-Community': (10, 8, 'node'),  # Two BA communities with house motifs, 8 classes
    'Tree-Cycles': (10, 2, 'node'),   # Tree with cycle motifs (syn3)
    'Tree-Cycles-Noise': (10, 2, 'node'),  # Tree-Cycles with deterministic Gaussian feature noise
    'Tree-Cycles-Noise0.5': (10, 2, 'node'),  # Tree-Cycles with deterministic Gaussian feature noise, std=0.5
    'Tree-Cycles-Noise0.1': (10, 2, 'node'),  # Tree-Cycles with deterministic Gaussian feature noise, std=0.1
    'Tree-Cycles-Noise0.05': (10, 2, 'node'),  # Tree-Cycles with deterministic Gaussian feature noise, std=0.05
    'Tree-Cycles-Uniform0.5': (10, 2, 'node'),  # Tree-Cycles with deterministic uniform feature noise, width=0.5
    'Tree-Grid': (10, 2, 'node'),     # Tree with grid motifs (syn4)
    'PubMed': (500, 3, 'node'),
    'Cora': (1433, 7, 'node'),
    'Infection4': (2, 6, 'node'),
    'Infection4-one-hot': (2, 6, 'node'),
    'Infection3': (2, 5, 'node'),
    'Infection3-one-hot': (2, 5, 'node'),
    'Infection3-one-hot-binary': (2, 2, 'node'),
    'Infection3-one-hot-binary-Noise': (2, 2, 'node'),
    'Infection3-one-hot-binary-Bernoulli4': (4, 2, 'node'),
    'PyG-Infection-ER-L3': (2, 5, 'node'),
    'PyG-Infection-ER-L3-Split43': (2, 5, 'node'),
    'PyG-Infection-ER-L3-Binary-Split43': (2, 2, 'node'),
    'PyG-Infection-BA-L3': (2, 5, 'node'),
    'PyG-Infection-BA-L3-Split43': (2, 5, 'node'),
    'PyG-Infection-BA-L3-Binary-Split43': (2, 2, 'node'),
    'BA-2Motifs': (10, 2, 'graph'),
    'BA2Grid-Longa': (10, 2, 'graph'),
    'BA-HouseGrid': (1, 2, 'graph'),
    'BA-HouseGrid-wb60-m3': (1, 2, 'graph'),
    'BA-HouseANDGrid': (1, 2, 'graph'),
    'BA-HouseANDGrid-wb15-m1': (1, 2, 'graph'),
    'BA-HouseORGrid': (1, 2, 'graph'),
    'MarkerPath-Exact3': (3, 2, 'graph'),
    'InfectedSteiner-3': (2, 2, 'graph'),
    'InfectedPair-Unique3': (2, 2, 'graph'),
    'InfectedPair-Dist3': (2, 2, 'graph'),
    'InfectedPair-ER-Dist3-Split43': (2, 2, 'graph'),
    'InfectedPair-BA-Dist3-Split43': (2, 2, 'graph'),
    'InfectedPair-ER-L3-Multiclass-Split43': (2, 4, 'graph'),
    'InfectedPair-BA-L3-Multiclass-Split43': (2, 4, 'graph'),
    'InfectedPair-ER-Dist5-Split43': (2, 2, 'graph'),
    'InfectedPair-BA-Dist5-Split43': (2, 2, 'graph'),
    'InfectedPair-Dist3-Noise': (2, 2, 'graph'),
    'InfectedPair-Dist3-Noise0.05': (2, 2, 'graph'),
    'InfectedPair-Dist6': (2, 2, 'graph'),
    'InfectedPair-MultiPath3': (2, 2, 'graph'),
    'InfectedHouse': (2, 2, 'graph'),
    'InfectedPentagon': (2, 2, 'graph'),
    'RandomInfectedTriCenter-3': (2, 2, 'graph'),
    'MUTAG': (7, 2, 'graph'),         # PyG TUDataset MUTAG (188 graphs)
    'PROTEINS': (3, 2, 'graph'),      # PyG TUDataset PROTEINS (1113 graphs)
}

# Node vs Graph classification datasets
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
    DATASET_CONFIGS[_canonical] = (2, 5, "node")
    NODE_DATASETS.append(_canonical)
    ALL_DATASETS.append(_canonical)


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to canonical (friendly) format."""
    # Already canonical
    if name in DATASET_CONFIGS:
        return name
    # Legacy/internal name
    if name in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[name]
    return name


def get_path_name(dataset: str) -> str:
    """Get file path name for a dataset (for pretrained models, pkl files)."""
    canonical = normalize_dataset_name(dataset)
    return DATASET_PATH_NAMES.get(canonical, canonical)


def get_dataset_config(dataset: str):
    """Get dataset configuration (num_features, num_classes, task_type)."""
    canonical_name = normalize_dataset_name(dataset)
    if canonical_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[canonical_name]
    raise ValueError(f"Unknown dataset: {dataset}. Available: {ALL_DATASETS}")

def string_to_model(paper, dataset, num_layers=3, hidden_dim=20, use_batchnorm=False, use_jk_concat=True):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. Supports both friendly and internal names.
    :param num_layers: number of GNN layers (2 or 3). Only used for Vanilla models.
    :param hidden_dim: hidden dimension for Vanilla models.
    :param use_batchnorm: whether to use batch normalization. Only used for Vanilla models.
    :param use_jk_concat: whether to use JK-concat (concatenate all layer outputs). 
                          If False, uses only the last layer. Only used for Vanilla models.
    :returns: torch.nn.module models
    """
    # Normalize dataset name to canonical format
    canonical_name = normalize_dataset_name(dataset)
    
    # Get dataset configuration
    try:
        num_features, num_classes, task_type = get_dataset_config(canonical_name)
    except ValueError:
        raise NotImplementedError(f"Dataset '{dataset}' is not supported. Available: {ALL_DATASETS}")
    
    if paper == "Vanilla":
        # Vanilla GNN: MILP-friendly, with optional batch normalization and JK-concat
        if task_type == 'node':
            return Vanilla_NodeGNN(num_features, num_classes, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)
        else:  # graph classification
            return Vanilla_GraphGNN(num_features, num_classes, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)
    if paper == "GCN":
        model_cls = NodeGCN if task_type == 'node' else GraphGCN
        return model_cls(
            num_features,
            num_classes,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
        )
    raise NotImplementedError(f"Paper '{paper}' is not supported. Use 'Vanilla' (GIN) or 'GCN'.")


def get_pretrained_path(
    paper,
    dataset,
    num_layers=3,
    hidden_dim=20,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features: bool = True,
    nn_seed: Optional[int] = 42,
):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param num_layers: number of GNN layers (2 or 3). Only used for Vanilla models.
    :param hidden_dim: hidden dimension for Vanilla models.
    :param use_batchnorm: whether batch normalization is used. Only for Vanilla models.
    :param use_jk_concat: whether JK-concat is used. Only for Vanilla models.
    :param nn_seed: neural network seed used in checkpoint directory naming.
        If None, only legacy (unseeded) naming is considered.
    :returns: str; the path to the pre-trined model parameters.
    """
    # Normalize to canonical name (new standard)
    canonical_name = normalize_dataset_name(dataset)
    # Get path name for backward compatibility with old pretrained models
    path_name = get_path_name(dataset)

    # Compatibility aliases for renamed Infection datasets.
    old_name_alias = {
        'Infection4': ('Infection', 'infection'),
        'Infection4-one-hot': ('Infection-one-hot', 'infection_one_hot'),
    }.get(canonical_name)
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Build model name suffix
    bn_suffix = "_BN" if use_batchnorm else ""
    jk_suffix = "_JK" if use_jk_concat else ""
    
    if paper in {"Vanilla", "GCN"}:
        # Search canonical/legacy folders, including pre-rename aliases.
        candidate_prefixes = [canonical_name]
        if canonical_name != 'BA-House':
            candidate_prefixes.append(path_name)
        if old_name_alias is not None:
            candidate_prefixes.extend([old_name_alias[0], old_name_alias[1]])
        # Preserve order while removing duplicates.
        candidate_prefixes = list(dict.fromkeys(candidate_prefixes))

        seed_suffix = f"_seed{int(nn_seed)}" if nn_seed is not None else ""
        feat_suffix = "" if normalize_features else "_rawfeat"
        candidate_paths = []

        # Preferred: exact architecture match with requested nn_seed.
        if seed_suffix:
            for prefix in candidate_prefixes:
                candidate_paths.append(
                    f"{dir_path}/pretrained/{paper}/{prefix}_{num_layers}L_{hidden_dim}H{bn_suffix}{jk_suffix}{feat_suffix}{seed_suffix}/best_model"
                )

        # Backward-compatible: exact architecture without seed suffix.
        for prefix in candidate_prefixes:
            candidate_paths.append(
                f"{dir_path}/pretrained/{paper}/{prefix}_{num_layers}L_{hidden_dim}H{bn_suffix}{jk_suffix}{feat_suffix}/best_model"
            )

        # Legacy Vanilla compatibility: older folders omitted the JK suffix.
        if paper == "Vanilla":
            if seed_suffix:
                for prefix in candidate_prefixes:
                    candidate_paths.append(
                        f"{dir_path}/pretrained/{paper}/{prefix}_{num_layers}L_{hidden_dim}H{bn_suffix}{feat_suffix}{seed_suffix}/best_model"
                    )
            for prefix in candidate_prefixes:
                candidate_paths.append(
                    f"{dir_path}/pretrained/{paper}/{prefix}_{num_layers}L_{hidden_dim}H{bn_suffix}{feat_suffix}/best_model"
                )

        existing = next((p for p in candidate_paths if os.path.exists(p)), None)
        path = existing if existing is not None else candidate_paths[0]

    return path


def _detect_batchnorm_from_state_dict(state_dict):
    """Detect whether the checkpoint was trained with BatchNorm by checking for BN keys."""
    for key in state_dict.keys():
        if '.bn.' in key or key.endswith('.bn.weight') or key.startswith(('bn1.', 'bn2.', 'bn3.', 'bn4.')):
            return True
    return False


def _detect_jk_concat_from_state_dict(state_dict, num_layers, hidden_dim):
    """Detect whether the checkpoint was trained with JK-concat by checking lin.weight shape."""
    lin_weight_key = 'lin.weight'
    if lin_weight_key in state_dict:
        weight_shape = state_dict[lin_weight_key].shape
        # JK-concat: lin.weight shape is (num_classes, hidden_dim * num_layers)
        # Last-layer only: lin.weight shape is (num_classes, hidden_dim)
        expected_jk_size = hidden_dim * num_layers
        actual_in_features = weight_shape[1]
        return actual_in_features == expected_jk_size
    return True  # Default to JK-concat if we can't detect


def model_selector(
    paper,
    dataset,
    pretrained=True,
    return_checkpoint=False,
    num_layers=3,
    hidden_dim=20,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features: bool = True,
    nn_seed: Optional[int] = 42,
):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :param num_layers: number of GNN layers (2 or 3). Only used for Vanilla models.
    :param hidden_dim: hidden dimension for Vanilla models.
    :param use_batchnorm: whether to use batch normalization. Only used for Vanilla models.
        If pretrained=True, this is auto-detected from checkpoint and the argument is ignored.
    :param use_jk_concat: whether to use JK-concat. Only used for Vanilla models.
        If pretrained=True, this is auto-detected from checkpoint.
    :param nn_seed: neural network seed used in checkpoint directory naming.
        Only used when pretrained=True.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    if pretrained:
        # Try to load checkpoint with the specified settings
        path = get_pretrained_path(
            paper,
            dataset,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
            normalize_features=normalize_features,
            nn_seed=nn_seed,
        )
        
        # If path doesn't exist and we're using Vanilla, try the other BatchNorm setting
        if not os.path.exists(path) and paper in {"Vanilla", "GCN"}:
            alt_path = get_pretrained_path(
                paper,
                dataset,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                use_batchnorm=not use_batchnorm,
                use_jk_concat=use_jk_concat,
                normalize_features=normalize_features,
                nn_seed=nn_seed,
            )
            if os.path.exists(alt_path):
                print(f"Note: Checkpoint with use_batchnorm={use_batchnorm} not found at {path}")
                print(f"      Using checkpoint with use_batchnorm={not use_batchnorm} at {alt_path}")
                path = alt_path
                use_batchnorm = not use_batchnorm

        if not os.path.exists(path):
            canonical_name = normalize_dataset_name(dataset)
            path_name = get_path_name(dataset)
            old_name_alias = {
                'Infection4': ('Infection', 'infection'),
                'Infection4-one-hot': ('Infection-one-hot', 'infection_one_hot'),
            }.get(canonical_name)
            candidate_prefixes = [canonical_name]
            if canonical_name != 'BA-House':
                candidate_prefixes.append(path_name)
            if old_name_alias is not None:
                candidate_prefixes.extend([old_name_alias[0], old_name_alias[1]])
            candidate_prefixes = list(dict.fromkeys(candidate_prefixes))

            pretrained_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pretrained", paper)
            available_variants = []
            if os.path.isdir(pretrained_root):
                available_variants = sorted(
                    entry
                    for entry in os.listdir(pretrained_root)
                    if os.path.isdir(os.path.join(pretrained_root, entry))
                    and any(entry.startswith(f"{prefix}_") for prefix in candidate_prefixes)
                )

            available_text = ", ".join(available_variants[:12]) if available_variants else "none found"
            raise FileNotFoundError(
                "Pretrained checkpoint not found.\n"
                f"  dataset={canonical_name}, paper={paper}, num_layers={num_layers}, hidden_dim={hidden_dim}, "
                f"use_batchnorm={use_batchnorm}, use_jk_concat={use_jk_concat}, nn_seed={nn_seed}\n"
                f"  expected path: {path}\n"
                f"  available variants for this dataset: {available_text}"
            )
        
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        
        # Auto-detect BatchNorm and JK-concat from supported checkpoints as a safety check.
        if paper in {"Vanilla", "GCN"}:
            detected_bn = _detect_batchnorm_from_state_dict(state_dict)
            if detected_bn != use_batchnorm:
                print(f"Warning: Checkpoint BatchNorm mismatch. Detected={detected_bn}, Expected={use_batchnorm}. Using detected.")
                use_batchnorm = detected_bn
            
            detected_jk = _detect_jk_concat_from_state_dict(state_dict, num_layers, hidden_dim)
            if detected_jk != use_jk_concat:
                print(f"Warning: Checkpoint JK-concat mismatch. Detected={detected_jk}, Expected={use_jk_concat}. Using detected.")
                use_jk_concat = detected_jk
        
        model = string_to_model(paper, dataset, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # Compatibility: older PyG checkpoints used `conv*.weight` with shape (in, out),
            # while newer `GCNConv` uses `conv*.lin.weight` with shape (out, in).
            adapted = dict(state_dict)
            for i in (1, 2, 3):
                old_key = f"conv{i}.weight"
                new_key = f"conv{i}.lin.weight"
                if old_key in adapted and new_key not in adapted:
                    adapted[new_key] = adapted.pop(old_key).T
            model.load_state_dict(adapted, strict=True)
        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
        return model
    else:
        model = string_to_model(paper, dataset, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)
        return model
