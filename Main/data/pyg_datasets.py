"""
PyG-based dataset loaders with caching support.

This module provides fast loading by:
1. Using PyG official loaders for MUTAG, PROTEINS, PubMed, Cora (TUDataset, Planetoid)
2. Converting existing pkl files (and synthetic generators) to PyG InMemoryDataset with processed/data.pt caching

Supported Datasets:
- Node classification: BA-Shapes, BA-House, BA-Community, Tree-Cycles, Tree-Cycles-Noise, Tree-Cycles-Noise0.5, Tree-Cycles-Noise0.1, Tree-Cycles-Noise0.05, Tree-Cycles-Uniform0.5, Tree-Grid, PubMed, Cora, Infection4, Infection4-one-hot, Infection3, Infection3-one-hot, Infection3-one-hot-binary, Infection3-one-hot-binary-Noise
- Graph classification: BA-2Motifs, BA2Grid-Longa, MarkerPath-Exact3, InfectedSteiner-3, InfectedPair-Unique3, InfectedPair-Dist3, InfectedPair-Dist3-Noise, InfectedPair-Dist3-Noise0.05, InfectedPair-Dist6, InfectedPair-MultiPath3, InfectedHouse, InfectedPentagon, RandomInfectedTriCenter-3, BA-HouseGrid, BA-HouseGrid-wb60-m3, BA-HouseANDGrid, BA-HouseANDGrid-wb15-m1, MUTAG, PROTEINS

Usage:
    from Main.data.pyg_datasets import load_dataset_pyg
    data = load_dataset_pyg('BA-Shapes')   # or 'syn1'
    data = load_dataset_pyg('BA-House')
    data = load_dataset_pyg('BA-Community') # or 'syn2'
    data = load_dataset_pyg('Tree-Cycles')  # or 'syn3'
    data = load_dataset_pyg('Tree-Cycles-Noise')  # or 'syn3_noise'
    data = load_dataset_pyg('Tree-Cycles-Noise0.5')  # or 'syn3_noise0p5'
    data = load_dataset_pyg('Tree-Cycles-Noise0.1')  # or 'syn3_noise0p1'
    data = load_dataset_pyg('Tree-Cycles-Noise0.05')  # or 'syn3_noise0p05'
    data = load_dataset_pyg('Tree-Cycles-Uniform0.5')  # or 'syn3_uniform0p5'
    data = load_dataset_pyg('Tree-Grid')    # or 'syn4'
    data = load_dataset_pyg('BA-2Motifs')   # or 'ba2'
    data = load_dataset_pyg('BA2Grid-Longa')   # aliases: BA2Grid, ba2grid
    data = load_dataset_pyg('BA-HouseGrid')   # or 'ba_house_grid'
    data = load_dataset_pyg('BA-HouseGrid-wb60-m3')
    data = load_dataset_pyg('BA-HouseANDGrid')   # or 'ba_house_and_grid'
    data = load_dataset_pyg('BA-HouseANDGrid-wb15-m1')
    data = load_dataset_pyg('MarkerPath-Exact3')   # or 'marker_path_exact3'
    data = load_dataset_pyg('InfectedSteiner-3')   # or 'infected_steiner_3'
    data = load_dataset_pyg('InfectedPair-Unique3')   # or 'infected_pair_unique3'
    data = load_dataset_pyg('InfectedPair-Dist3')   # or 'infected_pair_dist3'
    data = load_dataset_pyg('InfectedPair-Dist3-Noise')   # or 'infected_pair_dist3_noise'
    data = load_dataset_pyg('InfectedPair-Dist3-Noise0.05')   # or 'infected_pair_dist3_noise0p05'
    data = load_dataset_pyg('InfectedPair-Dist6')   # or 'infected_pair_dist6'
    data = load_dataset_pyg('InfectedPair-MultiPath3')   # or 'infected_pair_multipath3'
    data = load_dataset_pyg('InfectedHouse')   # or 'infected_house'
    data = load_dataset_pyg('InfectedPentagon')   # or 'infected_pentagon'
    data = load_dataset_pyg('RandomInfectedTriCenter-3')   # or 'random_infected_tricenter_3'
    data = load_dataset_pyg('MUTAG')        # PyG TUDataset (188 graphs)
    data = load_dataset_pyg('PROTEINS')     # PyG TUDataset (1113 graphs)
    data = load_dataset_pyg('PubMed')       # or 'pubmed'
    data = load_dataset_pyg('Cora')         # or 'cora'
    data = load_dataset_pyg('Infection4')    # or 'infection4' (legacy: 'Infection')
    data = load_dataset_pyg('Infection4-one-hot')    # or 'infection4_one_hot' (legacy: 'Infection-one-hot')
    data = load_dataset_pyg('Infection3')    # or 'infection3'
    data = load_dataset_pyg('Infection3-one-hot')    # or 'infection3_one_hot'
    data = load_dataset_pyg('Infection3-one-hot-binary')    # label 3 -> 1, others -> 0
    data = load_dataset_pyg('Infection3-one-hot-binary-Noise')    # non-infected [0, 1 + N(0,1)]
"""

import os
import hashlib
import pickle as pkl
import random
import numpy as np
import torch
import networkx as nx
from typing import Tuple, List, Optional, Union, Dict, Set

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from numpy.random import RandomState

from Main.data.infection_dataset import (
    generate_infection_dataset,
    DEFAULT_NUM_NODES as INFECTION_DEFAULT_NUM_NODES,
    DEFAULT_EDGE_PROB as INFECTION_DEFAULT_EDGE_PROB,
    DEFAULT_NUM_INFECTED as INFECTION_DEFAULT_NUM_INFECTED,
    DEFAULT_SEED as INFECTION_DEFAULT_SEED,
)
from Main.data.pyg_infection_dataset import (
    build_label3_unique_path_ground_truth,
    generate_pyg_infection_data,
)
from Main.data.marker_path_dataset import (
    DEFAULT_BRANCH_EDGE_PROB as MARKER_PATH_DEFAULT_BRANCH_EDGE_PROB,
    DEFAULT_NUM_GRAPHS as MARKER_PATH_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as MARKER_PATH_DEFAULT_NUM_NODES,
    DEFAULT_SEED as MARKER_PATH_DEFAULT_SEED,
    generate_marker_path_exact3_dataset,
)
from Main.data.infected_steiner_dataset import (
    DEFAULT_DECOY_ARMS_PER_INFECTED as INFECTED_STEINER_DEFAULT_DECOY_ARMS_PER_INFECTED,
    DEFAULT_NUM_GRAPHS as INFECTED_STEINER_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as INFECTED_STEINER_DEFAULT_NUM_NODES,
    DEFAULT_SEED as INFECTED_STEINER_DEFAULT_SEED,
    DEFAULT_TARGET_AVG_NUM_EDGES as INFECTED_STEINER_DEFAULT_TARGET_AVG_NUM_EDGES,
    generate_infected_steiner3_dataset,
)
from Main.data.infected_pair_dataset import (
    DEFAULT_NUM_GRAPHS as INFECTED_PAIR_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as INFECTED_PAIR_DEFAULT_NUM_NODES,
    DEFAULT_SEED as INFECTED_PAIR_DEFAULT_SEED,
    DEFAULT_TARGET_AVG_NUM_EDGES as INFECTED_PAIR_DEFAULT_TARGET_AVG_NUM_EDGES,
    generate_infected_pair_unique3_dataset,
)
from Main.data.infected_pair_dist3_dataset import (
    DEFAULT_NUM_GRAPHS as INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
    DEFAULT_SEED as INFECTED_PAIR_DIST3_DEFAULT_SEED,
    DEFAULT_TARGET_AVG_NUM_EDGES as INFECTED_PAIR_DIST3_DEFAULT_TARGET_AVG_NUM_EDGES,
    DEFAULT_DENSITY_MATCHED_ER_EDGE_PROB as INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB,
    DEFAULT_BA_NUM_EDGES as INFECTED_PAIR_DIST3_DEFAULT_BA_NUM_EDGES,
    generate_infected_pair_distance_dataset,
    generate_infected_pair_multiclass_dataset,
    generate_infected_pair_multipath3_dataset,
)
from Main.data.infected_house_dataset import (
    DEFAULT_NUM_GRAPHS as INFECTED_HOUSE_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as INFECTED_HOUSE_DEFAULT_NUM_NODES,
    DEFAULT_SEED as INFECTED_HOUSE_DEFAULT_SEED,
    DEFAULT_TARGET_AVG_NUM_EDGES as INFECTED_HOUSE_DEFAULT_TARGET_AVG_NUM_EDGES,
    generate_infected_house_dataset,
)
from Main.data.infected_pentagon_dataset import (
    DEFAULT_NUM_GRAPHS as INFECTED_PENTAGON_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as INFECTED_PENTAGON_DEFAULT_NUM_NODES,
    DEFAULT_SEED as INFECTED_PENTAGON_DEFAULT_SEED,
    DEFAULT_TARGET_AVG_NUM_EDGES as INFECTED_PENTAGON_DEFAULT_TARGET_AVG_NUM_EDGES,
    generate_infected_pentagon_dataset,
)
from Main.data.random_infected_tricenter_dataset import (
    DEFAULT_NUM_GRAPHS as RANDOM_TRICENTER_DEFAULT_NUM_GRAPHS,
    DEFAULT_NUM_NODES as RANDOM_TRICENTER_DEFAULT_NUM_NODES,
    DEFAULT_SEED as RANDOM_TRICENTER_DEFAULT_SEED,
    DEFAULT_TARGET_AVG_NUM_EDGES as RANDOM_TRICENTER_DEFAULT_TARGET_AVG_NUM_EDGES,
    generate_random_infected_tricenter3_dataset,
)


# ============================================================================
# Dataset Name Mappings
# ============================================================================

# Canonical (friendly) names -> path names (for pkl files, cache directories)
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


def _collapse_ba_house_labels(labels) -> np.ndarray:
    """Merge BA-Shapes motif labels {1,2,3} into a single positive class."""
    labels_np = np.array(labels)
    if labels_np.ndim > 1:
        labels_np = np.argmax(labels_np, axis=1)
    labels_np = labels_np.astype(np.int64, copy=False).reshape(-1)
    return (labels_np != 0).astype(np.int64, copy=False)


INFECTION_VARIANTS = {
    'infection4': {'feature_mode': 'original', 'max_distance_class': 5},
    'infection4_one_hot': {'feature_mode': 'one_hot', 'max_distance_class': 5},
    'infection3': {'feature_mode': 'original', 'max_distance_class': 4},
    'infection3_one_hot': {'feature_mode': 'one_hot', 'max_distance_class': 4},
    'infection3_one_hot_binary': {
        'feature_mode': 'one_hot',
        'max_distance_class': 4,
        'binary_positive_label': 3,
    },
    'infection3_one_hot_binary_noise': {
        'feature_mode': 'one_hot',
        'max_distance_class': 4,
        'binary_positive_label': 3,
        'noninfected_noise_std': 1.0,
    },
    'infection3_one_hot_binary_bernoulli4': {
        'feature_mode': 'infected_bernoulli',
        'max_distance_class': 4,
        'binary_positive_label': 3,
        'bernoulli_feature_dim': 4,
        'bernoulli_feature_prob': 0.5,
    },
}

PYG_INFECTION_VARIANTS = {
    'pyg_infection_er_l3': {'graph_family': 'er', 'split_seed': 42},
    'pyg_infection_er_l3_split43': {'graph_family': 'er', 'split_seed': 43},
    'pyg_infection_er_l3_binary_split43': {
        'graph_family': 'er', 'split_seed': 43, 'binary_positive_label': 3,
    },
    'pyg_infection_ba_l3': {'graph_family': 'ba', 'split_seed': 42},
    'pyg_infection_ba_l3_split43': {'graph_family': 'ba', 'split_seed': 43},
    'pyg_infection_ba_l3_binary_split43': {
        'graph_family': 'ba', 'split_seed': 43, 'binary_positive_label': 3,
    },
}

from Main.data.pyg_infection_seed_registry import seeded_infection_names

for _canonical, _path, _family, _graph_seed in seeded_infection_names():
    DATASET_PATH_NAMES[_canonical] = _path
    LEGACY_TO_CANONICAL[_path] = _canonical
    PYG_INFECTION_VARIANTS[_path] = {
        'graph_family': _family,
        'dataset_seed': _graph_seed,
        'split_seed': 43,
    }
    NODE_DATASETS.append(_canonical)
    ALL_DATASETS.append(_canonical)


MUTAG_ATOM_TYPE_TO_INDEX = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "I": 4,
    "Cl": 5,
    "Br": 6,
}


def _edge_index_to_undirected_neighbors(edge_index: np.ndarray, num_nodes: int) -> List[Set[int]]:
    """Build an undirected adjacency list from a COO edge index."""
    neighbors: List[Set[int]] = [set() for _ in range(int(num_nodes))]
    for k in range(int(edge_index.shape[1])):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)
    return neighbors


def detect_mutag_motif_edges(
    edge_index: np.ndarray,
    node_features: np.ndarray,
) -> List[Tuple[int, int]]:
    """Detect MUTAG motif edges using NO2/NH2-like nitrogen-centered patterns.

    We treat MUTAG as a pseudo-synthetic motif dataset:
    - NO2 motif: an N atom with at least two O neighbors contributes the two N-O edges.
    - NH2 motif: an N atom with exactly two non-O neighbors contributes its two incident edges.

    Returned edges are undirected `(min(u, v), max(u, v))` pairs with duplicates removed.
    """
    if node_features.ndim != 2:
        raise ValueError(f"Expected node_features shape (num_nodes, num_features), got {tuple(node_features.shape)}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index shape (2, num_edges), got {tuple(edge_index.shape)}")

    num_nodes = int(node_features.shape[0])
    atom_types = np.argmax(node_features, axis=1).astype(int)
    neighbors = _edge_index_to_undirected_neighbors(edge_index, num_nodes)

    nitrogen_idx = MUTAG_ATOM_TYPE_TO_INDEX["N"]
    oxygen_idx = MUTAG_ATOM_TYPE_TO_INDEX["O"]
    motif_edges: Set[Tuple[int, int]] = set()

    for center in range(num_nodes):
        if int(atom_types[center]) != nitrogen_idx:
            continue

        center_neighbors = sorted(int(v) for v in neighbors[center])
        oxygen_neighbors = [v for v in center_neighbors if int(atom_types[v]) == oxygen_idx]
        if len(oxygen_neighbors) >= 2:
            for v in oxygen_neighbors[:2]:
                a, b = (center, v) if center < v else (v, center)
                motif_edges.add((a, b))
            continue

        non_oxygen_neighbors = [v for v in center_neighbors if int(atom_types[v]) != oxygen_idx]
        if len(center_neighbors) == 2 and len(non_oxygen_neighbors) == 2:
            for v in non_oxygen_neighbors:
                a, b = (center, v) if center < v else (v, center)
                motif_edges.add((a, b))

    return sorted(motif_edges)


def build_mutag_ground_truth_edge_labels(
    edge_index: np.ndarray,
    node_features: np.ndarray,
) -> np.ndarray:
    """Build per-edge binary labels aligned with MUTAG edge_index columns."""
    motif_edge_set = set(detect_mutag_motif_edges(edge_index=edge_index, node_features=node_features))
    edge_labels: List[int] = []
    for k in range(int(edge_index.shape[1])):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])
        if i == j:
            edge_labels.append(0)
            continue
        a, b = (i, j) if i < j else (j, i)
        edge_labels.append(1 if (a, b) in motif_edge_set else 0)
    return np.asarray(edge_labels, dtype=np.int64)


def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to canonical (friendly) format."""
    # Already canonical
    if name in DATASET_PATH_NAMES:
        return name
    # Legacy/internal name
    if name in LEGACY_TO_CANONICAL:
        return LEGACY_TO_CANONICAL[name]
    raise ValueError(f"Unknown dataset: {name}. Available: {ALL_DATASETS}")


def get_path_name(dataset: str) -> str:
    """Get file path name for a dataset (for pkl files, cache directories)."""
    canonical = normalize_dataset_name(dataset)
    return DATASET_PATH_NAMES.get(canonical, canonical)


def get_display_name(name: str) -> str:
    """Get display name (same as canonical name)."""
    return normalize_dataset_name(name)


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


def _synthetic_node_variant(path_name: str) -> Dict[str, Union[str, float, int]]:
    """Return synthetic-node variant settings for a path name."""
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


def _float_token(value: float) -> str:
    """Render a float safely for cache filenames."""
    return str(float(value)).replace('-', 'm').replace('.', 'p')


# ============================================================================
# Base Synthetic Node Dataset (BA-Shapes, Tree-Cycles, Tree-Grid, BA-Community)
# ============================================================================

class  SyntheticNodeDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for synthetic node classification datasets.
    Converts pkl files to PyG format with caching in processed/data.pt

    Supported datasets:
    - syn1 / BA-Shapes (house motifs)
    - syn2 / BA-Community (house motifs on 2 communities)
    - syn3 / Tree-Cycles (6-cycle motifs on tree)
    - syn3_noise / Tree-Cycles-Noise (syn3 with deterministic Gaussian feature noise)
    - syn3_noise0p5 / Tree-Cycles-Noise0.5 (syn3 with deterministic Gaussian feature noise)
    - syn3_noise0p1 / Tree-Cycles-Noise0.1 (syn3 with deterministic Gaussian feature noise)
    - syn3_noise0p05 / Tree-Cycles-Noise0.05 (syn3 with deterministic Gaussian feature noise)
    - syn3_uniform0p5 / Tree-Cycles-Uniform0.5 (syn3 with deterministic uniform feature noise)
    - syn4 / Tree-Grid (grid motifs on tree)
    """

    DATASET_NAMES = {
        'syn1': 'BA-Shapes',
        'syn2': 'BA-Community',
        'syn3': 'Tree-Cycles',
        'syn4': 'Tree-Grid'
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        normalize_features: bool = True,
        feature_noise_std: float = 0.0,
        feature_noise_seed: int = 1000003,
        feature_noise_distribution: str = 'gaussian',
    ):
        """
        Args:
            root: Root directory where the dataset should be saved (contains processed/ folder)
            name: Dataset name (syn1, syn2, syn3, syn4)
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply before saving
        """
        self.name = name
        self.normalize_features = bool(normalize_features)
        self.feature_noise_std = float(feature_noise_std)
        self.feature_noise_seed = int(feature_noise_seed)
        self.feature_noise_distribution = str(feature_noise_distribution).lower()
        if self.feature_noise_std < 0.0:
            raise ValueError("feature_noise_std must be non-negative")
        if self.feature_noise_distribution not in {'gaussian', 'uniform'}:
            raise ValueError(f"Unsupported feature noise distribution: {self.feature_noise_distribution}")
        assert name in self.DATASET_NAMES, f"Unknown dataset: {name}. Choose from {list(self.DATASET_NAMES.keys())}"

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}.pkl']

    @property
    def processed_file_names(self) -> List[str]:
        base = 'data' if self.normalize_features else 'data_rawfeat'
        if self.feature_noise_std > 0.0:
            if self.feature_noise_distribution == 'gaussian':
                base += f'_noise_std{_float_token(self.feature_noise_std)}_seed{self.feature_noise_seed}'
            else:
                base += f'_{self.feature_noise_distribution}_width{_float_token(self.feature_noise_std)}_seed{self.feature_noise_seed}'
        return [f'{base}.pt']

    def download(self):
        # PKL files should already exist in pkls/ directory
        pass

    def process(self):
        """Convert pkl file to PyG Data object and save to processed/"""
        # Find pkl file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pkl_path = os.path.join(dir_path, 'pkls', f'{self.name}.pkl')

        if not os.path.exists(pkl_path):
            if self.feature_noise_std > 0.0:
                self._process_from_base_pyg_cache(pkl_path)
                return
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")

        print(f"Converting {self.name} pkl to PyG format...")

        with open(pkl_path, 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)

        if self.normalize_features:
            features = self._preprocess_features(features).astype('float32')
        else:
            features = np.asarray(features, dtype=np.float32)
        features = self._apply_feature_noise(features)

        # Convert adjacency to edge_index
        edge_index = self._adj_to_edge_index(adj)

        # Combine labels
        labels = y_train.copy()
        labels[val_mask] = y_val[val_mask]
        labels[test_mask] = y_test[test_mask]
        labels = np.argmax(labels, axis=1)  # One-hot to class index

        # Extract edge labels from matrix
        edge_labels = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_labels.append(edge_label_matrix[src, dst])
        edge_labels = np.array(edge_labels)

        # Create PyG Data object
        data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            edge_labels=torch.tensor(edge_labels, dtype=torch.long),
            edge_label_matrix=torch.tensor(edge_label_matrix, dtype=torch.long),
            feature_noise_std=torch.tensor([self.feature_noise_std], dtype=torch.float32),
            feature_noise_seed=torch.tensor([self.feature_noise_seed], dtype=torch.long),
            feature_noise_distribution_id=torch.tensor(
                [0 if self.feature_noise_distribution == 'gaussian' else 1],
                dtype=torch.long,
            ),
        )

        # Store original adj for ground truth loading
        data.adj = torch.tensor(adj, dtype=torch.float32) if isinstance(adj, np.ndarray) else adj

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset to {self.processed_paths[0]}")

    def _process_from_base_pyg_cache(self, missing_pkl_path: str) -> None:
        """Build a noisy synthetic-node variant from the existing clean PyG cache."""
        base_root = os.path.join(get_pyg_dataset_root(), self.name)
        base_processed = os.path.join(
            base_root,
            'processed',
            'data.pt' if self.normalize_features else 'data_rawfeat.pt',
        )
        if not os.path.exists(base_processed):
            raise FileNotFoundError(
                f"PKL file not found: {missing_pkl_path}; base PyG cache not found: {base_processed}"
            )

        print(f"Deriving noisy {self.name} PyG dataset from {base_processed}...")
        base_dataset = SyntheticNodeDataset(
            root=base_root,
            name=self.name,
            normalize_features=self.normalize_features,
        )
        data = base_dataset[0].clone()
        data.x = torch.tensor(
            self._apply_feature_noise(data.x.detach().cpu().numpy()),
            dtype=torch.float32,
        )
        data.feature_noise_std = torch.tensor([self.feature_noise_std], dtype=torch.float32)
        data.feature_noise_seed = torch.tensor([self.feature_noise_seed], dtype=torch.long)
        data.feature_noise_distribution_id = torch.tensor(
            [0 if self.feature_noise_distribution == 'gaussian' else 1],
            dtype=torch.long,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
        print(f"Saved noisy {self.name} PyG dataset to {self.processed_paths[0]}")

    def _apply_feature_noise(self, features: np.ndarray) -> np.ndarray:
        """Apply deterministic feature noise when requested."""
        features = np.asarray(features, dtype=np.float32)
        if self.feature_noise_std <= 0.0:
            return features
        rng = np.random.RandomState(self.feature_noise_seed)
        if self.feature_noise_distribution == 'gaussian':
            noise = rng.normal(loc=0.0, scale=self.feature_noise_std, size=features.shape).astype(np.float32)
        elif self.feature_noise_distribution == 'uniform':
            noise = rng.uniform(
                low=-self.feature_noise_std,
                high=self.feature_noise_std,
                size=features.shape,
            ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported feature noise distribution: {self.feature_noise_distribution}")
        return (features + noise).astype(np.float32, copy=False)

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features by row sum"""
        import scipy.sparse as sp
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features).astype(np.float32)
        try:
            return np.array(features.todense())
        except:
            return features

    def _adj_to_edge_index(self, adj) -> np.ndarray:
        """Convert adjacency matrix to edge_index"""
        import scipy.sparse as sp
        if sp.issparse(adj):
            adj = adj.tocoo()
            edge_index = np.vstack([adj.row, adj.col])
        else:
            coo = sp.coo_matrix(adj)
            edge_index = np.vstack([coo.row, coo.col])
        return edge_index.astype(np.int64)


# ============================================================================
# Infection Node Classification Dataset (Synthetic Generator)
# ============================================================================

class InfectionNodeDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for the synthetic Infection-Path node classification task.

    Ground truth for explanation is node-specific and stored in:
    processed/ground_truth.pt
    """
    def __init__(
        self,
        root: str,
        feature_mode: str = "original",
        max_distance_class: int = 5,
        binary_positive_label: Optional[int] = None,
        noninfected_noise_std: float = 0.0,
        bernoulli_feature_dim: int = 2,
        bernoulli_feature_prob: float = 0.5,
        transform=None,
        pre_transform=None,
    ):
        if feature_mode not in ("original", "one_hot", "infected_bernoulli"):
            raise ValueError(f"Unknown Infection feature_mode: {feature_mode}")
        if max_distance_class not in (4, 5):
            raise ValueError(f"Unknown max_distance_class: {max_distance_class}. Use 4 or 5.")
        if binary_positive_label is not None and not (0 <= int(binary_positive_label) <= int(max_distance_class)):
            raise ValueError(
                f"binary_positive_label must be in [0, {max_distance_class}], got {binary_positive_label}"
            )
        if int(bernoulli_feature_dim) < 2:
            raise ValueError("bernoulli_feature_dim must be at least 2")
        if not (0.0 <= float(bernoulli_feature_prob) <= 1.0):
            raise ValueError("bernoulli_feature_prob must be in [0, 1]")
        self.name = "infection"
        self.feature_mode = feature_mode
        self.max_distance_class = int(max_distance_class)
        self.binary_positive_label = None if binary_positive_label is None else int(binary_positive_label)
        self.noninfected_noise_std = float(noninfected_noise_std)
        self.bernoulli_feature_dim = int(bernoulli_feature_dim)
        self.bernoulli_feature_prob = float(bernoulli_feature_prob)
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        self.gt_payload = torch.load(self.processed_paths[1], map_location="cpu")

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "ground_truth.pt"]

    def download(self):
        pass

    def process(self):
        payload = generate_infection_dataset(
            num_nodes=INFECTION_DEFAULT_NUM_NODES,
            edge_prob=INFECTION_DEFAULT_EDGE_PROB,
            num_infected=INFECTION_DEFAULT_NUM_INFECTED,
            seed=INFECTION_DEFAULT_SEED,
            feature_mode=self.feature_mode,
            max_distance_class=self.max_distance_class,
            noninfected_noise_std=self.noninfected_noise_std,
            bernoulli_feature_dim=self.bernoulli_feature_dim,
            bernoulli_feature_prob=self.bernoulli_feature_prob,
        )
        original_labels = np.asarray(payload["labels"], dtype=np.int64)
        if self.binary_positive_label is None:
            labels = original_labels
            gt_test_indices = [int(i) for i in payload["gt_test_indices"]]
            node_to_gt_edges = payload["node_to_gt_edges"]
        else:
            labels = (original_labels == int(self.binary_positive_label)).astype(np.int64)
            gt_test_indices = [
                int(i)
                for i in payload["gt_test_indices"]
                if int(original_labels[int(i)]) == int(self.binary_positive_label)
            ]
            positive_index_set = set(gt_test_indices)
            node_to_gt_edges = {
                int(node_idx): edges
                for node_idx, edges in payload["node_to_gt_edges"].items()
                if int(node_idx) in positive_index_set
            }

        data = Data(
            x=torch.tensor(payload["features"], dtype=torch.float32),
            edge_index=torch.tensor(payload["edge_index"], dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
            train_mask=torch.tensor(payload["train_mask"], dtype=torch.bool),
            val_mask=torch.tensor(payload["val_mask"], dtype=torch.bool),
            test_mask=torch.tensor(payload["test_mask"], dtype=torch.bool),
            infected_mask=torch.tensor(payload["infected_mask"], dtype=torch.bool),
            distance_label=torch.tensor(payload["distances"], dtype=torch.long),
            shortest_path_count_clipped=torch.tensor(payload["shortest_path_count_clipped"], dtype=torch.long),
            noninfected_noise_std=torch.tensor([self.noninfected_noise_std], dtype=torch.float32),
            bernoulli_feature_dim=torch.tensor([self.bernoulli_feature_dim], dtype=torch.long),
            bernoulli_feature_prob=torch.tensor([self.bernoulli_feature_prob], dtype=torch.float32),
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

        gt_payload = {
            "node_to_gt_edges": {
                int(node_idx): [(int(a), int(b)) for (a, b) in edges]
                for node_idx, edges in node_to_gt_edges.items()
            },
            "gt_test_indices": gt_test_indices,
            "metadata": {
                **payload["metadata"],
                "binary_positive_label": self.binary_positive_label,
                "bernoulli_feature_dim": self.bernoulli_feature_dim,
                "bernoulli_feature_prob": self.bernoulli_feature_prob,
                "num_binary_positive_nodes": int(labels.sum()) if self.binary_positive_label is not None else None,
                "num_binary_positive_gt_test_nodes": int(len(gt_test_indices)) if self.binary_positive_label is not None else None,
            },
        }
        torch.save(gt_payload, self.processed_paths[1])


# ============================================================================
# Official PyG InfectionDataset wrappers (ER/BA, L=3 multiclass)
# ============================================================================

class PyGInfectionNodeDataset(InMemoryDataset):
    """Cached ER/BA variants generated by PyG's official InfectionDataset."""

    def __init__(
        self,
        root: str,
        graph_family: str,
        dataset_seed: int = 42,
        split_seed: int = 42,
        binary_positive_label: Optional[int] = None,
        transform=None,
        pre_transform=None,
    ):
        family = str(graph_family).lower()
        if family not in ("er", "ba"):
            raise ValueError(f"graph_family must be 'er' or 'ba', got {graph_family!r}")
        self.graph_family = family
        self.dataset_seed = int(dataset_seed)
        self.split_seed = int(split_seed)
        self.binary_positive_label = (
            None if binary_positive_label is None else int(binary_positive_label)
        )
        if self.binary_positive_label is not None and self.binary_positive_label != 3:
            raise ValueError("PyG Infection binary variants currently support distance label 3 only")
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        self.metadata = torch.load(self.processed_paths[1], map_location="cpu")
        self.gt_payload = build_label3_unique_path_ground_truth(self[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "metadata.pt"]

    def download(self):
        pass

    def process(self):
        data, metadata = generate_pyg_infection_data(
            self.graph_family,
            dataset_seed=self.dataset_seed,
            split_seed=self.split_seed,
        )
        if self.binary_positive_label is not None:
            distance_labels = data.distance_label.clone()
            data.y = (distance_labels == self.binary_positive_label).to(torch.long)
            metadata = {
                **metadata,
                "label_map": f"binary_indicator_distance_{self.binary_positive_label}",
                "binary_positive_label": self.binary_positive_label,
                "distance_class_counts": metadata["class_counts"],
                "distance_split_class_counts": metadata["split_class_counts"],
                "num_classes": 2,
                "class_counts": torch.bincount(data.y, minlength=2).tolist(),
                "split_class_counts": {
                    split_name: torch.bincount(
                        data.y[getattr(data, f"{split_name}_mask")], minlength=2
                    ).tolist()
                    for split_name in ("train", "val", "test")
                },
            }
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        self.save([data], self.processed_paths[0])
        torch.save(metadata, self.processed_paths[1])


# ============================================================================
# BA-HouseGrid / BA-HouseANDGrid Graph Classification Datasets
# ============================================================================

HOUSE_MOTIF_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (2, 4),
    (3, 4),
)

GRID_MOTIF_EDGES: Tuple[Tuple[int, int], ...] = tuple(
    [(3 * r + c, 3 * r + c + 1) for r in range(3) for c in range(2)]
    + [(3 * r + c, 3 * (r + 1) + c) for r in range(2) for c in range(3)]
)

BA_HOUSE_GRID_VARIANTS: Dict[str, Dict[str, object]] = {
    "ba_house_grid": {
        "kind": "house_grid",
        # MAGE official BA-HouseGrid configuration.  Dataset generation is
        # preceded by fix_random_seed(42), while its random train/val/test
        # split uses torch.Generator(seed=2).
        "num_graphs": 10000,
        "num_shapes": 1,
        "width_basis": 20,
        "seed": 42,
        "split_seed": 2,
        "base_m": 1,
    },
    "ba_house_grid_wb60_m3": {
        "kind": "house_grid",
        "num_graphs": 2000,
        "num_shapes": 1,
        "width_basis": 60,
        "seed": 0,
        "base_m": 3,
    },
    "ba_house_and_grid": {
        "kind": "house_and_grid",
        "num_graphs": 10000,
        "num_shapes": 1,
        "width_basis": 20,
        "seed": 2,
        "base_m": None,
    },
    "ba_house_and_grid_wb15_m1": {
        "kind": "house_and_grid",
        "num_graphs": 10000,
        "num_shapes": 1,
        "width_basis": 15,
        "seed": 2,
        "base_m": 1,
    },
    "ba_house_or_grid": {
        "kind": "house_or_grid",
        "num_graphs": 10000,
        "num_shapes": 1,
        "width_basis": 20,
        # MAGE calls fix_random_seed(42) before lazy dataset generation.
        "seed": 42,
        "split_seed": 2,
        "base_m": None,
    },
}


def _is_ba_house_grid_variant(path_name: str) -> bool:
    return str(path_name) in BA_HOUSE_GRID_VARIANTS


def _ba_house_grid_variant_kind(path_name: str) -> str:
    return str(BA_HOUSE_GRID_VARIANTS[str(path_name)]["kind"])


def _ba_house_grid_shuffled_indices(path_name: str, n_graphs: int) -> np.ndarray:
    """Return the frozen graph order used to construct train/val/test masks."""
    variant = BA_HOUSE_GRID_VARIANTS[str(path_name)]
    if "split_seed" in variant:
        # Match MAGE's torch.utils.data.random_split(..., generator=seed 2).
        generator = torch.Generator().manual_seed(int(variant["split_seed"]))
        return torch.randperm(int(n_graphs), generator=generator).cpu().numpy()
    return RandomState(42).permutation(np.arange(int(n_graphs)))


def _undirected_pair(u: int, v: int) -> Tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


def _attach_synthetic_motif(
    graph: nx.Graph,
    anchor: int,
    motif_name: str,
) -> Set[Tuple[int, int]]:
    """Attach one motif to a BA base node and return its internal GT edges."""
    if motif_name == "house":
        motif_edges = HOUSE_MOTIF_EDGES
        num_motif_nodes = 5
    elif motif_name == "grid":
        motif_edges = GRID_MOTIF_EDGES
        num_motif_nodes = 9
    else:
        raise ValueError(f"Unknown motif: {motif_name}")

    offset = int(graph.number_of_nodes())
    motif_nodes = [offset + i for i in range(num_motif_nodes)]
    graph.add_nodes_from(motif_nodes)

    gt_edges: Set[Tuple[int, int]] = set()
    for local_u, local_v in motif_edges:
        u = offset + int(local_u)
        v = offset + int(local_v)
        graph.add_edge(u, v)
        gt_edges.add(_undirected_pair(u, v))

    # The anchor edge connects the motif to the BA structure, but is not part
    # of the motif explanation in the GInX/MAGE convention.
    graph.add_edge(int(anchor), offset)
    return gt_edges


def _edge_index_and_labels_from_graph(
    graph: nx.Graph,
    gt_edges: Set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    edge_pairs: List[Tuple[int, int]] = []
    edge_labels: List[int] = []

    for u, v in sorted(graph.edges()):
        label = 1 if _undirected_pair(u, v) in gt_edges else 0
        edge_pairs.append((int(u), int(v)))
        edge_labels.append(label)
        edge_pairs.append((int(v), int(u)))
        edge_labels.append(label)

    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(edge_pairs, dtype=np.int64).T, np.asarray(edge_labels, dtype=np.int64)


class BAHouseGridDataset(InMemoryDataset):
    """PyG datasets for GInX-Eval BA-HouseGrid and MAGE BA-HouseANDGrid."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_shapes: Optional[int] = None,
        width_basis: Optional[int] = None,
        num_node_features: int = 1,
        base_m: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.name = str(name)
        if self.name not in BA_HOUSE_GRID_VARIANTS:
            raise ValueError(f"Unknown BA-HouseGrid variant: {name}")

        variant = BA_HOUSE_GRID_VARIANTS[self.name]
        self.variant_kind = str(variant["kind"])
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_shapes = int(variant["num_shapes"]) if num_shapes is None else int(num_shapes)
        self.width_basis = int(variant["width_basis"]) if width_basis is None else int(width_basis)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        default_base_m = variant["base_m"]
        self.base_m = (
            None
            if base_m is None and default_base_m is None
            else int(default_base_m if base_m is None else base_m)
        )

        self.synthetic_num_node_features = int(num_node_features)
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        rng = RandomState(self.seed)
        if self.variant_kind == "house_grid":
            data_list = self._generate_ba_house_grid(rng)
        elif self.variant_kind == "house_or_grid":
            data_list = self._generate_ba_house_or_grid(rng)
        else:
            data_list = self._generate_ba_house_and_grid(rng)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")

    def _base_graph(self, rng: RandomState, base_m: int) -> nx.Graph:
        graph_seed = int(rng.randint(0, np.iinfo(np.int32).max))
        return nx.barabasi_albert_graph(self.width_basis, int(base_m), seed=graph_seed)

    def _to_data(
        self,
        graph: nx.Graph,
        graph_idx: int,
        label: int,
        gt_edges: Set[Tuple[int, int]],
    ) -> Data:
        edge_index_np, edge_labels_np = _edge_index_and_labels_from_graph(graph, gt_edges)
        x = np.ones((int(graph.number_of_nodes()), self.synthetic_num_node_features), dtype=np.float32)

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            y=torch.tensor([int(label)], dtype=torch.long),
            edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
            edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
            idx=torch.tensor([int(graph_idx)], dtype=torch.long),
        )
        data.edge_mask = data.edge_labels.float()
        return data

    def _generate_ba_house_grid(self, rng: RandomState) -> List[Data]:
        if self.name == "ba_house_grid":
            # MAGE samples house/grid independently with equal probability
            # after fix_random_seed(42); it does not force exact balance.
            motif_ids = rng.choice(
                np.asarray([0, 1], dtype=np.int64),
                size=self.num_graphs,
                p=np.asarray([0.5, 0.5], dtype=np.float64),
            )
            # NetworkX consumes Python's RNG in the MAGE implementation.
            ba_rng = random.Random(self.seed)
        else:
            motif_ids = np.asarray(
                [0] * (self.num_graphs // 2)
                + [1] * (self.num_graphs - self.num_graphs // 2),
                dtype=np.int64,
            )
            rng.shuffle(motif_ids)
            ba_rng = None

        data_list: List[Data] = []
        for graph_idx, motif_id in enumerate(motif_ids):
            motif_name = "house" if int(motif_id) == 0 else "grid"
            if self.name == "ba_house_grid":
                graph = nx.barabasi_albert_graph(
                    self.width_basis,
                    int(self.base_m or 1),
                    seed=ba_rng,
                )
                # MAGE's build_weighted_graph uses plugins=[0] when one motif
                # is attached.  Its subsequent 1% perturbation adds zero edges
                # for these 26/32-edge graphs because int(0.01 * |E|) == 0.
                anchors = np.asarray([0], dtype=np.int64)
            else:
                graph = self._base_graph(rng, base_m=int(self.base_m or 5))
                if int(self.num_shapes) <= 1:
                    num_motifs = 1
                else:
                    high = max(3, int(self.num_shapes))
                    num_motifs = int(rng.randint(2, high))
                anchors = rng.choice(self.width_basis, size=num_motifs, replace=False)

            gt_edges: Set[Tuple[int, int]] = set()
            for anchor in anchors:
                gt_edges.update(_attach_synthetic_motif(graph, int(anchor), motif_name))

            data_list.append(self._to_data(graph, graph_idx, int(motif_id), gt_edges))

        return data_list

    def _generate_ba_house_and_grid(self, rng: RandomState) -> List[Data]:
        cases = ("none", "house", "grid", "house_grid")
        labels = {
            "none": 0,
            "house": 0,
            "grid": 0,
            "house_grid": 1,
        }
        probs = np.asarray([0.5 / 3.0, 0.5 / 3.0, 0.5 / 3.0, 0.5], dtype=np.float64)

        data_list: List[Data] = []
        for graph_idx in range(self.num_graphs):
            case = str(rng.choice(cases, p=probs))
            graph_base_m = int(self.base_m) if self.base_m is not None else int(rng.randint(1, 3))
            graph = self._base_graph(rng, base_m=graph_base_m)
            gt_edges: Set[Tuple[int, int]] = set()

            if case == "house":
                anchor = int(rng.choice(self.width_basis))
                gt_edges.update(_attach_synthetic_motif(graph, anchor, "house"))
            elif case == "grid":
                anchor = int(rng.choice(self.width_basis))
                gt_edges.update(_attach_synthetic_motif(graph, anchor, "grid"))
            elif case == "house_grid":
                anchors = rng.choice(self.width_basis, size=2, replace=False)
                gt_edges.update(_attach_synthetic_motif(graph, int(anchors[0]), "house"))
                gt_edges.update(_attach_synthetic_motif(graph, int(anchors[1]), "grid"))

            data_list.append(self._to_data(graph, graph_idx, labels[case], gt_edges))

        return data_list

    def _generate_ba_house_or_grid(self, rng: RandomState) -> List[Data]:
        """Reproduce MAGE's BA-HouseORGrid generator with isolated RNG state.

        MAGE seeds Python, NumPy, and PyTorch with 42 before constructing the
        dataset. NetworkX's BA generator consumes Python's RNG, while motif-case
        and ``m`` sampling consume NumPy's RNG. Keeping both streams local makes
        regeneration deterministic without mutating process-global state.
        """
        cases = ("none", "house", "grid", "house_grid")
        labels = {"none": 0, "house": 1, "grid": 1, "house_grid": 1}
        probs = np.asarray([0.5, 0.5 / 3.0, 0.5 / 3.0, 0.5 / 3.0], dtype=np.float64)
        python_rng = random.Random(self.seed)

        # These are the rooted motif definitions used by MAGE. In particular,
        # the house attachment node is one of the two roof-adjacent corners.
        mage_house_edges: Tuple[Tuple[int, int], ...] = (
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 1)
        )

        def attach(graph: nx.Graph, anchor: int, motif_name: str) -> Set[Tuple[int, int]]:
            motif_edges = mage_house_edges if motif_name == "house" else GRID_MOTIF_EDGES
            num_nodes = 5 if motif_name == "house" else 9
            offset = int(graph.number_of_nodes())
            graph.add_nodes_from(range(offset, offset + num_nodes))
            gt_edges: Set[Tuple[int, int]] = set()
            for local_u, local_v in motif_edges:
                u, v = offset + int(local_u), offset + int(local_v)
                graph.add_edge(u, v)
                gt_edges.add(_undirected_pair(u, v))
            graph.add_edge(offset, int(anchor))
            return gt_edges

        data_list: List[Data] = []
        for graph_idx in range(self.num_graphs):
            case_idx = int(rng.choice(len(cases), p=probs))
            case = cases[case_idx]
            graph_base_m = int(rng.randint(1, 3))
            graph = nx.barabasi_albert_graph(
                self.width_basis,
                graph_base_m,
                seed=python_rng,
            )
            gt_edges: Set[Tuple[int, int]] = set()

            # MAGE places one motif at BA node 0; with two motifs it uses the
            # regularly spaced BA nodes 0 and width_basis/2.
            if case == "house":
                gt_edges.update(attach(graph, 0, "house"))
            elif case == "grid":
                gt_edges.update(attach(graph, 0, "grid"))
            elif case == "house_grid":
                gt_edges.update(attach(graph, 0, "house"))
                gt_edges.update(attach(graph, self.width_basis // 2, "grid"))

            data = self._to_data(graph, graph_idx, labels[case], gt_edges)
            data.motif_case = torch.tensor([case_idx], dtype=torch.long)
            data.gt_valid = torch.tensor(
                [case in ("house", "grid")],
                dtype=torch.bool,
            )
            data.base_m = torch.tensor([graph_base_m], dtype=torch.long)
            data_list.append(data)

        return data_list


# ============================================================================
# MarkerPath-Exact3 Graph Classification Dataset
# ============================================================================

MARKER_PATH_VARIANTS: Dict[str, Dict[str, object]] = {
    "marker_path_exact3": {
        "num_graphs": MARKER_PATH_DEFAULT_NUM_GRAPHS,
        "num_nodes": MARKER_PATH_DEFAULT_NUM_NODES,
        "seed": MARKER_PATH_DEFAULT_SEED,
        "branch_edge_prob": MARKER_PATH_DEFAULT_BRANCH_EDGE_PROB,
    },
}


def _is_marker_path_variant(path_name: str) -> bool:
    return str(path_name) in MARKER_PATH_VARIANTS


class MarkerPathDataset(InMemoryDataset):
    """PyG dataset for MarkerPath-Exact3 graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        branch_edge_prob: Optional[float] = None,
    ):
        self.name = str(name)
        if self.name not in MARKER_PATH_VARIANTS:
            raise ValueError(f"Unknown MarkerPath variant: {name}")

        variant = MARKER_PATH_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.branch_edge_prob = (
            float(variant["branch_edge_prob"]) if branch_edge_prob is None else float(branch_edge_prob)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        payload = generate_marker_path_exact3_dataset(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            seed=self.seed,
            branch_edge_prob=self.branch_edge_prob,
        )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                marker_nodes=torch.tensor(graph_payload["marker_nodes"], dtype=torch.long),
                target_distance=torch.tensor([int(graph_payload["target_distance"])], dtype=torch.long),
                shortest_path_count=torch.tensor([int(graph_payload["shortest_path_count"])], dtype=torch.long),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# InfectedSteiner-3 Graph Classification Dataset
# ============================================================================

INFECTED_STEINER_VARIANTS: Dict[str, Dict[str, object]] = {
    "infected_steiner_3": {
        "num_graphs": INFECTED_STEINER_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_STEINER_DEFAULT_NUM_NODES,
        "seed": INFECTED_STEINER_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_STEINER_DEFAULT_TARGET_AVG_NUM_EDGES,
        "decoy_arms_per_infected": INFECTED_STEINER_DEFAULT_DECOY_ARMS_PER_INFECTED,
    },
}


def _is_infected_steiner_variant(path_name: str) -> bool:
    return str(path_name) in INFECTED_STEINER_VARIANTS


def _infected_steiner_negative_type_code(negative_type: str) -> int:
    if str(negative_type) == "pairwise":
        return 0
    if str(negative_type) == "near_common":
        return 1
    return -1


class InfectedSteinerDataset(InMemoryDataset):
    """PyG dataset for InfectedSteiner-3 graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        target_avg_num_edges: Optional[float] = None,
        decoy_arms_per_infected: Optional[int] = None,
    ):
        self.name = str(name)
        if self.name not in INFECTED_STEINER_VARIANTS:
            raise ValueError(f"Unknown InfectedSteiner variant: {name}")

        variant = INFECTED_STEINER_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.target_avg_num_edges = (
            float(variant["target_avg_num_edges"])
            if target_avg_num_edges is None
            else float(target_avg_num_edges)
        )
        self.decoy_arms_per_infected = (
            int(variant["decoy_arms_per_infected"])
            if decoy_arms_per_infected is None
            else int(decoy_arms_per_infected)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        payload = generate_infected_steiner3_dataset(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            seed=self.seed,
            target_avg_num_edges=self.target_avg_num_edges,
            decoy_arms_per_infected=self.decoy_arms_per_infected,
        )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                infected_nodes=torch.tensor(graph_payload["infected_nodes"], dtype=torch.long),
                gt_valid=torch.tensor([bool(graph_payload["gt_valid"])], dtype=torch.bool),
                gt_undirected_budget=torch.tensor([int(graph_payload["gt_undirected_budget"])], dtype=torch.long),
                gt_directed_label_count=torch.tensor([int(graph_payload["gt_directed_label_count"])], dtype=torch.long),
                num_valid_certificates=torch.tensor([int(graph_payload["num_valid_certificates"])], dtype=torch.long),
                num_edges_undirected=torch.tensor([int(graph_payload["num_edges"])], dtype=torch.long),
                structural_edges=torch.tensor([int(graph_payload["structural_edges"])], dtype=torch.long),
                background_edge_count=torch.tensor([int(graph_payload["background_edge_count"])], dtype=torch.long),
                background_edge_prob=torch.tensor([float(graph_payload["background_edge_prob"])], dtype=torch.float32),
                target_avg_num_edges=torch.tensor([float(graph_payload["target_avg_num_edges"])], dtype=torch.float32),
                certificate_center=torch.tensor([int(graph_payload["certificate_center"])], dtype=torch.long),
                negative_type_code=torch.tensor(
                    [_infected_steiner_negative_type_code(str(graph_payload["negative_type"]))],
                    dtype=torch.long,
                ),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# InfectedPair-Unique3 Graph Classification Dataset
# ============================================================================

INFECTED_PAIR_VARIANTS: Dict[str, Dict[str, object]] = {
    "infected_pair_unique3": {
        "num_graphs": INFECTED_PAIR_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PAIR_DEFAULT_TARGET_AVG_NUM_EDGES,
    },
}


def _is_infected_pair_variant(path_name: str) -> bool:
    return str(path_name) in INFECTED_PAIR_VARIANTS


def _infected_pair_bucket_code(bucket: str) -> int:
    if str(bucket) == "unique3":
        return 0
    if str(bucket) == "dist1":
        return 1
    if str(bucket) == "dist2":
        return 2
    if str(bucket) == "dist3_nonunique":
        return 3
    if str(bucket) == "other":
        return 4
    return -1


class InfectedPairDataset(InMemoryDataset):
    """PyG dataset for InfectedPair-Unique3 graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        target_avg_num_edges: Optional[float] = None,
    ):
        self.name = str(name)
        if self.name not in INFECTED_PAIR_VARIANTS:
            raise ValueError(f"Unknown InfectedPair variant: {name}")

        variant = INFECTED_PAIR_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.target_avg_num_edges = (
            float(variant["target_avg_num_edges"])
            if target_avg_num_edges is None
            else float(target_avg_num_edges)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        payload = generate_infected_pair_unique3_dataset(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            seed=self.seed,
            target_avg_num_edges=self.target_avg_num_edges,
        )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                infected_nodes=torch.tensor(graph_payload["infected_nodes"], dtype=torch.long),
                gt_valid=torch.tensor([bool(graph_payload["gt_valid"])], dtype=torch.bool),
                gt_undirected_budget=torch.tensor([int(graph_payload["gt_undirected_budget"])], dtype=torch.long),
                gt_directed_label_count=torch.tensor([int(graph_payload["gt_directed_label_count"])], dtype=torch.long),
                target_distance=torch.tensor([int(graph_payload["target_distance"])], dtype=torch.long),
                shortest_path_count=torch.tensor([int(graph_payload["shortest_path_count"])], dtype=torch.long),
                num_edges_undirected=torch.tensor([int(graph_payload["num_edges"])], dtype=torch.long),
                er_edge_prob=torch.tensor([float(graph_payload["er_edge_prob"])], dtype=torch.float32),
                target_avg_num_edges=torch.tensor([float(graph_payload["target_avg_num_edges"])], dtype=torch.float32),
                target_bucket_code=torch.tensor(
                    [_infected_pair_bucket_code(str(graph_payload["target_bucket"]))],
                    dtype=torch.long,
                ),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# InfectedPair-DistK Graph Classification Dataset
# ============================================================================

INFECTED_PAIR_DIST3_VARIANTS: Dict[str, Dict[str, object]] = {
    "infected_pair_dist3": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PAIR_DIST3_DEFAULT_TARGET_AVG_NUM_EDGES,
        "target_distance": 3,
        "mode": "distance",
        "graph_family": "er",
        "split_seed": 42,
    },
    "infected_pair_er_dist3_split43": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": (
            INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB
            * INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES
            * (INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES - 1)
            / 2.0
        ),
        "er_edge_prob": INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB,
        "target_distance": 3,
        "mode": "distance",
        "graph_family": "er",
        "split_seed": 43,
    },
    "infected_pair_ba_dist3_split43": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": (
            INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB
            * INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES
            * (INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES - 1)
            / 2.0
        ),
        "ba_num_edges": INFECTED_PAIR_DIST3_DEFAULT_BA_NUM_EDGES,
        "target_distance": 3,
        "mode": "distance",
        "graph_family": "ba",
        "split_seed": 43,
    },
    "infected_pair_er_l3_multiclass_split43": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": (
            INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB
            * INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES
            * (INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES - 1)
            / 2.0
        ),
        "er_edge_prob": INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB,
        "target_distance": 3,
        "mode": "multiclass",
        "graph_family": "er",
        "split_seed": 43,
    },
    "infected_pair_ba_l3_multiclass_split43": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": (
            INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB
            * INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES
            * (INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES - 1)
            / 2.0
        ),
        "ba_num_edges": INFECTED_PAIR_DIST3_DEFAULT_BA_NUM_EDGES,
        "target_distance": 3,
        "mode": "multiclass",
        "graph_family": "ba",
        "split_seed": 43,
    },
    "infected_pair_er_dist5_split43": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": (
            INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB
            * INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES
            * (INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES - 1)
            / 2.0
        ),
        "er_edge_prob": INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB,
        "target_distance": 5,
        "mode": "distance",
        "graph_family": "er",
        "split_seed": 43,
    },
    "infected_pair_ba_dist5_split43": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": (
            INFECTED_PAIR_DIST3_MATCHED_ER_EDGE_PROB
            * INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES
            * (INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES - 1)
            / 2.0
        ),
        "ba_num_edges": INFECTED_PAIR_DIST3_DEFAULT_BA_NUM_EDGES,
        "target_distance": 5,
        "mode": "distance",
        "graph_family": "ba",
        "split_seed": 43,
    },
    "infected_pair_dist3_noise": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PAIR_DIST3_DEFAULT_TARGET_AVG_NUM_EDGES,
        "target_distance": 3,
        "mode": "distance",
        "graph_family": "er",
        "split_seed": 42,
        "noninfected_noise_std": 1.0,
    },
    "infected_pair_dist3_noise0p05": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PAIR_DIST3_DEFAULT_TARGET_AVG_NUM_EDGES,
        "target_distance": 3,
        "mode": "distance",
        "graph_family": "er",
        "split_seed": 42,
        "noninfected_noise_std": 0.05,
    },
    "infected_pair_dist6": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PAIR_DIST3_DEFAULT_TARGET_AVG_NUM_EDGES,
        "target_distance": 6,
        "mode": "distance",
        "graph_family": "er",
        "split_seed": 42,
    },
    "infected_pair_multipath3": {
        "num_graphs": INFECTED_PAIR_DIST3_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PAIR_DIST3_DEFAULT_NUM_NODES,
        "seed": INFECTED_PAIR_DIST3_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PAIR_DIST3_DEFAULT_TARGET_AVG_NUM_EDGES,
        "target_distance": 3,
        "mode": "multipath3",
        "graph_family": "er",
        "split_seed": 42,
    },
}


def _is_infected_pair_dist3_variant(path_name: str) -> bool:
    return str(path_name) in INFECTED_PAIR_DIST3_VARIANTS


def _infected_pair_dist3_bucket_code(bucket: str, target_distance: int = 3) -> int:
    bucket = str(bucket)
    if bucket.startswith("dist_ge"):
        return int(bucket[len("dist_ge"):])
    if bucket.startswith("dist"):
        return int(bucket[len("dist"):])
    if str(bucket) == "disconnected":
        return int(target_distance) + 2
    return -1


class InfectedPairDist3Dataset(InMemoryDataset):
    """PyG dataset for InfectedPair-DistK graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        target_avg_num_edges: Optional[float] = None,
    ):
        self.name = str(name)
        if self.name not in INFECTED_PAIR_DIST3_VARIANTS:
            raise ValueError(f"Unknown InfectedPair-Dist variant: {name}")

        variant = INFECTED_PAIR_DIST3_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.target_distance = int(variant["target_distance"])
        self.mode = str(variant.get("mode", "distance"))
        self.graph_family = str(variant.get("graph_family", "er"))
        self.split_seed = int(variant.get("split_seed", 42))
        self.er_edge_prob = (
            None if variant.get("er_edge_prob") is None else float(variant["er_edge_prob"])
        )
        self.ba_num_edges = int(variant.get("ba_num_edges", INFECTED_PAIR_DIST3_DEFAULT_BA_NUM_EDGES))
        self.noninfected_noise_std = float(variant.get("noninfected_noise_std", 0.0))
        self.target_avg_num_edges = (
            float(variant["target_avg_num_edges"])
            if target_avg_num_edges is None
            else float(target_avg_num_edges)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        if self.mode == "multipath3":
            payload = generate_infected_pair_multipath3_dataset(
                num_graphs=self.num_graphs,
                num_nodes=self.num_nodes,
                seed=self.seed,
                target_avg_num_edges=self.target_avg_num_edges,
            )
        elif self.mode == "multiclass":
            payload = generate_infected_pair_multiclass_dataset(
                num_graphs=self.num_graphs,
                num_nodes=self.num_nodes,
                seed=self.seed,
                target_avg_num_edges=self.target_avg_num_edges,
                graph_family=self.graph_family,
                er_edge_prob=self.er_edge_prob,
                ba_num_edges=self.ba_num_edges,
            )
        else:
            payload = generate_infected_pair_distance_dataset(
                num_graphs=self.num_graphs,
                num_nodes=self.num_nodes,
                seed=self.seed,
                target_avg_num_edges=self.target_avg_num_edges,
                target_distance=self.target_distance,
                noninfected_noise_std=self.noninfected_noise_std,
                graph_family=self.graph_family,
                er_edge_prob=self.er_edge_prob,
                ba_num_edges=self.ba_num_edges,
            )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                infected_nodes=torch.tensor(graph_payload["infected_nodes"], dtype=torch.long),
                gt_valid=torch.tensor([bool(graph_payload["gt_valid"])], dtype=torch.bool),
                gt_undirected_budget=torch.tensor([int(graph_payload["gt_undirected_budget"])], dtype=torch.long),
                gt_directed_label_count=torch.tensor([int(graph_payload["gt_directed_label_count"])], dtype=torch.long),
                target_distance=torch.tensor([int(graph_payload["target_distance"])], dtype=torch.long),
                positive_target_distance=torch.tensor(
                    [int(graph_payload["positive_target_distance"])],
                    dtype=torch.long,
                ),
                shortest_path_count=torch.tensor([int(graph_payload["shortest_path_count"])], dtype=torch.long),
                length3_path_count=torch.tensor(
                    [int(graph_payload.get("length3_path_count", 0))],
                    dtype=torch.long,
                ),
                length3_edge_union_size=torch.tensor(
                    [int(graph_payload.get("length3_edge_union_size", 0))],
                    dtype=torch.long,
                ),
                num_edges_undirected=torch.tensor([int(graph_payload["num_edges"])], dtype=torch.long),
                graph_family_code=torch.tensor(
                    [0 if str(graph_payload.get("graph_family", "er")) == "er" else 1],
                    dtype=torch.long,
                ),
                graph_seed=torch.tensor(
                    [-1 if graph_payload.get("graph_seed") is None else int(graph_payload["graph_seed"])],
                    dtype=torch.long,
                ),
                er_edge_prob=torch.tensor([float(graph_payload["er_edge_prob"])], dtype=torch.float32),
                ba_num_edges=torch.tensor([int(graph_payload.get("ba_num_edges", 0))], dtype=torch.long),
                split_seed=torch.tensor([int(self.split_seed)], dtype=torch.long),
                target_avg_num_edges=torch.tensor([float(graph_payload["target_avg_num_edges"])], dtype=torch.float32),
                noninfected_noise_std=torch.tensor(
                    [float(graph_payload.get("noninfected_noise_std", self.noninfected_noise_std))],
                    dtype=torch.float32,
                ),
                distance_bucket_code=torch.tensor(
                    [
                        _infected_pair_dist3_bucket_code(
                            str(graph_payload["distance_bucket"]),
                            target_distance=self.target_distance,
                        )
                    ],
                    dtype=torch.long,
                ),
                distance_class=torch.tensor(
                    [int(graph_payload.get("distance_class", graph_payload["label"]))],
                    dtype=torch.long,
                ),
                distance_class_cap=torch.tensor(
                    [int(graph_payload.get("distance_class_cap", 0))],
                    dtype=torch.long,
                ),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# InfectedHouse Graph Classification Dataset
# ============================================================================

INFECTED_HOUSE_VARIANTS: Dict[str, Dict[str, object]] = {
    "infected_house": {
        "num_graphs": INFECTED_HOUSE_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_HOUSE_DEFAULT_NUM_NODES,
        "seed": INFECTED_HOUSE_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_HOUSE_DEFAULT_TARGET_AVG_NUM_EDGES,
    },
}


def _is_infected_house_variant(path_name: str) -> bool:
    return str(path_name) in INFECTED_HOUSE_VARIANTS


class InfectedHouseDataset(InMemoryDataset):
    """PyG dataset for InfectedHouse graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        target_avg_num_edges: Optional[float] = None,
    ):
        self.name = str(name)
        if self.name not in INFECTED_HOUSE_VARIANTS:
            raise ValueError(f"Unknown InfectedHouse variant: {name}")

        variant = INFECTED_HOUSE_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.target_avg_num_edges = (
            float(variant["target_avg_num_edges"])
            if target_avg_num_edges is None
            else float(target_avg_num_edges)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        edge_token = str(float(self.target_avg_num_edges)).replace(".", "p")
        return [
            f"data_graphs{self.num_graphs}_nodes{self.num_nodes}_edges{edge_token}_seed{self.seed}.pt"
        ]

    def download(self):
        pass

    def process(self):
        payload = generate_infected_house_dataset(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            seed=self.seed,
            target_avg_num_edges=self.target_avg_num_edges,
        )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                infected_node=torch.tensor([int(graph_payload["infected_node"])], dtype=torch.long),
                infected_nodes=torch.tensor(graph_payload["infected_nodes"], dtype=torch.long),
                house_nodes=torch.tensor(graph_payload["house_nodes"], dtype=torch.long),
                gt_valid=torch.tensor([bool(graph_payload["gt_valid"])], dtype=torch.bool),
                gt_undirected_budget=torch.tensor([int(graph_payload["gt_undirected_budget"])], dtype=torch.long),
                gt_directed_label_count=torch.tensor([int(graph_payload["gt_directed_label_count"])], dtype=torch.long),
                num_edges_undirected=torch.tensor([int(graph_payload["num_edges"])], dtype=torch.long),
                er_edge_prob=torch.tensor([float(graph_payload["er_edge_prob"])], dtype=torch.float32),
                target_avg_num_edges=torch.tensor([float(graph_payload["target_avg_num_edges"])], dtype=torch.float32),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# InfectedPentagon Graph Classification Dataset
# ============================================================================

INFECTED_PENTAGON_VARIANTS: Dict[str, Dict[str, object]] = {
    "infected_pentagon": {
        "num_graphs": INFECTED_PENTAGON_DEFAULT_NUM_GRAPHS,
        "num_nodes": INFECTED_PENTAGON_DEFAULT_NUM_NODES,
        "seed": INFECTED_PENTAGON_DEFAULT_SEED,
        "target_avg_num_edges": INFECTED_PENTAGON_DEFAULT_TARGET_AVG_NUM_EDGES,
    },
}


def _is_infected_pentagon_variant(path_name: str) -> bool:
    return str(path_name) in INFECTED_PENTAGON_VARIANTS


class InfectedPentagonDataset(InMemoryDataset):
    """PyG dataset for InfectedPentagon graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        target_avg_num_edges: Optional[float] = None,
    ):
        self.name = str(name)
        if self.name not in INFECTED_PENTAGON_VARIANTS:
            raise ValueError(f"Unknown InfectedPentagon variant: {name}")

        variant = INFECTED_PENTAGON_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.target_avg_num_edges = (
            float(variant["target_avg_num_edges"])
            if target_avg_num_edges is None
            else float(target_avg_num_edges)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        edge_token = str(float(self.target_avg_num_edges)).replace(".", "p")
        return [
            f"data_graphs{self.num_graphs}_nodes{self.num_nodes}_edges{edge_token}_seed{self.seed}.pt"
        ]

    def download(self):
        pass

    def process(self):
        payload = generate_infected_pentagon_dataset(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            seed=self.seed,
            target_avg_num_edges=self.target_avg_num_edges,
        )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                infected_nodes=torch.tensor(graph_payload["infected_nodes"], dtype=torch.long),
                pentagon_nodes=torch.tensor(graph_payload["pentagon_nodes"], dtype=torch.long),
                gt_valid=torch.tensor([bool(graph_payload["gt_valid"])], dtype=torch.bool),
                gt_undirected_budget=torch.tensor([int(graph_payload["gt_undirected_budget"])], dtype=torch.long),
                gt_directed_label_count=torch.tensor([int(graph_payload["gt_directed_label_count"])], dtype=torch.long),
                pentagon_path_lengths=torch.tensor(graph_payload["pentagon_path_lengths"], dtype=torch.long),
                num_edges_undirected=torch.tensor([int(graph_payload["num_edges"])], dtype=torch.long),
                er_edge_prob=torch.tensor([float(graph_payload["er_edge_prob"])], dtype=torch.float32),
                target_avg_num_edges=torch.tensor([float(graph_payload["target_avg_num_edges"])], dtype=torch.float32),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# RandomInfectedTriCenter-3 Graph Classification Dataset
# ============================================================================

RANDOM_TRICENTER_VARIANTS: Dict[str, Dict[str, object]] = {
    "random_infected_tricenter_3": {
        "num_graphs": RANDOM_TRICENTER_DEFAULT_NUM_GRAPHS,
        "num_nodes": RANDOM_TRICENTER_DEFAULT_NUM_NODES,
        "seed": RANDOM_TRICENTER_DEFAULT_SEED,
        "target_avg_num_edges": RANDOM_TRICENTER_DEFAULT_TARGET_AVG_NUM_EDGES,
    },
}


def _is_random_tricenter_variant(path_name: str) -> bool:
    return str(path_name) in RANDOM_TRICENTER_VARIANTS


class RandomInfectedTriCenterDataset(InMemoryDataset):
    """PyG dataset for RandomInfectedTriCenter-3 graph classification."""

    def __init__(
        self,
        root: str,
        name: str,
        transform=None,
        pre_transform=None,
        num_graphs: Optional[int] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
        target_avg_num_edges: Optional[float] = None,
    ):
        self.name = str(name)
        if self.name not in RANDOM_TRICENTER_VARIANTS:
            raise ValueError(f"Unknown RandomInfectedTriCenter variant: {name}")

        variant = RANDOM_TRICENTER_VARIANTS[self.name]
        self.num_graphs = int(variant["num_graphs"]) if num_graphs is None else int(num_graphs)
        self.num_nodes = int(variant["num_nodes"]) if num_nodes is None else int(num_nodes)
        self.seed = int(variant["seed"]) if seed is None else int(seed)
        self.target_avg_num_edges = (
            float(variant["target_avg_num_edges"])
            if target_avg_num_edges is None
            else float(target_avg_num_edges)
        )

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        payload = generate_random_infected_tricenter3_dataset(
            num_graphs=self.num_graphs,
            num_nodes=self.num_nodes,
            seed=self.seed,
            target_avg_num_edges=self.target_avg_num_edges,
        )

        data_list: List[Data] = []
        for graph_payload in payload["graphs"]:
            edge_index_np = np.asarray(graph_payload["edge_index"], dtype=np.int64)
            edge_labels_np = np.asarray(graph_payload["edge_labels"], dtype=np.int64)

            data = Data(
                x=torch.tensor(graph_payload["features"], dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([int(graph_payload["label"])], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_payload["graph_idx"])], dtype=torch.long),
                infected_nodes=torch.tensor(graph_payload["infected_nodes"], dtype=torch.long),
                gt_valid=torch.tensor([bool(graph_payload["gt_valid"])], dtype=torch.bool),
                gt_center=torch.tensor([int(graph_payload["gt_center"])], dtype=torch.long),
                gt_undirected_budget=torch.tensor([int(graph_payload["gt_undirected_budget"])], dtype=torch.long),
                gt_directed_label_count=torch.tensor([int(graph_payload["gt_directed_label_count"])], dtype=torch.long),
                num_valid_centers=torch.tensor([int(graph_payload["num_valid_centers"])], dtype=torch.long),
                pairwise_radius2_pair_count=torch.tensor(
                    [int(graph_payload["pairwise_radius2_pair_count"])],
                    dtype=torch.long,
                ),
                num_edges_undirected=torch.tensor([int(graph_payload["num_edges"])], dtype=torch.long),
                er_edge_prob=torch.tensor([float(graph_payload["er_edge_prob"])], dtype=torch.float32),
                target_avg_num_edges=torch.tensor([float(graph_payload["target_avg_num_edges"])], dtype=torch.float32),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved {self.name} PyG dataset ({len(data_list)} graphs) to {self.processed_paths[0]}")


# ============================================================================
# BA-2Motifs Graph Classification Dataset
# ============================================================================

class BA2MotifsDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for BA-2Motifs graph classification dataset.
    Uses caching in processed/data.pt
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['BA-2motif.pkl']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """Convert pkl file to PyG Data objects and save"""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        pkl_path = os.path.join(dir_path, 'pkls', 'BA-2motif.pkl')

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"PKL file not found: {pkl_path}")

        print("Converting BA-2Motifs pkl to PyG format...")

        with open(pkl_path, 'rb') as f:
            adjs, features, labels = pkl.load(f)

        n_graphs = adjs.shape[0]

        # Create Data objects for each graph
        data_list = []
        for i in range(n_graphs):
            adj = adjs[i]
            feat = features[i].astype('float32')
            label = labels[i]

            # Convert adj to edge_index
            edge_index = np.argwhere(adj > 0.).T

            # Get label as class index
            if len(label.shape) > 0 and label.shape[0] > 1:
                y = np.argmax(label)
            else:
                y = int(label)

            data = Data(
                x=torch.tensor(feat, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor([y], dtype=torch.long)
            )
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        print(f"Saved BA-2Motifs PyG dataset ({n_graphs} graphs) to {self.processed_paths[0]}")


# ============================================================================
# BA2Grid Graph Classification Dataset (Longa et al. artifact)
# ============================================================================

LONGA_BA2GRID_SHA256 = "bf3dfc8337cbb9ab8a319ecda70bcfc7b8f30a9ef05ddd387b05318a5ea555d9"
LONGA_BA2GRID_COMMIT = "73eca19ca10007ee2866186bb63ddaf59a9b0396"


class BA2GridLongaDataset(InMemoryDataset):
    """PyG conversion of Longa et al.'s precomputed ``BA-2grid.pkl``.

    The raw artifact is verified before unpickling. Positive graphs contain a
    3x3 grid in their last nine node positions. Ground-truth edge labels mark
    the 12 undirected grid edges (24 directed ``edge_index`` entries), excluding
    the single BA-to-grid attachment edge. Negative graphs have an empty mask.
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        self._verify_raw_artifact()
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @staticmethod
    def _raw_artifact_path() -> str:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_path, "pkls", "BA-2grid.pkl")

    @classmethod
    def _verify_raw_artifact(cls) -> str:
        pkl_path = cls._raw_artifact_path()
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Longa BA2Grid artifact not found: {pkl_path}. "
                f"See Version5/260700-datasets.md for the pinned download URL."
            )

        digest = hashlib.sha256()
        with open(pkl_path, "rb") as raw_file:
            for chunk in iter(lambda: raw_file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != LONGA_BA2GRID_SHA256:
            raise ValueError(
                "Longa BA2Grid SHA-256 mismatch: "
                f"expected {LONGA_BA2GRID_SHA256}, got {actual_sha256}"
            )
        return pkl_path

    @property
    def raw_file_names(self) -> List[str]:
        return ["BA-2grid.pkl"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        pkl_path = self._verify_raw_artifact()

        with open(pkl_path, "rb") as fin:
            adjs, features, labels = pkl.load(fin)

        if not (len(adjs) == len(features) == len(labels) == 2000):
            raise ValueError("Expected 2,000 aligned BA2Grid graphs, features, and labels")
        label_values, label_counts = np.unique(np.asarray(labels), return_counts=True)
        if not (
            np.array_equal(label_values, np.asarray([0, 1]))
            and np.array_equal(label_counts, np.asarray([1000, 1000]))
        ):
            raise ValueError(
                f"Unexpected BA2Grid class distribution: {dict(zip(label_values, label_counts))}"
            )

        data_list: List[Data] = []
        for graph_idx, (adj, feat, label) in enumerate(zip(adjs, features, labels)):
            adj_np = np.asarray(adj)
            feat_np = np.asarray(feat, dtype=np.float32)
            if adj_np.ndim != 2 or adj_np.shape[0] != adj_np.shape[1]:
                raise ValueError(f"Graph {graph_idx} has a non-square adjacency matrix")
            if feat_np.shape[0] != adj_np.shape[0]:
                raise ValueError(f"Graph {graph_idx} has misaligned node features")
            if not np.array_equal(adj_np, adj_np.T):
                raise ValueError(f"Graph {graph_idx} adjacency is not symmetric")
            if np.any(np.diag(adj_np) != 0):
                raise ValueError(f"Graph {graph_idx} adjacency contains a self-loop")

            edge_index_np = np.argwhere(adj_np > 0).T.astype(np.int64, copy=False)
            y = int(label)
            if y == 1:
                grid_start = int(adj_np.shape[0]) - 9
                edge_labels_np = (
                    (edge_index_np[0] >= grid_start)
                    & (edge_index_np[1] >= grid_start)
                ).astype(np.int64, copy=False)
                if int(edge_labels_np.sum()) != 24:
                    raise ValueError(
                        f"Positive graph {graph_idx} does not contain 24 directed grid edges"
                    )
            else:
                edge_labels_np = np.zeros(edge_index_np.shape[1], dtype=np.int64)

            data = Data(
                x=torch.tensor(feat_np, dtype=torch.float32),
                edge_index=torch.tensor(edge_index_np, dtype=torch.long),
                y=torch.tensor([y], dtype=torch.long),
                edge_labels=torch.tensor(edge_labels_np, dtype=torch.long),
                edge_attr=torch.ones((edge_index_np.shape[1], 1), dtype=torch.float32),
                idx=torch.tensor([int(graph_idx)], dtype=torch.long),
            )
            data.edge_mask = data.edge_labels.float()
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print(
            f"Saved BA2Grid-Longa PyG dataset ({len(data_list)} graphs, "
            f"source commit {LONGA_BA2GRID_COMMIT}) to {self.processed_paths[0]}"
        )


# ============================================================================
# MUTAG Dataset using PyG TUDataset (Official Loader)
# ============================================================================

def load_mutag_pyg(root: str = None):
    """
    Load MUTAG dataset using PyG's official TUDataset loader.
    This is much faster than loading from text files.

    Args:
        root: Root directory to save/load the dataset

    Returns:
        dataset: PyG TUDataset for MUTAG
    """
    from torch_geometric.datasets import TUDataset

    if root is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(dir_path, 'pyg_data')

    # Use PyG's TUDataset for official MUTAG
    # Note: This is the standard MUTAG, different from Mutagenicity
    dataset = TUDataset(root=root, name='MUTAG')
    return dataset


# ============================================================================
# PROTEINS Dataset using PyG TUDataset (Official Loader)
# ============================================================================

def load_proteins_pyg(root: str = None):
    """
    Load PROTEINS dataset using PyG's official TUDataset loader.

    Args:
        root: Root directory to save/load the dataset

    Returns:
        dataset: PyG TUDataset for PROTEINS
    """
    from torch_geometric.datasets import TUDataset

    if root is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(dir_path, 'pyg_data')

    dataset = TUDataset(root=root, name='PROTEINS')
    return dataset


# ============================================================================
# PubMed/Cora Datasets using PyG Planetoid (Official Loader)
# ============================================================================

def load_pubmed_pyg(root: str = None):
    """
    Load PubMed dataset using PyG's official Planetoid loader.

    Args:
        root: Root directory to save/load the dataset

    Returns:
        data: PyG Data object for PubMed
    """
    from torch_geometric.datasets import Planetoid

    if root is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(dir_path, 'pyg_data')

    dataset = Planetoid(root=root, name='PubMed')
    return dataset[0]  # Return the single graph


def load_cora_pyg(root: str = None):
    """
    Load Cora dataset using PyG's official Planetoid loader.

    Args:
        root: Root directory to save/load the dataset

    Returns:
        data: PyG Data object for Cora
    """
    from torch_geometric.datasets import Planetoid

    if root is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(dir_path, 'pyg_data')

    dataset = Planetoid(root=root, name='Cora')
    return dataset[0]  # Return the single graph


# ============================================================================
# Unified Dataset Loader Interface
# ============================================================================

def get_pyg_dataset_root() -> str:
    """Get the root directory for PyG processed datasets"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'pyg_processed')


def load_dataset_pyg(dataset_name: str, shuffle: bool = True, normalize_features: bool = True):
    """
    Load dataset using PyG with caching support.

    Supported datasets:
    - Node classification: 'BA-Shapes', 'BA-House', 'BA-Community', 'Tree-Cycles', 'Tree-Cycles-Noise', 'Tree-Cycles-Noise0.5', 'Tree-Cycles-Noise0.1', 'Tree-Cycles-Noise0.05', 'Tree-Cycles-Uniform0.5', 'Tree-Grid', 'PubMed', 'Cora',
      'Infection4', 'Infection4-one-hot', 'Infection3', 'Infection3-one-hot', 'Infection3-one-hot-binary', 'Infection3-one-hot-binary-Noise'
    - Graph classification: 'BA-2Motifs', 'BA2Grid-Longa', 'MarkerPath-Exact3', 'InfectedSteiner-3', 'InfectedPair-Unique3', 'InfectedPair-Dist3', 'InfectedPair-Dist3-Noise', 'InfectedPair-Dist3-Noise0.05', 'InfectedPair-Dist6', 'InfectedPair-MultiPath3', 'InfectedHouse', 'RandomInfectedTriCenter-3', 'BA-HouseGrid', 'BA-HouseGrid-wb60-m3',
      'BA-HouseANDGrid', 'BA-HouseANDGrid-wb15-m1', 'MUTAG', 'PROTEINS'

    Args:
        dataset_name: Dataset name (supports both friendly and internal names)
        shuffle: Whether to shuffle (only applies to graph datasets)

    Returns:
        Tuple of (edge_index, features, labels, train_mask, val_mask, test_mask)
        in the same format as the original load_dataset function for compatibility
    """
    # Normalize to canonical name, then get path name
    canonical_name = normalize_dataset_name(dataset_name)
    path_name = get_path_name(dataset_name)
    root = get_pyg_dataset_root()

    if path_name in SYNTHETIC_NODE_BASE_NAMES or path_name in SYNTHETIC_NODE_VARIANTS:
        # Node classification - synthetic datasets (from pkl)
        variant = _synthetic_node_variant(path_name)
        base_path_name = str(variant['base_path_name'])
        dataset_root = os.path.join(root, path_name)
        dataset = SyntheticNodeDataset(
            root=dataset_root,
            name=base_path_name,
            normalize_features=normalize_features,
            feature_noise_std=float(variant.get('feature_noise_std', 0.0)),
            feature_noise_seed=int(variant.get('feature_noise_seed', 1000003)),
            feature_noise_distribution=str(variant.get('feature_noise_distribution', 'gaussian')),
        )
        data = dataset[0]

        # Return in original format for compatibility
        graph = data.edge_index.numpy()
        features = data.x.numpy()
        labels = data.y.numpy()
        if canonical_name == 'BA-House':
            labels = _collapse_ba_house_labels(labels)
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        return graph, features, labels, train_mask, val_mask, test_mask

    elif path_name == 'ba2':
        # Graph classification - BA-2Motifs (from pkl)
        dataset_root = os.path.join(root, 'ba2')
        dataset = BA2MotifsDataset(root=dataset_root)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            indices = RandomState(42).permutation(indices)

        # Convert to original format
        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        # Create masks
        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif path_name == 'ba2grid_longa':
        # Graph classification - Longa et al. precomputed BA2Grid artifact.
        dataset_root = os.path.join(root, path_name)
        dataset = BA2GridLongaDataset(root=dataset_root)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            indices = RandomState(42).permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []
        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)
        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_ba_house_grid_variant(path_name):
        # Graph classification - synthetic BA motif datasets generated in PyG format
        dataset_root = os.path.join(root, path_name)
        dataset = BAHouseGridDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            indices = _ba_house_grid_shuffled_indices(path_name, n_graphs)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_infected_house_variant(path_name):
        # Graph classification - ER graph with one planted house and one random infected node
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedHouseDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_infected_pentagon_variant(path_name):
        # Graph classification - ER graph with one planted pentagon and two infected nodes
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedPentagonDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_marker_path_variant(path_name):
        # Graph classification - synthetic marker-path distance dataset
        dataset_root = os.path.join(root, path_name)
        dataset = MarkerPathDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_infected_steiner_variant(path_name):
        # Graph classification - synthetic infected Steiner certificate dataset
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedSteinerDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_infected_pair_variant(path_name):
        # Graph classification - ER graph with two infected nodes and unique-distance-3 positives
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedPairDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_infected_pair_dist3_variant(path_name):
        # Graph classification - ER/BA graph with two random infected nodes; label is distance exactly K
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedPairDist3Dataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(int(dataset.split_seed))
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif _is_random_tricenter_variant(path_name):
        # Graph classification - random graph and random infected triple with intrinsic tri-center certificate
        dataset_root = os.path.join(root, path_name)
        dataset = RandomInfectedTriCenterDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif path_name == 'MUTAG':
        # MUTAG - PyG TUDataset (188 graphs, official benchmark)
        dataset = load_mutag_pyg()

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        # Convert to original format
        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        # Create masks (80/10/10 split)
        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif path_name == 'PROTEINS':
        # PROTEINS - PyG TUDataset (1113 graphs, official benchmark)
        dataset = load_proteins_pyg()

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        if shuffle:
            prng = RandomState(42)
            indices = prng.permutation(indices)

        # Convert to original format
        edge_index_list = []
        features_list = []
        labels_list = []

        for i in indices:
            data = dataset[int(i)]
            if data.x is None:
                raise ValueError("PROTEINS dataset did not provide node features.")
            edge_index_list.append(data.edge_index.numpy())
            features_list.append(data.x.numpy())
            labels_list.append(data.y.numpy())

        # Create masks (80/10/10 split)
        train_indices = np.arange(0, int(n_graphs * 0.8))
        val_indices = np.arange(int(n_graphs * 0.8), int(n_graphs * 0.9))
        test_indices = np.arange(int(n_graphs * 0.9), n_graphs)

        train_mask = np.full(n_graphs, False)
        train_mask[train_indices] = True
        val_mask = np.full(n_graphs, False)
        val_mask[val_indices] = True
        test_mask = np.full(n_graphs, False)
        test_mask[test_indices] = True

        return edge_index_list, features_list, np.array(labels_list), train_mask, val_mask, test_mask

    elif path_name in ('pubmed', 'cora'):
        # Planetoid node classification datasets
        data = load_pubmed_pyg() if path_name == 'pubmed' else load_cora_pyg()

        graph = data.edge_index.numpy()
        features = data.x.numpy()
        labels = data.y.numpy()
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        return graph, features, labels, train_mask, val_mask, test_mask

    elif path_name in PYG_INFECTION_VARIANTS:
        variant = PYG_INFECTION_VARIANTS[path_name]
        dataset_root = os.path.join(root, path_name)
        dataset = PyGInfectionNodeDataset(
            root=dataset_root,
            graph_family=str(variant["graph_family"]),
            dataset_seed=int(variant.get("dataset_seed", 42)),
            split_seed=int(variant["split_seed"]),
            binary_positive_label=variant.get("binary_positive_label"),
        )
        data = dataset[0]

        graph = data.edge_index.numpy()
        features = data.x.numpy()
        labels = data.y.numpy()
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        return graph, features, labels, train_mask, val_mask, test_mask

    elif path_name in INFECTION_VARIANTS:
        # Infection synthetic node datasets with distance caps:
        # infection4*: class 5 means dist>=5, infection3*: class 4 means dist>=4.
        variant = INFECTION_VARIANTS[path_name]
        dataset_root = os.path.join(root, path_name)
        dataset = InfectionNodeDataset(
            root=dataset_root,
            feature_mode=variant["feature_mode"],
            max_distance_class=variant["max_distance_class"],
            binary_positive_label=variant.get("binary_positive_label"),
            noninfected_noise_std=float(variant.get("noninfected_noise_std", 0.0)),
            bernoulli_feature_dim=int(variant.get("bernoulli_feature_dim", 2)),
            bernoulli_feature_prob=float(variant.get("bernoulli_feature_prob", 0.5)),
        )
        data = dataset[0]

        graph = data.edge_index.numpy()
        features = data.x.numpy()
        labels = data.y.numpy()
        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        return graph, features, labels, train_mask, val_mask, test_mask

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {ALL_DATASETS}")


# ============================================================================
# Ground Truth Loaders (PyG compatible)
# ============================================================================

def load_dataset_ground_truth_pyg(dataset_name: str, test_indices=None):
    """
    Load ground truth using PyG cached datasets.
    Returns same format as original load_dataset_ground_truth for compatibility.

    Supports both friendly names and internal names.

    Args:
        dataset_name: Dataset name (supports both friendly and internal names)
        test_indices: Optional list of test indices to filter

    Returns:
        (ground_truth_data, indices) tuple
        - Synthetic motif datasets: ground_truth_data = (graph_edge_index, edge_labels)
        - Infection3/Infection4 variants: ground_truth_data = (graph_edge_index, node_to_gt_edges_dict)
    """
    # Normalize to canonical and path names
    canonical_name = normalize_dataset_name(dataset_name)
    path_name = get_path_name(dataset_name)
    root = get_pyg_dataset_root()

    if path_name in SYNTHETIC_NODE_BASE_NAMES or path_name in SYNTHETIC_NODE_VARIANTS:
        variant = _synthetic_node_variant(path_name)
        base_path_name = str(variant['base_path_name'])
        dataset_root = os.path.join(root, path_name)
        dataset = SyntheticNodeDataset(
            root=dataset_root,
            name=base_path_name,
            feature_noise_std=float(variant.get('feature_noise_std', 0.0)),
            feature_noise_seed=int(variant.get('feature_noise_seed', 1000003)),
            feature_noise_distribution=str(variant.get('feature_noise_distribution', 'gaussian')),
        )
        data = dataset[0]

        graph = data.edge_index.numpy()
        edge_labels = data.edge_labels.numpy()

        if canonical_name in ('BA-Shapes', 'BA-House', 'BA-Community'):
            test_nodes = np.where(data.test_mask.numpy())[0].astype(np.int64)
            gt_nodes: Set[int] = set()
            for k in range(int(graph.shape[1])):
                if int(edge_labels[k]) <= 0:
                    continue
                gt_nodes.add(int(graph[0, k]))
                gt_nodes.add(int(graph[1, k]))

            default_test_indices = [int(i) for i in test_nodes if int(i) in gt_nodes]
            if test_indices is None:
                return (graph, edge_labels), default_test_indices
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in default_test_indices if int(i) in test_index_set]
            return (graph, edge_labels), filtered

        # Define test ranges
        if base_path_name in ['syn1', 'syn2']:
            default_range = range(400, 700, 5)
            all_range = range(400, 700, 1)
        elif base_path_name == 'syn3':
            default_range = range(511, 871, 6)
            all_range = range(511, 871, 1)
        elif base_path_name == 'syn4':
            default_range = range(511, 800, 1)
            all_range = range(511, 800, 1)

        if test_indices is None:
            return (graph, edge_labels), list(default_range)
        else:
            filtered = [i for i in all_range if i in test_indices]
            return (graph, edge_labels), filtered

    elif path_name == 'ba2':
        dataset_root = os.path.join(root, 'ba2')
        dataset = BA2MotifsDataset(root=dataset_root)

        n_graphs = len(dataset)
        shuffled_indices = RandomState(42).permutation(np.arange(n_graphs))

        # Extract edge labels based on node positions (same as original)
        edge_index_list = []
        edge_labels_list = []

        insert = 20
        skip = 5

        for orig_idx in shuffled_indices:
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_index_list.append(edge_index)

            labels = []
            for pair in edge_index.T:
                r, c = pair[0], pair[1]
                if r >= insert and r < insert + skip and c >= insert and c < insert + skip:
                    labels.append(1)
                else:
                    labels.append(0)
            edge_labels_list.append(np.array(labels))

        motif_graph_indices = [
            int(i) for i, edge_labels in enumerate(edge_labels_list)
            if np.any(np.asarray(edge_labels) > 0)
        ]
        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(motif_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in motif_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif path_name == 'ba2grid_longa':
        dataset_root = os.path.join(root, path_name)
        dataset = BA2GridLongaDataset(root=dataset_root)

        n_graphs = len(dataset)
        shuffled_indices = RandomState(42).permutation(np.arange(n_graphs))
        edge_index_list = []
        edge_labels_list = []
        motif_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)
            if int(data.y.view(-1)[0].item()) == 1 and int(edge_labels.sum()) == 24:
                motif_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        motif_graph_index_set = set(motif_graph_indices)
        default_test_indices = [
            int(i) for i in range(test_start, n_graphs)
            if int(i) in motif_graph_index_set
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        test_index_set = {int(i) for i in test_indices}
        filtered = [int(i) for i in motif_graph_indices if int(i) in test_index_set]
        return (edge_index_list, edge_labels_list), filtered

    elif _is_ba_house_grid_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = BAHouseGridDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        shuffled_indices = _ba_house_grid_shuffled_indices(path_name, n_graphs)

        edge_index_list = []
        edge_labels_list = []
        motif_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            variant_kind = _ba_house_grid_variant_kind(path_name)
            if variant_kind == 'house_and_grid':
                has_target_motif = label == 1 and int(edge_labels.sum()) > 0
            elif variant_kind == 'house_or_grid':
                has_target_motif = (
                    label == 1
                    and hasattr(data, "gt_valid")
                    and bool(data.gt_valid.view(-1)[0].item())
                    and int(edge_labels.sum()) > 0
                )
            else:
                has_target_motif = int(edge_labels.sum()) > 0
            if has_target_motif:
                motif_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(motif_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in motif_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_infected_house_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedHouseDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())
            gt_valid = bool(data.gt_valid.view(-1)[0].item()) if hasattr(data, "gt_valid") else False

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if label == 1 and gt_valid and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_infected_pentagon_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedPentagonDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())
            gt_valid = bool(data.gt_valid.view(-1)[0].item()) if hasattr(data, "gt_valid") else False

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if label == 1 and gt_valid and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_marker_path_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = MarkerPathDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if label == 1 and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_infected_steiner_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedSteinerDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())
            gt_valid = bool(data.gt_valid.view(-1)[0].item()) if hasattr(data, "gt_valid") else False

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if label == 1 and gt_valid and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_infected_pair_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedPairDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())
            gt_valid = bool(data.gt_valid.view(-1)[0].item()) if hasattr(data, "gt_valid") else False

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if label == 1 and gt_valid and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_infected_pair_dist3_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = InfectedPairDist3Dataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(int(dataset.split_seed))
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())
            gt_valid = bool(data.gt_valid.view(-1)[0].item()) if hasattr(data, "gt_valid") else False

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            # Binary variants use label 1 for distance 3, while the
            # four-class variants use label 2. GT validity is the common
            # task-independent criterion.
            if gt_valid and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif _is_random_tricenter_variant(path_name):
        dataset_root = os.path.join(root, path_name)
        dataset = RandomInfectedTriCenterDataset(root=dataset_root, name=path_name)

        n_graphs = len(dataset)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(np.arange(n_graphs))

        edge_index_list = []
        edge_labels_list = []
        positive_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            edge_labels = data.edge_labels.numpy()
            label = int(data.y.view(-1)[0].item())
            gt_valid = bool(data.gt_valid.view(-1)[0].item()) if hasattr(data, "gt_valid") else False

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if label == 1 and gt_valid and int(edge_labels.sum()) > 0:
                positive_graph_indices.append(int(shuffled_pos))

        test_start = int(n_graphs * 0.9)
        test_end = int(n_graphs)
        default_test_indices = [
            int(i) for i in range(test_start, test_end) if int(i) in set(positive_graph_indices)
        ]

        if test_indices is None:
            return (edge_index_list, edge_labels_list), default_test_indices
        else:
            test_index_set = {int(i) for i in test_indices}
            filtered = [int(i) for i in positive_graph_indices if int(i) in test_index_set]
            return (edge_index_list, edge_labels_list), filtered

    elif path_name == 'MUTAG':
        dataset = load_mutag_pyg()

        n_graphs = len(dataset)
        indices = np.arange(n_graphs)
        prng = RandomState(42)
        shuffled_indices = prng.permutation(indices)

        edge_index_list = []
        edge_labels_list = []
        motif_graph_indices: List[int] = []

        for shuffled_pos, orig_idx in enumerate(shuffled_indices):
            data = dataset[int(orig_idx)]
            edge_index = data.edge_index.numpy()
            node_features = data.x.numpy()
            edge_labels = build_mutag_ground_truth_edge_labels(
                edge_index=edge_index,
                node_features=node_features,
            )

            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)

            if int(edge_labels.sum()) > 0:
                motif_graph_indices.append(int(shuffled_pos))

        if test_indices is None:
            return (edge_index_list, edge_labels_list), motif_graph_indices
        else:
            filtered = [i for i in motif_graph_indices if i in test_indices]
            return (edge_index_list, edge_labels_list), filtered

    elif path_name in PYG_INFECTION_VARIANTS:
        variant = PYG_INFECTION_VARIANTS[path_name]
        dataset_root = os.path.join(root, path_name)
        dataset = PyGInfectionNodeDataset(
            root=dataset_root,
            graph_family=str(variant["graph_family"]),
            dataset_seed=int(variant.get("dataset_seed", 42)),
            split_seed=int(variant["split_seed"]),
            binary_positive_label=variant.get("binary_positive_label"),
        )
        data = dataset[0]
        gt_payload = dataset.gt_payload
        graph = data.edge_index.numpy()
        node_to_gt_edges: Dict[int, List[Tuple[int, int]]] = {
            int(node_idx): [(int(a), int(b)) for (a, b) in edges]
            for node_idx, edges in gt_payload.get("node_to_gt_edges", {}).items()
        }
        gt_test_indices = [int(i) for i in gt_payload.get("gt_test_indices", [])]
        if test_indices is None:
            return (graph, node_to_gt_edges), gt_test_indices
        test_index_set = {int(i) for i in test_indices}
        filtered = [int(i) for i in gt_test_indices if int(i) in test_index_set]
        return (graph, node_to_gt_edges), filtered

    elif path_name in INFECTION_VARIANTS:
        variant = INFECTION_VARIANTS[path_name]
        dataset_root = os.path.join(root, path_name)
        dataset = InfectionNodeDataset(
            root=dataset_root,
            feature_mode=variant["feature_mode"],
            max_distance_class=variant["max_distance_class"],
            binary_positive_label=variant.get("binary_positive_label"),
            noninfected_noise_std=float(variant.get("noninfected_noise_std", 0.0)),
            bernoulli_feature_dim=int(variant.get("bernoulli_feature_dim", 2)),
            bernoulli_feature_prob=float(variant.get("bernoulli_feature_prob", 0.5)),
        )
        data = dataset[0]
        gt_payload = dataset.gt_payload

        graph = data.edge_index.numpy()
        node_to_gt_edges: Dict[int, List[Tuple[int, int]]] = {
            int(node_idx): [(int(a), int(b)) for (a, b) in edges]
            for node_idx, edges in gt_payload.get("node_to_gt_edges", {}).items()
        }
        gt_test_indices = [int(i) for i in gt_payload.get("gt_test_indices", [])]
        if variant.get("binary_positive_label") is not None:
            gt_test_indices = [
                int(i)
                for i in gt_test_indices
                if int(i) < int(data.y.numel()) and int(data.y[int(i)].item()) == 1
            ]
            positive_index_set = set(gt_test_indices)
            node_to_gt_edges = {
                int(node_idx): edges
                for node_idx, edges in node_to_gt_edges.items()
                if int(node_idx) in positive_index_set
            }

        if test_indices is None:
            return (graph, node_to_gt_edges), gt_test_indices
        else:
            filtered = [i for i in gt_test_indices if i in test_indices]
            return (graph, node_to_gt_edges), filtered

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# Utility: Clear PyG cache
# ============================================================================

def clear_pyg_cache(dataset_name: str = None):
    """
    Clear PyG processed cache for a dataset or all datasets.

    Args:
        dataset_name: Specific dataset to clear, or None for all
    """
    import shutil
    root = get_pyg_dataset_root()

    if dataset_name is not None:
        cache_path = os.path.join(root, dataset_name, 'processed')
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"Cleared cache for {dataset_name}")
    else:
        if os.path.exists(root):
            shutil.rmtree(root)
            print(f"Cleared all PyG cache")


# ============================================================================
# Test function
# ============================================================================

if __name__ == '__main__':
    import time

    print("Testing PyG dataset loaders...")

    # Test syn1
    print("\n--- Testing syn1 ---")
    t0 = time.time()
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset_pyg('syn1')
    t1 = time.time()
    print(f"Loaded in {t1-t0:.3f}s")
    print(f"Graph shape: {graph.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test again (should be faster from cache)
    t0 = time.time()
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset_pyg('syn1')
    t1 = time.time()
    print(f"Loaded from cache in {t1-t0:.3f}s")

    # Test ba2
    print("\n--- Testing ba2 ---")
    t0 = time.time()
    edge_index, features, labels, train_mask, val_mask, test_mask = load_dataset_pyg('ba2')
    t1 = time.time()
    print(f"Loaded in {t1-t0:.3f}s")
    print(f"Number of graphs: {len(edge_index)}")

    # Test ground truth
    print("\n--- Testing ground truth ---")
    gt_data, indices = load_dataset_ground_truth_pyg('syn1')
    print(f"Ground truth loaded, {len(indices)} test indices")

    print("\nAll tests passed!")
