"""
Dataset loaders with PyG caching support.

Usage:
    from Main.data.dataset_loaders import load_dataset
    
    # Use canonical (friendly) names:
    data = load_dataset('BA-Shapes')
    data = load_dataset('BA-Community')
    data = load_dataset('Tree-Grid')
    data = load_dataset('Tree-Cycles')
    data = load_dataset('PubMed')
    data = load_dataset('Cora')
    data = load_dataset('Infection4')
    data = load_dataset('Infection4-one-hot')
    data = load_dataset('Infection3')
    data = load_dataset('Infection3-one-hot')
    data = load_dataset('Infection3-one-hot-binary')
    data = load_dataset('Infection3-one-hot-binary-Noise')
    data = load_dataset('Infection3-one-hot-binary-Bernoulli4')
    data = load_dataset('BA-2Motifs')
    data = load_dataset('BA-HouseGrid')
    data = load_dataset('BA-HouseGrid-wb60-m3')
    data = load_dataset('BA-HouseANDGrid')
    data = load_dataset('BA-HouseANDGrid-wb15-m1')
    data = load_dataset('BA-HouseORGrid')
    data = load_dataset('MarkerPath-Exact3')
    data = load_dataset('InfectedSteiner-3')
    data = load_dataset('InfectedPair-Unique3')
    data = load_dataset('InfectedPair-Dist3')
    data = load_dataset('InfectedPair-Dist3-Noise')
    data = load_dataset('InfectedPair-Dist3-Noise0.05')
    data = load_dataset('InfectedPair-Dist6')
    data = load_dataset('InfectedPair-MultiPath3')
    data = load_dataset('InfectedHouse')
    data = load_dataset('InfectedPentagon')
    data = load_dataset('RandomInfectedTriCenter-3')
    data = load_dataset('MUTAG')        # PyG TUDataset (188 graphs)
    data = load_dataset('PROTEINS')     # PyG TUDataset (1113 graphs)
    
    # Legacy names still supported for backward compatibility:
    data = load_dataset('syn1')  # -> BA-Shapes
    data = load_dataset('syn2')  # -> BA-Community
    data = load_dataset('syn3')  # -> BA-Grid  
    data = load_dataset('syn4')  # -> Tree-Cycles
    data = load_dataset('ba2')   # -> BA-2Motifs
    data = load_dataset('ba_house_grid')   # -> BA-HouseGrid
    data = load_dataset('ba_house_and_grid')   # -> BA-HouseANDGrid
    data = load_dataset('ba_house_or_grid')   # -> BA-HouseORGrid
    data = load_dataset('cora')  # -> Cora
"""

from Main.data.dataset_loaders import (
    load_dataset,
    load_dataset_legacy,
    DATASET_PATH_NAMES,
    LEGACY_TO_CANONICAL,
    normalize_dataset_name,
    get_path_name,
    USE_PYG_CACHE,
    NODE_DATASETS,
    GRAPH_DATASETS,
    ALL_DATASETS,
)

from Main.data.ground_truth_loaders import (
    has_ground_truth,
    load_dataset_ground_truth,
    DATASETS_WITH_GROUND_TRUTH,
    DATASETS_WITHOUT_GROUND_TRUTH,
)

__all__ = [
    'load_dataset',
    'load_dataset_legacy', 
    'DATASET_PATH_NAMES',
    'LEGACY_TO_CANONICAL',
    'normalize_dataset_name',
    'get_path_name',
    'USE_PYG_CACHE',
    'NODE_DATASETS',
    'GRAPH_DATASETS',
    'ALL_DATASETS',
    'has_ground_truth',
    'load_dataset_ground_truth',
    'DATASETS_WITH_GROUND_TRUTH',
    'DATASETS_WITHOUT_GROUND_TRUTH',
]
