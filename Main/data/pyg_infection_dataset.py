"""Deterministic wrappers around PyG's official InfectionDataset."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import InfectionDataset
from torch_geometric.datasets.graph_generator import BAGraph, ERGraph


GraphFamily = Literal["er", "ba"]

DEFAULT_NUM_NODES = 1000
DEFAULT_NUM_INFECTED_NODES = 50
DEFAULT_MAX_PATH_LENGTH = 3
DEFAULT_ER_EDGE_PROB = 0.004
DEFAULT_BA_NUM_EDGES = 2
DEFAULT_DATASET_SEED = 42
DEFAULT_SPLIT_SEED = 42


def build_label3_unique_path_ground_truth(data: Data) -> Dict[str, object]:
    """Build target-specific GT for test nodes with a unique distance-3 path.

    Uniqueness is measured from the complete infected-node set, not from an
    arbitrarily chosen infected source. The official PyG global edge mask is
    intentionally not used as a target-level explanation.
    """
    if data.edge_index.dim() != 2 or int(data.edge_index.size(0)) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if not hasattr(data, "infected_mask") or not hasattr(data, "test_mask"):
        raise ValueError("data must contain infected_mask and test_mask")

    num_nodes = int(data.num_nodes)
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for src, dst in data.edge_index.t().tolist():
        neighbors[int(src)].append(int(dst))

    infected_nodes = torch.where(data.infected_mask)[0].tolist()
    unreachable = num_nodes + 1
    distance = [unreachable] * num_nodes
    path_count = [0] * num_nodes
    queue: deque[int] = deque()
    for node in infected_nodes:
        distance[int(node)] = 0
        path_count[int(node)] = 1
        queue.append(int(node))

    while queue:
        src = queue.popleft()
        for dst in neighbors[src]:
            if distance[dst] == unreachable:
                distance[dst] = distance[src] + 1
                queue.append(dst)

    finite_max = max((value for value in distance if value != unreachable), default=0)
    for hop in range(1, finite_max + 1):
        for node in range(num_nodes):
            if distance[node] != hop:
                continue
            count = 0
            for predecessor in neighbors[node]:
                if distance[predecessor] == hop - 1:
                    count += path_count[predecessor]
                    if count >= 2:
                        count = 2
                        break
            path_count[node] = count

    node_to_gt_edges: Dict[int, List[Tuple[int, int]]] = {}
    distance_labels = data.distance_label if hasattr(data, "distance_label") else data.y
    label3_test_nodes = [
        int(node)
        for node in torch.where(data.test_mask & (distance_labels == 3))[0].tolist()
    ]
    for target in label3_test_nodes:
        if distance[target] != 3 or path_count[target] != 1:
            continue
        path_nodes = [target]
        current = target
        while distance[current] > 0:
            predecessors = [
                node
                for node in neighbors[current]
                if distance[node] == distance[current] - 1 and path_count[node] > 0
            ]
            if len(predecessors) != 1:
                path_nodes = []
                break
            current = int(predecessors[0])
            path_nodes.append(current)
        if len(path_nodes) != 4 or not bool(data.infected_mask[current]):
            continue
        path_nodes.reverse()
        edges = []
        for src, dst in zip(path_nodes[:-1], path_nodes[1:]):
            edges.append((min(int(src), int(dst)), max(int(src), int(dst))))
        if len(set(edges)) == 3:
            node_to_gt_edges[int(target)] = sorted(set(edges))

    gt_test_indices = sorted(node_to_gt_edges)
    return {
        "node_to_gt_edges": node_to_gt_edges,
        "gt_test_indices": gt_test_indices,
        "metadata": {
            "target_label": 3,
            "path_length": 3,
            "uniqueness_scope": "all_infected_sources",
            "num_label3_test_nodes": len(label3_test_nodes),
            "num_unique_path_label3_test_nodes": len(gt_test_indices),
            "excluded_nonunique_label3_test_nodes": len(label3_test_nodes) - len(gt_test_indices),
        },
    }


def _node_split_masks(
    num_nodes: int,
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must lie strictly between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    rng = np.random.RandomState(int(seed))
    permutation = rng.permutation(int(num_nodes))
    num_train = int(np.floor(train_ratio * num_nodes))
    num_val = int(np.floor(val_ratio * num_nodes))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(permutation[:num_train]).long()] = True
    val_mask[torch.from_numpy(permutation[num_train:num_train + num_val]).long()] = True
    test_mask[torch.from_numpy(permutation[num_train + num_val:]).long()] = True
    return train_mask, val_mask, test_mask


def generate_pyg_infection_data(
    graph_family: GraphFamily,
    *,
    num_nodes: int = DEFAULT_NUM_NODES,
    num_infected_nodes: int = DEFAULT_NUM_INFECTED_NODES,
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
    er_edge_prob: float = DEFAULT_ER_EDGE_PROB,
    ba_num_edges: int = DEFAULT_BA_NUM_EDGES,
    dataset_seed: int = DEFAULT_DATASET_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[Data, Dict[str, object]]:
    """Generate one official PyG Infection graph and deterministic node splits.

    PyG's ER generator uses Torch randomness, while its BA generator also uses
    NumPy randomness. The caller's global RNG states are restored after dataset
    construction so loading this cached dataset cannot perturb model seeding.
    """
    family = str(graph_family).lower()
    if family not in ("er", "ba"):
        raise ValueError(f"graph_family must be 'er' or 'ba', got {graph_family!r}")

    torch_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    try:
        torch.manual_seed(int(dataset_seed))
        np.random.seed(int(dataset_seed))
        if family == "er":
            graph_generator = ERGraph(
                num_nodes=int(num_nodes),
                edge_prob=float(er_edge_prob),
            )
            generator_parameters = {"edge_prob": float(er_edge_prob)}
        else:
            graph_generator = BAGraph(
                num_nodes=int(num_nodes),
                num_edges=int(ba_num_edges),
            )
            generator_parameters = {"num_edges": int(ba_num_edges)}

        source = InfectionDataset(
            graph_generator=graph_generator,
            num_infected_nodes=int(num_infected_nodes),
            max_path_length=int(max_path_length),
        )[0]
    finally:
        torch.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)

    data = Data(
        x=source.x.detach().clone().to(torch.float32),
        edge_index=source.edge_index.detach().clone().to(torch.long),
        y=source.y.detach().clone().to(torch.long),
        pyg_infection_edge_mask=source.edge_mask.detach().clone().to(torch.float32),
    )
    data.train_mask, data.val_mask, data.test_mask = _node_split_masks(
        int(data.num_nodes),
        seed=int(split_seed),
    )
    # PyG InfectionDataset encodes infected nodes as [0, 1].
    data.infected_mask = data.x[:, 1].to(torch.bool)
    data.distance_label = data.y.clone()

    class_counts = torch.bincount(
        data.y,
        minlength=int(max_path_length) + 2,
    )
    split_class_counts = {}
    for split_name in ("train", "val", "test"):
        mask = getattr(data, f"{split_name}_mask")
        split_class_counts[split_name] = torch.bincount(
            data.y[mask],
            minlength=int(max_path_length) + 2,
        ).tolist()

    metadata: Dict[str, object] = {
        "source_class": "torch_geometric.datasets.InfectionDataset",
        "graph_generator_class": type(graph_generator).__name__,
        "graph_family": family,
        "generator_parameters": generator_parameters,
        "num_nodes": int(data.num_nodes),
        "num_directed_edge_entries": int(data.num_edges),
        "num_undirected_edges": int(data.num_edges // 2),
        "num_infected_nodes": int(data.infected_mask.sum().item()),
        "max_path_length": int(max_path_length),
        "num_classes": int(max_path_length) + 2,
        "class_counts": class_counts.tolist(),
        "split_class_counts": split_class_counts,
        "dataset_seed": int(dataset_seed),
        "split_seed": int(split_seed),
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "feature_convention": {"healthy": [1.0, 0.0], "infected": [0.0, 1.0]},
        "pyg_edge_mask_scope": "global_infection_hop_expansion_not_target_specific",
    }
    return data, metadata
