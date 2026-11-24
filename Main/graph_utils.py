"""Neighborhood extraction and planted-rationale edge selection."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


def extract_l_hop_subgraph(
    center_node: int,
    l_hop: int,
    edge_index: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Extract the L-hop neighborhood with local node indices."""
    from collections import deque

    # Build adjacency
    neighbors = {i: set() for i in range(num_nodes)}
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        neighbors[i].add(j)
        neighbors[j].add(i)

    # BFS
    visited = {center_node}
    frontier = {center_node}
    for _ in range(l_hop):
        new_frontier = set()
        for node in frontier:
            for neighbor in neighbors.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_frontier.add(neighbor)
        frontier = new_frontier
        if not frontier:
            break

    sub_nodes = np.array(sorted(visited))
    node_to_sub = {node: idx for idx, node in enumerate(sub_nodes)}

    # Filter edges
    sub_edges = []
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i in visited and j in visited:
            sub_edges.append((node_to_sub[i], node_to_sub[j]))

    if sub_edges:
        sub_edge_index = np.array(sub_edges, dtype=np.int64).T
    else:
        sub_edge_index = np.zeros((2, 0), dtype=np.int64)

    return sub_edge_index, sub_nodes, node_to_sub


def get_node_own_motif_edges(
    node_idx: int,
    gt_edges_in_khop: List[Tuple[int, int]],
    dataset: str,
) -> List[Tuple[int, int]]:
    """
    Filter GT edges to only include edges from the target node's own motif.

    For BA-Shapes and similar datasets, a node's K-hop neighborhood may include
    multiple motifs. This function returns only the edges belonging to the
    specific motif that contains the target node.

    Args:
        node_idx: Target node index
        gt_edges_in_khop: All GT edges within K-hop neighborhood
        dataset: Dataset name

    Returns:
        List of edges belonging to the target node's own motif only
    """
    normalized_gt_edges = sorted(set(
        (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        for (a, b) in gt_edges_in_khop
        if int(a) != int(b)
    ))
    if not normalized_gt_edges:
        return []

    # Prefer a component-based split of the GT subgraph. This is robust to the
    # actual synthetic dataset layout and isolates the motif/path that contains
    # the target node when multiple GT components fall inside the same K-hop
    # working graph.
    gt_neighbors: Dict[int, set] = {}
    for a, b in normalized_gt_edges:
        gt_neighbors.setdefault(int(a), set()).add(int(b))
        gt_neighbors.setdefault(int(b), set()).add(int(a))
    if int(node_idx) in gt_neighbors:
        component_nodes = {int(node_idx)}
        frontier = [int(node_idx)]
        while frontier:
            cur = frontier.pop()
            for nbr in gt_neighbors.get(cur, ()):
                if nbr not in component_nodes:
                    component_nodes.add(int(nbr))
                    frontier.append(int(nbr))
        component_edges = [
            (a, b) for (a, b) in normalized_gt_edges
            if int(a) in component_nodes and int(b) in component_nodes
        ]
        if component_edges:
            return component_edges

    if dataset in ("BA-Shapes", "BA-House", "syn1"):
        # BA-Shapes: 300 base nodes (0-299), 80 house motifs (5 nodes each, 300-699)
        # Each house: nodes [300 + i*5, 300 + i*5 + 4] for i in 0..79
        BASE_NODES = 300
        MOTIF_SIZE = 5
        if node_idx >= BASE_NODES:
            house_id = (node_idx - BASE_NODES) // MOTIF_SIZE
            house_start = BASE_NODES + house_id * MOTIF_SIZE
            house_nodes = set(range(house_start, house_start + MOTIF_SIZE))
            return [
                (a, b) for (a, b) in gt_edges_in_khop
                if a in house_nodes and b in house_nodes
            ]
        # Node is in base graph (class 0), no motif
        return []

    elif dataset in ("BA-Community", "syn2"):
        # BA-Community: 2 communities x 350 nodes = 700 total
        # Each community: 175 base nodes + 35 houses (5 nodes each)
        # Community 0: base 0-174, motifs 175-349
        # Community 1: base 350-524, motifs 525-699
        COMMUNITY_SIZE = 350
        BASE_PER_COMMUNITY = 175
        MOTIF_SIZE = 5
        community = node_idx // COMMUNITY_SIZE
        local_idx = node_idx % COMMUNITY_SIZE
        if local_idx >= BASE_PER_COMMUNITY:
            house_id = (local_idx - BASE_PER_COMMUNITY) // MOTIF_SIZE
            house_start = community * COMMUNITY_SIZE + BASE_PER_COMMUNITY + house_id * MOTIF_SIZE
            house_nodes = set(range(house_start, house_start + MOTIF_SIZE))
            return [
                (a, b) for (a, b) in gt_edges_in_khop
                if a in house_nodes and b in house_nodes
            ]
        return []

    elif dataset in (
        "Tree-Cycles",
        "Tree-Cycles-Noise",
        "Tree-Cycles-Noise0.5",
        "Tree-Cycles-Noise0.1",
        "Tree-Cycles-Noise0.05",
        "Tree-Cycles-Uniform0.5",
        "syn3",
        "syn3_noise",
        "syn3_noise0p5",
        "syn3_noise0p1",
        "syn3_noise0p05",
        "syn3_uniform0p5",
    ):
        # Tree-Cycles: 511 tree nodes + 80 6-cycles (6 nodes each)
        # Tree nodes: 0-510, Cycle nodes: 511-990
        BASE_NODES = 511
        MOTIF_SIZE = 6
        if node_idx >= BASE_NODES:
            cycle_id = (node_idx - BASE_NODES) // MOTIF_SIZE
            cycle_start = BASE_NODES + cycle_id * MOTIF_SIZE
            cycle_nodes = set(range(cycle_start, cycle_start + MOTIF_SIZE))
            return [
                (a, b) for (a, b) in gt_edges_in_khop
                if a in cycle_nodes and b in cycle_nodes
            ]
        return []

    elif dataset in ("Tree-Grid", "syn4"):
        # Tree-Grid: 511 tree nodes + 80 3x3 grids (9 nodes each)
        # Tree nodes: 0-510, Grid nodes: 511-1230
        BASE_NODES = 511
        MOTIF_SIZE = 9
        if node_idx >= BASE_NODES:
            grid_id = (node_idx - BASE_NODES) // MOTIF_SIZE
            grid_start = BASE_NODES + grid_id * MOTIF_SIZE
            grid_nodes = set(range(grid_start, grid_start + MOTIF_SIZE))
            return [
                (a, b) for (a, b) in gt_edges_in_khop
                if a in grid_nodes and b in grid_nodes
            ]
        return []

    # For other datasets, return all GT edges (no filtering)
    return normalized_gt_edges


def extract_node_ground_truth_edges_within_hops(
    node_idx: int,
    k_hop: int,
    edge_index: np.ndarray,
    num_nodes: int,
    gt_graph: np.ndarray,
    gt_edge_labels: Any,
    dataset: str,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Return the target node's GT motif edges restricted to the given k-hop neighborhood."""
    _, khop_nodes, _ = extract_l_hop_subgraph(node_idx, k_hop, edge_index, num_nodes)
    khop_node_set = set(int(v) for v in khop_nodes.tolist())

    if isinstance(gt_edge_labels, dict):
        gt_edges_in_khop = [
            (int(a), int(b))
            for (a, b) in gt_edge_labels.get(node_idx, [])
            if int(a) in khop_node_set and int(b) in khop_node_set
        ]
        gt_edges_in_khop = list(set(gt_edges_in_khop))
    else:
        gt_edges_in_khop = []
        for k in range(gt_graph.shape[1]):
            if gt_edge_labels[k] > 0:
                i, j = int(gt_graph[0, k]), int(gt_graph[1, k])
                if i in khop_node_set and j in khop_node_set:
                    a, b = (i, j) if i < j else (j, i)
                    gt_edges_in_khop.append((a, b))
        gt_edges_in_khop = list(set(gt_edges_in_khop))

    gt_edges = get_node_own_motif_edges(node_idx, gt_edges_in_khop, dataset)
    return gt_edges, khop_nodes
