from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


DEFAULT_NUM_GRAPHS = 2000
DEFAULT_NUM_NODES = 50
DEFAULT_SEED = 0
DEFAULT_TARGET_AVG_NUM_EDGES = 60.0
GT_UNDIRECTED_BUDGET = 6

HOUSE_MOTIF_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (2, 4),
    (3, 4),
)


def undirected_pair(u: int, v: int) -> Tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


def _sample_er_edges(
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Tuple[Set[Tuple[int, int]], float]:
    possible_edges = int(num_nodes) * (int(num_nodes) - 1) // 2
    edge_prob = min(1.0, max(0.0, float(target_avg_num_edges) / float(possible_edges)))
    edges: Set[Tuple[int, int]] = set()
    for u in range(int(num_nodes)):
        for v in range(u + 1, int(num_nodes)):
            if float(rng.rand()) < edge_prob:
                edges.add((u, v))
    return edges, float(edge_prob)


def _edge_index_and_labels_from_edges(
    undirected_edges: Sequence[Tuple[int, int]],
    gt_edges: Set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    edge_pairs: List[Tuple[int, int]] = []
    edge_labels: List[int] = []

    for u, v in sorted({undirected_pair(u, v) for (u, v) in undirected_edges if int(u) != int(v)}):
        label = 1 if undirected_pair(u, v) in gt_edges else 0
        edge_pairs.append((int(u), int(v)))
        edge_labels.append(label)
        edge_pairs.append((int(v), int(u)))
        edge_labels.append(label)

    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(edge_pairs, dtype=np.int64).T, np.asarray(edge_labels, dtype=np.int64)


def _features(num_nodes: int, infected_node: int) -> np.ndarray:
    x = np.zeros((int(num_nodes), 2), dtype=np.float32)
    x[:, 1] = 1.0
    x[int(infected_node), 0] = 1.0
    x[int(infected_node), 1] = 0.0
    return x


def _plant_house(
    num_nodes: int,
    edges: Set[Tuple[int, int]],
    rng: np.random.RandomState,
) -> Tuple[Tuple[int, ...], Set[Tuple[int, int]]]:
    house_nodes = tuple(int(node) for node in rng.choice(int(num_nodes), 5, replace=False).tolist())
    gt_edges: Set[Tuple[int, int]] = set()
    for local_u, local_v in HOUSE_MOTIF_EDGES:
        u = int(house_nodes[int(local_u)])
        v = int(house_nodes[int(local_v)])
        edge = undirected_pair(u, v)
        edges.add(edge)
        gt_edges.add(edge)
    return house_nodes, gt_edges


def _apply_node_permutation(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_node: int,
    house_nodes: Sequence[int],
    gt_edges: Set[Tuple[int, int]],
    rng: np.random.RandomState,
) -> Tuple[List[Tuple[int, int]], int, Tuple[int, ...], Set[Tuple[int, int]], Tuple[int, ...]]:
    permutation = rng.permutation(int(num_nodes)).astype(np.int64)
    remapped_edges = sorted(
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in undirected_edges
    )
    remapped_infected_node = int(permutation[int(infected_node)])
    remapped_house_nodes = tuple(int(permutation[int(node)]) for node in house_nodes)
    remapped_gt_edges = {
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in gt_edges
    }
    return remapped_edges, remapped_infected_node, remapped_house_nodes, remapped_gt_edges, tuple(
        int(node) for node in permutation
    )


def _sample_candidate(
    sample_idx: int,
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    edges, edge_prob = _sample_er_edges(
        num_nodes=num_nodes,
        target_avg_num_edges=target_avg_num_edges,
        rng=rng,
    )
    house_nodes, house_edges = _plant_house(num_nodes=num_nodes, edges=edges, rng=rng)
    infected_node = int(rng.randint(int(num_nodes)))
    label = 1 if int(infected_node) in set(house_nodes) else 0
    gt_edges = set(house_edges) if label == 1 else set()

    (
        remapped_edges,
        remapped_infected_node,
        remapped_house_nodes,
        remapped_gt_edges,
        node_permutation,
    ) = _apply_node_permutation(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_node=infected_node,
        house_nodes=house_nodes,
        gt_edges=gt_edges,
        rng=rng,
    )

    observed_label = int(remapped_infected_node) in set(remapped_house_nodes)
    if observed_label != bool(label):
        raise ValueError(
            f"label validation failed: label={label}, infected={remapped_infected_node}, "
            f"house_nodes={remapped_house_nodes}"
        )
    if label == 1 and len(remapped_gt_edges) != GT_UNDIRECTED_BUDGET:
        raise ValueError(f"positive GT has {len(remapped_gt_edges)} edges")

    edge_index, edge_labels = _edge_index_and_labels_from_edges(
        remapped_edges,
        remapped_gt_edges if label == 1 else set(),
    )

    return {
        "sample_idx": int(sample_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes, remapped_infected_node),
        "label": int(label),
        "infected_node": int(remapped_infected_node),
        "infected_nodes": (int(remapped_infected_node),),
        "house_nodes": tuple(int(node) for node in remapped_house_nodes),
        "gt_valid": bool(label == 1),
        "gt_edges": sorted(remapped_gt_edges) if label == 1 else [],
        "gt_undirected_budget": int(len(remapped_gt_edges)) if label == 1 else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "undirected_edges": sorted(remapped_edges),
        "num_edges": int(len(remapped_edges)),
        "er_edge_prob": float(edge_prob),
        "target_avg_num_edges": float(target_avg_num_edges),
        "node_permutation": node_permutation,
    }


def _with_graph_idx(graph: Dict[str, object], graph_idx: int) -> Dict[str, object]:
    updated = dict(graph)
    updated["graph_idx"] = int(graph_idx)
    return updated


def generate_infected_house_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """
    Generate an InfectedHouse graph-classification dataset.

    Each graph has an ER background over ``num_nodes`` nodes and one planted
    house motif. Exactly one infected node is sampled uniformly. The label is 1
    iff that infected node is one of the planted house nodes; positives expose
    the six internal house edges as ground-truth.
    """
    if int(num_nodes) < 5:
        raise ValueError("InfectedHouse expects num_nodes >= 5")
    if int(num_graphs) % 2 != 0:
        raise ValueError("InfectedHouse expects an even num_graphs for 50/50 balance")

    rng = np.random.RandomState(int(seed))
    positive_count = int(num_graphs) // 2
    negative_count = int(num_graphs) - positive_count

    positives: List[Dict[str, object]] = []
    negative_pool: List[Dict[str, object]] = []

    attempts = 0
    while (len(positives) < positive_count or len(negative_pool) < negative_count) and attempts < int(max_attempts):
        candidate = _sample_candidate(
            sample_idx=attempts,
            num_nodes=int(num_nodes),
            target_avg_num_edges=float(target_avg_num_edges),
            rng=rng,
        )
        attempts += 1
        if int(candidate["label"]) == 1:
            if len(positives) < positive_count:
                positives.append(candidate)
        else:
            negative_pool.append(candidate)

    if len(positives) < positive_count or len(negative_pool) < negative_count:
        raise ValueError(
            "failed to collect balanced InfectedHouse dataset: "
            f"positives={len(positives)}/{positive_count}, negatives={len(negative_pool)}/{negative_count}, "
            f"attempts={attempts}/{max_attempts}"
        )

    selected_negative_indices = rng.choice(len(negative_pool), size=negative_count, replace=False)
    selected_negatives = [negative_pool[int(idx)] for idx in selected_negative_indices.tolist()]

    graphs = positives + selected_negatives
    shuffled_order = rng.permutation(len(graphs))
    graphs = [_with_graph_idx(graphs[int(idx)], graph_idx=pos) for pos, idx in enumerate(shuffled_order)]

    labels = np.asarray([int(graph["label"]) for graph in graphs], dtype=np.int64)
    num_edges = np.asarray([int(graph["num_edges"]) for graph in graphs], dtype=np.int64)
    possible_edges = int(num_nodes) * (int(num_nodes) - 1) / 2.0
    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "target_avg_num_edges": float(target_avg_num_edges),
        "er_edge_prob": float(min(1.0, max(0.0, float(target_avg_num_edges) / possible_edges))),
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "positive_rule": "infected_node_in_planted_house",
        "num_house_nodes": 5,
        "gt_undirected_budget": int(GT_UNDIRECTED_BUDGET),
        "gt_directed_label_count_positive": int(2 * GT_UNDIRECTED_BUDGET),
        "negative_pool_size_before_downsample": int(len(negative_pool)),
        "num_attempts": int(attempts),
        "mean_num_edges_positive": float(num_edges[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_num_edges_negative": float(num_edges[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "label_counts": {
            int(value): int(count)
            for value, count in sorted(Counter(int(label) for label in labels).items())
        },
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }
