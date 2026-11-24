from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


DEFAULT_NUM_GRAPHS = 2000
DEFAULT_NUM_NODES = 50
DEFAULT_SEED = 0
DEFAULT_TARGET_AVG_NUM_EDGES = 60.0
GT_UNDIRECTED_BUDGET = 5

PENTAGON_MOTIF_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0),
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


def _features(num_nodes: int, infected_nodes: Sequence[int]) -> np.ndarray:
    infected_set = {int(node) for node in infected_nodes}
    x = np.zeros((int(num_nodes), 2), dtype=np.float32)
    x[:, 1] = 1.0
    for node_idx in infected_set:
        x[int(node_idx), 0] = 1.0
        x[int(node_idx), 1] = 0.0
    return x


def _pentagon_nonadjacent_local_pairs() -> Tuple[Tuple[int, int], ...]:
    cycle_edges = {undirected_pair(u, v) for (u, v) in PENTAGON_MOTIF_EDGES}
    pairs: List[Tuple[int, int]] = []
    for u in range(5):
        for v in range(u + 1, 5):
            pair = undirected_pair(u, v)
            if pair not in cycle_edges:
                pairs.append(pair)
    return tuple(pairs)


def _plant_pentagon(
    num_nodes: int,
    edges: Set[Tuple[int, int]],
    rng: np.random.RandomState,
) -> Tuple[Tuple[int, ...], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    pentagon_nodes = tuple(int(node) for node in rng.choice(int(num_nodes), 5, replace=False).tolist())
    pentagon_node_set = set(pentagon_nodes)

    # Keep the planted motif clean: no ER chord among the five pentagon nodes.
    for u in pentagon_node_set:
        for v in pentagon_node_set:
            if int(u) < int(v):
                edges.discard(undirected_pair(u, v))

    gt_edges: Set[Tuple[int, int]] = set()
    cycle_pairs: Set[Tuple[int, int]] = set()
    for local_u, local_v in PENTAGON_MOTIF_EDGES:
        u = int(pentagon_nodes[int(local_u)])
        v = int(pentagon_nodes[int(local_v)])
        edge = undirected_pair(u, v)
        edges.add(edge)
        gt_edges.add(edge)
        cycle_pairs.add(edge)
    return pentagon_nodes, gt_edges, cycle_pairs


def _positive_pairs(pentagon_nodes: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    pairs = []
    for local_u, local_v in _pentagon_nonadjacent_local_pairs():
        pairs.append(undirected_pair(pentagon_nodes[int(local_u)], pentagon_nodes[int(local_v)]))
    return tuple(sorted(pairs))


def _negative_pairs(num_nodes: int, positive_pairs: Set[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    pairs: List[Tuple[int, int]] = []
    for u in range(int(num_nodes)):
        for v in range(u + 1, int(num_nodes)):
            pair = (u, v)
            if pair not in positive_pairs:
                pairs.append(pair)
    return tuple(pairs)


def _is_positive_pair(
    infected_pair: Tuple[int, int],
    pentagon_nodes: Sequence[int],
) -> bool:
    return undirected_pair(*infected_pair) in set(_positive_pairs(pentagon_nodes))


def _apply_node_permutation(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_pair: Tuple[int, int],
    pentagon_nodes: Sequence[int],
    gt_edges: Set[Tuple[int, int]],
    rng: np.random.RandomState,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Tuple[int, ...], Set[Tuple[int, int]], Tuple[int, ...]]:
    permutation = rng.permutation(int(num_nodes)).astype(np.int64)
    remapped_edges = sorted(
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in undirected_edges
    )
    remapped_infected_pair = tuple(sorted((int(permutation[infected_pair[0]]), int(permutation[infected_pair[1]]))))
    remapped_pentagon_nodes = tuple(int(permutation[int(node)]) for node in pentagon_nodes)
    remapped_gt_edges = {
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in gt_edges
    }
    return remapped_edges, remapped_infected_pair, remapped_pentagon_nodes, remapped_gt_edges, tuple(
        int(node) for node in permutation
    )


def _sample_graph(
    graph_idx: int,
    label: int,
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    edges, edge_prob = _sample_er_edges(
        num_nodes=num_nodes,
        target_avg_num_edges=target_avg_num_edges,
        rng=rng,
    )
    pentagon_nodes, pentagon_edges, _ = _plant_pentagon(num_nodes=num_nodes, edges=edges, rng=rng)
    positive_candidates = _positive_pairs(pentagon_nodes)
    positive_pair_set = set(positive_candidates)

    if int(label) == 1:
        infected_pair = positive_candidates[int(rng.randint(len(positive_candidates)))]
    else:
        negative_candidates = _negative_pairs(num_nodes=num_nodes, positive_pairs=positive_pair_set)
        infected_pair = negative_candidates[int(rng.randint(len(negative_candidates)))]

    infected_pair = tuple(int(node) for node in infected_pair)
    gt_edges = set(pentagon_edges) if int(label) == 1 else set()

    (
        remapped_edges,
        remapped_infected_pair,
        remapped_pentagon_nodes,
        remapped_gt_edges,
        node_permutation,
    ) = _apply_node_permutation(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_pair=infected_pair,
        pentagon_nodes=pentagon_nodes,
        gt_edges=gt_edges,
        rng=rng,
    )

    observed_label = _is_positive_pair(remapped_infected_pair, remapped_pentagon_nodes)
    if observed_label != bool(label):
        raise ValueError(
            f"label validation failed: label={label}, infected={remapped_infected_pair}, "
            f"pentagon_nodes={remapped_pentagon_nodes}"
        )
    if int(label) == 1 and len(remapped_gt_edges) != GT_UNDIRECTED_BUDGET:
        raise ValueError(f"positive GT has {len(remapped_gt_edges)} edges")

    edge_index, edge_labels = _edge_index_and_labels_from_edges(
        remapped_edges,
        remapped_gt_edges if int(label) == 1 else set(),
    )

    return {
        "graph_idx": int(graph_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes, remapped_infected_pair),
        "label": int(label),
        "infected_nodes": tuple(int(node) for node in remapped_infected_pair),
        "pentagon_nodes": tuple(int(node) for node in remapped_pentagon_nodes),
        "gt_valid": bool(int(label) == 1),
        "gt_edges": sorted(remapped_gt_edges) if int(label) == 1 else [],
        "gt_undirected_budget": int(len(remapped_gt_edges)) if int(label) == 1 else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "pentagon_path_lengths": (2, 3) if int(label) == 1 else tuple(),
        "undirected_edges": sorted(remapped_edges),
        "num_edges": int(len(remapped_edges)),
        "er_edge_prob": float(edge_prob),
        "target_avg_num_edges": float(target_avg_num_edges),
        "node_permutation": node_permutation,
    }


def generate_infected_pentagon_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
) -> Dict[str, object]:
    """
    Generate an InfectedPentagon graph-classification dataset.

    Each graph has an ER background over ``num_nodes`` nodes and one clean
    planted pentagon. Exactly two infected nodes are selected. The label is 1
    iff the infected nodes are a non-adjacent pair on the planted pentagon, so
    the pentagon contains one length-2 path and one length-3 path between them.
    Positive GT is the five-edge pentagon cycle.
    """
    if int(num_nodes) < 5:
        raise ValueError("InfectedPentagon expects num_nodes >= 5")
    if int(num_graphs) % 2 != 0:
        raise ValueError("InfectedPentagon expects an even num_graphs for 50/50 balance")

    rng = np.random.RandomState(int(seed))
    labels = np.asarray(
        [1] * (int(num_graphs) // 2) + [0] * (int(num_graphs) - int(num_graphs) // 2),
        dtype=np.int64,
    )
    rng.shuffle(labels)

    graphs = [
        _sample_graph(
            graph_idx=int(graph_idx),
            label=int(label),
            num_nodes=int(num_nodes),
            target_avg_num_edges=float(target_avg_num_edges),
            rng=rng,
        )
        for graph_idx, label in enumerate(labels)
    ]

    observed_labels = np.asarray([int(graph["label"]) for graph in graphs], dtype=np.int64)
    num_edges = np.asarray([int(graph["num_edges"]) for graph in graphs], dtype=np.int64)
    possible_edges = int(num_nodes) * (int(num_nodes) - 1) / 2.0
    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "target_avg_num_edges": float(target_avg_num_edges),
        "er_edge_prob": float(min(1.0, max(0.0, float(target_avg_num_edges) / possible_edges))),
        "num_positive_graphs": int(observed_labels.sum()),
        "num_negative_graphs": int(observed_labels.shape[0] - observed_labels.sum()),
        "positive_rule": "two_infected_nodes_are_nonadjacent_on_planted_pentagon",
        "negative_rule": "all_other_infected_pairs",
        "num_pentagon_nodes": 5,
        "gt_undirected_budget": int(GT_UNDIRECTED_BUDGET),
        "gt_directed_label_count_positive": int(2 * GT_UNDIRECTED_BUDGET),
        "mean_num_edges_positive": float(num_edges[observed_labels == 1].mean()) if np.any(observed_labels == 1) else 0.0,
        "mean_num_edges_negative": float(num_edges[observed_labels == 0].mean()) if np.any(observed_labels == 0) else 0.0,
        "label_counts": {
            int(value): int(count)
            for value, count in sorted(Counter(int(label) for label in observed_labels).items())
        },
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }
