from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


DEFAULT_NUM_GRAPHS = 1000
DEFAULT_NUM_NODES = 50
DEFAULT_SEED = 0
DEFAULT_TARGET_AVG_NUM_EDGES = 80.0
GT_UNDIRECTED_BUDGET = 6


def undirected_pair(u: int, v: int) -> Tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u <= v else (v, u)


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


def _build_neighbors(num_nodes: int, undirected_edges: Sequence[Tuple[int, int]]) -> List[Set[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(int(num_nodes))]
    for u, v in undirected_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        neighbors[u].add(v)
        neighbors[v].add(u)
    return neighbors


def find_valid_tricenters(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_nodes: Sequence[int],
) -> List[Dict[str, object]]:
    infected = tuple(int(node) for node in infected_nodes)
    infected_set = set(infected)
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=undirected_edges)

    certificates: List[Dict[str, object]] = []
    for center in range(int(num_nodes)):
        if center in infected_set:
            continue

        internal_nodes: List[int] = []
        paths: List[Tuple[int, int, int]] = []
        is_valid = True
        for marker in infected:
            if center in neighbors[marker]:
                is_valid = False
                break

            common = sorted(
                node for node in (neighbors[marker] & neighbors[center])
                if node not in infected_set and int(node) != int(center)
            )
            if len(common) != 1:
                is_valid = False
                break

            internal = int(common[0])
            internal_nodes.append(internal)
            paths.append((int(marker), internal, int(center)))

        if not is_valid:
            continue
        if len(set(internal_nodes)) != len(infected):
            continue

        gt_edges: Set[Tuple[int, int]] = set()
        for marker, internal, center_node in paths:
            gt_edges.add(undirected_pair(marker, internal))
            gt_edges.add(undirected_pair(internal, center_node))

        certificates.append({
            "center": int(center),
            "internal_nodes": tuple(int(node) for node in internal_nodes),
            "paths": tuple(paths),
            "gt_edges": sorted(gt_edges),
        })

    return certificates


def pairwise_radius2_pair_count(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_nodes: Sequence[int],
) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    infected = tuple(int(node) for node in infected_nodes)
    infected_set = set(infected)
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=undirected_edges)

    valid_pairs: List[Tuple[int, int]] = []
    for left_pos in range(len(infected)):
        for right_pos in range(left_pos + 1, len(infected)):
            left = infected[left_pos]
            right = infected[right_pos]
            has_pairwise_center = False
            for center in range(int(num_nodes)):
                if center in infected_set:
                    continue
                if center in neighbors[left] or center in neighbors[right]:
                    continue
                left_common = [
                    node for node in (neighbors[left] & neighbors[center])
                    if node not in infected_set and int(node) != int(center)
                ]
                right_common = [
                    node for node in (neighbors[right] & neighbors[center])
                    if node not in infected_set and int(node) != int(center)
                ]
                if left_common and right_common:
                    has_pairwise_center = True
                    break
            if has_pairwise_center:
                valid_pairs.append((int(left), int(right)))

    return int(len(valid_pairs)), tuple(valid_pairs)


def _features(num_nodes: int, infected_nodes: Sequence[int]) -> np.ndarray:
    infected_set = {int(node) for node in infected_nodes}
    x = np.zeros((int(num_nodes), 2), dtype=np.float32)
    for node_idx in range(int(num_nodes)):
        if node_idx in infected_set:
            x[node_idx, 0] = 1.0
        else:
            x[node_idx, 1] = 1.0
    return x


def _schedule(num_graphs: int, rng: np.random.RandomState) -> List[int]:
    positive_count = int(num_graphs) // 2
    negative_count = int(num_graphs) - positive_count
    schedule = [1] * positive_count + [0] * negative_count
    rng.shuffle(schedule)
    return [int(label) for label in schedule]


def _build_single_graph(
    graph_idx: int,
    target_label: int,
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    edges, edge_prob = _sample_er_edges(
        num_nodes=num_nodes,
        target_avg_num_edges=target_avg_num_edges,
        rng=rng,
    )
    infected_nodes = tuple(sorted(int(node) for node in rng.choice(int(num_nodes), 3, replace=False)))
    certificates = find_valid_tricenters(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_nodes=infected_nodes,
    )
    pairwise_count, pairwise_pairs = pairwise_radius2_pair_count(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_nodes=infected_nodes,
    )

    if int(target_label) == 1:
        if len(certificates) != 1:
            raise ValueError(f"positive rejected: num_valid_centers={len(certificates)}")
        certificate = certificates[0]
        gt_edges = {undirected_pair(u, v) for (u, v) in certificate["gt_edges"]}
        if len(gt_edges) != GT_UNDIRECTED_BUDGET:
            raise ValueError(f"positive GT has {len(gt_edges)} undirected edges")
        gt_center = int(certificate["center"])
        gt_paths = tuple(certificate["paths"])
        gt_internal_nodes = tuple(int(node) for node in certificate["internal_nodes"])
    else:
        if len(certificates) != 0 or pairwise_count < 2:
            raise ValueError(
                f"negative rejected: num_valid_centers={len(certificates)}, pairwise_count={pairwise_count}"
            )
        gt_edges = set()
        gt_center = -1
        gt_paths = tuple()
        gt_internal_nodes = tuple()

    edge_index, edge_labels = _edge_index_and_labels_from_edges(sorted(edges), gt_edges)

    return {
        "graph_idx": int(graph_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes, infected_nodes),
        "label": int(target_label),
        "infected_nodes": tuple(int(node) for node in infected_nodes),
        "gt_valid": bool(int(target_label) == 1),
        "gt_edges": sorted(gt_edges) if int(target_label) == 1 else [],
        "gt_center": int(gt_center),
        "gt_paths": tuple(gt_paths),
        "gt_internal_nodes": tuple(gt_internal_nodes),
        "gt_undirected_budget": int(len(gt_edges)) if int(target_label) == 1 else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "num_valid_centers": int(len(certificates)),
        "pairwise_radius2_pair_count": int(pairwise_count),
        "pairwise_radius2_pairs": tuple(pairwise_pairs),
        "undirected_edges": sorted(edges),
        "num_edges": int(len(edges)),
        "er_edge_prob": float(edge_prob),
        "target_avg_num_edges": float(target_avg_num_edges),
    }


def generate_random_infected_tricenter3_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    max_attempts_per_graph: int = 2000,
) -> Dict[str, object]:
    """
    Generate RandomInfectedTriCenter-3 graph classification data.

    Each attempt samples an ER graph and three infected nodes uniformly at
    random. Positive examples are accepted when the final graph has exactly
    one valid radius-2 common center. Negative examples are hard negatives:
    no valid common center, but at least two infected pairs have a pairwise
    radius-2 center. No explanatory structure is planted.
    """
    rng = np.random.RandomState(int(seed))
    schedule = _schedule(int(num_graphs), rng)

    graphs: List[Dict[str, object]] = []
    reject_count = 0
    reject_counts_by_label = {0: 0, 1: 0}
    attempts_by_label = {0: 0, 1: 0}

    for graph_idx, target_label in enumerate(schedule):
        for attempt in range(int(max_attempts_per_graph)):
            attempts_by_label[int(target_label)] += 1
            try:
                graph = _build_single_graph(
                    graph_idx=int(graph_idx),
                    target_label=int(target_label),
                    num_nodes=int(num_nodes),
                    target_avg_num_edges=float(target_avg_num_edges),
                    rng=rng,
                )
                graphs.append(graph)
                break
            except ValueError:
                reject_count += 1
                reject_counts_by_label[int(target_label)] += 1
                if attempt + 1 >= int(max_attempts_per_graph):
                    raise

    labels = np.asarray([int(graph["label"]) for graph in graphs], dtype=np.int64)
    num_edges = np.asarray([int(graph["num_edges"]) for graph in graphs], dtype=np.int64)
    pairwise_counts = np.asarray([int(graph["pairwise_radius2_pair_count"]) for graph in graphs], dtype=np.int64)

    possible_edges = int(num_nodes) * (int(num_nodes) - 1) / 2.0
    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "target_avg_num_edges": float(target_avg_num_edges),
        "er_edge_prob": float(min(1.0, max(0.0, float(target_avg_num_edges) / possible_edges))),
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "negative_rule": "num_valid_centers == 0 and pairwise_radius2_pair_count >= 2",
        "mean_num_edges_positive": float(num_edges[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_num_edges_negative": float(num_edges[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "mean_pairwise_count_positive": float(pairwise_counts[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_pairwise_count_negative": float(pairwise_counts[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "gt_undirected_budget": int(GT_UNDIRECTED_BUDGET),
        "gt_directed_label_count_positive": int(2 * GT_UNDIRECTED_BUDGET),
        "reject_count": int(reject_count),
        "reject_counts_by_label": {int(k): int(v) for k, v in reject_counts_by_label.items()},
        "attempts_by_label": {int(k): int(v) for k, v in attempts_by_label.items()},
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }
