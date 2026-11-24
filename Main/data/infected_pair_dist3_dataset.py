from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch_geometric.datasets.graph_generator import BAGraph


DEFAULT_NUM_GRAPHS = 1000
DEFAULT_NUM_NODES = 50
DEFAULT_SEED = 0
DEFAULT_TARGET_AVG_NUM_EDGES = 60.0
DEFAULT_TARGET_DISTANCE = 3
DEFAULT_DENSITY_MATCHED_ER_EDGE_PROB = 0.0725
DEFAULT_BA_NUM_EDGES = 2
GT_UNDIRECTED_BUDGET = DEFAULT_TARGET_DISTANCE


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


def _build_neighbors(num_nodes: int, undirected_edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    neighbors: List[Set[int]] = [set() for _ in range(int(num_nodes))]
    for u, v in undirected_edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        neighbors[u].add(v)
        neighbors[v].add(u)
    return [sorted(items) for items in neighbors]


def _bfs_dist_count_pred(
    neighbors: Sequence[Sequence[int]],
    source: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_nodes = len(neighbors)
    inf_dist = np.iinfo(np.int32).max
    dist = np.full(num_nodes, inf_dist, dtype=np.int32)
    count = np.zeros(num_nodes, dtype=np.int8)
    pred = np.full(num_nodes, -1, dtype=np.int32)

    source = int(source)
    dist[source] = 0
    count[source] = 1
    q: deque[int] = deque([source])

    while q:
        u = q.popleft()
        next_dist = int(dist[u]) + 1
        for v in neighbors[u]:
            if dist[v] == inf_dist:
                dist[v] = next_dist
                count[v] = count[u]
                pred[v] = int(u) if int(count[u]) == 1 else -2
                q.append(v)
            elif dist[v] == next_dist:
                count[v] = min(2, int(count[v]) + int(count[u]))
                pred[v] = -2

    return dist, count, pred


def _unique_path_from_pred(pred: np.ndarray, source: int, target: int) -> Tuple[int, ...]:
    path = [int(target)]
    cur = int(target)
    source = int(source)
    while cur != source:
        cur = int(pred[cur])
        if cur < 0:
            raise ValueError("predecessor chain is not unique")
        path.append(cur)
    path.reverse()
    return tuple(path)


def _path_edges(path_nodes: Sequence[int]) -> Set[Tuple[int, int]]:
    return {
        undirected_pair(path_nodes[idx], path_nodes[idx + 1])
        for idx in range(len(path_nodes) - 1)
    }


def _length3_paths(
    neighbors: Sequence[Sequence[int]],
    source: int,
    target: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    source = int(source)
    target = int(target)
    paths: Set[Tuple[int, int, int, int]] = set()
    for first_hop in neighbors[source]:
        first_hop = int(first_hop)
        if first_hop == target:
            continue
        for second_hop in neighbors[first_hop]:
            second_hop = int(second_hop)
            if second_hop in (source, target, first_hop):
                continue
            if target in neighbors[second_hop]:
                paths.add((source, first_hop, second_hop, target))
    return tuple(sorted(paths))


def _path_edge_union(paths: Sequence[Sequence[int]]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for path in paths:
        edges.update(_path_edges(path))
    return edges


def shortest_distance_and_count_clipped(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    source: int,
    target: int,
) -> Tuple[int, int]:
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=undirected_edges)
    dist, count, _ = _bfs_dist_count_pred(neighbors, source=int(source))
    distance = int(dist[int(target)])
    if distance == np.iinfo(np.int32).max:
        return distance, 0
    return distance, int(count[int(target)])


def _sample_er_edges(
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
    edge_prob: Optional[float] = None,
) -> Tuple[Set[Tuple[int, int]], float]:
    possible_edges = int(num_nodes) * (int(num_nodes) - 1) // 2
    if edge_prob is None:
        edge_prob = float(target_avg_num_edges) / float(possible_edges)
    edge_prob = min(1.0, max(0.0, float(edge_prob)))
    edges: Set[Tuple[int, int]] = set()
    for u in range(int(num_nodes)):
        for v in range(u + 1, int(num_nodes)):
            if float(rng.rand()) < edge_prob:
                edges.add((u, v))
    return edges, float(edge_prob)


def _sample_ba_edges(
    num_nodes: int,
    num_edges: int,
    rng: np.random.RandomState,
) -> Tuple[Set[Tuple[int, int]], int]:
    """Sample the official PyG BAGraph while keeping RNG effects local."""
    graph_seed = int(rng.randint(0, np.iinfo(np.int32).max))
    numpy_state = np.random.get_state()
    try:
        np.random.seed(graph_seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(graph_seed)
            edge_index = BAGraph(
                num_nodes=int(num_nodes),
                num_edges=int(num_edges),
            )().edge_index.detach().cpu().numpy()
    finally:
        np.random.set_state(numpy_state)

    edges = {
        undirected_pair(int(u), int(v))
        for u, v in edge_index.T
        if int(u) != int(v)
    }
    return edges, graph_seed


def _features(
    num_nodes: int,
    infected_nodes: Sequence[int],
    noninfected_noise_std: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    infected_set = {int(node) for node in infected_nodes}
    x = np.zeros((int(num_nodes), 2), dtype=np.float32)
    noise_std = float(noninfected_noise_std)
    for node_idx in range(int(num_nodes)):
        if node_idx in infected_set:
            x[node_idx, 0] = 1.0
        else:
            feature_value = 1.0
            if noise_std > 0.0:
                if rng is None:
                    raise ValueError("rng is required when noninfected_noise_std > 0")
                feature_value += float(rng.normal(loc=0.0, scale=noise_std))
            x[node_idx, 1] = feature_value
    return x


def _distance_bucket(distance: int, target_distance: int = DEFAULT_TARGET_DISTANCE) -> str:
    inf_dist = np.iinfo(np.int32).max
    if int(distance) == inf_dist:
        return "disconnected"
    overflow_start = int(target_distance) + 1
    if int(distance) >= overflow_start:
        return f"dist_ge{overflow_start}"
    return f"dist{int(distance)}"


def _apply_node_permutation(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_pair: Tuple[int, int],
    gt_edges: Set[Tuple[int, int]],
    path_nodes: Sequence[int],
    rng: np.random.RandomState,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Set[Tuple[int, int]], Tuple[int, ...], Tuple[int, ...]]:
    permutation = rng.permutation(int(num_nodes)).astype(np.int64)
    remapped_edges = sorted(
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in undirected_edges
    )
    remapped_infected_pair = tuple(sorted((int(permutation[infected_pair[0]]), int(permutation[infected_pair[1]]))))
    remapped_gt_edges = {
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in gt_edges
    }
    remapped_path_nodes = tuple(int(permutation[node]) for node in path_nodes)
    return remapped_edges, remapped_infected_pair, remapped_gt_edges, remapped_path_nodes, tuple(
        int(node) for node in permutation
    )


def _apply_node_permutation_to_paths(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_pair: Tuple[int, int],
    gt_edges: Set[Tuple[int, int]],
    path_nodes_list: Sequence[Sequence[int]],
    rng: np.random.RandomState,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int], Set[Tuple[int, int]], Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    permutation = rng.permutation(int(num_nodes)).astype(np.int64)
    remapped_edges = sorted(
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in undirected_edges
    )
    remapped_infected_pair = tuple(sorted((int(permutation[infected_pair[0]]), int(permutation[infected_pair[1]]))))
    remapped_gt_edges = {
        undirected_pair(int(permutation[u]), int(permutation[v]))
        for (u, v) in gt_edges
    }
    remapped_path_nodes = tuple(
        tuple(int(permutation[node]) for node in path)
        for path in path_nodes_list
    )
    return remapped_edges, remapped_infected_pair, remapped_gt_edges, remapped_path_nodes, tuple(
        int(node) for node in permutation
    )


def _sample_candidate(
    sample_idx: int,
    num_nodes: int,
    target_avg_num_edges: float,
    target_distance: int,
    rng: np.random.RandomState,
    noninfected_noise_std: float = 0.0,
    feature_rng: Optional[np.random.RandomState] = None,
    graph_family: str = "er",
    er_edge_prob: Optional[float] = None,
    ba_num_edges: int = DEFAULT_BA_NUM_EDGES,
) -> Dict[str, object]:
    graph_family = str(graph_family).lower()
    graph_seed: Optional[int] = None
    if graph_family == "er":
        edges, edge_prob = _sample_er_edges(
            num_nodes=num_nodes,
            target_avg_num_edges=target_avg_num_edges,
            rng=rng,
            edge_prob=er_edge_prob,
        )
    elif graph_family == "ba":
        edges, graph_seed = _sample_ba_edges(
            num_nodes=num_nodes,
            num_edges=int(ba_num_edges),
            rng=rng,
        )
        edge_prob = float("nan")
    else:
        raise ValueError(f"unknown graph_family={graph_family!r}; expected 'er' or 'ba'")
    infected_pair = tuple(sorted(int(node) for node in rng.choice(int(num_nodes), 2, replace=False)))
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=sorted(edges))
    dist, count, pred = _bfs_dist_count_pred(neighbors, source=infected_pair[0])

    distance = int(dist[infected_pair[1]])
    shortest_count = 0 if distance == np.iinfo(np.int32).max else int(count[infected_pair[1]])
    label = 1 if distance == int(target_distance) else 0

    gt_edges: Set[Tuple[int, int]] = set()
    path_nodes: Tuple[int, ...] = tuple()
    gt_valid = False
    if label == 1 and shortest_count == 1:
        path_nodes = _unique_path_from_pred(pred, source=infected_pair[0], target=infected_pair[1])
        gt_edges = _path_edges(path_nodes)
        gt_valid = len(gt_edges) == int(target_distance)
    if not gt_valid:
        gt_edges = set()
        path_nodes = tuple()

    (
        remapped_edges,
        remapped_infected_pair,
        remapped_gt_edges,
        remapped_path_nodes,
        node_permutation,
    ) = _apply_node_permutation(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_pair=infected_pair,
        gt_edges=gt_edges,
        path_nodes=path_nodes,
        rng=rng,
    )

    observed_distance, observed_count = shortest_distance_and_count_clipped(
        num_nodes=num_nodes,
        undirected_edges=remapped_edges,
        source=remapped_infected_pair[0],
        target=remapped_infected_pair[1],
    )
    if (observed_distance == int(target_distance)) != bool(label):
        raise ValueError(
            f"label validation failed: label={label}, distance={observed_distance}, count={observed_count}"
        )
    if bool(gt_valid) and (
        observed_distance != int(target_distance)
        or observed_count != 1
        or len(remapped_gt_edges) != int(target_distance)
    ):
        raise ValueError(
            f"GT validation failed: distance={observed_distance}, count={observed_count}, gt={len(remapped_gt_edges)}"
        )

    edge_index, edge_labels = _edge_index_and_labels_from_edges(
        remapped_edges,
        remapped_gt_edges if bool(gt_valid) else set(),
    )

    return {
        "sample_idx": int(sample_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(
            num_nodes,
            remapped_infected_pair,
            noninfected_noise_std=float(noninfected_noise_std),
            rng=feature_rng if feature_rng is not None else rng,
        ),
        "label": int(label),
        "infected_nodes": tuple(int(node) for node in remapped_infected_pair),
        "gt_valid": bool(gt_valid),
        "gt_edges": sorted(remapped_gt_edges) if bool(gt_valid) else [],
        "gt_path_nodes": tuple(int(node) for node in remapped_path_nodes),
        "gt_undirected_budget": int(len(remapped_gt_edges)) if bool(gt_valid) else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "target_distance": int(observed_distance),
        "positive_target_distance": int(target_distance),
        "distance_bucket": _distance_bucket(observed_distance, target_distance=int(target_distance)),
        "shortest_path_count": int(observed_count),
        "undirected_edges": sorted(remapped_edges),
        "num_edges": int(len(remapped_edges)),
        "graph_family": graph_family,
        "graph_seed": graph_seed,
        "er_edge_prob": float(edge_prob),
        "ba_num_edges": int(ba_num_edges) if graph_family == "ba" else 0,
        "target_avg_num_edges": float(target_avg_num_edges),
        "noninfected_noise_std": float(noninfected_noise_std),
        "node_permutation": node_permutation,
    }


def _sample_multipath3_candidate(
    sample_idx: int,
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    target_distance = 3
    edges, edge_prob = _sample_er_edges(
        num_nodes=num_nodes,
        target_avg_num_edges=target_avg_num_edges,
        rng=rng,
    )
    infected_pair = tuple(sorted(int(node) for node in rng.choice(int(num_nodes), 2, replace=False)))
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=sorted(edges))
    dist, count, _ = _bfs_dist_count_pred(neighbors, source=infected_pair[0])

    distance = int(dist[infected_pair[1]])
    shortest_count = 0 if distance == np.iinfo(np.int32).max else int(count[infected_pair[1]])
    length3_paths = _length3_paths(neighbors, source=infected_pair[0], target=infected_pair[1]) if distance == 3 else tuple()
    length3_path_count = int(len(length3_paths))
    label = 1 if distance == 3 and length3_path_count >= 2 else 0

    gt_edges: Set[Tuple[int, int]] = set()
    gt_path_nodes: Tuple[Tuple[int, ...], ...] = tuple()
    gt_valid = False
    length3_edge_union_size = 0
    if label == 1:
        length3_edge_union = _path_edge_union(length3_paths)
        length3_edge_union_size = int(len(length3_edge_union))
        if length3_path_count == 2 and length3_edge_union_size == 6:
            gt_edges = length3_edge_union
            gt_path_nodes = tuple(tuple(int(node) for node in path) for path in length3_paths)
            gt_valid = True

    (
        remapped_edges,
        remapped_infected_pair,
        remapped_gt_edges,
        remapped_path_nodes,
        node_permutation,
    ) = _apply_node_permutation_to_paths(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_pair=infected_pair,
        gt_edges=gt_edges,
        path_nodes_list=gt_path_nodes,
        rng=rng,
    )

    observed_neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=remapped_edges)
    observed_dist, observed_count_arr, _ = _bfs_dist_count_pred(
        observed_neighbors,
        source=remapped_infected_pair[0],
    )
    observed_distance = int(observed_dist[remapped_infected_pair[1]])
    observed_count = 0 if observed_distance == np.iinfo(np.int32).max else int(observed_count_arr[remapped_infected_pair[1]])
    observed_length3_paths = (
        _length3_paths(observed_neighbors, source=remapped_infected_pair[0], target=remapped_infected_pair[1])
        if observed_distance == 3
        else tuple()
    )
    observed_length3_count = int(len(observed_length3_paths))
    observed_label = observed_distance == 3 and observed_length3_count >= 2
    if observed_label != bool(label):
        raise ValueError(
            f"label validation failed: label={label}, distance={observed_distance}, "
            f"length3_paths={observed_length3_count}"
        )
    if bool(gt_valid):
        observed_gt_edges = _path_edge_union(observed_length3_paths)
        if (
            observed_distance != 3
            or observed_length3_count != 2
            or len(remapped_gt_edges) != 6
            or observed_gt_edges != remapped_gt_edges
        ):
            raise ValueError(
                "GT validation failed: "
                f"distance={observed_distance}, length3_paths={observed_length3_count}, "
                f"gt={len(remapped_gt_edges)}, observed_gt={len(observed_gt_edges)}"
            )

    edge_index, edge_labels = _edge_index_and_labels_from_edges(
        remapped_edges,
        remapped_gt_edges if bool(gt_valid) else set(),
    )

    shortest_path_count = observed_length3_count if observed_distance == 3 else observed_count
    return {
        "sample_idx": int(sample_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes, remapped_infected_pair),
        "label": int(label),
        "infected_nodes": tuple(int(node) for node in remapped_infected_pair),
        "gt_valid": bool(gt_valid),
        "gt_edges": sorted(remapped_gt_edges) if bool(gt_valid) else [],
        "gt_path_nodes": remapped_path_nodes,
        "gt_undirected_budget": int(len(remapped_gt_edges)) if bool(gt_valid) else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "target_distance": int(observed_distance),
        "positive_target_distance": int(target_distance),
        "distance_bucket": _distance_bucket(observed_distance, target_distance=target_distance),
        "shortest_path_count": int(shortest_path_count),
        "length3_path_count": int(observed_length3_count),
        "length3_edge_union_size": int(len(_path_edge_union(observed_length3_paths))) if observed_length3_count > 0 else 0,
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


def generate_infected_pair_distance_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    target_distance: int = DEFAULT_TARGET_DISTANCE,
    noninfected_noise_std: float = 0.0,
    graph_family: str = "er",
    er_edge_prob: Optional[float] = None,
    ba_num_edges: int = DEFAULT_BA_NUM_EDGES,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """
    Generate InfectedPair-DistK graph classification data.

    Each candidate graph is sampled from G(n, p), then two infected nodes are
    sampled uniformly. The graph label is 1 iff the infected pair has shortest
    distance exactly K. Extra label-0 candidates are randomly downsampled to
    balance the dataset. Edge-level GT is exposed only for label-1 graphs whose
    length-K shortest path is unique. When noninfected_noise_std > 0, infected
    nodes keep feature [1, 0] and non-infected nodes receive [0, 1 + N(0, std)].
    """
    if int(num_graphs) % 2 != 0:
        raise ValueError("InfectedPair-DistK expects an even num_graphs for 50/50 balance")
    if int(target_distance) <= 0:
        raise ValueError("target_distance must be positive")
    if float(noninfected_noise_std) < 0.0:
        raise ValueError("noninfected_noise_std must be non-negative")
    graph_family = str(graph_family).lower()
    if graph_family not in {"er", "ba"}:
        raise ValueError("graph_family must be 'er' or 'ba'")
    if graph_family == "ba" and not 0 < int(ba_num_edges) < int(num_nodes):
        raise ValueError("ba_num_edges must be in [1, num_nodes-1]")
    if er_edge_prob is not None and not 0.0 <= float(er_edge_prob) <= 1.0:
        raise ValueError("er_edge_prob must be in [0, 1]")

    rng = np.random.RandomState(int(seed))
    noise_seed = (int(seed) + 1000003) % (2 ** 32 - 1)
    feature_rng = np.random.RandomState(noise_seed) if float(noninfected_noise_std) > 0.0 else None
    target_distance = int(target_distance)
    dataset_name = f"InfectedPair-{graph_family.upper()}-Dist{target_distance}"
    if float(noninfected_noise_std) > 0.0:
        dataset_name = f"{dataset_name}-Noise"
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
            target_distance=target_distance,
            rng=rng,
            noninfected_noise_std=float(noninfected_noise_std),
            feature_rng=feature_rng,
            graph_family=graph_family,
            er_edge_prob=er_edge_prob,
            ba_num_edges=int(ba_num_edges),
        )
        attempts += 1
        if int(candidate["label"]) == 1:
            if len(positives) < positive_count:
                positives.append(candidate)
        else:
            negative_pool.append(candidate)

    if len(positives) < positive_count or len(negative_pool) < negative_count:
        raise ValueError(
            f"failed to collect balanced {dataset_name} dataset: "
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
    valid_gt_flags = np.asarray([bool(graph["gt_valid"]) for graph in graphs], dtype=bool)
    positive_shortest_counts = [
        int(graph["shortest_path_count"])
        for graph in graphs
        if int(graph["label"]) == 1
    ]
    negative_buckets = Counter(
        str(graph["distance_bucket"])
        for graph in graphs
        if int(graph["label"]) == 0
    )

    possible_edges = int(num_nodes) * (int(num_nodes) - 1) / 2.0
    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "graph_family": graph_family,
        "target_avg_num_edges": float(target_avg_num_edges),
        "noninfected_noise_std": float(noninfected_noise_std),
        "positive_target_distance": int(target_distance),
        "er_edge_prob": (
            float(min(1.0, max(0.0, float(er_edge_prob))))
            if graph_family == "er" and er_edge_prob is not None
            else (
                float(min(1.0, max(0.0, float(target_avg_num_edges) / possible_edges)))
                if graph_family == "er"
                else None
            )
        ),
        "ba_num_edges": int(ba_num_edges) if graph_family == "ba" else None,
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "num_gt_valid_positive_graphs": int(np.sum(valid_gt_flags & (labels == 1))),
        "num_gt_invalid_positive_graphs": int(np.sum((~valid_gt_flags) & (labels == 1))),
        "positive_shortest_path_count_distribution": {
            int(value): int(sum(1 for observed in positive_shortest_counts if observed == int(value)))
            for value in sorted(set(positive_shortest_counts))
        },
        "negative_distance_bucket_counts": {
            bucket: int(count) for bucket, count in sorted(negative_buckets.items())
        },
        "negative_pool_size_before_downsample": int(len(negative_pool)),
        "num_attempts": int(attempts),
        "mean_num_edges_positive": float(num_edges[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_num_edges_negative": float(num_edges[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "gt_undirected_budget": int(target_distance),
        "gt_directed_label_count_positive_valid": int(2 * target_distance),
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }


def _distance_class_four(distance: int) -> int:
    """Map endpoint distance to {dist1, dist2, dist3, dist>=4/disconnected}."""
    distance = int(distance)
    if distance == np.iinfo(np.int32).max or distance >= 4:
        return 3
    if distance < 1:
        raise ValueError(f"infected endpoints must be distinct, got distance={distance}")
    return distance - 1


def generate_infected_pair_multiclass_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    graph_family: str = "er",
    er_edge_prob: Optional[float] = None,
    ba_num_edges: int = DEFAULT_BA_NUM_EDGES,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """Generate a class-balanced four-class infected-pair distance dataset.

    Labels are zero-based: 0=distance 1, 1=distance 2, 2=distance 3, and
    3=distance at least 4 or disconnected. The graph generator parameters
    match the corresponding binary ER/BA variants. Unique distance-3 paths
    retain the same three-edge explanation target used by the binary task.
    """
    num_graphs = int(num_graphs)
    if num_graphs % 4 != 0:
        raise ValueError("four-class InfectedPair expects num_graphs divisible by 4")
    graph_family = str(graph_family).lower()
    if graph_family not in {"er", "ba"}:
        raise ValueError("graph_family must be 'er' or 'ba'")
    if graph_family == "ba" and not 0 < int(ba_num_edges) < int(num_nodes):
        raise ValueError("ba_num_edges must be in [1, num_nodes-1]")
    if er_edge_prob is not None and not 0.0 <= float(er_edge_prob) <= 1.0:
        raise ValueError("er_edge_prob must be in [0, 1]")

    rng = np.random.RandomState(int(seed))
    per_class = num_graphs // 4
    class_graphs: List[List[Dict[str, object]]] = [[] for _ in range(4)]
    attempts = 0
    while any(len(items) < per_class for items in class_graphs) and attempts < int(max_attempts):
        candidate = _sample_candidate(
            sample_idx=attempts,
            num_nodes=int(num_nodes),
            target_avg_num_edges=float(target_avg_num_edges),
            target_distance=3,
            rng=rng,
            graph_family=graph_family,
            er_edge_prob=er_edge_prob,
            ba_num_edges=int(ba_num_edges),
        )
        attempts += 1
        distance_class = _distance_class_four(int(candidate["target_distance"]))
        if len(class_graphs[distance_class]) >= per_class:
            continue
        candidate = dict(candidate)
        candidate["label"] = int(distance_class)
        candidate["distance_class"] = int(distance_class)
        candidate["distance_class_cap"] = 4
        class_graphs[distance_class].append(candidate)

    observed_counts = [len(items) for items in class_graphs]
    if any(count < per_class for count in observed_counts):
        raise ValueError(
            "failed to collect balanced four-class InfectedPair dataset: "
            f"counts={observed_counts}, target={per_class}, "
            f"attempts={attempts}/{max_attempts}"
        )

    graphs = [graph for items in class_graphs for graph in items]
    shuffled_order = rng.permutation(len(graphs))
    graphs = [_with_graph_idx(graphs[int(idx)], graph_idx=pos) for pos, idx in enumerate(shuffled_order)]

    labels = np.asarray([int(graph["label"]) for graph in graphs], dtype=np.int64)
    num_edges = np.asarray([int(graph["num_edges"]) for graph in graphs], dtype=np.int64)
    possible_edges = int(num_nodes) * (int(num_nodes) - 1) / 2.0
    class_counts = np.bincount(labels, minlength=4)
    metadata = {
        "num_graphs": num_graphs,
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "graph_family": graph_family,
        "target_avg_num_edges": float(target_avg_num_edges),
        "er_edge_prob": (
            float(er_edge_prob)
            if graph_family == "er" and er_edge_prob is not None
            else (
                float(target_avg_num_edges) / possible_edges
                if graph_family == "er"
                else None
            )
        ),
        "ba_num_edges": int(ba_num_edges) if graph_family == "ba" else None,
        "num_classes": 4,
        "class_mapping": {
            0: "distance_1",
            1: "distance_2",
            2: "distance_3",
            3: "distance_ge_4_or_disconnected",
        },
        "class_balance_rule": "equal_rejection_sampling_per_distance_class",
        "class_counts": class_counts.tolist(),
        "num_attempts": int(attempts),
        "num_gt_valid_distance3_graphs": int(
            sum(bool(graph["gt_valid"]) and int(graph["label"]) == 2 for graph in graphs)
        ),
        "mean_num_edges_by_class": {
            int(label): float(num_edges[labels == label].mean())
            for label in range(4)
        },
        "gt_undirected_budget": 3,
        "gt_directed_label_count_distance3_valid": 6,
    }
    return {"graphs": graphs, "metadata": metadata}


def generate_infected_pair_dist3_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """Generate InfectedPair-Dist3 graph classification data."""
    return generate_infected_pair_distance_dataset(
        num_graphs=num_graphs,
        num_nodes=num_nodes,
        seed=seed,
        target_avg_num_edges=target_avg_num_edges,
        target_distance=3,
        max_attempts=max_attempts,
    )


def generate_infected_pair_dist3_noise_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    noninfected_noise_std: float = 1.0,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """Generate InfectedPair-Dist3 graph data with N(0, std) noise on non-infected feature channel 1."""
    return generate_infected_pair_distance_dataset(
        num_graphs=num_graphs,
        num_nodes=num_nodes,
        seed=seed,
        target_avg_num_edges=target_avg_num_edges,
        target_distance=3,
        noninfected_noise_std=noninfected_noise_std,
        max_attempts=max_attempts,
    )


def generate_infected_pair_dist6_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """Generate InfectedPair-Dist6 graph classification data."""
    return generate_infected_pair_distance_dataset(
        num_graphs=num_graphs,
        num_nodes=num_nodes,
        seed=seed,
        target_avg_num_edges=target_avg_num_edges,
        target_distance=6,
        max_attempts=max_attempts,
    )


def generate_infected_pair_multipath3_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    max_attempts: int = 100000,
) -> Dict[str, object]:
    """
    Generate InfectedPair-MultiPath3 graph classification data.

    The graph label is 1 iff the two infected nodes have shortest distance 3
    and at least two distinct shortest length-3 paths. Edge-level GT is exposed
    only for positives with exactly two such paths whose undirected edge union
    has size 6.
    """
    if int(num_graphs) % 2 != 0:
        raise ValueError("InfectedPair-MultiPath3 expects an even num_graphs for 50/50 balance")

    rng = np.random.RandomState(int(seed))
    positive_count = int(num_graphs) // 2
    negative_count = int(num_graphs) - positive_count

    positives: List[Dict[str, object]] = []
    negative_pool: List[Dict[str, object]] = []

    attempts = 0
    while (len(positives) < positive_count or len(negative_pool) < negative_count) and attempts < int(max_attempts):
        candidate = _sample_multipath3_candidate(
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
            "failed to collect balanced InfectedPair-MultiPath3 dataset: "
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
    valid_gt_flags = np.asarray([bool(graph["gt_valid"]) for graph in graphs], dtype=bool)
    positive_length3_counts = [
        int(graph["length3_path_count"])
        for graph in graphs
        if int(graph["label"]) == 1
    ]
    positive_union_sizes = [
        int(graph["length3_edge_union_size"])
        for graph in graphs
        if int(graph["label"]) == 1 and int(graph["length3_path_count"]) == 2
    ]
    negative_buckets = Counter(
        str(graph["distance_bucket"])
        for graph in graphs
        if int(graph["label"]) == 0
    )

    possible_edges = int(num_nodes) * (int(num_nodes) - 1) / 2.0
    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "target_avg_num_edges": float(target_avg_num_edges),
        "positive_target_distance": 3,
        "positive_rule": "shortest_distance_eq_3_and_length3_path_count_ge_2",
        "gt_rule": "positive_and_length3_path_count_eq_2_and_edge_union_size_eq_6",
        "er_edge_prob": float(min(1.0, max(0.0, float(target_avg_num_edges) / possible_edges))),
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "num_gt_valid_positive_graphs": int(np.sum(valid_gt_flags & (labels == 1))),
        "num_gt_invalid_positive_graphs": int(np.sum((~valid_gt_flags) & (labels == 1))),
        "positive_length3_path_count_distribution": {
            int(value): int(sum(1 for observed in positive_length3_counts if observed == int(value)))
            for value in sorted(set(positive_length3_counts))
        },
        "positive_exact_two_path_edge_union_size_distribution": {
            int(value): int(sum(1 for observed in positive_union_sizes if observed == int(value)))
            for value in sorted(set(positive_union_sizes))
        },
        "negative_distance_bucket_counts": {
            bucket: int(count) for bucket, count in sorted(negative_buckets.items())
        },
        "negative_pool_size_before_downsample": int(len(negative_pool)),
        "num_attempts": int(attempts),
        "mean_num_edges_positive": float(num_edges[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_num_edges_negative": float(num_edges[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "gt_undirected_budget": 6,
        "gt_directed_label_count_positive_valid": 12,
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }
