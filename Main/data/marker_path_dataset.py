from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


DEFAULT_NUM_GRAPHS = 1000
DEFAULT_NUM_NODES = 25
DEFAULT_SEED = 0
DEFAULT_BRANCH_EDGE_PROB = 0.15
DEFAULT_POSITIVE_DISTANCE = 3
DEFAULT_NEGATIVE_DISTANCES = (1, 2, 4)
MARKER_SOURCE = 0
MARKER_TARGET = 1


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


def shortest_distance_and_count_clipped(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    source: int = MARKER_SOURCE,
    target: int = MARKER_TARGET,
) -> Tuple[int, int]:
    """Return `(dist(source, target), num_shortest_paths_clipped_at_2)`."""
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=undirected_edges)
    inf_dist = np.iinfo(np.int32).max
    dist = np.full(int(num_nodes), inf_dist, dtype=np.int32)
    count = np.zeros(int(num_nodes), dtype=np.int8)

    source = int(source)
    target = int(target)
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
                q.append(v)
            elif dist[v] == next_dist:
                count[v] = min(2, int(count[v]) + int(count[u]))

    if dist[target] == inf_dist:
        return inf_dist, 0
    return int(dist[target]), int(count[target])


def _features(num_nodes: int) -> np.ndarray:
    x = np.zeros((int(num_nodes), 3), dtype=np.float32)
    x[MARKER_SOURCE, 0] = 1.0
    x[MARKER_TARGET, 1] = 1.0
    for node_idx in range(int(num_nodes)):
        if node_idx not in (MARKER_SOURCE, MARKER_TARGET):
            x[node_idx, 2] = 1.0
    return x


def _distance_schedule(num_graphs: int, rng: np.random.RandomState) -> List[int]:
    positive_count = int(num_graphs) // 2
    negative_count = int(num_graphs) - positive_count

    schedule: List[int] = [DEFAULT_POSITIVE_DISTANCE] * positive_count
    base_negative_count = negative_count // len(DEFAULT_NEGATIVE_DISTANCES)
    remainder = negative_count % len(DEFAULT_NEGATIVE_DISTANCES)
    for offset, distance in enumerate(DEFAULT_NEGATIVE_DISTANCES):
        count = base_negative_count + (1 if offset < remainder else 0)
        schedule.extend([int(distance)] * int(count))

    rng.shuffle(schedule)
    return schedule


def _build_single_graph(
    graph_idx: int,
    target_distance: int,
    num_nodes: int,
    branch_edge_prob: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    if int(num_nodes) < int(target_distance) + 1:
        raise ValueError(f"num_nodes={num_nodes} is too small for target_distance={target_distance}")

    internal_count = int(target_distance) - 1
    internal_nodes = list(range(2, 2 + internal_count))
    path_nodes = [MARKER_SOURCE] + internal_nodes + [MARKER_TARGET]

    edges: Set[Tuple[int, int]] = set()
    gt_path_edges: Set[Tuple[int, int]] = set()
    for left, right in zip(path_nodes[:-1], path_nodes[1:]):
        edge = undirected_pair(left, right)
        edges.add(edge)
        gt_path_edges.add(edge)

    remaining_nodes = list(range(2 + internal_count, int(num_nodes)))
    branch_nodes_by_anchor: Dict[int, List[int]] = {int(anchor): [] for anchor in path_nodes}
    for node_idx in remaining_nodes:
        anchor = int(rng.choice(path_nodes))
        branch_nodes_by_anchor[anchor].append(int(node_idx))
        edges.add(undirected_pair(anchor, node_idx))

    for branch_nodes in branch_nodes_by_anchor.values():
        for pos, u in enumerate(branch_nodes):
            for v in branch_nodes[pos + 1:]:
                if float(rng.rand()) < float(branch_edge_prob):
                    edges.add(undirected_pair(u, v))

    distance, path_count = shortest_distance_and_count_clipped(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        source=MARKER_SOURCE,
        target=MARKER_TARGET,
    )
    if distance != int(target_distance) or path_count != 1:
        raise ValueError(
            f"invalid MarkerPath graph: graph_idx={graph_idx}, "
            f"target_distance={target_distance}, observed=({distance}, {path_count})"
        )

    label = 1 if int(target_distance) == DEFAULT_POSITIVE_DISTANCE else 0
    gt_edges = gt_path_edges if label == 1 else set()
    edge_index, edge_labels = _edge_index_and_labels_from_edges(sorted(edges), gt_edges)

    return {
        "graph_idx": int(graph_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes),
        "label": int(label),
        "target_distance": int(target_distance),
        "shortest_path_count": int(path_count),
        "marker_nodes": (MARKER_SOURCE, MARKER_TARGET),
        "gt_edges": sorted(gt_edges),
        "undirected_edges": sorted(edges),
    }


def generate_marker_path_exact3_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    branch_edge_prob: float = DEFAULT_BRANCH_EDGE_PROB,
    max_attempts_per_graph: int = 100,
) -> Dict[str, object]:
    """
    Generate MarkerPath-Exact3 graph classification data.

    Positive graphs have a unique marker-to-marker shortest path of length 3.
    Negative graphs have a unique marker-to-marker shortest path of length 1, 2, or 4.
    Only positive graphs expose edge-level ground truth.
    """
    rng = np.random.RandomState(int(seed))
    schedule = _distance_schedule(int(num_graphs), rng)

    graphs: List[Dict[str, object]] = []
    reject_count = 0
    for graph_idx, target_distance in enumerate(schedule):
        for attempt in range(int(max_attempts_per_graph)):
            try:
                graph = _build_single_graph(
                    graph_idx=int(graph_idx),
                    target_distance=int(target_distance),
                    num_nodes=int(num_nodes),
                    branch_edge_prob=float(branch_edge_prob),
                    rng=rng,
                )
                graphs.append(graph)
                break
            except ValueError:
                reject_count += 1
                if attempt + 1 >= int(max_attempts_per_graph):
                    raise

    labels = np.asarray([int(graph["label"]) for graph in graphs], dtype=np.int64)
    target_distances = np.asarray([int(graph["target_distance"]) for graph in graphs], dtype=np.int64)

    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "branch_edge_prob": float(branch_edge_prob),
        "positive_distance": int(DEFAULT_POSITIVE_DISTANCE),
        "negative_distances": tuple(int(d) for d in DEFAULT_NEGATIVE_DISTANCES),
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "distance_counts": {
            int(distance): int(np.sum(target_distances == int(distance)))
            for distance in sorted(set(target_distances.tolist()))
        },
        "reject_count": int(reject_count),
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }
