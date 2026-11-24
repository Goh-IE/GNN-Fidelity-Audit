from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np


DEFAULT_NUM_GRAPHS = 1000
DEFAULT_NUM_NODES = 50
DEFAULT_SEED = 0
DEFAULT_TARGET_AVG_NUM_EDGES = 60.0
POSITIVE_BUCKET = "unique3"
NEGATIVE_BUCKETS = ("dist1", "dist2", "dist3_nonunique", "other")
GT_UNDIRECTED_BUDGET = 3


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
) -> Tuple[Set[Tuple[int, int]], float]:
    possible_edges = int(num_nodes) * (int(num_nodes) - 1) // 2
    edge_prob = min(1.0, max(0.0, float(target_avg_num_edges) / float(possible_edges)))
    edges: Set[Tuple[int, int]] = set()
    for u in range(int(num_nodes)):
        for v in range(u + 1, int(num_nodes)):
            if float(rng.rand()) < edge_prob:
                edges.add((u, v))
    return edges, float(edge_prob)


def _classify_pair_candidates(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
) -> Dict[str, List[Dict[str, object]]]:
    neighbors = _build_neighbors(num_nodes=num_nodes, undirected_edges=undirected_edges)
    buckets: Dict[str, List[Dict[str, object]]] = {
        POSITIVE_BUCKET: [],
        "dist1": [],
        "dist2": [],
        "dist3_nonunique": [],
        "other": [],
    }

    inf_dist = np.iinfo(np.int32).max
    for source in range(int(num_nodes)):
        dist, count, pred = _bfs_dist_count_pred(neighbors, source=source)
        for target in range(source + 1, int(num_nodes)):
            distance = int(dist[target])
            path_nodes: Tuple[int, ...] = tuple()
            gt_edges: Set[Tuple[int, int]] = set()
            shortest_count = 0 if distance == inf_dist else int(count[target])

            if distance == 3 and shortest_count == 1:
                path_nodes = _unique_path_from_pred(pred, source=source, target=target)
                gt_edges = _path_edges(path_nodes)
                bucket = POSITIVE_BUCKET
            elif distance == 1:
                bucket = "dist1"
            elif distance == 2:
                bucket = "dist2"
            elif distance == 3 and shortest_count > 1:
                bucket = "dist3_nonunique"
            else:
                bucket = "other"

            buckets[bucket].append({
                "pair": (int(source), int(target)),
                "distance": int(distance),
                "shortest_path_count": int(shortest_count),
                "path_nodes": tuple(int(node) for node in path_nodes),
                "gt_edges": sorted(gt_edges),
                "bucket": bucket,
            })

    return buckets


def _features(num_nodes: int, infected_nodes: Sequence[int]) -> np.ndarray:
    infected_set = {int(node) for node in infected_nodes}
    x = np.zeros((int(num_nodes), 2), dtype=np.float32)
    for node_idx in range(int(num_nodes)):
        if node_idx in infected_set:
            x[node_idx, 0] = 1.0
        else:
            x[node_idx, 1] = 1.0
    return x


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


def _schedule(num_graphs: int, rng: np.random.RandomState) -> List[str]:
    positive_count = int(num_graphs) // 2
    negative_count = int(num_graphs) - positive_count
    schedule: List[str] = [POSITIVE_BUCKET] * positive_count

    base_count = negative_count // len(NEGATIVE_BUCKETS)
    remainder = negative_count % len(NEGATIVE_BUCKETS)
    for offset, bucket in enumerate(NEGATIVE_BUCKETS):
        count = base_count + (1 if offset < remainder else 0)
        schedule.extend([bucket] * count)

    rng.shuffle(schedule)
    return schedule


def _build_single_graph(
    graph_idx: int,
    target_bucket: str,
    num_nodes: int,
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    edges, edge_prob = _sample_er_edges(
        num_nodes=num_nodes,
        target_avg_num_edges=target_avg_num_edges,
        rng=rng,
    )
    buckets = _classify_pair_candidates(num_nodes=num_nodes, undirected_edges=sorted(edges))
    candidates = buckets[str(target_bucket)]
    if not candidates:
        raise ValueError(f"no candidate pairs for bucket={target_bucket}")

    candidate = candidates[int(rng.randint(len(candidates)))]
    infected_pair = tuple(int(node) for node in candidate["pair"])
    label = 1 if str(target_bucket) == POSITIVE_BUCKET else 0
    gt_edges = {undirected_pair(u, v) for (u, v) in candidate["gt_edges"]} if label == 1 else set()
    path_nodes = tuple(int(node) for node in candidate["path_nodes"]) if label == 1 else tuple()

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
    if label == 1:
        if observed_distance != 3 or observed_count != 1:
            raise ValueError(
                f"positive validation failed: dist={observed_distance}, count={observed_count}"
            )
        if len(remapped_gt_edges) != GT_UNDIRECTED_BUDGET:
            raise ValueError(f"positive GT has {len(remapped_gt_edges)} edges")
    elif observed_distance == 3 and observed_count == 1:
        raise ValueError("negative validation failed: infected pair is positive")

    edge_index, edge_labels = _edge_index_and_labels_from_edges(remapped_edges, remapped_gt_edges)

    return {
        "graph_idx": int(graph_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes, remapped_infected_pair),
        "label": int(label),
        "infected_nodes": tuple(int(node) for node in remapped_infected_pair),
        "gt_valid": bool(label == 1),
        "gt_edges": sorted(remapped_gt_edges) if label == 1 else [],
        "gt_path_nodes": tuple(int(node) for node in remapped_path_nodes),
        "gt_undirected_budget": int(len(remapped_gt_edges)) if label == 1 else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "target_bucket": str(target_bucket),
        "target_distance": int(observed_distance),
        "shortest_path_count": int(observed_count),
        "undirected_edges": sorted(remapped_edges),
        "num_edges": int(len(remapped_edges)),
        "er_edge_prob": float(edge_prob),
        "target_avg_num_edges": float(target_avg_num_edges),
        "node_permutation": node_permutation,
        "candidate_counts": {bucket: int(len(items)) for bucket, items in buckets.items()},
    }


def generate_infected_pair_unique3_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    max_attempts_per_graph: int = 100,
) -> Dict[str, object]:
    """
    Generate InfectedPair-Unique3 graph classification data.

    Graph topology is sampled from G(n, p). Exactly two nodes are marked
    infected after graph sampling. Positive graphs mark a pair with a unique
    shortest path of length 3; negative graphs mark a non-positive pair from
    balanced negative buckets. Edge-level ground truth is exposed only for
    positives and is the unique length-3 shortest path.
    """
    rng = np.random.RandomState(int(seed))
    schedule = _schedule(int(num_graphs), rng)

    graphs: List[Dict[str, object]] = []
    reject_count = 0
    for graph_idx, target_bucket in enumerate(schedule):
        for attempt in range(int(max_attempts_per_graph)):
            try:
                graph = _build_single_graph(
                    graph_idx=int(graph_idx),
                    target_bucket=str(target_bucket),
                    num_nodes=int(num_nodes),
                    target_avg_num_edges=float(target_avg_num_edges),
                    rng=rng,
                )
                graphs.append(graph)
                break
            except ValueError:
                reject_count += 1
                if attempt + 1 >= int(max_attempts_per_graph):
                    raise

    labels = np.asarray([int(graph["label"]) for graph in graphs], dtype=np.int64)
    num_edges = np.asarray([int(graph["num_edges"]) for graph in graphs], dtype=np.int64)
    target_buckets = [str(graph["target_bucket"]) for graph in graphs]

    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "target_avg_num_edges": float(target_avg_num_edges),
        "er_edge_prob": float(min(1.0, max(0.0, float(target_avg_num_edges) / (int(num_nodes) * (int(num_nodes) - 1) / 2.0)))),
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "bucket_counts": {
            bucket: int(sum(1 for value in target_buckets if value == bucket))
            for bucket in [POSITIVE_BUCKET] + list(NEGATIVE_BUCKETS)
        },
        "mean_num_edges_positive": float(num_edges[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_num_edges_negative": float(num_edges[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "gt_undirected_budget": int(GT_UNDIRECTED_BUDGET),
        "gt_directed_label_count_positive": int(2 * GT_UNDIRECTED_BUDGET),
        "reject_count": int(reject_count),
    }

    return {
        "graphs": graphs,
        "metadata": metadata,
    }
