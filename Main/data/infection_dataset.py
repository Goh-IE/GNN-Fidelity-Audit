import math
from collections import deque
from typing import Dict, List, Tuple, Literal

import numpy as np


DEFAULT_NUM_NODES = 1000
DEFAULT_EDGE_PROB = 0.004
DEFAULT_NUM_INFECTED = 50
DEFAULT_SEED = 42

InfectionFeatureMode = Literal["original", "one_hot", "infected_bernoulli"]
InfectionDistanceMode = Literal[4, 5]


def _build_undirected_er_graph(
    num_nodes: int,
    edge_prob: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Return an undirected adjacency matrix (without self-loops)."""
    upper = rng.rand(num_nodes, num_nodes) < edge_prob
    upper = np.triu(upper, k=1)
    adj = upper | upper.T
    return adj


def _adj_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """Convert adjacency matrix to edge_index with both directions."""
    src, dst = np.where(adj)
    return np.vstack([src, dst]).astype(np.int64)


def _build_neighbors(edge_index: np.ndarray, num_nodes: int) -> List[List[int]]:
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for k in range(edge_index.shape[1]):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])
        neighbors[u].append(v)
    return neighbors


def _multi_source_shortest_distance(
    neighbors: List[List[int]],
    sources: np.ndarray,
) -> np.ndarray:
    """Compute shortest-path distance from source set in an unweighted graph."""
    num_nodes = len(neighbors)
    inf_dist = np.iinfo(np.int32).max
    dist = np.full(num_nodes, inf_dist, dtype=np.int32)
    q: deque[int] = deque()

    for s in sources.tolist():
        dist[s] = 0
        q.append(int(s))

    while q:
        u = q.popleft()
        next_dist = int(dist[u]) + 1
        for v in neighbors[u]:
            if dist[v] == inf_dist:
                dist[v] = next_dist
                q.append(v)

    return dist


def _count_shortest_paths_from_source_set(
    neighbors: List[List[int]],
    dist: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    """
    Count shortest paths from source set to each node, clipped at 2.
    `count[v] == 1` means unique shortest path, `>=2` means non-unique.
    """
    count = np.zeros(len(neighbors), dtype=np.int8)
    count[sources] = 1

    finite_dist = dist[dist < np.iinfo(np.int32).max]
    if finite_dist.size == 0:
        return count

    max_dist = int(finite_dist.max())
    for d in range(1, max_dist + 1):
        nodes_at_d = np.where(dist == d)[0]
        for v in nodes_at_d:
            total = 0
            for u in neighbors[int(v)]:
                if dist[u] == d - 1 and count[u] > 0:
                    total += int(count[u])
                    if total >= 2:
                        total = 2
                        break
            count[int(v)] = int(total)

    return count


def _label_from_distance(dist: np.ndarray, max_distance_class: int) -> np.ndarray:
    """
    Bucket shortest-path distances into classes 0..max_distance_class.
    The last class means "distance >= max_distance_class" (or unreachable).
    """
    if max_distance_class < 1:
        raise ValueError(f"max_distance_class must be >= 1 (got {max_distance_class})")

    labels = np.full(dist.shape[0], int(max_distance_class), dtype=np.int64)
    for d in range(max_distance_class):
        labels[dist == d] = d
    return labels


def _reconstruct_unique_shortest_path_edges(
    node_idx: int,
    neighbors: List[List[int]],
    dist: np.ndarray,
    path_count: np.ndarray,
    infected_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    """Reconstruct unique shortest-path edges from infected set to node."""
    if path_count[node_idx] != 1:
        return []
    if dist[node_idx] <= 0 or dist[node_idx] >= np.iinfo(np.int32).max:
        return []

    path_nodes = [int(node_idx)]
    cur = int(node_idx)
    while dist[cur] > 0:
        preds = [u for u in neighbors[cur] if dist[u] == dist[cur] - 1 and path_count[u] > 0]
        if len(preds) != 1:
            return []
        cur = int(preds[0])
        path_nodes.append(cur)

    if not infected_mask[cur]:
        return []

    path_nodes = list(reversed(path_nodes))
    path_edges: List[Tuple[int, int]] = []
    for i in range(1, len(path_nodes)):
        a, b = int(path_nodes[i - 1]), int(path_nodes[i])
        u, v = (a, b) if a < b else (b, a)
        path_edges.append((u, v))

    return sorted(set(path_edges))


def _build_node_splits(
    num_nodes: int,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0.0 < train_ratio < 1.0
    assert 0.0 < val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0

    perm = rng.permutation(np.arange(num_nodes))
    n_train = int(math.floor(train_ratio * num_nodes))
    n_val = int(math.floor(val_ratio * num_nodes))
    n_test = num_nodes - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]

    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def generate_infection_dataset(
    num_nodes: int = DEFAULT_NUM_NODES,
    edge_prob: float = DEFAULT_EDGE_PROB,
    num_infected: int = DEFAULT_NUM_INFECTED,
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    feature_mode: InfectionFeatureMode = "original",
    max_distance_class: InfectionDistanceMode = 5,
    noninfected_noise_std: float = 0.0,
    bernoulli_feature_dim: int = 2,
    bernoulli_feature_prob: float = 0.5,
) -> Dict[str, object]:
    """
    Generate undirected Infection-Path node classification dataset.
    Labels:
      0: infected node
      1..(max_distance_class-1): shortest distance to infected set
      max_distance_class: distance >= max_distance_class or unreachable
    """
    if float(noninfected_noise_std) < 0.0:
        raise ValueError("noninfected_noise_std must be non-negative")
    if int(bernoulli_feature_dim) < 2:
        raise ValueError("bernoulli_feature_dim must be at least 2")
    if not (0.0 <= float(bernoulli_feature_prob) <= 1.0):
        raise ValueError("bernoulli_feature_prob must be in [0, 1]")
    if not (1 <= int(num_infected) <= int(num_nodes)):
        raise ValueError(
            f"num_infected must be in [1, num_nodes], got {num_infected} for num_nodes={num_nodes}"
        )

    rng = np.random.RandomState(seed)

    adj = _build_undirected_er_graph(num_nodes=num_nodes, edge_prob=edge_prob, rng=rng)
    edge_index = _adj_to_edge_index(adj)
    neighbors = _build_neighbors(edge_index=edge_index, num_nodes=num_nodes)

    infected_nodes = np.sort(
        rng.choice(int(num_nodes), size=int(num_infected), replace=False)
    ).astype(np.int64)
    infected_mask = np.zeros(int(num_nodes), dtype=bool)
    infected_mask[infected_nodes] = True

    dist = _multi_source_shortest_distance(neighbors=neighbors, sources=infected_nodes)
    path_count = _count_shortest_paths_from_source_set(
        neighbors=neighbors,
        dist=dist,
        sources=infected_nodes,
    )
    labels = _label_from_distance(dist, max_distance_class=int(max_distance_class))

    if feature_mode == "original":
        features = np.zeros((num_nodes, 2), dtype=np.float32)
        features[:, 0] = infected_mask.astype(np.float32)
        features[:, 1] = 1.0
    elif feature_mode == "one_hot":
        features = np.zeros((num_nodes, 2), dtype=np.float32)
        features[infected_mask, 0] = 1.0
        features[~infected_mask, 1] = 1.0
    elif feature_mode == "infected_bernoulli":
        feature_dim = int(bernoulli_feature_dim)
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        features[infected_mask, 0] = 1.0
        feature_rng = np.random.RandomState((int(seed) + 2000003) % (2 ** 32 - 1))
        features[:, 1:] = (
            feature_rng.rand(num_nodes, feature_dim - 1) < float(bernoulli_feature_prob)
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    noise_std = float(noninfected_noise_std)
    if noise_std > 0.0:
        feature_rng = np.random.RandomState((int(seed) + 1000003) % (2 ** 32 - 1))
        features[~infected_mask, 1] = (
            1.0 + feature_rng.normal(loc=0.0, scale=noise_std, size=int((~infected_mask).sum()))
        ).astype(np.float32)

    train_mask, val_mask, test_mask = _build_node_splits(
        num_nodes=num_nodes,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        rng=rng,
    )

    node_to_gt_edges: Dict[int, List[Tuple[int, int]]] = {}
    gt_test_indices: List[int] = []
    test_nodes = np.where(test_mask)[0]
    for node_idx in test_nodes.tolist():
        node_label = int(labels[node_idx])
        # Only include nodes with valid ground truth paths:
        # label 0 = infected (no path needed), label >= max_distance_class = bucket class (path unclear)
        if node_label < 1 or node_label >= max_distance_class:
            continue
        path_edges = _reconstruct_unique_shortest_path_edges(
            node_idx=node_idx,
            neighbors=neighbors,
            dist=dist,
            path_count=path_count,
            infected_mask=infected_mask,
        )
        if len(path_edges) != node_label:
            continue
        node_to_gt_edges[int(node_idx)] = path_edges
        gt_test_indices.append(int(node_idx))

    metadata = {
        "num_nodes": int(num_nodes),
        "edge_prob": float(edge_prob),
        "num_infected_requested": int(num_infected),
        "infected_sampling": "uniform_without_replacement",
        "seed": int(seed),
        "feature_mode": str(feature_mode),
        "max_distance_class": int(max_distance_class),
        "noninfected_noise_std": float(noninfected_noise_std),
        "bernoulli_feature_dim": int(bernoulli_feature_dim),
        "bernoulli_feature_prob": float(bernoulli_feature_prob),
        "num_undirected_edges": int(adj.sum() // 2),
        "num_infected_nodes": int(infected_mask.sum()),
        "num_gt_test_nodes": int(len(gt_test_indices)),
    }

    return {
        "edge_index": edge_index,
        "features": features,
        "labels": labels.astype(np.int64),
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "infected_mask": infected_mask.astype(bool),
        "distances": dist.astype(np.int64),
        "shortest_path_count_clipped": path_count.astype(np.int64),
        "node_to_gt_edges": node_to_gt_edges,
        "gt_test_indices": gt_test_indices,
        "metadata": metadata,
    }
