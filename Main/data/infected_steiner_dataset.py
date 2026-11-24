from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


DEFAULT_NUM_GRAPHS = 1000
DEFAULT_NUM_NODES = 50
DEFAULT_SEED = 0
DEFAULT_TARGET_AVG_NUM_EDGES = 40.0
DEFAULT_DECOY_ARMS_PER_INFECTED = 3
INFECTED_NODES = (0, 1, 2)
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


def find_valid_steiner_certificates(
    num_nodes: int,
    undirected_edges: Sequence[Tuple[int, int]],
    infected_nodes: Sequence[int] = INFECTED_NODES,
) -> List[Dict[str, object]]:
    """Find centers with unique internally-disjoint length-2 paths from all infected nodes."""
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


class _NodeAllocator:
    def __init__(self, start: int, stop: int):
        self._next = int(start)
        self._stop = int(stop)

    def take(self, count: int = 1):
        count = int(count)
        if self._next + count > self._stop:
            raise ValueError("num_nodes is too small for InfectedSteiner-3 construction")
        nodes = list(range(self._next, self._next + count))
        self._next += count
        if count == 1:
            return int(nodes[0])
        return nodes


def _features(num_nodes: int, infected_nodes: Sequence[int]) -> np.ndarray:
    infected_set = {int(node) for node in infected_nodes}
    x = np.zeros((int(num_nodes), 2), dtype=np.float32)
    for node_idx in range(int(num_nodes)):
        if node_idx in infected_set:
            x[node_idx, 0] = 1.0
        else:
            x[node_idx, 1] = 1.0
    return x


def _add_edge(edges: Set[Tuple[int, int]], u: int, v: int) -> None:
    if int(u) == int(v):
        return
    edges.add(undirected_pair(int(u), int(v)))


def _add_length2_arm(
    edges: Set[Tuple[int, int]],
    allocator: _NodeAllocator,
    marker: int,
    center: Optional[int] = None,
) -> Tuple[int, int]:
    internal = int(allocator.take())
    if center is None:
        center = int(allocator.take())
    else:
        center = int(center)
    _add_edge(edges, marker, internal)
    _add_edge(edges, internal, center)
    return internal, center


def _add_dummy_degree3_center(edges: Set[Tuple[int, int]], allocator: _NodeAllocator) -> int:
    center = int(allocator.take())
    leaves = allocator.take(3)
    for leaf in leaves:
        _add_edge(edges, center, int(leaf))
    return center


def _add_background_edges(
    edges: Set[Tuple[int, int]],
    num_nodes: int,
    protected_nodes: Set[int],
    target_avg_num_edges: float,
    rng: np.random.RandomState,
) -> Tuple[float, int, Tuple[int, ...]]:
    background_nodes = tuple(
        int(node) for node in range(int(num_nodes)) if int(node) not in protected_nodes
    )
    structural_edges = len(edges)
    possible_edges = len(background_nodes) * (len(background_nodes) - 1) // 2
    missing_expected = max(0.0, float(target_avg_num_edges) - float(structural_edges))
    if possible_edges <= 0:
        return 0.0, 0, background_nodes
    edge_prob = min(1.0, missing_expected / float(possible_edges))

    before = len(edges)
    for pos, u in enumerate(background_nodes):
        for v in background_nodes[pos + 1:]:
            if float(rng.rand()) < edge_prob:
                _add_edge(edges, int(u), int(v))
    return float(edge_prob), int(len(edges) - before), background_nodes


def _build_positive_core(
    num_nodes: int,
    decoy_arms_per_infected: int,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[int], Dict[str, object]]:
    edges: Set[Tuple[int, int]] = set()
    gt_edges: Set[Tuple[int, int]] = set()
    infected = tuple(int(node) for node in INFECTED_NODES)
    protected_nodes: Set[int] = set(infected)
    allocator = _NodeAllocator(start=len(infected), stop=int(num_nodes))

    center = int(allocator.take())
    true_paths: List[Tuple[int, int, int]] = []
    for marker in infected:
        internal, _ = _add_length2_arm(edges, allocator, marker, center=center)
        protected_nodes.update((internal, center))
        true_paths.append((int(marker), int(internal), center))
        gt_edges.add(undirected_pair(marker, internal))
        gt_edges.add(undirected_pair(internal, center))

    decoy_centers: List[int] = []
    for marker in infected:
        for _ in range(int(decoy_arms_per_infected)):
            internal, decoy_center = _add_length2_arm(edges, allocator, marker)
            protected_nodes.update((internal, decoy_center))
            decoy_centers.append(decoy_center)

    dummy_centers = []
    for _ in range(2):
        before = allocator._next
        dummy_center = _add_dummy_degree3_center(edges, allocator)
        protected_nodes.update(range(before, allocator._next))
        dummy_centers.append(dummy_center)

    metadata = {
        "certificate_center": int(center),
        "certificate_paths": tuple(true_paths),
        "decoy_centers": tuple(int(node) for node in decoy_centers),
        "dummy_degree3_centers": tuple(int(node) for node in dummy_centers),
        "negative_type": "",
    }
    return edges, gt_edges, protected_nodes, metadata


def _add_pairwise_center(
    edges: Set[Tuple[int, int]],
    allocator: _NodeAllocator,
    marker_a: int,
    marker_b: int,
) -> int:
    center = int(allocator.take())
    internal_a = int(allocator.take())
    internal_b = int(allocator.take())
    dummy_leaf = int(allocator.take())
    _add_edge(edges, marker_a, internal_a)
    _add_edge(edges, internal_a, center)
    _add_edge(edges, marker_b, internal_b)
    _add_edge(edges, internal_b, center)
    _add_edge(edges, center, dummy_leaf)
    return center


def _build_pairwise_negative_core(
    num_nodes: int,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[int], Dict[str, object]]:
    edges: Set[Tuple[int, int]] = set()
    infected = tuple(int(node) for node in INFECTED_NODES)
    protected_nodes: Set[int] = set(infected)
    allocator = _NodeAllocator(start=len(infected), stop=int(num_nodes))

    pairwise_centers = []
    for marker_a, marker_b in ((infected[0], infected[1]), (infected[1], infected[2]), (infected[0], infected[2])):
        before = allocator._next
        pairwise_centers.append(_add_pairwise_center(edges, allocator, marker_a, marker_b))
        protected_nodes.update(range(before, allocator._next))

    private_centers: List[int] = []
    for marker in infected:
        for _ in range(2):
            internal, private_center = _add_length2_arm(edges, allocator, marker)
            protected_nodes.update((internal, private_center))
            private_centers.append(private_center)

    filler_edges: List[Tuple[int, int]] = []
    for _ in range(3):
        u = int(allocator.take())
        v = int(allocator.take())
        _add_edge(edges, u, v)
        protected_nodes.update((u, v))
        filler_edges.append(undirected_pair(u, v))

    metadata = {
        "certificate_center": -1,
        "certificate_paths": tuple(),
        "pairwise_centers": tuple(int(node) for node in pairwise_centers),
        "private_centers": tuple(int(node) for node in private_centers),
        "filler_edges": tuple(filler_edges),
        "negative_type": "pairwise",
    }
    return edges, set(), protected_nodes, metadata


def _build_near_common_negative_core(
    num_nodes: int,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[int], Dict[str, object]]:
    edges: Set[Tuple[int, int]] = set()
    infected = tuple(int(node) for node in INFECTED_NODES)
    protected_nodes: Set[int] = set(infected)
    allocator = _NodeAllocator(start=len(infected), stop=int(num_nodes))

    center = int(allocator.take())
    u0 = int(allocator.take())
    u1 = int(allocator.take())
    w2 = int(allocator.take())
    z2 = int(allocator.take())
    _add_edge(edges, infected[0], u0)
    _add_edge(edges, u0, center)
    _add_edge(edges, infected[1], u1)
    _add_edge(edges, u1, center)
    _add_edge(edges, infected[2], w2)
    _add_edge(edges, w2, z2)
    _add_edge(edges, z2, center)
    protected_nodes.update((center, u0, u1, w2, z2))

    private_centers: List[int] = []
    short_leaf = None
    for marker_pos, marker in enumerate(infected):
        for arm_idx in range(3):
            if marker_pos == 0 and arm_idx == 0:
                short_leaf = int(allocator.take())
                _add_edge(edges, marker, short_leaf)
                protected_nodes.add(short_leaf)
                continue
            internal, private_center = _add_length2_arm(edges, allocator, marker)
            protected_nodes.update((internal, private_center))
            private_centers.append(private_center)

    dummy_centers = []
    for _ in range(2):
        before = allocator._next
        dummy_center = _add_dummy_degree3_center(edges, allocator)
        protected_nodes.update(range(before, allocator._next))
        dummy_centers.append(dummy_center)

    metadata = {
        "certificate_center": -1,
        "certificate_paths": tuple(),
        "near_common_center": int(center),
        "near_common_distances": (2, 2, 3),
        "private_centers": tuple(int(node) for node in private_centers),
        "short_private_leaf": -1 if short_leaf is None else int(short_leaf),
        "dummy_degree3_centers": tuple(int(node) for node in dummy_centers),
        "negative_type": "near_common",
    }
    return edges, set(), protected_nodes, metadata


def _build_single_graph(
    graph_idx: int,
    label: int,
    negative_type: str,
    num_nodes: int,
    target_avg_num_edges: float,
    decoy_arms_per_infected: int,
    rng: np.random.RandomState,
) -> Dict[str, object]:
    min_required_nodes = 33
    if int(num_nodes) < min_required_nodes:
        raise ValueError(f"num_nodes={num_nodes} is too small; InfectedSteiner-3 needs at least {min_required_nodes}")

    if int(label) == 1:
        edges, gt_edges, protected_nodes, metadata = _build_positive_core(
            num_nodes=num_nodes,
            decoy_arms_per_infected=decoy_arms_per_infected,
        )
    elif str(negative_type) == "near_common":
        edges, gt_edges, protected_nodes, metadata = _build_near_common_negative_core(num_nodes=num_nodes)
    else:
        edges, gt_edges, protected_nodes, metadata = _build_pairwise_negative_core(num_nodes=num_nodes)

    structural_edges = len(edges)
    background_edge_prob, background_edge_count, background_nodes = _add_background_edges(
        edges=edges,
        num_nodes=num_nodes,
        protected_nodes=protected_nodes,
        target_avg_num_edges=target_avg_num_edges,
        rng=rng,
    )

    certificates = find_valid_steiner_certificates(
        num_nodes=num_nodes,
        undirected_edges=sorted(edges),
        infected_nodes=INFECTED_NODES,
    )
    if int(label) == 1:
        if len(certificates) != 1:
            raise ValueError(f"positive graph has {len(certificates)} valid certificates")
        observed_gt_edges = {undirected_pair(u, v) for (u, v) in certificates[0]["gt_edges"]}
        if observed_gt_edges != {undirected_pair(u, v) for (u, v) in gt_edges}:
            raise ValueError("positive graph constructed GT does not match certificate scanner")
    elif len(certificates) != 0:
        raise ValueError(f"negative graph has {len(certificates)} valid certificates")

    if int(label) == 1 and len(gt_edges) != GT_UNDIRECTED_BUDGET:
        raise ValueError(f"positive graph GT has {len(gt_edges)} undirected edges, expected {GT_UNDIRECTED_BUDGET}")

    edge_index, edge_labels = _edge_index_and_labels_from_edges(sorted(edges), gt_edges if int(label) == 1 else set())

    return {
        "graph_idx": int(graph_idx),
        "edge_index": edge_index,
        "edge_labels": edge_labels,
        "features": _features(num_nodes, INFECTED_NODES),
        "label": int(label),
        "infected_nodes": tuple(int(node) for node in INFECTED_NODES),
        "gt_valid": bool(int(label) == 1),
        "gt_edges": sorted(gt_edges) if int(label) == 1 else [],
        "gt_undirected_budget": int(len(gt_edges)) if int(label) == 1 else 0,
        "gt_directed_label_count": int(edge_labels.sum()),
        "num_valid_certificates": int(len(certificates)),
        "undirected_edges": sorted(edges),
        "num_edges": int(len(edges)),
        "structural_edges": int(structural_edges),
        "background_edge_prob": float(background_edge_prob),
        "background_edge_count": int(background_edge_count),
        "background_nodes": tuple(int(node) for node in background_nodes),
        "protected_nodes": tuple(sorted(int(node) for node in protected_nodes)),
        "target_avg_num_edges": float(target_avg_num_edges),
        **metadata,
    }


def _schedule(num_graphs: int, rng: np.random.RandomState) -> List[Tuple[int, str]]:
    positive_count = int(num_graphs) // 2
    negative_count = int(num_graphs) - positive_count
    schedule: List[Tuple[int, str]] = [(1, "") for _ in range(positive_count)]
    for idx in range(negative_count):
        negative_type = "pairwise" if idx % 2 == 0 else "near_common"
        schedule.append((0, negative_type))
    rng.shuffle(schedule)
    return schedule


def generate_infected_steiner3_dataset(
    num_graphs: int = DEFAULT_NUM_GRAPHS,
    num_nodes: int = DEFAULT_NUM_NODES,
    seed: int = DEFAULT_SEED,
    target_avg_num_edges: float = DEFAULT_TARGET_AVG_NUM_EDGES,
    decoy_arms_per_infected: int = DEFAULT_DECOY_ARMS_PER_INFECTED,
    max_attempts_per_graph: int = 100,
) -> Dict[str, object]:
    """
    Generate InfectedSteiner-3 graph classification data.

    Positive graphs have one unique ordinary center at distance 2 from all
    three infected nodes. Negative graphs preserve local marker/candidate
    structure but have no valid triple common center. Ground truth edge labels
    are exposed only for positives.
    """
    rng = np.random.RandomState(int(seed))
    schedule = _schedule(int(num_graphs), rng)

    graphs: List[Dict[str, object]] = []
    reject_count = 0
    for graph_idx, (label, negative_type) in enumerate(schedule):
        for attempt in range(int(max_attempts_per_graph)):
            try:
                graph = _build_single_graph(
                    graph_idx=int(graph_idx),
                    label=int(label),
                    negative_type=str(negative_type),
                    num_nodes=int(num_nodes),
                    target_avg_num_edges=float(target_avg_num_edges),
                    decoy_arms_per_infected=int(decoy_arms_per_infected),
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
    negative_types = [str(graph["negative_type"]) for graph in graphs if int(graph["label"]) == 0]

    metadata = {
        "num_graphs": int(num_graphs),
        "num_nodes": int(num_nodes),
        "seed": int(seed),
        "target_avg_num_edges": float(target_avg_num_edges),
        "decoy_arms_per_infected": int(decoy_arms_per_infected),
        "num_positive_graphs": int(labels.sum()),
        "num_negative_graphs": int(labels.shape[0] - labels.sum()),
        "negative_type_counts": {
            negative_type: int(sum(1 for value in negative_types if value == negative_type))
            for negative_type in sorted(set(negative_types))
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
