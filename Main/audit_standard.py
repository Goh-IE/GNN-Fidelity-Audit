#!/usr/bin/env python3
"""Run certified standard-fidelity GT-size and budget experiments.

Main is self-contained:
- datasets, pretrained Vanilla models, and MILP builders live under Main
- this script owns the search policy, filtering, metrics, and output tables

The public objective choices cover maximum/minimum sufficiency and necessity.
MILP solves the requested extreme. GT-size enumeration scores the feasible
candidate family once and always reports both extremes together with both
directions of the planted-GT rank.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch


MAIN_DIR = Path(__file__).resolve().parent
REPO_DIR = MAIN_DIR.parent
RESULTS_DIR = MAIN_DIR / "results"
ROOTED_COMPONENT_CPP_SOURCE = MAIN_DIR / "enumeration" / "native" / "rooted_component_enumerator.cpp"
ROOTED_COMPONENT_CPP_PROTOCOL = "ROOTED_COMPONENT_ENUM_V1"
CONCRETE_SUBSET_CPP_SOURCE = MAIN_DIR / "enumeration" / "native" / "concrete_edge_subset_enumerator.cpp"
CONCRETE_SUBSET_CPP_PROTOCOL = "CONCRETE_EDGE_SUBSET_ENUM_V1"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from Main.data.dataset_loaders import load_dataset
from Main.data.ground_truth_loaders import load_dataset_ground_truth
from Main.solvers.MIPExplainer import MIPExplainer
from Main.solvers.mip_gnn_vanilla import CurrentGurobiSolveInterrupted
from Main.models.model_selector import (
    ALL_DATASETS,
    get_dataset_config,
    model_selector,
    normalize_dataset_name,
)
from Main.graph_utils import (
    extract_l_hop_subgraph,
    extract_node_ground_truth_edges_within_hops,
)


DEFAULT_DATASETS = "Tree-Cycles,Infection3-one-hot-binary,BA-2Motifs,BA-HouseANDGrid"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_MULTIPLIERS = "2"
DEFAULT_STAGES = "gt_size"
DEFAULT_MAX_INSTANCES = 30
DEFAULT_TIME_LIMIT = 600.0
DEFAULT_GUROBI_NOREL_HEUR_WORK = 10.0
PILOT_INSTANCE_COUNT = 3
FIXED_HIDDEN_DIM = 8
COUNTEREXAMPLE_TOLERANCE = 1e-6
ENUMERATION_RANK_TOLERANCE = 1e-6
from Main.data.pyg_infection_seed_registry import seeded_infection_names
NODE_EXPLANATION_LABEL_FILTERS = {
    "Infection3-one-hot-binary": 1,
    "PyG-Infection-ER-L3": 3,
    "PyG-Infection-ER-L3-Split43": 3,
    "PyG-Infection-ER-L3-Binary-Split43": 1,
    "PyG-Infection-BA-L3": 3,
    "PyG-Infection-BA-L3-Split43": 3,
    "PyG-Infection-BA-L3-Binary-Split43": 1,
}

for _canonical, _, _, _ in seeded_infection_names():
    NODE_EXPLANATION_LABEL_FILTERS[_canonical] = 3

SUCCESS_STATUSES = {"optimal", "time_limit"}
SKIP_STATUSES = {
    "skipped_incorrect_prediction",
    "skipped_nonpositive_margin",
    "skipped_nonbinary_min_sufficiency",
    "skipped_nonbinary_necessity_milp",
    "skipped_nonbinary_delta_milp",
    "skipped_delta_requires_milp",
    "skipped_delta_requires_exact_budget",
    "skipped_no_ground_truth",
    "skipped_empty_candidate_set",
    "skipped_invalid_budget",
    "skipped_enumeration_too_large",
    "skipped_padding_infeasible",
    "skipped_necessity_graph_only",
    "skipped_necessity_requires_enumeration",
    "skipped_rank_requires_enumeration",
}
SUFFICIENCY_OBJECTIVES = {"max_sufficiency", "min_sufficiency"}
NECESSITY_OBJECTIVES = {"max_necessity", "min_necessity"}
DELTA_OBJECTIVES = {"max_delta"}
RANK_SUFFICIENCY_OBJECTIVE = "rank_sufficiency"
RANK_NECESSITY_OBJECTIVE = "rank_necessity"
RANK_OBJECTIVES = {RANK_SUFFICIENCY_OBJECTIVE, RANK_NECESSITY_OBJECTIVE}
NECESSITY_SCORING_OBJECTIVES = NECESSITY_OBJECTIVES | {RANK_NECESSITY_OBJECTIVE}
DEFAULT_ENUMERATION_BATCH_SIZE = 1024
ENUMERATION_CACHE_VERSION = 2
DEFAULT_ENUMERATION_CACHE_DIR = MAIN_DIR / "standard_enumeration_cache"
MODEL_PAPERS = {"gin": "Vanilla", "gcn": "GCN"}
_EDGE_MASK_PATTERN_CACHE: Dict[int, List[Tuple[Tuple[int, int], ...]]] = {}
_ROOTED_TOPOLOGY_PATTERN_CACHE: Dict[Tuple[int, ...], List[nx.Graph]] = {}
ENUMERATION_TIMING_FIELDS = [
    "recorded_at",
    "dataset",
    "task",
    "cache_kind",
    "instance_id",
    "budget_k",
    "candidate_edge_count",
    "exclude_gt_solution",
    "cache_status",
    "enumeration_performed",
    "prewarm_elapsed_sec",
]

FIELDNAMES = [
    "row_id",
    "dataset",
    "task",
    "nn_seed",
    "instance_id",
    "stage",
    "budget_multiplier",
    "budget_k",
    "budget_mode",
    "connectivity",
    "edge_budget_mode",
    "oracle_objective",
    "solver_backend",
    "gt_incumbent",
    "exclude_gt_solution",
    "gt_solution_representable",
    "gt_no_good_cut_added",
    "enumerated_subgraphs",
    "enumerated_signatures",
    "enumeration_pruned",
    "enumeration_cache_status",
    "enumeration_cache_file",
    "acceleration_profile",
    "use_o1_tightening",
    "cap_mode",
    "use_o3_symmetry_breaking",
    "use_o4_star_constraints",
    # Node-specific O5-O8 accelerations are intentionally excluded from this
    # standard-fidelity driver; the paper uses the task-agnostic bundle.
    "use_o9_connectivity_lb",
    "use_anderson_cuts",
    "anderson_max_node_count",
    "anderson_max_cuts",
    "edge_branch_priority",
    "use_obbt_u",
    "max_obbt_u_layers",
    "obbt_rebuild",
    "degree_table_layer1",
    # Nested cuts and back cuts are intentionally excluded.
    "max_root_sep_rounds",
    "cut_set_scope",
    "k_hop",
    "candidate_size",
    "gt_size",
    "budget_context_size",
    "status",
    "pred",
    "label",
    "is_correct",
    "c_star",
    "num_classes",
    "min_sufficiency_other_class",
    "bar_m",
    "rho_optimal",
    "sufficiency_raw",
    "necessity_raw",
    "sufficiency_probability",
    "complement_probability",
    "fidelity_minus",
    "fidelity_plus",
    "fidelity_delta",
    "objective_value",
    "objective_value_true",
    "pwl_approx_error_at_solution",
    "enum_objective_max",
    "enum_objective_min",
    "enum_rho_at_objective_max",
    "enum_rho_at_objective_min",
    "enum_sufficiency_raw_at_objective_max",
    "enum_sufficiency_raw_at_objective_min",
    "enum_necessity_raw_at_objective_max",
    "enum_necessity_raw_at_objective_min",
    "enum_fidelity_at_objective_max",
    "enum_fidelity_at_objective_min",
    "gt_rho",
    "gt_sufficiency_raw",
    "gt_necessity_raw",
    "gt_sufficiency_probability",
    "gt_complement_probability",
    "gt_fidelity_minus",
    "gt_fidelity_plus",
    "gt_fidelity_delta",
    "gt_objective_value",
    "counterexample_advantage",
    "counterexample_status",
    "counterexample_found",
    "counterexample_definition",
    "gt_rank_max_objective",
    "gt_rank_min_objective",
    "gt_rank_position",
    "gt_rank_total",
    "gt_rank_tie_count",
    "gt_rank_direction",
    "gt_rank_scope",
    "enum_padding_feasible",
    "enum_padding_filtered",
    "selected_size",
    "D",
    "H",
    "precision",
    "recall",
    "f1",
    "padded_size",
    "padded_D",
    "padded_H",
    "padded_precision",
    "padded_recall",
    "padded_f1",
    "padding_policy",
    "padding_seed",
    "padding_needed",
    "padding_candidate_size",
    "padding_added",
    "padding_forbidden_node_count",
    "padding_status",
    "rooted_connected",
    "time_s",
    "final_mip_gap",
    "best_bound",
    "incumbent_obj",
    "model_size",
    "root_lp_value",
    "root_lp_time",
    "root_lp_source",
    "callback_time_sec",
    "callback_calls",
    "cuts_mipnode",
    "cuts_mipsol_lazy",
    "anderson_cuts_added",
    "connectivity_lazy_cuts_added",
    "bnb_node_count",
    "gurobi_log",
    "terminal_log",
    "edge_file",
    "gt_edge_file",
    "padded_edge_file",
    "enum_max_edge_file",
    "enum_min_edge_file",
    "enum_extrema_file",
    "figure_file",
    "gt_figure_file",
    "enum_max_figure_file",
    "enum_min_figure_file",
    "error",
]

COUNTEREXAMPLE_RESULT_FIELDS = [
    "row_id",
    "dataset",
    "task",
    "nn_seed",
    "instance_id",
    "stage",
    "budget_k",
    "oracle_objective",
    "status",
    "exclude_gt_solution",
    "gt_no_good_cut_added",
    "fidelity_minus",
    "gt_fidelity_minus",
    "fidelity_plus",
    "gt_fidelity_plus",
    "fidelity_delta",
    "gt_fidelity_delta",
    "counterexample_advantage",
    "counterexample_status",
    "counterexample_found",
    "counterexample_definition",
    "gt_rank_position",
    "gt_rank_total",
    "gt_rank_tie_count",
    "gt_rank_direction",
    "gt_rank_scope",
    "edge_file",
    "figure_file",
]

SUMMARY_ACCEL_FIELDS = [
    "acceleration_profile",
    "use_o1_tightening",
    "cap_mode",
    "use_o3_symmetry_breaking",
    "use_o4_star_constraints",
    # O5-O8 are intentionally not part of the reported acceleration bundle.
    "use_o9_connectivity_lb",
    "use_anderson_cuts",
    "edge_branch_priority",
    "use_obbt_u",
    "max_obbt_u_layers",
    "obbt_rebuild",
    "degree_table_layer1",
    # Nested cuts and back cuts are not reported.
    "cut_set_scope",
]


CONNECTIVITY_DISABLED_ACCEL_FIELDS = (
    "use_o4_star_constraints",
    "use_o9_connectivity_lb",
)


@dataclass(frozen=True)
class RunConfig:
    preset: str
    datasets: List[str]
    nn_seeds: List[int]
    instance_filter: str
    budget_multipliers: List[float]
    stages: List[str]
    max_instances: Optional[int]
    min_target_degree: Optional[int]
    time_limit_s: Optional[float]
    output_dir: str
    run_name: str
    architecture: str
    model_paper: str
    num_layers: int
    hidden_dim: int
    use_jk_concat: bool
    normalize_features: bool
    readout: str
    k_hop: Optional[int]
    connectivity: str
    node_edge_budget_mode: str
    gt_size_solver: Literal["milp", "enumeration"]
    enumeration_max_subgraphs: int
    enumeration_batch_size: int
    use_enumeration_cache: bool
    enumeration_cache_dir: str
    oracle_objective: str
    gt_incumbent: bool
    exclude_gt_solution: bool
    acceleration_profile: str
    use_o1_tightening: bool
    cap_mode: str
    use_o3_symmetry_breaking: bool
    use_o4_star_constraints: bool
    # Node-only O5-O8 fields are intentionally omitted.
    use_o9_connectivity_lb: bool
    use_anderson_cuts: bool
    anderson_max_node_count: int
    anderson_max_cuts: int
    edge_branch_priority: int
    use_obbt_u: bool
    max_obbt_u_layers: Optional[int]
    obbt_rebuild: bool
    degree_table_layer1: bool
    # Nested cuts and back cuts are intentionally omitted.
    max_root_sep_rounds: int
    cut_set_scope: str
    save_figures: bool
    padding_seed: int
    metric: Optional[str]
    opt_samples: Optional[int]
    eval_samples: Optional[int]
    gurobi_method: Optional[int]
    gurobi_threads: Optional[int]
    gurobi_norel_hybrid: bool
    gurobi_norel_heur_work: Optional[float]
    gurobi_mem_limit: Optional[float]
    experiment_template: Optional[str]
    verbose: bool


def parse_csv_strings(value: str) -> List[str]:
    items = [part.strip() for part in str(value).split(",")]
    return [item for item in items if item]


def effective_gurobi_norel_heur_work(config: RunConfig) -> float:
    if not config.gurobi_norel_hybrid or config.gurobi_norel_heur_work is None:
        return 0.0
    return max(0.0, float(config.gurobi_norel_heur_work))


def gurobi_param_overrides(config: RunConfig) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if config.gurobi_method is not None:
        params["Method"] = int(config.gurobi_method)
    if config.gurobi_threads is not None:
        params["Threads"] = int(config.gurobi_threads)
    params["NoRelHeurWork"] = effective_gurobi_norel_heur_work(config)
    if config.gurobi_mem_limit is not None:
        params["MemLimit"] = float(config.gurobi_mem_limit)
    params["Symmetry"] = 2 if config.use_o3_symmetry_breaking else 0
    return params


def format_gurobi_param_overrides(config: RunConfig) -> str:
    params = gurobi_param_overrides(config)
    if not params:
        return "default"
    return ", ".join(f"{name}={value}" for name, value in params.items())


def expand_dataset_selectors(selectors: Sequence[str]) -> List[str]:
    """Expand dataset wildcard selectors like BA-HouseGrid* before config validation."""
    expanded: List[str] = []
    seen = set()
    available = [normalize_dataset_name(dataset) for dataset in ALL_DATASETS]

    for selector in selectors:
        token = str(selector).strip()
        if not token:
            continue

        if any(ch in token for ch in "*?[]"):
            matches = [
                dataset
                for dataset in available
                if fnmatch.fnmatchcase(dataset, token)
                or fnmatch.fnmatchcase(dataset.lower(), token.lower())
            ]
        else:
            matches = [normalize_dataset_name(token)]

        if not matches:
            matches = [token]

        for dataset in matches:
            canonical = normalize_dataset_name(dataset)
            if canonical not in seen:
                expanded.append(canonical)
                seen.add(canonical)

    return expanded


def parse_csv_ints(value: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for item in parse_csv_strings(value):
        seed = int(item)
        if seed not in seen:
            out.append(seed)
            seen.add(seed)
    return out


def parse_csv_nonnegative_floats(value: str, *, label: str) -> List[float]:
    out: List[float] = []
    seen = set()
    for item in parse_csv_strings(value):
        value_f = float(item)
        if value_f < 0.0:
            raise ValueError(f"{label} values must be nonnegative, got {value_f}.")
        key = round(value_f, 12)
        if key not in seen:
            out.append(value_f)
            seen.add(key)
    return out


def scalar_int(value: Any) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Cannot convert empty label value to int.")
    return int(arr.reshape(-1)[0].item())


def to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def normalize_edge(edge: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        return None
    return (a, b) if a < b else (b, a)


def edge_set(edges: Iterable[Tuple[int, int]]) -> set[Tuple[int, int]]:
    out: set[Tuple[int, int]] = set()
    for edge in edges:
        normalized = normalize_edge(edge)
        if normalized is not None:
            out.add(normalized)
    return out


def unique_nonloop_edges(edge_index: np.ndarray) -> List[Tuple[int, int]]:
    edges: set[Tuple[int, int]] = set()
    edge_index = np.asarray(edge_index)
    for col in range(int(edge_index.shape[1])):
        normalized = normalize_edge((int(edge_index[0, col]), int(edge_index[1, col])))
        if normalized is not None:
            edges.add(normalized)
    return sorted(edges)


def node_degrees_from_edge_index(edge_index: np.ndarray) -> Counter[int]:
    degrees: Counter[int] = Counter()
    for a, b in unique_nonloop_edges(edge_index):
        degrees[int(a)] += 1
        degrees[int(b)] += 1
    return degrees


def compute_margin(logits: torch.Tensor, c_star: int) -> float:
    vec = logits.detach().cpu().reshape(-1)
    if vec.numel() <= 1:
        return 0.0
    if c_star < 0 or c_star >= int(vec.numel()):
        return 0.0
    others = torch.cat((vec[:c_star], vec[c_star + 1 :]))
    return float(vec[c_star].item() - others.max().item())


def compute_metrics(
    selected_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
) -> Dict[str, Optional[float]]:
    selected = edge_set(selected_edges)
    gt = edge_set(gt_edges)
    if not gt:
        return {
            "D": None,
            "H": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    tp = len(selected & gt)
    fp = len(selected - gt)
    fn = len(gt - selected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metadata = {
        "D": recall,
        "H": 1.0 if gt.issubset(selected) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metadata


def candidate_edges_for_node(
    *,
    node_idx: int,
    k_hop: int,
    edge_index: np.ndarray,
    num_nodes: int,
) -> List[Tuple[int, int]]:
    sub_edge_index, _, node_to_sub = extract_l_hop_subgraph(
        center_node=int(node_idx),
        l_hop=int(k_hop),
        edge_index=np.asarray(edge_index),
        num_nodes=int(num_nodes),
    )
    sub_to_node = {int(local): int(orig) for orig, local in node_to_sub.items()}
    edges: set[Tuple[int, int]] = set()
    for col in range(int(sub_edge_index.shape[1])):
        local_a, local_b = int(sub_edge_index[0, col]), int(sub_edge_index[1, col])
        if local_a not in sub_to_node or local_b not in sub_to_node:
            continue
        normalized = normalize_edge((sub_to_node[local_a], sub_to_node[local_b]))
        if normalized is not None:
            edges.add(normalized)
    return sorted(edges)


def graph_prediction_and_margin(
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
) -> Tuple[int, int, float, int]:
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        edge_tensor = torch.tensor(np.asarray(edge_index), dtype=torch.long)
        logits = model(x_tensor, edge_tensor).reshape(-1)
    c_star = int(logits.argmax(dim=-1).item())
    return c_star, c_star, compute_margin(logits, c_star), int(logits.numel())


def node_prediction_and_margin(
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    node_idx: int,
) -> Tuple[int, int, float, int]:
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        edge_tensor = torch.tensor(np.asarray(edge_index), dtype=torch.long)
        logits = model(x_tensor, edge_tensor)
    target_logits = logits[int(node_idx)].reshape(-1)
    c_star = int(target_logits.argmax(dim=-1).item())
    return c_star, c_star, compute_margin(target_logits, c_star), int(target_logits.numel())


def rooted_connected(
    root: int,
    selected_edges: Sequence[Tuple[int, int]],
) -> bool:
    selected = edge_set(selected_edges)
    if not selected:
        return True
    adj: Dict[int, List[int]] = {}
    endpoints: set[int] = set()
    for a, b in selected:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        endpoints.add(a)
        endpoints.add(b)
    if int(root) not in endpoints:
        return False
    seen = {int(root)}
    queue: deque[int] = deque([int(root)])
    while queue:
        cur = queue.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return endpoints.issubset(seen)


def root_distances(root: int, candidate_edges: Sequence[Tuple[int, int]]) -> Dict[int, int]:
    adj: Dict[int, List[int]] = {}
    for a, b in edge_set(candidate_edges):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    dist = {int(root): 0}
    queue: deque[int] = deque([int(root)])
    while queue:
        cur = queue.popleft()
        for nxt in sorted(adj.get(cur, [])):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def filter_edge_index_by_selected_edges(
    edge_index: np.ndarray,
    selected_edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Keep selected undirected edges and preserve existing self-loop columns."""
    edge_index = np.asarray(edge_index)
    if edge_index.size == 0:
        return edge_index[:, :0]

    selected = edge_set(selected_edges)
    keep_cols: List[int] = []
    for col in range(int(edge_index.shape[1])):
        i, j = int(edge_index[0, col]), int(edge_index[1, col])
        if i == j:
            keep_cols.append(col)
            continue
        normalized = normalize_edge((i, j))
        if normalized is not None and normalized in selected:
            keep_cols.append(col)

    if not keep_cols:
        return np.zeros((2, 0), dtype=np.int64)
    return edge_index[:, np.asarray(keep_cols, dtype=np.int64)]


def compact_node_inference_graph(
    *,
    x: np.ndarray,
    edge_index: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    root_node: int,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, Dict[int, int]]:
    """Build the exact compact graph needed to score node edge masks.

    Node explanation masks only activate ``candidate_edges`` (plus any explicit
    self-loop columns).  Nodes and edge columns outside that support cannot
    affect the target of the local message-passing NodeGNN, but retaining them
    in every replicated inference graph is very expensive.  Cache/artifact edge
    IDs stay global; this helper is only an inference-time reindexing.

    Returns ``(local_x, local_edge_index, local_root, original_edge_columns,
    node_to_local)``.  ``original_edge_columns`` maps each local directed edge
    column back to its column in the supplied full ``edge_index`` so robust
    edge-weight batches can be sliced without changing mask semantics.
    """
    x_array = np.asarray(x, dtype=np.float32)
    full_edge_index = np.asarray(edge_index, dtype=np.int64)
    if full_edge_index.ndim != 2 or int(full_edge_index.shape[0]) != 2:
        raise ValueError(f"Expected edge_index shape (2, m), got {full_edge_index.shape}.")
    if not 0 <= int(root_node) < int(x_array.shape[0]):
        raise ValueError("root_node is outside the node feature array.")

    allowed_edges = edge_set(candidate_edges)
    original_nodes = sorted(
        {int(root_node)}
        | {int(node) for edge in allowed_edges for node in edge}
    )
    node_to_local = {node: local for local, node in enumerate(original_nodes)}

    kept_columns: List[int] = []
    local_columns: List[Tuple[int, int]] = []
    for col in range(int(full_edge_index.shape[1])):
        source = int(full_edge_index[0, col])
        target = int(full_edge_index[1, col])
        if source not in node_to_local or target not in node_to_local:
            continue
        if source != target:
            normalized = normalize_edge((source, target))
            if normalized is None or normalized not in allowed_edges:
                continue
        kept_columns.append(col)
        local_columns.append((node_to_local[source], node_to_local[target]))

    local_edge_index = (
        np.asarray(local_columns, dtype=np.int64).T
        if local_columns
        else np.zeros((2, 0), dtype=np.int64)
    )
    return (
        x_array[np.asarray(original_nodes, dtype=np.int64)],
        local_edge_index,
        int(node_to_local[int(root_node)]),
        np.asarray(kept_columns, dtype=np.int64),
        node_to_local,
    )


def feature_only_label_for_signature(x: np.ndarray, node: int) -> str:
    feature = tuple(round(float(v), 10) for v in np.asarray(x[int(node)]).reshape(-1))
    return json.dumps(feature, separators=(",", ":"))


def fixed_self_loop_counts_from_edge_index(
    edge_index: np.ndarray,
    *,
    num_nodes: int,
) -> np.ndarray:
    """Return the fixed input self-loop multiplicity of every graph node."""
    counts = np.zeros(int(num_nodes), dtype=np.int64)
    array = np.asarray(edge_index)
    if array.size == 0:
        return counts
    for col in range(int(array.shape[1])):
        source, target = int(array[0, col]), int(array[1, col])
        if source == target and 0 <= source < int(num_nodes):
            counts[source] += 1
    return counts


def graph_structural_node_colors(
    *,
    x: np.ndarray,
    fixed_self_loop_counts: Sequence[int],
) -> Tuple[str, ...]:
    """Exact Standard-Sufficiency colors: input feature plus fixed loop count."""
    array = np.asarray(x)
    loop_counts = np.asarray(fixed_self_loop_counts, dtype=np.int64).reshape(-1)
    if int(array.shape[0]) != int(loop_counts.shape[0]):
        raise ValueError(
            "Graph structural colors require one fixed self-loop count per node."
        )
    colors: List[str] = []
    for node in range(int(array.shape[0])):
        feature_row = np.ascontiguousarray(array[node])
        colors.append(
            json.dumps(
                {
                    "feature_dtype": str(feature_row.dtype),
                    "feature_shape": [int(value) for value in feature_row.shape],
                    "feature_bytes_hex": feature_row.tobytes().hex(),
                    "fixed_self_loops": int(loop_counts[node]),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return tuple(colors)


def candidate_nx_graph(
    *,
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    root_node: Optional[int],
    node_colors: Optional[Sequence[str]] = None,
) -> nx.Graph:
    graph = nx.Graph()

    def node_color(node: int) -> str:
        if node_colors is None:
            return feature_only_label_for_signature(x, int(node))
        return str(node_colors[int(node)])

    if root_node is not None:
        graph.add_node(
            int(root_node),
            root=1,
            feature=node_color(int(root_node)),
        )
    for a, b in sorted(edge_set(candidate_edges)):
        for node in (int(a), int(b)):
            graph.add_node(
                node,
                root=1 if root_node is not None and node == int(root_node) else 0,
                feature=node_color(node),
            )
        graph.add_edge(int(a), int(b))
    return graph


def exact_colored_graph_signature(graph: nx.Graph) -> str:
    """Return an exact color-preserving nauty certificate, including color identities."""
    try:
        import pynauty  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pynauty is required for exact colored Standard-Sufficiency enumeration."
        ) from exc

    nodes = sorted(int(node) for node in graph.nodes)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    color_by_node: Dict[int, str] = {}
    for node in nodes:
        data = graph.nodes[node]
        color_by_node[node] = json.dumps(
            {
                "root": int(data.get("root", 0)),
                "feature": str(data.get("feature", "")),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    color_labels = sorted(set(color_by_node.values()))
    coloring = [
        {node_to_index[node] for node in nodes if color_by_node[node] == color}
        for color in color_labels
    ]
    adjacency_dict = {
        node_to_index[node]: sorted(node_to_index[int(neighbor)] for neighbor in graph.neighbors(node))
        for node in nodes
    }
    canonical_graph = pynauty.Graph(
        len(nodes),
        adjacency_dict=adjacency_dict,
        vertex_coloring=coloring,
    )
    payload = {
        "node_count": len(nodes),
        "color_labels": color_labels,
        "certificate": pynauty.certificate(canonical_graph).hex(),
    }
    return sha256_json(payload)


def exact_colored_edge_subset_signature(
    *,
    selected_edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[str],
) -> str:
    """Exact colored topology of the non-isolated part of a selected edge mask."""
    graph = nx.Graph()
    for a, b in sorted(edge_set(selected_edges)):
        for node in (int(a), int(b)):
            graph.add_node(int(node), root=0, feature=str(node_colors[int(node)]))
        graph.add_edge(int(a), int(b))
    return exact_colored_graph_signature(graph)


def exact_rooted_colored_edge_subset_signature(
    *,
    selected_edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[str],
    root_node: int,
) -> str:
    """Exact colored topology of a selected mask with a distinguished target root."""
    graph = nx.Graph()
    graph.add_node(
        int(root_node),
        root=1,
        feature=str(node_colors[int(root_node)]),
    )
    for a, b in sorted(edge_set(selected_edges)):
        for node in (int(a), int(b)):
            graph.add_node(
                node,
                root=1 if node == int(root_node) else 0,
                feature=str(node_colors[node]),
            )
        graph.add_edge(int(a), int(b))
    return exact_colored_graph_signature(graph)


def rooted_pattern_signature(pattern: nx.Graph) -> str:
    """Exact rooted/color-preserving certificate for an abstract pattern."""
    return exact_colored_graph_signature(pattern)


def homogeneous_node_features(x: np.ndarray) -> bool:
    features = np.asarray(x, dtype=float)
    if features.ndim != 2 or int(features.shape[0]) <= 0:
        return False
    return bool(np.allclose(features, features[0:1], rtol=1e-9, atol=1e-9))


def uniform_graph_edge_index_convention(
    edge_index: np.ndarray,
    *,
    num_nodes: int,
) -> Optional[Tuple[int, int]]:
    """Return uniform (self_loop_count, nonloop_count_per_direction) if exact pattern scoring is stable."""
    edge_index = np.asarray(edge_index)
    if edge_index.size == 0:
        return 0, 1

    loop_counts = Counter()
    directed_counts = Counter()
    for col in range(int(edge_index.shape[1])):
        i, j = int(edge_index[0, col]), int(edge_index[1, col])
        if i == j:
            loop_counts[i] += 1
        else:
            directed_counts[(i, j)] += 1

    node_loop_counts = [int(loop_counts.get(node, 0)) for node in range(int(num_nodes))]
    if len(set(node_loop_counts)) != 1:
        return None
    self_loop_count = int(node_loop_counts[0]) if node_loop_counts else 0

    undirected_edges = {
        normalize_edge(edge)
        for edge in directed_counts
        if normalize_edge(edge) is not None
    }
    if not undirected_edges:
        return self_loop_count, 1

    per_direction_counts: set[int] = set()
    for edge in undirected_edges:
        if edge is None:
            continue
        a, b = int(edge[0]), int(edge[1])
        forward = int(directed_counts.get((a, b), 0))
        backward = int(directed_counts.get((b, a), 0))
        if forward <= 0 or backward <= 0 or forward != backward:
            return None
        per_direction_counts.add(forward)
    if len(per_direction_counts) != 1:
        return None
    return self_loop_count, int(next(iter(per_direction_counts)))


def _pynauty_certificate_for_adjacency(adj: Tuple[int, ...]) -> Tuple[int, bytes]:
    try:
        import pynauty  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pynauty is required for K>6 graph pattern generation. "
            "Install it or use the existing K<=6 topology enumeration path."
        ) from exc

    adjacency_dict = {
        int(node): [int(other) for other in range(len(adj)) if int(adj[node]) & (1 << other)]
        for node in range(len(adj))
    }
    graph = pynauty.Graph(len(adj), adjacency_dict=adjacency_dict)
    return len(adj), pynauty.certificate(graph)


def _add_pattern_edge_to_adjacency(adj: Tuple[int, ...], a: int, b: int) -> Tuple[int, ...]:
    values = list(adj)
    required_len = max(int(a), int(b)) + 1
    if required_len > len(values):
        values.extend([0] * (required_len - len(values)))
    values[int(a)] |= 1 << int(b)
    values[int(b)] |= 1 << int(a)
    return tuple(int(value) for value in values)


def _pattern_edges_from_adjacency(adj: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    edges: List[Tuple[int, int]] = []
    for a, mask in enumerate(adj):
        for b in range(a + 1, len(adj)):
            if int(mask) & (1 << b):
                edges.append((int(a), int(b)))
    return tuple(edges)


def edge_mask_topology_patterns(edge_count: int) -> List[Tuple[Tuple[int, int], ...]]:
    """Return all non-isomorphic simple edge-mask patterns with no isolated nodes."""
    edge_count = int(edge_count)
    if edge_count <= 0:
        return []
    cached = _EDGE_MASK_PATTERN_CACHE.get(edge_count)
    if cached is not None:
        return cached

    if 1 not in _EDGE_MASK_PATTERN_CACHE:
        _EDGE_MASK_PATTERN_CACHE[1] = [((0, 1),)]

    max_cached = max(_EDGE_MASK_PATTERN_CACHE)
    for cur_edges in range(max_cached + 1, edge_count + 1):
        previous_adjs: List[Tuple[int, ...]] = []
        for previous_edges in _EDGE_MASK_PATTERN_CACHE[cur_edges - 1]:
            node_count = 0
            for a, b in previous_edges:
                node_count = max(node_count, int(a) + 1, int(b) + 1)
            adj = tuple(0 for _ in range(node_count))
            for a, b in previous_edges:
                adj = _add_pattern_edge_to_adjacency(adj, int(a), int(b))
            previous_adjs.append(adj)

        unique: Dict[Tuple[int, bytes], Tuple[int, ...]] = {}
        for base in previous_adjs:
            n = len(base)
            for a in range(n):
                mask = int(base[a])
                for b in range(a + 1, n):
                    if mask & (1 << b):
                        continue
                    candidate = _add_pattern_edge_to_adjacency(base, a, b)
                    unique.setdefault(_pynauty_certificate_for_adjacency(candidate), candidate)
            for a in range(n):
                candidate = _add_pattern_edge_to_adjacency(base, a, n)
                unique.setdefault(_pynauty_certificate_for_adjacency(candidate), candidate)
            candidate = _add_pattern_edge_to_adjacency(base, n, n + 1)
            unique.setdefault(_pynauty_certificate_for_adjacency(candidate), candidate)

        patterns = [_pattern_edges_from_adjacency(adj) for adj in unique.values()]
        patterns.sort(key=lambda edges: (len({node for edge in edges for node in edge}), edges))
        _EDGE_MASK_PATTERN_CACHE[cur_edges] = patterns

    return _EDGE_MASK_PATTERN_CACHE[edge_count]


def pattern_edges_connected(pattern_edges: Sequence[Tuple[int, int]]) -> bool:
    edges = sorted(edge_set(pattern_edges))
    if not edges:
        return False
    nodes = {int(node) for edge in edges for node in edge}
    adj: Dict[int, List[int]] = {node: [] for node in nodes}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    start = min(nodes)
    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for nbr in adj.get(int(node), []):
            if int(nbr) in seen:
                continue
            seen.add(int(nbr))
            queue.append(int(nbr))
    return seen == nodes


def abstract_pattern_edges_from_edges(edges: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    selected = sorted(edge_set(edges))
    nodes = sorted({int(node) for edge in selected for node in edge})
    mapping = {node: idx for idx, node in enumerate(nodes)}
    return tuple(sorted((mapping[int(a)], mapping[int(b)]) for a, b in selected))


def pattern_certificate(pattern_edges: Sequence[Tuple[int, int]]) -> Tuple[int, bytes]:
    pattern = abstract_pattern_edges_from_edges(pattern_edges)
    node_count = 0
    for a, b in pattern:
        node_count = max(node_count, int(a) + 1, int(b) + 1)
    adj = tuple(0 for _ in range(node_count))
    for a, b in pattern:
        adj = _add_pattern_edge_to_adjacency(adj, int(a), int(b))
    return _pynauty_certificate_for_adjacency(adj)


def rooted_connected_topology_patterns(valid_sizes: Sequence[int]) -> List[nx.Graph]:
    """Return one representative per rooted connected topology for small K."""
    valid_size_set = {int(size) for size in valid_sizes if int(size) > 0}
    if not valid_size_set:
        return []
    cache_key = tuple(sorted(valid_size_set))
    cached = _ROOTED_TOPOLOGY_PATTERN_CACHE.get(cache_key)
    if cached is not None:
        return [pattern.copy() for pattern in cached]
    max_edges = max(valid_size_set)
    # graph_atlas_g covers all simple graphs up to 7 nodes, enough for the
    # current Main GT-size node datasets where K <= 6.
    if max_edges > 6:
        return []

    patterns: List[nx.Graph] = []
    seen: set[str] = set()
    for atlas_graph in nx.graph_atlas_g():
        if atlas_graph.number_of_edges() not in valid_size_set:
            continue
        if atlas_graph.number_of_edges() <= 0 or not nx.is_connected(atlas_graph):
            continue
        base = nx.convert_node_labels_to_integers(atlas_graph, ordering="sorted")
        for root in sorted(base.nodes):
            pattern = nx.Graph()
            pattern.add_nodes_from(sorted(base.nodes))
            pattern.add_edges_from(sorted(normalize_edge(edge) for edge in base.edges if normalize_edge(edge) is not None))
            for node in pattern.nodes:
                pattern.nodes[node]["root"] = 1 if int(node) == int(root) else 0
                pattern.nodes[node]["feature"] = ""
            signature = rooted_pattern_signature(pattern)
            if signature in seen:
                continue
            seen.add(signature)
            patterns.append(pattern)
    patterns.sort(
        key=lambda graph: (
            graph.number_of_edges(),
            graph.number_of_nodes(),
            rooted_pattern_signature(graph),
        )
    )
    _ROOTED_TOPOLOGY_PATTERN_CACHE[cache_key] = [pattern.copy() for pattern in patterns]
    return patterns


def connected_topology_patterns(valid_sizes: Sequence[int]) -> List[nx.Graph]:
    """Return one representative per unrooted connected topology for small K."""
    valid_size_set = {int(size) for size in valid_sizes if int(size) > 0}
    if not valid_size_set:
        return []
    max_edges = max(valid_size_set)
    if max_edges > 6:
        return []

    patterns: List[nx.Graph] = []
    seen: set[str] = set()
    for atlas_graph in nx.graph_atlas_g():
        if atlas_graph.number_of_edges() not in valid_size_set:
            continue
        if atlas_graph.number_of_edges() <= 0 or not nx.is_connected(atlas_graph):
            continue
        pattern = nx.convert_node_labels_to_integers(atlas_graph, ordering="sorted")
        for node in pattern.nodes:
            pattern.nodes[node]["root"] = 0
            pattern.nodes[node]["feature"] = ""
        signature = rooted_pattern_signature(pattern)
        if signature in seen:
            continue
        seen.add(signature)
        patterns.append(pattern)
    patterns.sort(
        key=lambda graph: (
            graph.number_of_edges(),
            graph.number_of_nodes(),
            rooted_pattern_signature(graph),
        )
    )
    return patterns


def rooted_pattern_root(pattern: nx.Graph) -> Optional[int]:
    roots = [int(node) for node, data in pattern.nodes(data=True) if int(data.get("root", 0)) == 1]
    if len(roots) != 1:
        return None
    return roots[0]


def rooted_pattern_order(pattern: nx.Graph, pattern_root: int) -> List[int]:
    distances = nx.single_source_shortest_path_length(pattern, int(pattern_root))
    return sorted(
        (int(node) for node in pattern.nodes),
        key=lambda node: (distances.get(int(node), 10**9), -int(pattern.degree[int(node)]), int(node)),
    )


def find_rooted_monomorphism_edges(
    *,
    candidate_graph: nx.Graph,
    pattern: nx.Graph,
    root_node: int,
    pattern_root: int,
    match_features: bool,
    excluded_edge_sets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Optional[Tuple[Tuple[int, int], ...]]:
    candidate_adj = {int(node): set(int(nbr) for nbr in candidate_graph.neighbors(node)) for node in candidate_graph.nodes}
    pattern_adj = {int(node): set(int(nbr) for nbr in pattern.neighbors(node)) for node in pattern.nodes}
    order = rooted_pattern_order(pattern, int(pattern_root))
    if not order or order[0] != int(pattern_root):
        return None
    excluded = {
        frozenset(edge_set(edges))
        for edges in (excluded_edge_sets or [])
    }

    def attrs_match(candidate_node: int, pattern_node: int) -> bool:
        candidate_attrs = candidate_graph.nodes[int(candidate_node)]
        pattern_attrs = pattern.nodes[int(pattern_node)]
        if int(candidate_attrs.get("root", 0)) != int(pattern_attrs.get("root", 0)):
            return False
        if not match_features:
            return True
        return str(candidate_attrs.get("feature", "")) == str(pattern_attrs.get("feature", ""))

    if int(root_node) not in candidate_graph or not attrs_match(int(root_node), int(pattern_root)):
        return None

    mapping: Dict[int, int] = {int(pattern_root): int(root_node)}
    used: set[int] = {int(root_node)}

    def backtrack(position: int) -> Optional[Dict[int, int]]:
        if position >= len(order):
            selected = frozenset(
                edge
                for edge in (
                    normalize_edge((mapping[int(a)], mapping[int(b)]))
                    for a, b in pattern.edges
                )
                if edge is not None
            )
            if selected in excluded:
                return None
            return dict(mapping)

        pattern_node = int(order[position])
        mapped_neighbors = [nbr for nbr in pattern_adj[pattern_node] if nbr in mapping]
        if not mapped_neighbors:
            candidates = set(int(node) for node in candidate_graph.nodes)
        else:
            candidates = set(candidate_adj[mapping[mapped_neighbors[0]]])
            for nbr in mapped_neighbors[1:]:
                candidates.intersection_update(candidate_adj[mapping[nbr]])

        for candidate_node in sorted(candidates):
            if candidate_node in used:
                continue
            if int(candidate_graph.degree[candidate_node]) < int(pattern.degree[pattern_node]):
                continue
            if not attrs_match(candidate_node, pattern_node):
                continue
            mapping[pattern_node] = candidate_node
            used.add(candidate_node)
            result = backtrack(position + 1)
            if result is not None:
                return result
            used.remove(candidate_node)
            del mapping[pattern_node]
        return None

    result_mapping = backtrack(1)
    if result_mapping is None:
        return None

    selected = sorted(
        edge
        for edge in (
            normalize_edge((result_mapping[int(a)], result_mapping[int(b)]))
            for a, b in pattern.edges
        )
        if edge is not None
    )
    return tuple(selected)


def unrooted_pattern_order(pattern: nx.Graph) -> List[int]:
    if pattern.number_of_nodes() == 0:
        return []
    start = min((int(node) for node in pattern.nodes), key=lambda node: (-int(pattern.degree[node]), node))
    distances = nx.single_source_shortest_path_length(pattern, start)
    return sorted(
        (int(node) for node in pattern.nodes),
        key=lambda node: (distances.get(int(node), 10**9), -int(pattern.degree[int(node)]), int(node)),
    )


def find_unrooted_monomorphism_edges(
    *,
    candidate_graph: nx.Graph,
    pattern: nx.Graph,
    match_features: bool,
    excluded_edge_sets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Optional[Tuple[Tuple[int, int], ...]]:
    candidate_adj = {int(node): set(int(nbr) for nbr in candidate_graph.neighbors(node)) for node in candidate_graph.nodes}
    pattern_adj = {int(node): set(int(nbr) for nbr in pattern.neighbors(node)) for node in pattern.nodes}
    order = unrooted_pattern_order(pattern)
    if not order:
        return None
    excluded = {
        frozenset(edge_set(edges))
        for edges in (excluded_edge_sets or [])
    }

    def attrs_match(candidate_node: int, pattern_node: int) -> bool:
        if not match_features:
            return True
        candidate_attrs = candidate_graph.nodes[int(candidate_node)]
        pattern_attrs = pattern.nodes[int(pattern_node)]
        return str(candidate_attrs.get("feature", "")) == str(pattern_attrs.get("feature", ""))

    mapping: Dict[int, int] = {}
    used: set[int] = set()

    def backtrack(position: int) -> Optional[Tuple[Tuple[int, int], ...]]:
        if position >= len(order):
            selected = tuple(
                sorted(
                    edge
                    for edge in (
                        normalize_edge((mapping[int(a)], mapping[int(b)]))
                        for a, b in pattern.edges
                    )
                    if edge is not None
                )
            )
            if frozenset(selected) in excluded:
                return None
            return selected

        pattern_node = int(order[position])
        mapped_neighbors = [nbr for nbr in pattern_adj[pattern_node] if nbr in mapping]
        if not mapped_neighbors:
            candidates = set(int(node) for node in candidate_graph.nodes)
        else:
            candidates = set(candidate_adj[mapping[mapped_neighbors[0]]])
            for nbr in mapped_neighbors[1:]:
                candidates.intersection_update(candidate_adj[mapping[nbr]])

        for candidate_node in sorted(candidates):
            if candidate_node in used:
                continue
            if int(candidate_graph.degree[candidate_node]) < int(pattern.degree[pattern_node]):
                continue
            if not attrs_match(candidate_node, pattern_node):
                continue
            mapping[pattern_node] = candidate_node
            used.add(candidate_node)
            result = backtrack(position + 1)
            if result is not None:
                return result
            used.remove(candidate_node)
            del mapping[pattern_node]
        return None

    return backtrack(0)


def node_shape_first_representatives(
    *,
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[str],
    root_node: int,
    valid_sizes: Sequence[int],
    max_subgraphs: int,
    include_empty: bool = False,
    excluded_edge_sets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Tuple[Optional[str], int, Dict[str, Tuple[Tuple[int, int], ...]]]:
    candidate_graph = candidate_nx_graph(
        x=x,
        candidate_edges=candidate_edges,
        root_node=int(root_node),
        node_colors=node_colors,
    )
    if int(root_node) not in candidate_graph:
        return None, 0, {}

    feature_labels = sorted({str(data["feature"]) for _, data in candidate_graph.nodes(data=True)})
    root_feature = str(candidate_graph.nodes[int(root_node)]["feature"])
    patterns = rooted_connected_topology_patterns(valid_sizes)
    if not patterns:
        return "fallback_embedding", 0, {}

    enumerated_patterns = 0
    representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    if include_empty:
        enumerated_patterns = 1
        if enumerated_patterns > int(max_subgraphs):
            return "skipped_enumeration_too_large", enumerated_patterns, representatives
        empty_signature = exact_rooted_colored_edge_subset_signature(
            selected_edges=(),
            node_colors=node_colors,
            root_node=int(root_node),
        )
        representatives[empty_signature] = tuple()

    for topology in patterns:
        pattern_root = rooted_pattern_root(topology)
        if pattern_root is None:
            continue
        nonroot_nodes = [int(node) for node in sorted(topology.nodes) if int(node) != pattern_root]
        feature_assignments = itertools.product(feature_labels, repeat=len(nonroot_nodes))
        for assigned_features in feature_assignments:
            pattern = topology.copy()
            pattern.nodes[pattern_root]["feature"] = root_feature
            for node, feature in zip(nonroot_nodes, assigned_features):
                pattern.nodes[int(node)]["feature"] = str(feature)

            enumerated_patterns += 1
            if enumerated_patterns > int(max_subgraphs):
                return "skipped_enumeration_too_large", enumerated_patterns, representatives

            pattern_signature = rooted_pattern_signature(pattern)
            if pattern_signature in representatives:
                continue

            selected = find_rooted_monomorphism_edges(
                candidate_graph=candidate_graph,
                pattern=pattern,
                root_node=int(root_node),
                pattern_root=int(pattern_root),
                match_features=True,
                excluded_edge_sets=excluded_edge_sets,
            )
            if selected is None:
                continue
            actual_signature = exact_rooted_colored_edge_subset_signature(
                selected_edges=selected,
                node_colors=node_colors,
                root_node=int(root_node),
            )
            if actual_signature != pattern_signature:
                raise RuntimeError(
                    "Rooted colored monomorphism produced a non-isomorphic representative."
                )
            representatives.setdefault(actual_signature, tuple(selected))

    return None, enumerated_patterns, representatives


def candidate_feature_labels(
    *,
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    root_node: Optional[int],
) -> List[str]:
    graph = candidate_nx_graph(x=x, candidate_edges=candidate_edges, root_node=root_node)
    return sorted({str(data["feature"]) for _, data in graph.nodes(data=True)})


def feature_pair_key(left: str, right: str) -> Tuple[str, str]:
    first, second = sorted((str(left), str(right)))
    return first, second


def candidate_feature_profile(
    candidate_graph: nx.Graph,
) -> Tuple[List[str], Counter, set[Tuple[str, str]], Dict[str, List[Tuple[int, Counter]]]]:
    feature_counts = Counter(str(data["feature"]) for _, data in candidate_graph.nodes(data=True))
    feature_labels = sorted(feature_counts, key=lambda label: (-feature_counts[label], label))
    edge_feature_pairs = {
        feature_pair_key(
            candidate_graph.nodes[int(a)]["feature"],
            candidate_graph.nodes[int(b)]["feature"],
        )
        for a, b in candidate_graph.edges
    }
    neighbor_profiles: Dict[str, List[Tuple[int, Counter]]] = {}
    for node, data in candidate_graph.nodes(data=True):
        feature = str(data["feature"])
        neighbor_features = Counter(
            str(candidate_graph.nodes[int(nbr)]["feature"]) for nbr in candidate_graph.neighbors(node)
        )
        neighbor_profiles.setdefault(feature, []).append((int(candidate_graph.degree[node]), neighbor_features))
    return feature_labels, feature_counts, edge_feature_pairs, neighbor_profiles


def feasible_feature_assignments(
    *,
    pattern: nx.Graph,
    pattern_nodes: Sequence[int],
    feature_labels: Sequence[str],
    feature_counts: Counter,
    edge_feature_pairs: set[Tuple[str, str]],
    neighbor_profiles: Dict[str, List[Tuple[int, Counter]]],
) -> Iterable[Tuple[str, ...]]:
    pattern_adj = {int(node): set(int(nbr) for nbr in pattern.neighbors(node)) for node in pattern.nodes}
    order = unrooted_pattern_order(pattern)
    assigned: Dict[int, str] = {}
    used_counts: Counter = Counter()

    def backtrack(position: int) -> Iterable[Tuple[str, ...]]:
        if position >= len(order):
            yield tuple(str(assigned[int(node)]) for node in pattern_nodes)
            return

        pattern_node = int(order[position])
        assigned_neighbors = [nbr for nbr in pattern_adj[pattern_node] if nbr in assigned]
        for feature in feature_labels:
            feature = str(feature)
            if used_counts[feature] >= feature_counts[feature]:
                continue
            if any(feature_pair_key(feature, assigned[nbr]) not in edge_feature_pairs for nbr in assigned_neighbors):
                continue

            assigned[pattern_node] = feature
            used_counts[feature] += 1
            locally_feasible = True
            for node, node_feature in assigned.items():
                required_neighbor_features = Counter(
                    assigned[nbr] for nbr in pattern_adj[int(node)] if nbr in assigned
                )
                required_degree = int(pattern.degree[int(node)])
                if not any(
                    degree >= required_degree
                    and all(neighbor_counts[color] >= count for color, count in required_neighbor_features.items())
                    for degree, neighbor_counts in neighbor_profiles.get(str(node_feature), [])
                ):
                    locally_feasible = False
                    break
            if locally_feasible:
                yield from backtrack(position + 1)
            used_counts[feature] -= 1
            if used_counts[feature] <= 0:
                del used_counts[feature]
            del assigned[pattern_node]

    yield from backtrack(0)


def graph_shape_first_representatives(
    *,
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[str],
    valid_sizes: Sequence[int],
    max_subgraphs: int,
    excluded_edge_sets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Tuple[Optional[str], int, Dict[str, Tuple[Tuple[int, int], ...]]]:
    candidate_graph = candidate_nx_graph(
        x=x,
        candidate_edges=candidate_edges,
        root_node=None,
        node_colors=node_colors,
    )
    if candidate_graph.number_of_edges() <= 0:
        return None, 0, {}

    feature_labels, feature_counts, edge_feature_pairs, neighbor_profiles = candidate_feature_profile(candidate_graph)
    # Always carry the exact color identity. Even a single-color instance must
    # produce the same certificate as its concrete GT mask.
    match_colors = True
    patterns = connected_topology_patterns(valid_sizes)
    if not patterns:
        return "fallback_embedding", 0, {}

    enumerated_patterns = 0
    representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}

    for topology in patterns:
        pattern_nodes = [int(node) for node in sorted(topology.nodes)]
        feature_assignments = feasible_feature_assignments(
            pattern=topology,
            pattern_nodes=pattern_nodes,
            feature_labels=feature_labels,
            feature_counts=feature_counts,
            edge_feature_pairs=edge_feature_pairs,
            neighbor_profiles=neighbor_profiles,
        )
        for assigned_features in feature_assignments:
            pattern = topology.copy()
            for node, feature in zip(pattern_nodes, assigned_features):
                pattern.nodes[int(node)]["feature"] = str(feature)

            enumerated_patterns += 1
            if enumerated_patterns > int(max_subgraphs):
                return "skipped_enumeration_too_large", enumerated_patterns, representatives

            pattern_signature = exact_colored_graph_signature(pattern)
            if pattern_signature in representatives:
                continue

            selected = find_unrooted_monomorphism_edges(
                candidate_graph=candidate_graph,
                pattern=pattern,
                match_features=match_colors,
                excluded_edge_sets=excluded_edge_sets,
            )
            if selected is None:
                continue
            representatives.setdefault(pattern_signature, tuple(selected))

    return None, enumerated_patterns, representatives


def graph_edge_mask_shape_first_representatives(
    *,
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[str],
    valid_sizes: Sequence[int],
    max_subgraphs: int,
    connected_only: bool,
    excluded_edge_sets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Tuple[Optional[str], int, Dict[str, Tuple[Tuple[int, int], ...]]]:
    candidate_graph = candidate_nx_graph(
        x=x,
        candidate_edges=candidate_edges,
        root_node=None,
        node_colors=node_colors,
    )
    if candidate_graph.number_of_edges() <= 0:
        return None, 0, {}

    feature_labels, feature_counts, edge_feature_pairs, neighbor_profiles = candidate_feature_profile(candidate_graph)
    # Always carry the exact color identity. Even a single-color instance must
    # produce the same certificate as its concrete GT mask.
    match_colors = True
    patterns: List[nx.Graph] = []
    try:
        for size in sorted({int(size) for size in valid_sizes if int(size) > 0}):
            for pattern_edges in edge_mask_topology_patterns(size):
                if connected_only and not pattern_edges_connected(pattern_edges):
                    continue
                pattern = graph_from_pattern_edges(pattern_edges)
                for node in pattern.nodes:
                    pattern.nodes[int(node)]["root"] = 0
                    pattern.nodes[int(node)]["feature"] = ""
                patterns.append(pattern)
    except RuntimeError:
        if connected_only:
            return "fallback_connected_topology", 0, {}
        return "skipped_enumeration_too_large", int(max_subgraphs) + 1, {}

    enumerated_patterns = 0
    representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}

    for topology in patterns:
        pattern_nodes = [int(node) for node in sorted(topology.nodes)]
        feature_assignments = feasible_feature_assignments(
            pattern=topology,
            pattern_nodes=pattern_nodes,
            feature_labels=feature_labels,
            feature_counts=feature_counts,
            edge_feature_pairs=edge_feature_pairs,
            neighbor_profiles=neighbor_profiles,
        )
        for assigned_features in feature_assignments:
            pattern = topology.copy()
            for node, feature in zip(pattern_nodes, assigned_features):
                pattern.nodes[int(node)]["feature"] = str(feature)

            enumerated_patterns += 1
            if enumerated_patterns > int(max_subgraphs):
                return "skipped_enumeration_too_large", enumerated_patterns, representatives

            pattern_signature = exact_colored_graph_signature(pattern)
            if pattern_signature in representatives:
                continue

            selected = find_unrooted_monomorphism_edges(
                candidate_graph=candidate_graph,
                pattern=pattern,
                match_features=match_colors,
                excluded_edge_sets=excluded_edge_sets,
            )
            if selected is None:
                continue
            representatives.setdefault(pattern_signature, tuple(selected))

    return None, enumerated_patterns, representatives


def connected_edge_subsets(
    candidate_edges: Sequence[Tuple[int, int]],
    *,
    valid_sizes: Sequence[int],
    root_node: Optional[int],
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    edges = sorted(edge_set(candidate_edges))
    valid_size_set = {int(size) for size in valid_sizes if int(size) > 0}
    if not edges or not valid_size_set:
        return

    max_size = max(valid_size_set)
    edge_nodes = [(int(a), int(b)) for a, b in edges]
    incident: Dict[int, List[int]] = {}
    for idx, (a, b) in enumerate(edge_nodes):
        incident.setdefault(a, []).append(idx)
        incident.setdefault(b, []).append(idx)

    seen: set[Tuple[Tuple[int, int], ...]] = set()

    def expand(
        selected_indices: frozenset[int],
        selected_nodes: frozenset[int],
        min_edge_idx: int,
    ) -> Iterable[Tuple[Tuple[int, int], ...]]:
        if len(selected_indices) in valid_size_set:
            subset = tuple(edges[idx] for idx in sorted(selected_indices))
            if subset not in seen:
                seen.add(subset)
                yield subset
        if len(selected_indices) >= max_size:
            return

        frontier: set[int] = set()
        for node in selected_nodes:
            frontier.update(incident.get(int(node), []))
        frontier.difference_update(selected_indices)

        for idx in sorted(frontier):
            if root_node is None and idx < min_edge_idx:
                continue
            a, b = edge_nodes[idx]
            next_indices = frozenset(set(selected_indices) | {idx})
            next_nodes = frozenset(set(selected_nodes) | {a, b})
            yield from expand(next_indices, next_nodes, min_edge_idx)

    if root_node is not None:
        for idx in sorted(incident.get(int(root_node), [])):
            a, b = edge_nodes[idx]
            yield from expand(frozenset({idx}), frozenset({a, b}), 0)
    else:
        for idx, (a, b) in enumerate(edge_nodes):
            yield from expand(frozenset({idx}), frozenset({a, b}), idx)


def arbitrary_edge_subsets(
    candidate_edges: Sequence[Tuple[int, int]],
    *,
    valid_sizes: Sequence[int],
) -> Iterable[Tuple[Tuple[int, int], ...]]:
    edges = sorted(edge_set(candidate_edges))
    valid_size_set = sorted({int(size) for size in valid_sizes if int(size) > 0})
    if not edges or not valid_size_set:
        return
    for size in valid_size_set:
        if int(size) > len(edges):
            continue
        for subset in itertools.combinations(edges, int(size)):
            yield tuple(subset)


def root_component_edges(
    *,
    root_node: int,
    selected_edges: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, int], ...]:
    """Return only the selected component that can influence the target node."""
    selected = sorted(edge_set(selected_edges))
    incident: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {}
    for edge in selected:
        a, b = edge
        incident.setdefault(int(a), []).append((int(b), edge))
        incident.setdefault(int(b), []).append((int(a), edge))
    if int(root_node) not in incident:
        return tuple()

    seen_nodes = {int(root_node)}
    component: set[Tuple[int, int]] = set()
    queue: deque[int] = deque([int(root_node)])
    while queue:
        node = queue.popleft()
        for neighbor, edge in incident.get(int(node), []):
            component.add(edge)
            if int(neighbor) not in seen_nodes:
                seen_nodes.add(int(neighbor))
                queue.append(int(neighbor))
    return tuple(sorted(component))


def node_sufficiency_root_component_signature(
    *,
    node_colors: Sequence[str],
    selected_edges: Sequence[Tuple[int, int]],
    root_node: int,
) -> str:
    component = root_component_edges(root_node=int(root_node), selected_edges=selected_edges)
    return exact_rooted_colored_edge_subset_signature(
        selected_edges=component,
        node_colors=node_colors,
        root_node=int(root_node),
    )


def compiled_cpp_enumerator_binary(*, source: Path, binary_stem: str) -> Path:
    """Build an immutable source-hashed C++ enumeration binary on demand."""
    if not source.is_file():
        raise RuntimeError(f"Missing C++ enumerator source: {source}")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required for standard enumeration.")

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    build_dir = Path("/tmp") / f"26gnnexplainer-cpp-{os.getuid()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = build_dir / f"{binary_stem}-{source_digest}"
    if binary_path.is_file() and os.access(binary_path, os.X_OK):
        return binary_path

    temporary_binary = build_dir / f"{binary_path.name}.tmp-{os.getpid()}"
    completed = subprocess.run(
        [
            compiler,
            "-O3",
            "-DNDEBUG",
            "-std=c++17",
            str(source),
            "-o",
            str(temporary_binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        temporary_binary.unlink(missing_ok=True)
        details = (completed.stderr or completed.stdout or "unknown compiler error").strip()
        raise RuntimeError(f"Failed to build C++ enumerator {binary_stem}: {details}")
    os.replace(temporary_binary, binary_path)
    return binary_path


def rooted_component_cpp_binary() -> Path:
    return compiled_cpp_enumerator_binary(
        source=ROOTED_COMPONENT_CPP_SOURCE,
        binary_stem="rooted_component_enumerator",
    )


def concrete_subset_cpp_binary() -> Path:
    return compiled_cpp_enumerator_binary(
        source=CONCRETE_SUBSET_CPP_SOURCE,
        binary_stem="concrete_edge_subset_enumerator",
    )


def visit_concrete_edge_subsets_cpp(
    *,
    candidate_edges: Sequence[Tuple[int, int]],
    mode: str,
    valid_sizes: Sequence[int],
    root_node: Optional[int],
    max_subgraphs: int,
    visitor: Callable[[Tuple[Tuple[int, int], ...]], None],
) -> Tuple[Optional[str], int]:
    """Visit every instance-feasible concrete edge mask emitted by C++."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"all", "rooted", "connected"}:
        raise ValueError(f"Unsupported concrete enumeration mode: {mode!r}")
    sizes = sorted({int(size) for size in valid_sizes if int(size) > 0})
    if not sizes:
        return None, 0
    if normalized_mode == "rooted" and root_node is None:
        raise ValueError("root_node is required for rooted concrete enumeration.")

    edges = sorted(edge_set(candidate_edges))
    root_value = int(root_node) if root_node is not None else -1
    header = (
        f"{CONCRETE_SUBSET_CPP_PROTOCOL} {normalized_mode} {len(edges)} {root_value} "
        f"{len(sizes)} {max(0, int(max_subgraphs))}\n"
    )
    input_text = header + " ".join(str(size) for size in sizes) + "\n"
    input_text += "".join(f"{int(a)} {int(b)}\n" for a, b in edges)
    process = subprocess.Popen(
        [str(concrete_subset_cpp_binary())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Failed to open pipes for the C++ concrete subset enumerator.")
    process.stdin.write(input_text)
    process.stdin.close()

    terminal_kind: Optional[str] = None
    terminal_count: Optional[int] = None
    visited_records = 0
    try:
        for line_number, raw_line in enumerate(process.stdout, start=1):
            parts = raw_line.split()
            if not parts:
                continue
            if parts[0] == "C":
                if terminal_kind is not None or len(parts) < 2:
                    raise RuntimeError(f"Invalid C++ concrete record at line {line_number}.")
                subset_size = int(parts[1])
                indices = [int(value) for value in parts[2:]]
                if subset_size != len(indices) or subset_size not in sizes:
                    raise RuntimeError(f"Invalid C++ concrete subset size at line {line_number}.")
                if indices != sorted(set(indices)) or any(index < 0 or index >= len(edges) for index in indices):
                    raise RuntimeError(f"Invalid C++ concrete edge indices at line {line_number}.")
                visitor(tuple(edges[index] for index in indices))
                visited_records += 1
            elif parts[0] in {"END", "LIMIT"} and len(parts) == 2:
                if terminal_kind is not None:
                    raise RuntimeError("C++ concrete enumerator returned multiple terminal records.")
                terminal_kind = parts[0]
                terminal_count = int(parts[1])
            else:
                raise RuntimeError(f"Invalid C++ concrete output at line {line_number}: {raw_line.rstrip()}")
    except BaseException:
        process.kill()
        process.wait()
        raise

    stderr_text = process.stderr.read().strip()
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"C++ concrete subset enumerator failed: {stderr_text or 'unknown runtime error'}")
    if terminal_kind is None or terminal_count is None:
        raise RuntimeError("C++ concrete subset enumerator returned no terminal record.")
    expected_records = terminal_count if terminal_kind == "END" else terminal_count - 1
    if visited_records != expected_records:
        raise RuntimeError(
            "C++ concrete subset count mismatch: "
            f"records={visited_records}, terminal={terminal_kind} {terminal_count}."
        )
    status = "skipped_enumeration_too_large" if terminal_kind == "LIMIT" else None
    return status, int(terminal_count)


def visit_rooted_components_cpp(
    *,
    candidate_edges: Sequence[Tuple[int, int]],
    root_node: int,
    max_size: int,
    max_subgraphs: int,
    visitor: Callable[[Tuple[Tuple[int, int], ...]], None],
) -> Tuple[Optional[str], int]:
    """Enumerate rooted connected edge sets in C++ and visit them in DFS order."""
    edges = sorted(edge_set(candidate_edges))
    header = (
        f"{ROOTED_COMPONENT_CPP_PROTOCOL} {len(edges)} {int(root_node)} "
        f"{int(max_size)} {max(0, int(max_subgraphs))}\n"
    )
    input_text = header + "".join(f"{int(a)} {int(b)}\n" for a, b in edges)
    process = subprocess.Popen(
        [str(rooted_component_cpp_binary())],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Failed to open pipes for the C++ rooted-component enumerator.")
    process.stdin.write(input_text)
    process.stdin.close()
    terminal_kind: Optional[str] = None
    terminal_count: Optional[int] = None
    visited_records = 0
    try:
        for line_number, raw_line in enumerate(process.stdout, start=1):
            parts = raw_line.split()
            if not parts:
                continue
            if parts[0] == "C":
                if terminal_kind is not None or len(parts) < 2:
                    raise RuntimeError(f"Invalid C++ component record at line {line_number}.")
                component_size = int(parts[1])
                indices = [int(value) for value in parts[2:]]
                if component_size != len(indices):
                    raise RuntimeError(f"Invalid C++ component size at line {line_number}.")
                if indices != sorted(set(indices)) or any(index < 0 or index >= len(edges) for index in indices):
                    raise RuntimeError(f"Invalid C++ edge indices at line {line_number}.")
                visitor(tuple(edges[index] for index in indices))
                visited_records += 1
            elif parts[0] in {"END", "LIMIT"} and len(parts) == 2:
                if terminal_kind is not None:
                    raise RuntimeError("C++ rooted-component enumerator returned multiple terminal records.")
                terminal_kind = parts[0]
                terminal_count = int(parts[1])
            else:
                raise RuntimeError(f"Invalid C++ enumerator output at line {line_number}: {raw_line.rstrip()}")
    except BaseException:
        process.kill()
        process.wait()
        raise

    stderr_text = process.stderr.read().strip()
    returncode = process.wait()
    if returncode != 0:
        details = stderr_text or "unknown runtime error"
        raise RuntimeError(f"C++ rooted-component enumerator failed: {details}")

    if terminal_kind is None or terminal_count is None:
        raise RuntimeError("C++ rooted-component enumerator returned no terminal record.")
    expected_records = terminal_count if terminal_kind == "END" else terminal_count - 1
    if visited_records != expected_records:
        raise RuntimeError(
            "C++ rooted-component count mismatch: "
            f"records={visited_records}, terminal={terminal_kind} {terminal_count}."
        )
    status = "skipped_enumeration_too_large" if terminal_kind == "LIMIT" else None
    return status, int(terminal_count)


def unrestricted_node_edge_representatives(
    *,
    node_colors: Sequence[str],
    candidate_edges: Sequence[Tuple[int, int]],
    root_node: int,
    valid_sizes: Sequence[int],
    max_subgraphs: int,
    excluded_edge_sets: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
) -> Tuple[Optional[str], int, Dict[str, Tuple[Tuple[int, int], ...]]]:
    """Enumerate target-component shapes and attach irrelevant exact-K filler.

    This is valid only for standard node sufficiency with connectivity=none:
    selected components disconnected from ``root_node`` cannot affect its GNN
    output, so their topology is irrelevant once exact-budget feasibility is
    established.
    """
    enumerated_subgraphs = 0
    representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    excluded = {
        frozenset(edge_set(edges))
        for edges in (excluded_edge_sets or [])
    }
    candidate_edge_set = edge_set(candidate_edges)
    target_sizes = sorted({int(size) for size in valid_sizes if int(size) > 0})
    if not candidate_edge_set or not target_sizes:
        return None, 0, representatives

    def full_mask_for_component(
        component: Sequence[Tuple[int, int]],
        *,
        target_size: int,
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        core = tuple(sorted(edge_set(component)))
        if len(core) > int(target_size):
            return None
        core_nodes = {int(node) for edge in core for node in edge}
        core_nodes.add(int(root_node))
        filler_count = int(target_size) - len(core)
        filler_candidates = sorted(
            edge
            for edge in candidate_edge_set - edge_set(core)
            if int(edge[0]) not in core_nodes and int(edge[1]) not in core_nodes
        )
        if filler_count > len(filler_candidates):
            return None
        for filler in itertools.combinations(filler_candidates, filler_count):
            full_mask = tuple(sorted((*core, *filler)))
            if frozenset(full_mask) not in excluded:
                return full_mask
        return None

    def add_component(component: Tuple[Tuple[int, int], ...]) -> None:
        signature = node_sufficiency_root_component_signature(
            node_colors=node_colors,
            selected_edges=component,
            root_node=int(root_node),
        )
        if signature in representatives:
            return
        for target_size in target_sizes:
            full_mask = full_mask_for_component(component, target_size=int(target_size))
            if full_mask is not None:
                representatives[signature] = full_mask
                break

    enum_status, enumerated_subgraphs = visit_rooted_components_cpp(
        candidate_edges=candidate_edges,
        root_node=int(root_node),
        max_size=max(target_sizes),
        max_subgraphs=int(max_subgraphs),
        visitor=add_component,
    )
    return enum_status, enumerated_subgraphs, representatives


def canonical_edge_list(edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    return [[int(a), int(b)] for a, b in sorted(edge_set(edges))]


def edge_list_from_json(value: Any) -> Tuple[Tuple[int, int], ...]:
    if not isinstance(value, list):
        return tuple()
    edges: List[Tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            edge = normalize_edge((int(item[0]), int(item[1])))
        except (TypeError, ValueError):
            continue
        if edge is not None:
            edges.append(edge)
    return tuple(sorted(edge_set(edges)))


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def numpy_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(v) for v in array.shape)).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def graph_necessity_enumeration_cache_metadata(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    valid_sizes: Sequence[int],
) -> Dict[str, Any]:
    array = np.asarray(x)
    exclude_gt_solution = bool(row.get("exclude_gt_solution"))
    metadata: Dict[str, Any] = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "kind": "graph_necessity_full_combinations",
        "dataset": normalize_dataset_name(str(row.get("dataset", ""))),
        "task": str(row.get("task")),
        "instance_id": int(row.get("instance_id")),
        "budget_k": int(row.get("budget_k")),
        "valid_sizes": sorted({int(size) for size in valid_sizes if int(size) > 0}),
        "connectivity": str(row.get("connectivity")),
        "edge_budget_mode": str(row.get("edge_budget_mode")),
        "candidate_edges_sha256": sha256_json(canonical_edge_list(candidate_edges)),
        "candidate_edge_count": len(edge_set(candidate_edges)),
        "exclude_gt_solution": exclude_gt_solution,
        "excluded_gt_edges_sha256": (
            sha256_json(canonical_edge_list(gt_edges)) if exclude_gt_solution else None
        ),
        "x_shape": [int(v) for v in array.shape],
        "x_dtype": str(array.dtype),
        "x_sha256": numpy_array_sha256(array),
        "signature_method": "none",
        "representative_method": "concrete_edge_subset_enumerator_cpp_v1",
        "candidate_family": (
            "connected_concrete_removal_masks"
            if str(row.get("connectivity")) in {"cut", "flow"}
            else "all_concrete_removal_masks"
        ),
    }
    return metadata


def graph_necessity_enumeration_cache_path(
    *,
    cache_dir: Path,
    metadata: Dict[str, Any],
) -> Path:
    cache_key = sha256_json(metadata)[:24]
    dataset_slug = sanitize_run_slug(metadata.get("dataset", "dataset"))
    instance_id = int(metadata.get("instance_id"))
    budget_k = int(metadata.get("budget_k"))
    connectivity = sanitize_run_slug(metadata.get("connectivity", "none"))
    filename = f"instance-{instance_id:06d}_k{budget_k}_conn-{connectivity}_{cache_key}.json"
    return Path(cache_dir) / dataset_slug / "graph_necessity" / filename


def load_graph_necessity_enumeration_manifest(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not cache_path.exists():
        return "miss", None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None

    if payload.get("cache_version") != ENUMERATION_CACHE_VERSION:
        return "stale", None
    if payload.get("metadata") != metadata:
        return "stale", None
    if not isinstance(payload.get("enumerated_subgraphs"), int):
        return "invalid", None
    if not isinstance(payload.get("eligible_subgraphs"), int):
        return "invalid", None
    if not isinstance(payload.get("enumeration_time_sec"), (int, float)):
        return "invalid", None
    return "hit", payload


def save_graph_necessity_enumeration_manifest(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
    enumerated_subgraphs: int,
    eligible_subgraphs: int,
    enumeration_time_sec: float,
) -> bool:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "metadata": metadata,
        "enumerated_subgraphs": int(enumerated_subgraphs),
        "eligible_subgraphs": int(eligible_subgraphs),
        "enumerated_signatures": int(eligible_subgraphs),
        "wl_grouping": False,
        "enumeration_time_sec": float(enumeration_time_sec),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def append_enumeration_cache_timing(
    *,
    cache_dir: Path,
    dataset: str,
    task: str,
    cache_kind: str,
    instance_id: int,
    budget_k: int,
    candidate_edge_count: int,
    exclude_gt_solution: bool,
    cache_status: str,
    elapsed_sec: float,
) -> bool:
    """Append one cache-prewarm timing row without making cache use mandatory."""
    dataset_slug = sanitize_run_slug(normalize_dataset_name(str(dataset)))
    cache_kind_slug = sanitize_run_slug(str(cache_kind))
    timing_path = Path(cache_dir) / dataset_slug / cache_kind_slug / "enumeration_timings.csv"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "dataset": normalize_dataset_name(str(dataset)),
        "task": str(task),
        "cache_kind": str(cache_kind),
        "instance_id": int(instance_id),
        "budget_k": int(budget_k),
        "candidate_edge_count": int(candidate_edge_count),
        "exclude_gt_solution": bool(exclude_gt_solution),
        "cache_status": str(cache_status),
        "enumeration_performed": str(cache_status) != "hit",
        "prewarm_elapsed_sec": float(elapsed_sec),
    }
    try:
        write_header = not timing_path.exists() or timing_path.stat().st_size == 0
        with timing_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ENUMERATION_TIMING_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError:
        return False
    return True


def graph_sufficiency_pattern_cache_metadata(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    patterns: Sequence[Tuple[Tuple[int, int], ...]],
    fixed_self_loop_counts: Sequence[int],
) -> Dict[str, Any]:
    array = np.asarray(x)
    loop_counts = np.asarray(fixed_self_loop_counts, dtype=np.int64).reshape(-1)
    node_colors = graph_structural_node_colors(x=array, fixed_self_loop_counts=loop_counts)
    pattern_payload = [canonical_edge_list(pattern) for pattern in patterns]
    exclude_gt_solution = bool(row.get("exclude_gt_solution"))
    return {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "kind": "graph_sufficiency_colored_topology_representatives",
        "cache_encoding": "homogeneous_pattern_and_concrete_representative_v1",
        "dataset": normalize_dataset_name(str(row.get("dataset", ""))),
        "task": str(row.get("task")),
        "instance_id": int(row.get("instance_id")),
        "budget_k": int(row.get("budget_k")),
        "connectivity": str(row.get("connectivity")),
        "edge_budget_mode": str(row.get("edge_budget_mode")),
        "candidate_edges_sha256": sha256_json(canonical_edge_list(candidate_edges)),
        "candidate_edge_count": len(edge_set(candidate_edges)),
        "exclude_gt_solution": exclude_gt_solution,
        "excluded_gt_edges_sha256": (
            sha256_json(canonical_edge_list(gt_edges)) if exclude_gt_solution else None
        ),
        "x_shape": [int(v) for v in array.shape],
        "x_dtype": str(array.dtype),
        "x_sha256": numpy_array_sha256(array),
        "color_schema": ["node_feature", "fixed_self_loop_count"],
        "color_encoding": "numpy_row_bytes_hex_v1",
        "color_count": len(set(node_colors)),
        "fixed_self_loop_counts_sha256": numpy_array_sha256(loop_counts),
        "pattern_count": len(pattern_payload),
        "patterns_sha256": sha256_json(pattern_payload),
        "pattern_method": "edge_mask_topology_patterns_v1",
        "signature_method": "exact_colored_nauty_certificate_v1",
        "representative_method": "exact_colored_topology_monomorphism_v1",
    }


def graph_sufficiency_pattern_cache_path(
    *,
    cache_dir: Path,
    metadata: Dict[str, Any],
) -> Path:
    cache_key = sha256_json(metadata)[:24]
    dataset_slug = sanitize_run_slug(metadata.get("dataset", "dataset"))
    instance_id = int(metadata.get("instance_id"))
    budget_k = int(metadata.get("budget_k"))
    connectivity = sanitize_run_slug(metadata.get("connectivity", "none"))
    filename = f"instance-{instance_id:06d}_k{budget_k}_conn-{connectivity}_{cache_key}.json"
    return Path(cache_dir) / dataset_slug / "graph_sufficiency" / filename


def load_graph_sufficiency_pattern_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[str, List[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]]]:
    if not cache_path.exists():
        return "miss", []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", []

    if payload.get("cache_version") != ENUMERATION_CACHE_VERSION:
        return "stale", []
    if payload.get("metadata") != metadata:
        return "stale", []

    raw_representatives = payload.get("representatives")
    if not isinstance(raw_representatives, list):
        return "invalid", []

    representatives: List[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]] = []
    for item in raw_representatives:
        if not isinstance(item, dict):
            continue
        pattern_edges = edge_list_from_json(item.get("pattern_edges"))
        selected_edges = edge_list_from_json(item.get("edges"))
        if not pattern_edges or not selected_edges:
            continue
        representatives.append((pattern_edges, selected_edges))
    if not representatives:
        return "invalid", []
    return "hit", representatives


def save_graph_sufficiency_pattern_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
    enumerated_subgraphs: int,
    representatives: Sequence[Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]],
) -> bool:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "metadata": metadata,
        "enumerated_subgraphs": int(enumerated_subgraphs),
        "enumerated_signatures": int(len(representatives)),
        "representatives": [
            {
                "pattern_edges": canonical_edge_list(pattern_edges),
                "edges": canonical_edge_list(selected_edges),
            }
            for pattern_edges, selected_edges in representatives
        ],
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def graph_sufficiency_colored_cache_metadata(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    valid_sizes: Sequence[int],
    fixed_self_loop_counts: Sequence[int],
    connected_only: bool,
) -> Dict[str, Any]:
    array = np.asarray(x)
    loop_counts = np.asarray(fixed_self_loop_counts, dtype=np.int64).reshape(-1)
    node_colors = graph_structural_node_colors(x=array, fixed_self_loop_counts=loop_counts)
    exclude_gt_solution = bool(row.get("exclude_gt_solution"))
    return {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "kind": "graph_sufficiency_colored_topology_representatives",
        "cache_encoding": "signature_and_concrete_representative_v1",
        "dataset": normalize_dataset_name(str(row.get("dataset", ""))),
        "task": str(row.get("task")),
        "instance_id": int(row.get("instance_id")),
        "budget_k": int(row.get("budget_k")),
        "valid_sizes": sorted({int(size) for size in valid_sizes if int(size) > 0}),
        "connectivity": str(row.get("connectivity")),
        "edge_budget_mode": str(row.get("edge_budget_mode")),
        "candidate_edges_sha256": sha256_json(canonical_edge_list(candidate_edges)),
        "candidate_edge_count": len(edge_set(candidate_edges)),
        "exclude_gt_solution": exclude_gt_solution,
        "excluded_gt_edges_sha256": (
            sha256_json(canonical_edge_list(gt_edges)) if exclude_gt_solution else None
        ),
        "x_shape": [int(v) for v in array.shape],
        "x_dtype": str(array.dtype),
        "x_sha256": numpy_array_sha256(array),
        "color_schema": ["node_feature", "fixed_self_loop_count"],
        "color_encoding": "numpy_row_bytes_hex_v1",
        "color_count": len(set(node_colors)),
        "fixed_self_loop_counts_sha256": numpy_array_sha256(loop_counts),
        "connected_only": bool(connected_only),
        "signature_method": "exact_colored_nauty_certificate_v1",
        "representative_method": "exact_colored_topology_monomorphism_v1",
    }


def graph_sufficiency_colored_cache_path(
    *,
    cache_dir: Path,
    metadata: Dict[str, Any],
) -> Path:
    cache_key = sha256_json(metadata)[:24]
    dataset_slug = sanitize_run_slug(metadata.get("dataset", "dataset"))
    instance_id = int(metadata.get("instance_id"))
    budget_k = int(metadata.get("budget_k"))
    connectivity = sanitize_run_slug(metadata.get("connectivity", "none"))
    filename = f"instance-{instance_id:06d}_k{budget_k}_conn-{connectivity}_{cache_key}.json"
    return Path(cache_dir) / dataset_slug / "graph_sufficiency" / filename


def load_graph_sufficiency_colored_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[str, Optional[int], Dict[str, Tuple[Tuple[int, int], ...]]]:
    if not cache_path.exists():
        return "miss", None, {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None, {}

    if payload.get("cache_version") != ENUMERATION_CACHE_VERSION:
        return "stale", None, {}
    if payload.get("metadata") != metadata:
        return "stale", None, {}

    raw_representatives = payload.get("representatives")
    if not isinstance(raw_representatives, list):
        return "invalid", None, {}

    representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for item in raw_representatives:
        if not isinstance(item, dict):
            continue
        signature = item.get("signature")
        if not isinstance(signature, str):
            continue
        if "edges" not in item:
            continue
        edges = edge_list_from_json(item.get("edges"))
        representatives[signature] = edges
    if not representatives:
        return "invalid", None, {}
    return "hit", int(payload.get("enumerated_subgraphs", 0)), representatives


def save_graph_sufficiency_colored_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
    enumerated_subgraphs: int,
    representatives: Dict[str, Tuple[Tuple[int, int], ...]],
) -> bool:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "metadata": metadata,
        "enumerated_subgraphs": int(enumerated_subgraphs),
        "enumerated_signatures": int(len(representatives)),
        "representatives": [
            {
                "signature": signature,
                "edges": canonical_edge_list(edges),
            }
            for signature, edges in sorted(representatives.items())
        ],
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def node_enumeration_cache_metadata(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    valid_sizes: Sequence[int],
    match_features: bool,
    cache_kind: str,
    fixed_self_loop_counts: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    array = np.asarray(x)
    exclude_gt_solution = bool(row.get("exclude_gt_solution"))
    root_component_reduction = bool(
        cache_kind == "node_sufficiency"
        and str(row.get("connectivity")) == "none"
    )
    concrete_necessity = cache_kind == "node_necessity"
    exact_sufficiency = cache_kind == "node_sufficiency"
    if exact_sufficiency and fixed_self_loop_counts is None:
        raise ValueError(
            "Node Sufficiency cache metadata requires fixed self-loop counts."
        )
    metadata: Dict[str, Any] = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "kind": (
            "node_necessity_full_concrete_removal_masks"
            if concrete_necessity
            else "node_sufficiency_exact_rooted_colored_representatives"
        ),
        "dataset": normalize_dataset_name(str(row.get("dataset", ""))),
        "task": str(row.get("task")),
        "instance_id": int(row.get("instance_id")),
        "root_node": int(row.get("instance_id")),
        "budget_k": int(row.get("budget_k")),
        "valid_sizes": sorted({int(size) for size in valid_sizes if int(size) > 0}),
        "connectivity": str(row.get("connectivity")),
        "edge_budget_mode": str(row.get("edge_budget_mode")),
        "candidate_edges_sha256": sha256_json(canonical_edge_list(candidate_edges)),
        "candidate_edge_count": len(edge_set(candidate_edges)),
        "exclude_gt_solution": exclude_gt_solution,
        "excluded_gt_edges_sha256": (
            sha256_json(canonical_edge_list(gt_edges)) if exclude_gt_solution else None
        ),
        "x_shape": [int(v) for v in array.shape],
        "x_dtype": str(array.dtype),
        "x_sha256": numpy_array_sha256(array),
        "match_features": bool(match_features),
        "signature_method": (
            "exact_rooted_colored_nauty_certificate_v1"
            if exact_sufficiency
            else "none"
        ),
        "representative_method": (
            "node_sufficiency_root_component_with_exact_k_filler_cpp_v1"
            if root_component_reduction
            else (
                "concrete_edge_subset_enumerator_cpp_v1"
                if concrete_necessity
                else "exhaustive_exact_rooted_colored_topology_monomorphism_v2"
            )
        ),
        "node_feasible_family": (
            "target_rooted_connected"
            if str(row.get("connectivity")) == "cut"
            else "unrestricted_exact_budget"
        ),
    }
    if exact_sufficiency:
        loop_counts = np.asarray(fixed_self_loop_counts, dtype=np.int64).reshape(-1)
        node_colors = graph_structural_node_colors(
            x=array,
            fixed_self_loop_counts=loop_counts,
        )
        metadata.update(
            {
                "color_schema": [
                    "node_feature",
                    "fixed_self_loop_count",
                    "root_flag",
                ],
                "color_encoding": "numpy_row_bytes_hex_v1",
                "color_count": len(set(node_colors)),
                "root_distinguished": True,
                "fixed_self_loop_counts_sha256": numpy_array_sha256(loop_counts),
            }
        )
    if root_component_reduction:
        metadata["component_enumerator"] = "rooted_component_enumerator_cpp_v1"
    if concrete_necessity:
        metadata["component_enumerator"] = "concrete_edge_subset_enumerator_cpp_v1"
        metadata["candidate_family"] = (
            "rooted_concrete_removal_masks"
            if str(row.get("connectivity")) in {"cut", "flow"}
            else "all_concrete_removal_masks"
        )
    return metadata


def node_enumeration_cache_path(
    *,
    cache_dir: Path,
    metadata: Dict[str, Any],
    cache_kind: str,
) -> Path:
    cache_key = sha256_json(metadata)[:24]
    dataset_slug = sanitize_run_slug(metadata.get("dataset", "dataset"))
    instance_id = int(metadata.get("instance_id"))
    budget_k = int(metadata.get("budget_k"))
    edge_budget_mode = sanitize_run_slug(metadata.get("edge_budget_mode", "eq"))
    connectivity = sanitize_run_slug(metadata.get("connectivity", "none"))
    filename = f"node-{instance_id:06d}_k{budget_k}_mode-{edge_budget_mode}_conn-{connectivity}_{cache_key}.json"
    return Path(cache_dir) / dataset_slug / cache_kind / filename


def load_node_enumeration_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[str, Optional[int], Dict[str, Tuple[Tuple[int, int], ...]]]:
    if not cache_path.exists():
        return "miss", None, {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None, {}

    if payload.get("cache_version") != ENUMERATION_CACHE_VERSION:
        return "stale", None, {}
    if payload.get("metadata") != metadata:
        return "stale", None, {}

    raw_representatives = payload.get("representatives")
    if not isinstance(raw_representatives, list):
        return "invalid", None, {}
    representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for item in raw_representatives:
        if not isinstance(item, dict):
            continue
        signature = item.get("signature")
        if not isinstance(signature, str):
            continue
        if "edges" not in item:
            continue
        edges = edge_list_from_json(item.get("edges"))
        representatives[signature] = edges
    if not representatives:
        return "invalid", None, {}
    return "hit", int(payload.get("enumerated_subgraphs", 0)), representatives


def save_node_enumeration_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
    enumerated_subgraphs: int,
    representatives: Dict[str, Tuple[Tuple[int, int], ...]],
) -> bool:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "metadata": metadata,
        "enumerated_subgraphs": int(enumerated_subgraphs),
        "enumerated_signatures": int(len(representatives)),
        "representatives": [
            {
                "signature": signature,
                "edges": canonical_edge_list(edges),
            }
            for signature, edges in sorted(representatives.items())
        ],
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def load_node_necessity_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
) -> Tuple[str, Optional[int], Dict[str, Dict[str, Tuple[Tuple[int, int], ...]]]]:
    if not cache_path.exists():
        return "miss", None, {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid", None, {}

    if payload.get("cache_version") != ENUMERATION_CACHE_VERSION:
        return "stale", None, {}
    if payload.get("metadata") != metadata:
        return "stale", None, {}

    raw_representatives = payload.get("representatives")
    if not isinstance(raw_representatives, list):
        return "invalid", None, {}
    representatives: Dict[str, Dict[str, Tuple[Tuple[int, int], ...]]] = {}
    for item in raw_representatives:
        if not isinstance(item, dict):
            continue
        signature = item.get("signature")
        if not isinstance(signature, str):
            continue
        score_edges = edge_list_from_json(item.get("score_edges"))
        display_edges = edge_list_from_json(item.get("edges"))
        if not score_edges or not display_edges:
            continue
        representatives[signature] = {
            "score_edges": score_edges,
            "edges": display_edges,
        }
    if not representatives:
        return "invalid", None, {}
    return "hit", int(payload.get("enumerated_subgraphs", 0)), representatives


def save_node_necessity_cache(
    *,
    cache_path: Path,
    metadata: Dict[str, Any],
    enumerated_subgraphs: int,
    representatives: Dict[str, Dict[str, Tuple[Tuple[int, int], ...]]],
) -> bool:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": ENUMERATION_CACHE_VERSION,
        "metadata": metadata,
        "enumerated_subgraphs": int(enumerated_subgraphs),
        "enumerated_signatures": int(len(representatives)),
        "representatives": [
            {
                "signature": signature,
                "score_edges": canonical_edge_list(entry["score_edges"]),
                "edges": canonical_edge_list(entry["edges"]),
            }
            for signature, entry in sorted(representatives.items())
        ],
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def node_sufficiency_shape_representatives(
    *,
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    node_colors: Sequence[str],
    gt_edges: Sequence[Tuple[int, int]],
    root_node: int,
    valid_sizes: Sequence[int],
    max_subgraphs: int,
    edge_budget_mode: str,
    exclude_gt_solution: bool = False,
) -> Tuple[Optional[str], int, Dict[str, Tuple[Tuple[int, int], ...]]]:
    excluded_edge_sets = [gt_edges] if bool(exclude_gt_solution) else None
    enum_status, enumerated_subgraphs, representatives = node_shape_first_representatives(
        x=x,
        candidate_edges=candidate_edges,
        node_colors=node_colors,
        root_node=int(root_node),
        valid_sizes=valid_sizes,
        max_subgraphs=int(max_subgraphs),
        include_empty=str(edge_budget_mode) == "le",
        excluded_edge_sets=excluded_edge_sets,
    )
    return enum_status, enumerated_subgraphs, representatives


def selected_margin_for_edges(
    *,
    model: torch.nn.Module,
    task: str,
    x: np.ndarray,
    edge_index: np.ndarray,
    selected_edges: Sequence[Tuple[int, int]],
    c_star: int,
    root_node: Optional[int],
) -> float:
    selected_edge_index = filter_edge_index_by_selected_edges(edge_index, selected_edges)
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        edge_tensor = torch.tensor(np.asarray(selected_edge_index), dtype=torch.long)
        logits = model(x_tensor, edge_tensor)
    if task == "node":
        if root_node is None:
            raise ValueError("root_node is required for node enumeration scoring.")
        logits = logits[int(root_node)]
    return compute_margin(logits.reshape(-1), int(c_star))


def graph_margin_probability_for_edge_sets_batch(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    selected_edge_sets: Sequence[Sequence[Tuple[int, int]]],
    c_star: int,
    batch_size: int = DEFAULT_ENUMERATION_BATCH_SIZE,
) -> List[Tuple[float, float]]:
    """Return (margin, exact c-star probability) for batched graph masks."""
    if not selected_edge_sets:
        return []
    if int(batch_size) <= 0:
        raise ValueError("graph enumeration batch_size must be positive.")

    x_array = np.asarray(x, dtype=np.float32)
    num_nodes = int(x_array.shape[0])
    if num_nodes <= 0:
        raise ValueError("graph enumeration requires at least one node.")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    scores: List[Tuple[float, float]] = []
    with torch.no_grad():
        for start in range(0, len(selected_edge_sets), int(batch_size)):
            chunk = selected_edge_sets[start : start + int(batch_size)]
            chunk_size = len(chunk)
            x_tensor = torch.tensor(
                np.tile(x_array, (chunk_size, 1)),
                dtype=torch.float32,
                device=device,
            )
            batch_tensor = torch.arange(
                chunk_size,
                dtype=torch.long,
                device=device,
            ).repeat_interleave(num_nodes)

            shifted_edge_indices: List[np.ndarray] = []
            for local_idx, selected_edges in enumerate(chunk):
                masked_edge_index = filter_edge_index_by_selected_edges(edge_index, selected_edges)
                if int(masked_edge_index.shape[1]) <= 0:
                    continue
                shifted_edge_indices.append(
                    np.asarray(masked_edge_index, dtype=np.int64) + int(local_idx) * num_nodes
                )
            if shifted_edge_indices:
                edge_array = np.concatenate(shifted_edge_indices, axis=1)
                edge_tensor = torch.tensor(edge_array, dtype=torch.long, device=device)
            else:
                edge_tensor = torch.zeros((2, 0), dtype=torch.long, device=device)

            logits = model(x_tensor, edge_tensor, batch=batch_tensor)
            chunk_margins = margin_vector_for_logits(logits, int(c_star)).detach().cpu().reshape(-1)
            chunk_probabilities = (
                torch.softmax(logits.reshape(chunk_size, -1), dim=1)[:, int(c_star)]
                .detach()
                .cpu()
                .reshape(-1)
            )
            if int(chunk_margins.numel()) != chunk_size:
                raise RuntimeError(
                    "Graph enumeration batch returned "
                    f"{int(chunk_margins.numel())} margins for {chunk_size} masks."
                )
            scores.extend(
                (float(margin), float(probability))
                for margin, probability in zip(
                    chunk_margins.tolist(),
                    chunk_probabilities.tolist(),
                )
            )
    return scores


def node_margin_probability_for_edge_sets_batch(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    selected_edge_sets: Sequence[Sequence[Tuple[int, int]]],
    c_star: int,
    root_node: int,
    batch_size: int = DEFAULT_ENUMERATION_BATCH_SIZE,
) -> List[Tuple[float, float]]:
    """Return (margin, exact c-star probability) for batched node masks."""
    if not selected_edge_sets:
        return []
    if int(batch_size) <= 0:
        raise ValueError("node enumeration batch_size must be positive.")

    global_edge_sets = [tuple(sorted(edge_set(edges))) for edges in selected_edge_sets]
    inference_candidate_edges = sorted(
        edge_set(edge for edges in global_edge_sets for edge in edges)
    )
    x_array, local_edge_index, local_root_node, _, node_to_local = compact_node_inference_graph(
        x=np.asarray(x, dtype=np.float32),
        edge_index=np.asarray(edge_index, dtype=np.int64),
        candidate_edges=inference_candidate_edges,
        root_node=int(root_node),
    )
    local_edge_sets = [
        tuple(
            sorted(
                edge_set(
                    (node_to_local[int(a)], node_to_local[int(b)])
                    for a, b in edges
                )
            )
        )
        for edges in global_edge_sets
    ]
    num_nodes = int(x_array.shape[0])
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    scores: List[Tuple[float, float]] = []
    with torch.no_grad():
        for start in range(0, len(local_edge_sets), int(batch_size)):
            chunk = local_edge_sets[start : start + int(batch_size)]
            chunk_size = len(chunk)
            x_tensor = torch.tensor(
                np.tile(x_array, (chunk_size, 1)),
                dtype=torch.float32,
                device=device,
            )
            shifted_edge_indices: List[np.ndarray] = []
            for local_idx, selected_edges in enumerate(chunk):
                masked_edge_index = filter_edge_index_by_selected_edges(local_edge_index, selected_edges)
                if int(masked_edge_index.shape[1]) <= 0:
                    continue
                shifted_edge_indices.append(
                    np.asarray(masked_edge_index, dtype=np.int64) + int(local_idx) * num_nodes
                )
            if shifted_edge_indices:
                edge_array = np.concatenate(shifted_edge_indices, axis=1)
                edge_tensor = torch.tensor(edge_array, dtype=torch.long, device=device)
            else:
                edge_tensor = torch.zeros((2, 0), dtype=torch.long, device=device)

            logits = model(x_tensor, edge_tensor)
            target_indices = (
                torch.arange(chunk_size, dtype=torch.long, device=device) * num_nodes
                + int(local_root_node)
            )
            target_logits = logits[target_indices]
            chunk_margins = margin_vector_for_logits(target_logits, int(c_star)).detach().cpu().reshape(-1)
            chunk_probabilities = (
                torch.softmax(target_logits.reshape(chunk_size, -1), dim=1)[:, int(c_star)]
                .detach()
                .cpu()
                .reshape(-1)
            )
            if int(chunk_margins.numel()) != chunk_size:
                raise RuntimeError(
                    "Node enumeration batch returned "
                    f"{int(chunk_margins.numel())} margins for {chunk_size} masks."
                )
            scores.extend(
                (float(margin), float(probability))
                for margin, probability in zip(
                    chunk_margins.tolist(),
                    chunk_probabilities.tolist(),
                )
            )
    return scores


def enumeration_margin_probability_for_edge_sets_batch(
    *,
    model: torch.nn.Module,
    task: str,
    x: np.ndarray,
    edge_index: np.ndarray,
    selected_edge_sets: Sequence[Sequence[Tuple[int, int]]],
    c_star: int,
    root_node: Optional[int],
    batch_size: int = DEFAULT_ENUMERATION_BATCH_SIZE,
) -> List[Tuple[float, float]]:
    """Score masks through one task-agnostic probability-fidelity batch API."""
    normalized_task = str(task)
    if normalized_task == "graph":
        return graph_margin_probability_for_edge_sets_batch(
            model=model,
            x=x,
            edge_index=edge_index,
            selected_edge_sets=selected_edge_sets,
            c_star=int(c_star),
            batch_size=int(batch_size),
        )
    if normalized_task == "node":
        if root_node is None:
            raise ValueError("root_node is required for node enumeration scoring.")
        return node_margin_probability_for_edge_sets_batch(
            model=model,
            x=x,
            edge_index=edge_index,
            selected_edge_sets=selected_edge_sets,
            c_star=int(c_star),
            root_node=int(root_node),
            batch_size=int(batch_size),
        )
    raise ValueError(f"Unsupported enumeration task: {task!r}")


def class_probability_for_edges(
    *,
    model: torch.nn.Module,
    task: str,
    x: np.ndarray,
    edge_index: np.ndarray,
    selected_edges: Sequence[Tuple[int, int]],
    c_star: int,
    root_node: Optional[int],
) -> float:
    """Evaluate the exact softmax probability of c_star on an edge-masked graph."""
    selected_edge_index = filter_edge_index_by_selected_edges(edge_index, selected_edges)
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32)
        edge_tensor = torch.tensor(np.asarray(selected_edge_index), dtype=torch.long)
        logits = model(x_tensor, edge_tensor)
    if task == "node":
        if root_node is None:
            raise ValueError("root_node is required for node probability scoring.")
        logits = logits[int(root_node)]
    probabilities = torch.softmax(logits.reshape(-1), dim=0)
    return float(probabilities[int(c_star)].item())


def standard_fidelity_components(
    *,
    model: torch.nn.Module,
    task: str,
    x: np.ndarray,
    edge_index: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    selected_edges: Sequence[Tuple[int, int]],
    c_star: int,
    full_margin: float,
    root_node: Optional[int],
) -> Dict[str, float]:
    selected_set = edge_set(selected_edges)
    candidate_set = edge_set(candidate_edges)
    kept_margin = selected_margin_for_edges(
        model=model,
        task=task,
        x=x,
        edge_index=edge_index,
        selected_edges=sorted(selected_set),
        c_star=int(c_star),
        root_node=root_node,
    )
    complement_margin = selected_margin_for_edges(
        model=model,
        task=task,
        x=x,
        edge_index=edge_index,
        selected_edges=sorted(candidate_set - selected_set),
        c_star=int(c_star),
        root_node=root_node,
    )
    # Evaluate exact model probabilities for reporting. For the binary paper
    # datasets this equals sigmoid(margin), while this remains well-defined for
    # any multiclass sufficiency-only diagnostic rows.
    p_full = class_probability_for_edges(
        model=model,
        task=task,
        x=x,
        edge_index=edge_index,
        selected_edges=sorted(candidate_set),
        c_star=int(c_star),
        root_node=root_node,
    )
    p_kept = class_probability_for_edges(
        model=model,
        task=task,
        x=x,
        edge_index=edge_index,
        selected_edges=sorted(selected_set),
        c_star=int(c_star),
        root_node=root_node,
    )
    p_complement = class_probability_for_edges(
        model=model,
        task=task,
        x=x,
        edge_index=edge_index,
        selected_edges=sorted(candidate_set - selected_set),
        c_star=int(c_star),
        root_node=root_node,
    )
    fidelity_minus = float(p_full - p_kept)
    fidelity_plus = float(p_full - p_complement)
    return {
        "sufficiency_raw": float(kept_margin),
        "necessity_raw": float(full_margin - complement_margin),
        "sufficiency_probability": p_kept,
        "complement_probability": p_complement,
        "fidelity_minus": fidelity_minus,
        "fidelity_plus": fidelity_plus,
        "fidelity_delta": float(fidelity_plus - fidelity_minus),
    }


ENUMERATION_DIAGNOSTIC_FIELDS = (
    "enum_objective_max",
    "enum_objective_min",
    "enum_rho_at_objective_max",
    "enum_rho_at_objective_min",
    "enum_sufficiency_raw_at_objective_max",
    "enum_sufficiency_raw_at_objective_min",
    "enum_necessity_raw_at_objective_max",
    "enum_necessity_raw_at_objective_min",
    "enum_fidelity_at_objective_max",
    "enum_fidelity_at_objective_min",
    "gt_rho",
    "gt_sufficiency_raw",
    "gt_necessity_raw",
    "gt_objective_value",
    "gt_rank_max_objective",
    "gt_rank_min_objective",
    "gt_rank_position",
    "gt_rank_total",
    "gt_rank_tie_count",
    "gt_rank_direction",
    "gt_rank_scope",
    "enum_padding_feasible",
    "enum_padding_filtered",
)


def strict_rank(values: Sequence[float], value: float, *, descending: bool) -> int:
    if descending:
        return 1 + sum(
            1
            for other in values
            if float(other) > float(value) + ENUMERATION_RANK_TOLERANCE
        )
    return 1 + sum(
        1
        for other in values
        if float(other) < float(value) - ENUMERATION_RANK_TOLERANCE
    )


def enumeration_diagnostics(
    scored_items: Sequence[Dict[str, Any]],
    *,
    gt_item: Optional[Dict[str, Any]],
    objective: str,
    rank_scope: str,
    gt_candidate_present: bool,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {field: None for field in ENUMERATION_DIAGNOSTIC_FIELDS}
    items = [item for item in scored_items if item.get("objective") is not None]
    if not items:
        return diagnostics

    max_item = min(items, key=lambda item: (-float(item["objective"]), len(item["edges"]), item["edges"]))
    min_item = min(items, key=lambda item: (float(item["objective"]), len(item["edges"]), item["edges"]))
    diagnostics["enum_objective_max"] = float(max_item["objective"])
    diagnostics["enum_objective_min"] = float(min_item["objective"])
    diagnostics["enum_rho_at_objective_max"] = float(max_item["rho"])
    diagnostics["enum_rho_at_objective_min"] = float(min_item["rho"])
    diagnostics["enum_sufficiency_raw_at_objective_max"] = float(max_item["sufficiency_raw"])
    diagnostics["enum_sufficiency_raw_at_objective_min"] = float(min_item["sufficiency_raw"])
    diagnostics["enum_necessity_raw_at_objective_max"] = to_float_or_none(max_item.get("necessity_raw"))
    diagnostics["enum_necessity_raw_at_objective_min"] = to_float_or_none(min_item.get("necessity_raw"))
    diagnostics["enum_fidelity_at_objective_max"] = to_float_or_none(max_item.get("fidelity"))
    diagnostics["enum_fidelity_at_objective_min"] = to_float_or_none(min_item.get("fidelity"))
    diagnostics["enum_objective_max_edges"] = list(max_item["edges"])
    diagnostics["enum_objective_min_edges"] = list(min_item["edges"])

    if gt_item is not None and gt_item.get("objective") is not None:
        gt_objective = float(gt_item["objective"])
        objective_values = [float(item["objective"]) for item in items]
        diagnostics["gt_rho"] = float(gt_item["rho"])
        diagnostics["gt_sufficiency_raw"] = float(gt_item["sufficiency_raw"])
        diagnostics["gt_necessity_raw"] = to_float_or_none(gt_item.get("necessity_raw"))
        diagnostics["gt_objective_value"] = gt_objective
        diagnostics["gt_rank_max_objective"] = strict_rank(objective_values, gt_objective, descending=True)
        diagnostics["gt_rank_min_objective"] = strict_rank(objective_values, gt_objective, descending=False)
        maximize = str(objective) not in {"min_sufficiency", "min_necessity"}
        diagnostics["gt_rank_position"] = diagnostics[
            "gt_rank_max_objective" if maximize else "gt_rank_min_objective"
        ]
        diagnostics["gt_rank_total"] = len(items) + (0 if gt_candidate_present else 1)
        diagnostics["gt_rank_tie_count"] = sum(
            1
            for item_value in objective_values
            if abs(float(item_value) - gt_objective) <= ENUMERATION_RANK_TOLERANCE
        ) + (0 if gt_candidate_present else 1)
        diagnostics["gt_rank_direction"] = "descending" if maximize else "ascending"
        diagnostics["gt_rank_scope"] = str(rank_scope)
    return diagnostics


def padding_feasible_rank_items(
    *,
    row: Dict[str, Any],
    scored_items: Sequence[Dict[str, Any]],
    candidate_edges: Sequence[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], int]:
    if row.get("task") != "node" or row.get("oracle_objective") != RANK_SUFFICIENCY_OBJECTIVE:
        return list(scored_items), 0

    feasible: List[Dict[str, Any]] = []
    filtered_count = 0
    for item in scored_items:
        edges = item.get("edges")
        if edges is None:
            filtered_count += 1
            continue
        _, padding_info = pad_node_gt_size_edges(
            row=row,
            padding_seed=0,
            root=int(row["instance_id"]),
            selected_edges=edge_set(edges),
            candidate_edges=candidate_edges,
            target_size=int(row["gt_size"]),
        )
        if padding_info.get("padding_status") in {"not_needed", "ok"}:
            feasible.append(dict(item))
        else:
            filtered_count += 1
    return feasible, filtered_count


def padding_aware_enumeration_diagnostics(
    scored_items: Sequence[Dict[str, Any]],
    *,
    gt_item: Optional[Dict[str, Any]],
    row: Dict[str, Any],
    candidate_edges: Sequence[Tuple[int, int]],
    rank_scope: str = "instance_feasible_automorphism_representatives",
    gt_candidate_present: bool = True,
) -> Dict[str, Any]:
    feasible_items, filtered_count = padding_feasible_rank_items(
        row=row,
        scored_items=scored_items,
        candidate_edges=candidate_edges,
    )
    diagnostics = enumeration_diagnostics(
        feasible_items,
        gt_item=gt_item,
        objective=str(row.get("oracle_objective")),
        rank_scope=rank_scope,
        gt_candidate_present=gt_candidate_present,
    )
    if row.get("task") == "node" and row.get("oracle_objective") == RANK_SUFFICIENCY_OBJECTIVE:
        diagnostics["enum_padding_feasible"] = len(feasible_items)
        diagnostics["enum_padding_filtered"] = filtered_count
    return diagnostics


def abstract_pattern_edge_columns(
    *,
    pattern_edges: Sequence[Tuple[int, int]],
    node_offset: int,
    self_loop_count: int,
    nonloop_count_per_direction: int,
    num_nodes: int,
) -> List[Tuple[int, int]]:
    columns: List[Tuple[int, int]] = []
    for node in range(int(num_nodes)):
        actual = int(node_offset) + int(node)
        for _ in range(int(self_loop_count)):
            columns.append((actual, actual))
    for a, b in sorted(edge_set(pattern_edges)):
        left = int(node_offset) + int(a)
        right = int(node_offset) + int(b)
        for _ in range(int(nonloop_count_per_direction)):
            columns.append((left, right))
            columns.append((right, left))
    return columns


def margin_vector_for_logits(logits: torch.Tensor, c_star: int) -> torch.Tensor:
    logits = logits.reshape(logits.size(0), -1)
    if logits.size(1) <= 1 or int(c_star) < 0 or int(c_star) >= int(logits.size(1)):
        return logits.new_zeros((logits.size(0),))
    other_logits = logits.clone()
    other_logits[:, int(c_star)] = -torch.inf
    return logits[:, int(c_star)] - other_logits.max(dim=1).values


def score_abstract_graph_patterns(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    patterns: Sequence[Tuple[Tuple[int, int], ...]],
    c_star: int,
    self_loop_count: int,
    nonloop_count_per_direction: int,
    maximize: bool,
    batch_size: int = DEFAULT_ENUMERATION_BATCH_SIZE,
) -> List[Tuple[float, float, Tuple[Tuple[int, int], ...]]]:
    if not patterns:
        return []
    num_nodes = int(np.asarray(x).shape[0])
    if num_nodes <= 0:
        return []
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    feature = torch.tensor(np.asarray(x[0:1]), dtype=torch.float32, device=device)
    scored: List[Tuple[float, float, Tuple[Tuple[int, int], ...]]] = []

    with torch.no_grad():
        for start in range(0, len(patterns), int(batch_size)):
            batch_patterns = list(patterns[start : start + int(batch_size)])
            batch_count = len(batch_patterns)
            x_tensor = feature.repeat(batch_count * num_nodes, 1)
            batch_tensor = torch.arange(batch_count, dtype=torch.long, device=device).repeat_interleave(num_nodes)
            edge_columns: List[Tuple[int, int]] = []
            for local_idx, pattern_edges in enumerate(batch_patterns):
                edge_columns.extend(
                    abstract_pattern_edge_columns(
                        pattern_edges=pattern_edges,
                        node_offset=int(local_idx) * num_nodes,
                        self_loop_count=int(self_loop_count),
                        nonloop_count_per_direction=int(nonloop_count_per_direction),
                        num_nodes=num_nodes,
                    )
                )
            if edge_columns:
                edge_tensor = torch.tensor(np.asarray(edge_columns, dtype=np.int64).T, dtype=torch.long, device=device)
            else:
                edge_tensor = torch.zeros((2, 0), dtype=torch.long, device=device)
            logits = model(x_tensor, edge_tensor, batch=batch_tensor)
            margins = margin_vector_for_logits(logits, int(c_star)).detach().cpu().reshape(-1)
            probabilities = (
                torch.softmax(logits.reshape(batch_count, -1), dim=1)[:, int(c_star)]
                .detach()
                .cpu()
                .reshape(-1)
            )
            for pattern_edges, margin, probability in zip(
                batch_patterns,
                margins,
                probabilities,
            ):
                scored.append(
                    (float(margin.item()), float(probability.item()), tuple(pattern_edges))
                )

    if maximize:
        scored.sort(key=lambda item: (-item[1], len({node for edge in item[2] for node in edge}), item[2]))
    else:
        scored.sort(key=lambda item: (item[1], len({node for edge in item[2] for node in edge}), item[2]))
    return scored


def graph_from_pattern_edges(pattern_edges: Sequence[Tuple[int, int]]) -> nx.Graph:
    graph = nx.Graph()
    for a, b in sorted(edge_set(pattern_edges)):
        graph.add_edge(int(a), int(b))
    return graph


def run_graph_homogeneous_pattern_first_enumeration(
    *,
    row: Dict[str, Any],
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    max_subgraphs: int,
    enumeration_batch_size: int = DEFAULT_ENUMERATION_BATCH_SIZE,
    use_enumeration_cache: bool = True,
    enumeration_cache_dir: Optional[Path] = None,
) -> Optional[SimpleNamespace]:
    if row.get("task") != "graph":
        return None
    objective = str(row.get("oracle_objective"))
    if objective not in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}:
        return None
    connectivity = str(row.get("connectivity"))
    if connectivity not in {"none", "cut"} or row.get("edge_budget_mode") != "eq":
        return None
    if not homogeneous_node_features(x):
        return None

    budget_k = int(row["budget_k"])
    if budget_k <= 0:
        return None

    convention = uniform_graph_edge_index_convention(edge_index, num_nodes=int(np.asarray(x).shape[0]))
    if convention is None:
        return None
    self_loop_count, nonloop_count_per_direction = convention
    fixed_self_loop_counts = fixed_self_loop_counts_from_edge_index(
        edge_index,
        num_nodes=int(np.asarray(x).shape[0]),
    )

    try:
        patterns = edge_mask_topology_patterns(budget_k)
    except RuntimeError:
        return None
    if connectivity == "cut":
        patterns = [pattern for pattern in patterns if pattern_edges_connected(pattern)]

    if len(patterns) > int(max_subgraphs):
        return SimpleNamespace(
            status="skipped_enumeration_too_large",
            selected_edges=[],
            rho_optimal=None,
            sufficiency_raw=None,
            necessity_raw=None,
            objective_value=None,
            enumerated_subgraphs=len(patterns),
            enumerated_signatures=0,
            enumeration_pruned=0,
        )

    scored_patterns = score_abstract_graph_patterns(
        model=model,
        x=x,
        patterns=patterns,
        c_star=int(row["c_star"]),
        self_loop_count=int(self_loop_count),
        nonloop_count_per_direction=int(nonloop_count_per_direction),
        maximize=objective != "min_sufficiency",
        batch_size=int(enumeration_batch_size),
    )
    bar_m = float(row["bar_m"])
    p_full = probability_from_binary_margin(bar_m)
    if p_full is None or int(row.get("num_classes") or 0) != 2:
        p_full = class_probability_for_edges(
            model=model,
            task="graph",
            x=x,
            edge_index=edge_index,
            selected_edges=candidate_edges,
            c_star=int(row["c_star"]),
            root_node=None,
        )
    diagnostics: Dict[str, Any] = {}
    gt_item = None
    pattern_cache_status: Optional[str] = None
    pattern_cache_file: Optional[str] = None
    candidate_edge_set = edge_set(candidate_edges)
    gt_edge_set = edge_set(gt_edges)
    excluded_edge_sets = [tuple(sorted(gt_edge_set))] if bool(row.get("exclude_gt_solution")) else None
    candidate_graph = candidate_nx_graph(x=x, candidate_edges=candidate_edges, root_node=None)
    candidate_node_count = int(candidate_graph.number_of_nodes())
    scored_desc = sorted(
        scored_patterns,
        key=lambda item: (-float(item[1]), len({node for edge in item[2] for node in edge}), item[2]),
    )
    scored_asc = sorted(
        scored_patterns,
        key=lambda item: (float(item[1]), len({node for edge in item[2] for node in edge}), item[2]),
    )

    def first_feasible_pattern(
        ordered_patterns: Sequence[
            Tuple[float, float, Tuple[Tuple[int, int], ...]]
        ],
    ) -> Tuple[Optional[Tuple[Tuple[int, int], ...]], Optional[float], int]:
        feasibility_checks = 0
        group_start = 0
        while group_start < len(ordered_patterns):
            group_probability = float(ordered_patterns[group_start][1])
            group_end = group_start + 1
            while group_end < len(ordered_patterns) and math.isclose(
                float(ordered_patterns[group_end][1]),
                group_probability,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                group_end += 1

            best_group_selected: Optional[Tuple[Tuple[int, int], ...]] = None
            best_group_margin: Optional[float] = None
            for pattern_margin, _, pattern_edges in ordered_patterns[group_start:group_end]:
                pattern_nodes = {node for edge in pattern_edges for node in edge}
                if len(pattern_nodes) > candidate_node_count:
                    continue
                pattern_graph = graph_from_pattern_edges(pattern_edges)
                feasibility_checks += 1
                selected = find_unrooted_monomorphism_edges(
                    candidate_graph=candidate_graph,
                    pattern=pattern_graph,
                    match_features=False,
                    excluded_edge_sets=excluded_edge_sets,
                )
                if selected is None:
                    continue
                selected = tuple(sorted(edge_set(selected)))
                if best_group_selected is None or selected < best_group_selected:
                    best_group_selected = selected
                    best_group_margin = float(pattern_margin)
            if best_group_selected is not None:
                return best_group_selected, best_group_margin, feasibility_checks
            group_start = group_end
        return None, None, feasibility_checks

    need_max = objective == "max_sufficiency"
    need_min = objective == "min_sufficiency"
    max_selected: Optional[Tuple[Tuple[int, int], ...]] = None
    min_selected: Optional[Tuple[Tuple[int, int], ...]] = None
    max_margin: Optional[float] = None
    min_margin: Optional[float] = None
    max_checks = 0
    min_checks = 0
    if need_max:
        max_selected, max_margin, max_checks = first_feasible_pattern(scored_desc)
    if need_min:
        min_selected, min_margin, min_checks = first_feasible_pattern(scored_asc)

    feasible_rank_items: List[Dict[str, Any]] = []
    if objective in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}:
        feasible_pattern_representatives: List[
            Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...]]
        ] = []
        pattern_cache_status = "disabled"
        pattern_cache_file = None
        if use_enumeration_cache and enumeration_cache_dir is not None:
            cache_metadata = graph_sufficiency_pattern_cache_metadata(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                patterns=patterns,
                fixed_self_loop_counts=fixed_self_loop_counts,
            )
            cache_path = graph_sufficiency_pattern_cache_path(
                cache_dir=Path(enumeration_cache_dir),
                metadata=cache_metadata,
            )
            pattern_cache_file = output_path_string(cache_path)
            pattern_cache_status, feasible_pattern_representatives = load_graph_sufficiency_pattern_cache(
                cache_path=cache_path,
                metadata=cache_metadata,
            )

        if pattern_cache_status != "hit":
            feasible_pattern_representatives = []
            for pattern_edges in patterns:
                pattern_nodes = {node for edge in pattern_edges for node in edge}
                if len(pattern_nodes) > candidate_node_count:
                    continue
                pattern_graph = graph_from_pattern_edges(pattern_edges)
                selected = find_unrooted_monomorphism_edges(
                    candidate_graph=candidate_graph,
                    pattern=pattern_graph,
                    match_features=False,
                    excluded_edge_sets=excluded_edge_sets,
                )
                if selected is None:
                    continue
                feasible_pattern_representatives.append(
                    (tuple(pattern_edges), tuple(sorted(edge_set(selected))))
                )
            if use_enumeration_cache and enumeration_cache_dir is not None:
                saved = save_graph_sufficiency_pattern_cache(
                    cache_path=cache_path,
                    metadata=cache_metadata,
                    enumerated_subgraphs=len(scored_patterns),
                    representatives=feasible_pattern_representatives,
                )
                pattern_cache_status = (
                    f"{pattern_cache_status}_saved" if saved else f"{pattern_cache_status}_save_failed"
                )

        score_by_pattern = {
            tuple(pattern_edges): (float(margin), float(probability))
            for margin, probability, pattern_edges in scored_patterns
        }
        gt_cert = None
        gt_candidate_present = False
        if (
            len(gt_edge_set) == int(budget_k)
            and gt_edge_set.issubset(candidate_edge_set)
            and (connectivity != "cut" or pattern_edges_connected(gt_edge_set))
        ):
            gt_cert = pattern_certificate(tuple(sorted(gt_edge_set)))

        for pattern_edges, selected in feasible_pattern_representatives:
            score_pair = score_by_pattern.get(tuple(pattern_edges))
            if score_pair is None:
                continue
            margin, masked_probability = score_pair
            rho_value = float(margin) / bar_m
            fidelity_value, fidelity_score = standard_enumeration_fidelity_values(
                p_full=float(p_full),
                p_masked=float(masked_probability),
                necessity=False,
            )
            item = {
                "edges": tuple(sorted(edge_set(selected))),
                "rho": float(rho_value),
                "sufficiency_raw": float(margin),
                "necessity_raw": None,
                "masked_probability": float(masked_probability),
                "fidelity": float(fidelity_value),
                "objective": float(fidelity_score),
            }
            feasible_rank_items.append(item)
            if gt_cert is not None and pattern_certificate(pattern_edges) == gt_cert:
                gt_item = {**item, "edges": tuple(sorted(gt_edge_set))}
                gt_candidate_present = True

        if gt_item is None and gt_cert is not None:
            for pattern_edges, (margin, masked_probability) in score_by_pattern.items():
                if pattern_certificate(pattern_edges) != gt_cert:
                    continue
                gt_rho = float(margin) / bar_m
                gt_fidelity, gt_fidelity_score = standard_enumeration_fidelity_values(
                    p_full=float(p_full),
                    p_masked=float(masked_probability),
                    necessity=False,
                )
                gt_item = {
                    "edges": tuple(sorted(gt_edge_set)),
                    "rho": gt_rho,
                    "sufficiency_raw": float(margin),
                    "necessity_raw": None,
                    "masked_probability": float(masked_probability),
                    "fidelity": float(gt_fidelity),
                    "objective": float(gt_fidelity_score),
                }
                break

        diagnostics = padding_aware_enumeration_diagnostics(
            feasible_rank_items,
            gt_item=gt_item,
            row=row,
            candidate_edges=candidate_edges,
            rank_scope="instance_feasible_colored_topology_representatives",
            gt_candidate_present=gt_candidate_present,
        )

    if objective == "max_sufficiency":
        best_selected = tuple(
            sorted(edge_set(diagnostics.get("enum_objective_max_edges") or max_selected or ()))
        )
        best_margin = to_float_or_none(diagnostics.get("enum_sufficiency_raw_at_objective_max"))
        best_objective_value = to_float_or_none(diagnostics.get("enum_objective_max"))
        enumerated_signatures = len(feasible_rank_items)
        enumeration_pruned = max(0, len(scored_patterns) - len(feasible_rank_items))
    elif objective == "min_sufficiency":
        best_selected = tuple(
            sorted(edge_set(diagnostics.get("enum_objective_min_edges") or min_selected or ()))
        )
        best_margin = to_float_or_none(diagnostics.get("enum_sufficiency_raw_at_objective_min"))
        best_objective_value = to_float_or_none(diagnostics.get("enum_objective_min"))
        enumerated_signatures = len(feasible_rank_items)
        enumeration_pruned = max(0, len(scored_patterns) - len(feasible_rank_items))
    elif gt_item is not None:
        best_selected = tuple(sorted(gt_edge_set))
        best_margin = float(gt_item["sufficiency_raw"])
        best_objective_value = float(gt_item["objective"])
        enumerated_signatures = len(feasible_rank_items)
        enumeration_pruned = max(0, len(scored_patterns) - len(feasible_rank_items))
    else:
        best_rank_item = diagnostics.get("enum_objective_max_edges")
        best_selected = tuple(sorted(edge_set(best_rank_item))) if best_rank_item is not None else max_selected
        best_margin = (
            float(diagnostics["enum_rho_at_objective_max"]) * bar_m
            if diagnostics.get("enum_rho_at_objective_max") is not None
            else max_margin
        )
        best_objective_value = to_float_or_none(diagnostics.get("enum_objective_max"))
        enumerated_signatures = len(feasible_rank_items) if objective == RANK_SUFFICIENCY_OBJECTIVE else len(scored_patterns)
        enumeration_pruned = (
            max(0, len(scored_patterns) - len(feasible_rank_items))
            if objective == RANK_SUFFICIENCY_OBJECTIVE
            else max(0, len(scored_patterns) - int(max_checks + min_checks))
        )
    if best_selected is not None and best_margin is not None:
        rho_value = float(best_margin) / bar_m
        return SimpleNamespace(
            status="optimal",
            selected_edges=list(best_selected),
            rho_optimal=float(rho_value),
            sufficiency_raw=float(best_margin),
            necessity_raw=None,
            objective_value=best_objective_value,
            final_mip_gap=0.0,
            bnb_node_count=0,
            enumerated_subgraphs=len(scored_patterns),
            enumerated_signatures=enumerated_signatures,
            enumeration_pruned=enumeration_pruned,
            enumeration_cache_status=pattern_cache_status,
            enumeration_cache_file=pattern_cache_file,
            **diagnostics,
        )

    return SimpleNamespace(
        status="skipped_empty_candidate_set",
        selected_edges=[],
        rho_optimal=None,
        sufficiency_raw=None,
        necessity_raw=None,
        objective_value=None,
        enumerated_subgraphs=len(scored_patterns),
        enumerated_signatures=len(feasible_rank_items) if objective == RANK_SUFFICIENCY_OBJECTIVE else len(scored_patterns),
        enumeration_pruned=(
            max(0, len(scored_patterns) - len(feasible_rank_items))
            if objective == RANK_SUFFICIENCY_OBJECTIVE
            else 0
        ),
        enumeration_cache_status=pattern_cache_status,
        enumeration_cache_file=pattern_cache_file,
    )


def run_gt_size_enumeration(
    *,
    row: Dict[str, Any],
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    root_node: Optional[int],
    max_subgraphs: int,
    enumeration_batch_size: int = DEFAULT_ENUMERATION_BATCH_SIZE,
    use_enumeration_cache: bool = True,
    enumeration_cache_dir: Optional[Path] = None,
) -> SimpleNamespace:
    if int(enumeration_batch_size) <= 0:
        raise ValueError("enumeration_batch_size must be positive.")
    budget_k = int(row["budget_k"])
    if row["task"] == "node" and str(row["edge_budget_mode"]) == "le":
        valid_sizes = list(range(1, budget_k + 1))
    else:
        valid_sizes = [budget_k]
    positive_valid_sizes = [int(size) for size in valid_sizes if int(size) > 0]

    enumerated_subgraphs = 0
    signatures: Dict[str, Dict[str, Any]] = {}
    objective = str(row["oracle_objective"])
    bar_m = float(row["bar_m"])
    candidate_edge_set = edge_set(candidate_edges)
    gt_edge_set = edge_set(gt_edges)
    exclude_gt_solution = bool(row.get("exclude_gt_solution"))
    uses_necessity_scoring = objective in NECESSITY_SCORING_OBJECTIVES
    is_graph_necessity = uses_necessity_scoring and str(row["task"]) == "graph"
    is_node_necessity = uses_necessity_scoring and str(row["task"]) == "node"
    enumeration_cache_status: Optional[str] = None
    enumeration_cache_file: Optional[str] = None
    concrete_necessity_scored_items: Optional[List[Dict[str, Any]]] = None
    graph_sufficiency_node_colors: Optional[Tuple[str, ...]] = None
    node_sufficiency_node_colors: Optional[Tuple[str, ...]] = None
    p_full = probability_from_binary_margin(bar_m)
    if p_full is None or int(row.get("num_classes") or 0) != 2:
        p_full = class_probability_for_edges(
            model=model,
            task=str(row["task"]),
            x=x,
            edge_index=edge_index,
            selected_edges=candidate_edges,
            c_star=int(row["c_star"]),
            root_node=root_node,
        )

    def score_subset_values_batch(
        subsets: Sequence[Sequence[Tuple[int, int]]],
    ) -> List[Dict[str, Any]]:
        normalized_subsets = [tuple(sorted(edge_set(subset))) for subset in subsets]
        evaluation_edge_sets = [
            tuple(sorted(candidate_edge_set - edge_set(subset)))
            if uses_necessity_scoring
            else subset
            for subset in normalized_subsets
        ]
        margin_probabilities = enumeration_margin_probability_for_edge_sets_batch(
            model=model,
            task=str(row["task"]),
            x=x,
            edge_index=edge_index,
            selected_edge_sets=evaluation_edge_sets,
            c_star=int(row["c_star"]),
            root_node=root_node,
            batch_size=int(enumeration_batch_size),
        )
        scored: List[Dict[str, Any]] = []
        for subset, (margin, masked_probability) in zip(
            normalized_subsets,
            margin_probabilities,
        ):
            rho_value = float(margin) / bar_m
            necessity_raw = float(bar_m - margin) if uses_necessity_scoring else None
            fidelity_value, objective_value = standard_enumeration_fidelity_values(
                p_full=float(p_full),
                p_masked=float(masked_probability),
                necessity=uses_necessity_scoring,
            )
            scored.append(
                {
                    "edges": subset,
                    "rho": float(rho_value),
                    "sufficiency_raw": float(margin),
                    "necessity_raw": necessity_raw,
                    "masked_probability": float(masked_probability),
                    "fidelity": float(fidelity_value),
                    "objective": objective_value,
                }
            )
        return scored

    def score_subset_values(subset: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
        scored = score_subset_values_batch([subset])
        if len(scored) != 1:
            raise RuntimeError("Single standard-enumeration mask did not produce one score.")
        return scored[0]

    def score_representatives_batch(
        representative_items: Sequence[Tuple[str, Sequence[Tuple[int, int]]]],
    ) -> None:
        if not representative_items:
            return
        scored_items = score_subset_values_batch(
            [subset for _, subset in representative_items]
        )
        for (signature, _), item in zip(representative_items, scored_items):
            signatures[str(signature)] = item

    def enumerate_and_score_concrete_subsets(
        *,
        mode: str,
        concrete_root_node: Optional[int],
    ) -> Tuple[Optional[str], int, int, float, List[Dict[str, Any]]]:
        """Stream C++ masks through bounded standard-fidelity inference batches."""
        pending_subsets: List[Tuple[Tuple[int, int], ...]] = []
        scored_items: List[Dict[str, Any]] = []
        eligible_subgraphs = 0
        scoring_time_sec = 0.0

        def flush_pending_subsets() -> None:
            nonlocal scoring_time_sec
            if not pending_subsets:
                return
            scoring_started = time.perf_counter()
            scored_items.extend(score_subset_values_batch(pending_subsets))
            scoring_time_sec += time.perf_counter() - scoring_started
            pending_subsets.clear()

        def add_subset(subset: Tuple[Tuple[int, int], ...]) -> None:
            nonlocal eligible_subgraphs
            if exclude_gt_solution and edge_set(subset) == gt_edge_set:
                return
            eligible_subgraphs += 1
            pending_subsets.append(tuple(subset))
            if len(pending_subsets) >= int(enumeration_batch_size):
                flush_pending_subsets()

        enumeration_started = time.perf_counter()
        enum_status, emitted_subgraphs = visit_concrete_edge_subsets_cpp(
            candidate_edges=candidate_edges,
            mode=mode,
            valid_sizes=positive_valid_sizes,
            root_node=concrete_root_node,
            max_subgraphs=int(max_subgraphs),
            visitor=add_subset,
        )
        if enum_status != "skipped_enumeration_too_large":
            flush_pending_subsets()
        enumeration_time_sec = max(
            0.0,
            time.perf_counter() - enumeration_started - scoring_time_sec,
        )
        return (
            enum_status,
            int(emitted_subgraphs),
            int(eligible_subgraphs),
            float(enumeration_time_sec),
            scored_items,
        )

    def subset_feasible_for_diagnostics(subset: Sequence[Tuple[int, int]]) -> bool:
        subset_set = edge_set(subset)
        if not subset_set.issubset(candidate_edge_set):
            return False
        if len(subset_set) not in set(positive_valid_sizes):
            return False
        if str(row.get("connectivity")) != "cut":
            return True
        if str(row.get("task")) == "node":
            return root_node is not None and rooted_connected(int(root_node), subset_set)
        return pattern_edges_connected(subset_set)

    use_embedding_enumeration = True
    if not positive_valid_sizes:
        use_embedding_enumeration = False
    elif is_node_necessity and root_node is not None:
        use_embedding_enumeration = False
        cache_status = "disabled"
        cache_file: Optional[str] = None
        cache_path: Optional[Path] = None
        cache_metadata: Optional[Dict[str, Any]] = None
        cached_manifest: Optional[Dict[str, Any]] = None
        if use_enumeration_cache and enumeration_cache_dir is not None:
            cache_metadata = node_enumeration_cache_metadata(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                valid_sizes=positive_valid_sizes,
                match_features=True,
                cache_kind="node_necessity",
            )
            cache_path = node_enumeration_cache_path(
                cache_dir=Path(enumeration_cache_dir),
                metadata=cache_metadata,
                cache_kind="node_necessity",
            )
            cache_file = output_path_string(cache_path)
            cache_status, cached_manifest = load_graph_necessity_enumeration_manifest(
                cache_path=cache_path,
                metadata=cache_metadata,
            )

        concrete_mode = (
            "rooted"
            if str(row.get("connectivity")) in {"cut", "flow"}
            else "all"
        )
        (
            enum_status,
            enumerated_subgraphs,
            eligible_subgraphs,
            enumeration_time_sec,
            concrete_necessity_scored_items,
        ) = enumerate_and_score_concrete_subsets(
            mode=concrete_mode,
            concrete_root_node=int(root_node),
        )
        if enum_status == "skipped_enumeration_too_large":
            return SimpleNamespace(
                status=enum_status,
                selected_edges=[],
                rho_optimal=None,
                sufficiency_raw=None,
                necessity_raw=None,
                objective_value=None,
                enumerated_subgraphs=enumerated_subgraphs,
                enumerated_signatures=0,
                enumeration_pruned=0,
                enumeration_cache_status=cache_status,
                enumeration_cache_file=cache_file,
            )
        manifest_matches = bool(
            cache_status == "hit"
            and cached_manifest is not None
            and int(cached_manifest["enumerated_subgraphs"]) == enumerated_subgraphs
            and int(cached_manifest["eligible_subgraphs"]) == eligible_subgraphs
        )
        if cache_path is not None and cache_metadata is not None and not manifest_matches:
            prior_status = cache_status
            saved = save_graph_necessity_enumeration_manifest(
                cache_path=cache_path,
                metadata=cache_metadata,
                enumerated_subgraphs=enumerated_subgraphs,
                eligible_subgraphs=eligible_subgraphs,
                enumeration_time_sec=enumeration_time_sec,
            )
            cache_status = f"{prior_status}_saved" if saved else f"{prior_status}_save_failed"

        enumeration_cache_status = cache_status
        enumeration_cache_file = cache_file
    elif row["task"] == "node" and root_node is not None:
        remaining_max_subgraphs = max(0, int(max_subgraphs) - int(enumerated_subgraphs))
        fixed_self_loop_counts = fixed_self_loop_counts_from_edge_index(
            edge_index,
            num_nodes=int(np.asarray(x).shape[0]),
        )
        node_sufficiency_node_colors = graph_structural_node_colors(
            x=x,
            fixed_self_loop_counts=fixed_self_loop_counts,
        )
        match_features = True
        cache_status: Optional[str] = None
        cache_file: Optional[str] = None
        cache_path: Optional[Path] = None
        cache_metadata: Optional[Dict[str, Any]] = None
        representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        positive_enumerated_subgraphs = 0
        enum_status: Optional[str] = None
        if (
            use_enumeration_cache
            and enumeration_cache_dir is not None
            and objective in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}
        ):
            cache_metadata = node_enumeration_cache_metadata(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                valid_sizes=positive_valid_sizes,
                match_features=match_features,
                cache_kind="node_sufficiency",
                fixed_self_loop_counts=fixed_self_loop_counts,
            )
            cache_path = node_enumeration_cache_path(
                cache_dir=Path(enumeration_cache_dir),
                metadata=cache_metadata,
                cache_kind="node_sufficiency",
            )
            cache_file = output_path_string(cache_path)
            cache_status, cached_count, representatives = load_node_enumeration_cache(
                cache_path=cache_path,
                metadata=cache_metadata,
            )
            if cache_status == "hit" and cached_count is not None:
                positive_enumerated_subgraphs = int(cached_count)
            else:
                representatives = {}

        if cache_status != "hit":
            if str(row.get("connectivity")) == "cut":
                enum_status, positive_enumerated_subgraphs, representatives = node_sufficiency_shape_representatives(
                    x=x,
                    candidate_edges=candidate_edges,
                    node_colors=node_sufficiency_node_colors,
                    gt_edges=gt_edges,
                    root_node=int(root_node),
                    valid_sizes=positive_valid_sizes,
                    max_subgraphs=remaining_max_subgraphs,
                    edge_budget_mode=str(row["edge_budget_mode"]),
                    exclude_gt_solution=exclude_gt_solution,
                )
            else:
                enum_status, positive_enumerated_subgraphs, representatives = unrestricted_node_edge_representatives(
                    node_colors=node_sufficiency_node_colors,
                    candidate_edges=candidate_edges,
                    root_node=int(root_node),
                    valid_sizes=positive_valid_sizes,
                    max_subgraphs=remaining_max_subgraphs,
                    excluded_edge_sets=[gt_edges] if exclude_gt_solution else None,
                )
        enumerated_subgraphs += int(positive_enumerated_subgraphs)
        if enum_status == "skipped_enumeration_too_large":
            return SimpleNamespace(
                status="skipped_enumeration_too_large",
                selected_edges=[],
                rho_optimal=None,
                objective_value=None,
                enumerated_subgraphs=enumerated_subgraphs,
                enumerated_signatures=len(representatives),
                enumeration_pruned=max(0, enumerated_subgraphs - len(representatives)),
                enumeration_cache_status=cache_status,
                enumeration_cache_file=cache_file,
            )
        if enum_status is None:
            if cache_path is not None and cache_metadata is not None and cache_status != "hit":
                saved = save_node_enumeration_cache(
                    cache_path=cache_path,
                    metadata=cache_metadata,
                    enumerated_subgraphs=enumerated_subgraphs,
                    representatives=representatives,
                )
                cache_status = f"{cache_status}_saved" if saved else f"{cache_status}_save_failed"
            use_embedding_enumeration = False
            score_representatives_batch(list(representatives.items()))
            enumeration_cache_status = cache_status
            enumeration_cache_file = cache_file
    elif row["task"] == "graph" and is_graph_necessity:
        use_embedding_enumeration = False
        cache_status = "disabled"
        cache_file: Optional[str] = None
        cache_path: Optional[Path] = None
        cache_metadata: Optional[Dict[str, Any]] = None
        cached_manifest: Optional[Dict[str, Any]] = None
        if use_enumeration_cache and enumeration_cache_dir is not None:
            cache_metadata = graph_necessity_enumeration_cache_metadata(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                valid_sizes=positive_valid_sizes,
            )
            cache_path = graph_necessity_enumeration_cache_path(
                cache_dir=Path(enumeration_cache_dir),
                metadata=cache_metadata,
            )
            cache_file = output_path_string(cache_path)
            cache_status, cached_manifest = load_graph_necessity_enumeration_manifest(
                cache_path=cache_path,
                metadata=cache_metadata,
            )
            if cache_status == "hit" and cached_manifest is not None:
                enumerated_subgraphs = int(cached_manifest["enumerated_subgraphs"])
                if enumerated_subgraphs > int(max_subgraphs):
                    return SimpleNamespace(
                        status="skipped_enumeration_too_large",
                        selected_edges=[],
                        rho_optimal=None,
                        sufficiency_raw=None,
                        necessity_raw=None,
                        objective_value=None,
                        enumerated_subgraphs=enumerated_subgraphs,
                        enumerated_signatures=0,
                        enumeration_pruned=0,
                        enumeration_cache_status="hit_too_large",
                        enumeration_cache_file=cache_file,
                    )

        concrete_mode = (
            "connected"
            if str(row.get("connectivity")) in {"cut", "flow"}
            else "all"
        )
        (
            enum_status,
            enumerated_subgraphs,
            eligible_subgraphs,
            enumeration_time_sec,
            concrete_necessity_scored_items,
        ) = enumerate_and_score_concrete_subsets(
            mode=concrete_mode,
            concrete_root_node=None,
        )
        if enum_status == "skipped_enumeration_too_large":
            return SimpleNamespace(
                status=enum_status,
                selected_edges=[],
                rho_optimal=None,
                sufficiency_raw=None,
                necessity_raw=None,
                objective_value=None,
                enumerated_subgraphs=enumerated_subgraphs,
                enumerated_signatures=0,
                enumeration_pruned=0,
                enumeration_cache_status=cache_status,
                enumeration_cache_file=cache_file,
            )
        manifest_matches = bool(
            cache_status == "hit"
            and cached_manifest is not None
            and int(cached_manifest["enumerated_subgraphs"]) == enumerated_subgraphs
            and int(cached_manifest["eligible_subgraphs"]) == eligible_subgraphs
        )
        if cache_path is not None and cache_metadata is not None and not manifest_matches:
            prior_status = cache_status
            saved = save_graph_necessity_enumeration_manifest(
                cache_path=cache_path,
                metadata=cache_metadata,
                enumerated_subgraphs=enumerated_subgraphs,
                eligible_subgraphs=eligible_subgraphs,
                enumeration_time_sec=enumeration_time_sec,
            )
            cache_status = f"{prior_status}_saved" if saved else f"{prior_status}_save_failed"

        enumeration_cache_status = cache_status
        enumeration_cache_file = cache_file
    elif row["task"] == "graph":
        pattern_first_solution = run_graph_homogeneous_pattern_first_enumeration(
            row=row,
            model=model,
            x=x,
            edge_index=edge_index,
            candidate_edges=candidate_edges,
            gt_edges=gt_edges,
            max_subgraphs=int(max_subgraphs),
            enumeration_batch_size=int(enumeration_batch_size),
            use_enumeration_cache=use_enumeration_cache,
            enumeration_cache_dir=enumeration_cache_dir,
        )
        if pattern_first_solution is not None:
            return pattern_first_solution
        fixed_self_loop_counts = fixed_self_loop_counts_from_edge_index(
            edge_index,
            num_nodes=int(np.asarray(x).shape[0]),
        )
        graph_sufficiency_node_colors = graph_structural_node_colors(
            x=x,
            fixed_self_loop_counts=fixed_self_loop_counts,
        )
        connected_only = str(row.get("connectivity")) == "cut"
        excluded_edge_sets = [gt_edges] if exclude_gt_solution else None
        enum_cache_status: Optional[str] = None
        enum_cache_file: Optional[str] = None
        representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        enum_status: Optional[str] = None
        if (
            use_enumeration_cache
            and enumeration_cache_dir is not None
            and str(row.get("oracle_objective"))
            in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}
        ):
            cache_metadata = graph_sufficiency_colored_cache_metadata(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                valid_sizes=positive_valid_sizes,
                fixed_self_loop_counts=fixed_self_loop_counts,
                connected_only=connected_only,
            )
            cache_path = graph_sufficiency_colored_cache_path(
                cache_dir=Path(enumeration_cache_dir),
                metadata=cache_metadata,
            )
            enum_cache_file = output_path_string(cache_path)
            enum_cache_status, cached_count, representatives = load_graph_sufficiency_colored_cache(
                cache_path=cache_path,
                metadata=cache_metadata,
            )
            if enum_cache_status == "hit" and cached_count is not None:
                enumerated_subgraphs = int(cached_count)
                if enumerated_subgraphs > int(max_subgraphs):
                    return SimpleNamespace(
                        status="skipped_enumeration_too_large",
                        selected_edges=[],
                        rho_optimal=None,
                        objective_value=None,
                        enumerated_subgraphs=enumerated_subgraphs,
                        enumerated_signatures=len(representatives),
                        enumeration_pruned=max(0, enumerated_subgraphs - len(representatives)),
                        enumeration_cache_status="hit_too_large",
                        enumeration_cache_file=enum_cache_file,
                    )
            else:
                representatives = {}

        if enum_cache_status != "hit":
            enum_status, enumerated_subgraphs, representatives = graph_edge_mask_shape_first_representatives(
                x=x,
                candidate_edges=candidate_edges,
                node_colors=graph_sufficiency_node_colors,
                valid_sizes=positive_valid_sizes,
                max_subgraphs=int(max_subgraphs),
                connected_only=connected_only,
                excluded_edge_sets=excluded_edge_sets,
            )
        if enum_status == "fallback_connected_topology":
            enum_status, enumerated_subgraphs, representatives = graph_shape_first_representatives(
                x=x,
                candidate_edges=candidate_edges,
                node_colors=graph_sufficiency_node_colors,
                valid_sizes=positive_valid_sizes,
                max_subgraphs=int(max_subgraphs),
                excluded_edge_sets=excluded_edge_sets,
            )
        if enum_status == "skipped_enumeration_too_large":
            return SimpleNamespace(
                status="skipped_enumeration_too_large",
                selected_edges=[],
                rho_optimal=None,
                objective_value=None,
                enumerated_subgraphs=enumerated_subgraphs,
                enumerated_signatures=len(representatives),
                enumeration_pruned=max(0, enumerated_subgraphs - len(representatives)),
                enumeration_cache_status=enum_cache_status,
                enumeration_cache_file=enum_cache_file,
            )
        if enum_status is None:
            if (
                enum_cache_status is not None
                and enum_cache_status != "hit"
                and str(row.get("oracle_objective"))
                in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}
            ):
                saved = save_graph_sufficiency_colored_cache(
                    cache_path=cache_path,
                    metadata=cache_metadata,
                    enumerated_subgraphs=enumerated_subgraphs,
                    representatives=representatives,
                )
                enum_cache_status = f"{enum_cache_status}_saved" if saved else f"{enum_cache_status}_save_failed"
            use_embedding_enumeration = False
            score_representatives_batch(list(representatives.items()))
            enumeration_cache_status = enum_cache_status
            enumeration_cache_file = enum_cache_file

    if use_embedding_enumeration:
        pending_representatives: Dict[str, Tuple[Tuple[int, int], ...]] = {}

        def flush_pending_representatives() -> None:
            if not pending_representatives:
                return
            score_representatives_batch(list(pending_representatives.items()))
            pending_representatives.clear()

        subset_iter = (
            connected_edge_subsets(
                candidate_edges,
                valid_sizes=positive_valid_sizes,
                root_node=root_node,
            )
            if str(row.get("connectivity")) == "cut"
            else arbitrary_edge_subsets(candidate_edges, valid_sizes=positive_valid_sizes)
        )
        for subset in subset_iter:
            enumerated_subgraphs += 1
            if enumerated_subgraphs > int(max_subgraphs):
                unique_count = len(signatures) + len(pending_representatives)
                return SimpleNamespace(
                    status="skipped_enumeration_too_large",
                    selected_edges=[],
                    rho_optimal=None,
                    objective_value=None,
                    enumerated_subgraphs=enumerated_subgraphs,
                    enumerated_signatures=unique_count,
                    enumeration_pruned=max(0, enumerated_subgraphs - unique_count),
                )

            if str(row.get("task")) == "node" and root_node is not None:
                if node_sufficiency_node_colors is None:
                    fixed_self_loop_counts = fixed_self_loop_counts_from_edge_index(
                        edge_index,
                        num_nodes=int(np.asarray(x).shape[0]),
                    )
                    node_sufficiency_node_colors = graph_structural_node_colors(
                        x=x,
                        fixed_self_loop_counts=fixed_self_loop_counts,
                    )
                signature = exact_rooted_colored_edge_subset_signature(
                    selected_edges=subset,
                    node_colors=node_sufficiency_node_colors,
                    root_node=int(root_node),
                )
            elif graph_sufficiency_node_colors is not None:
                signature = exact_colored_edge_subset_signature(
                    selected_edges=subset,
                    node_colors=graph_sufficiency_node_colors,
                )
            else:
                raise RuntimeError(
                    "Exact colored enumeration signature was not initialized."
                )
            normalized_subset = tuple(sorted(edge_set(subset)))
            existing = signatures.get(signature)
            if existing is not None:
                if normalized_subset < existing["edges"]:
                    existing["edges"] = normalized_subset
                continue
            pending = pending_representatives.get(signature)
            if pending is not None:
                if normalized_subset < pending:
                    pending_representatives[signature] = normalized_subset
                continue

            pending_representatives[signature] = normalized_subset
            if len(pending_representatives) >= int(enumeration_batch_size):
                flush_pending_representatives()

        flush_pending_representatives()

    scored_values = (
        concrete_necessity_scored_items
        if concrete_necessity_scored_items is not None
        else list(signatures.values())
    )
    if not scored_values:
        return SimpleNamespace(
                status="skipped_empty_candidate_set",
                selected_edges=[],
                rho_optimal=None,
                sufficiency_raw=None,
                necessity_raw=None,
                objective_value=None,
                enumerated_subgraphs=enumerated_subgraphs,
                enumerated_signatures=0,
                enumeration_pruned=0,
                enumeration_cache_status=enumeration_cache_status,
                enumeration_cache_file=enumeration_cache_file,
        )

    rank_scope = (
        "instance_feasible_concrete_edge_subsets"
        if uses_necessity_scoring
        else (
            "instance_feasible_exact_rooted_colored_component_representatives"
            if str(row.get("task")) == "node"
            and str(row.get("connectivity")) == "none"
            else (
                "instance_feasible_colored_topology_representatives"
                if str(row.get("task")) == "graph"
                else "instance_feasible_exact_rooted_colored_topology_representatives"
            )
        )
    )
    gt_subset = tuple(sorted(edge_set(gt_edges)))
    gt_item = score_subset_values(gt_subset) if subset_feasible_for_diagnostics(gt_subset) else None
    gt_candidate_present = False
    if gt_item is not None:
        if uses_necessity_scoring:
            gt_candidate_present = not exclude_gt_solution
        else:
            gt_evaluation_edges = (
                tuple(sorted(candidate_edge_set - gt_edge_set))
                if uses_necessity_scoring
                else gt_subset
            )
            if (
                str(row.get("task")) == "node"
                and not uses_necessity_scoring
                and str(row.get("connectivity")) == "none"
                and root_node is not None
            ):
                gt_signature = node_sufficiency_root_component_signature(
                    node_colors=node_sufficiency_node_colors,
                    selected_edges=gt_evaluation_edges,
                    root_node=int(root_node),
                )
            else:
                if str(row.get("task")) == "graph" and graph_sufficiency_node_colors is not None:
                    gt_signature = exact_colored_edge_subset_signature(
                        selected_edges=gt_evaluation_edges,
                        node_colors=graph_sufficiency_node_colors,
                    )
                elif str(row.get("task")) == "node" and root_node is not None:
                    if node_sufficiency_node_colors is None:
                        raise RuntimeError(
                            "Node Sufficiency exact colors were not initialized."
                        )
                    gt_signature = exact_rooted_colored_edge_subset_signature(
                        selected_edges=gt_evaluation_edges,
                        node_colors=node_sufficiency_node_colors,
                        root_node=int(root_node),
                    )
                else:
                    raise RuntimeError(
                        "Exact colored GT signature was not initialized."
                    )
            gt_candidate_present = gt_signature in signatures
    diagnostics = padding_aware_enumeration_diagnostics(
        scored_values,
        gt_item=gt_item,
        row=row,
        candidate_edges=candidate_edges,
        rank_scope=rank_scope,
        gt_candidate_present=gt_candidate_present,
    )

    if objective in RANK_OBJECTIVES and gt_item is not None:
        best = gt_item
    elif objective == "min_sufficiency":
        best = min(scored_values, key=lambda item: (item["objective"], len(item["edges"]), item["edges"]))
    elif objective == "max_necessity":
        best = min(scored_values, key=lambda item: (-item["objective"], len(item["edges"]), item["edges"]))
    elif objective == "min_necessity":
        best = min(scored_values, key=lambda item: (item["objective"], len(item["edges"]), item["edges"]))
    else:
        best = min(scored_values, key=lambda item: (-item["objective"], len(item["edges"]), item["edges"]))

    return SimpleNamespace(
        status="optimal",
        selected_edges=list(best["edges"]),
        rho_optimal=float(best["rho"]),
        sufficiency_raw=float(best["sufficiency_raw"]),
        necessity_raw=to_float_or_none(best.get("necessity_raw")),
        objective_value=float(best["objective"]),
        final_mip_gap=0.0,
        bnb_node_count=0,
        enumerated_subgraphs=enumerated_subgraphs,
        enumerated_signatures=len(scored_values),
        enumeration_pruned=max(0, enumerated_subgraphs - len(scored_values)),
        enumeration_cache_status=enumeration_cache_status,
        enumeration_cache_file=enumeration_cache_file,
        **diagnostics,
    )


def row_padding_seed(base_seed: int, row: Dict[str, Any]) -> int:
    parts = [
        str(base_seed),
        str(row.get("dataset", "")),
        str(row.get("nn_seed", "")),
        str(row.get("instance_id", "")),
        str(row.get("stage", "")),
        str(row.get("budget_k", "")),
        str(row.get("budget_multiplier", "")),
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def pad_node_gt_size_edges(
    *,
    row: Dict[str, Any],
    padding_seed: int,
    root: int,
    selected_edges: Sequence[Tuple[int, int]],
    candidate_edges: Sequence[Tuple[int, int]],
    target_size: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    selected = sorted(edge_set(selected_edges))
    target_size = max(0, int(target_size))
    needed = max(0, target_size - len(selected))
    forbidden_nodes = {int(root)}
    for a, b in selected:
        forbidden_nodes.add(int(a))
        forbidden_nodes.add(int(b))

    info: Dict[str, Any] = {
        "padding_policy": "random_node_disjoint",
        "padding_seed": None,
        "padding_needed": needed,
        "padding_candidate_size": 0,
        "padding_added": 0,
        "padding_forbidden_node_count": len(forbidden_nodes),
        "padding_status": "not_needed" if needed == 0 else None,
    }
    if needed <= 0:
        return selected, info

    selected_set = set(selected)
    disconnected_candidates = [
        edge
        for edge in sorted(edge_set(candidate_edges) - selected_set)
        if int(edge[0]) not in forbidden_nodes and int(edge[1]) not in forbidden_nodes
    ]
    seed = row_padding_seed(padding_seed, row)
    info["padding_seed"] = seed
    info["padding_candidate_size"] = len(disconnected_candidates)

    rng = random.Random(seed)
    if len(disconnected_candidates) <= needed:
        sampled = list(disconnected_candidates)
    else:
        sampled = rng.sample(disconnected_candidates, needed)
    sampled = sorted(sampled)

    info["padding_added"] = len(sampled)
    info["padding_status"] = (
        "ok" if len(sampled) == needed else "insufficient_disconnected_candidates"
    )
    return sorted(selected + sampled), info


def write_edges_csv(path: Path, edges: Sequence[Tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst"])
        for a, b in sorted(edge_set(edges)):
            writer.writerow([a, b])


def output_path_string(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_DIR.resolve()))
    except ValueError:
        return str(resolved)


def probability_from_binary_margin(value: Any) -> Optional[float]:
    margin = to_float_or_none(value)
    if margin is None:
        return None
    if margin >= 0.0:
        return float(1.0 / (1.0 + math.exp(-margin)))
    exp_margin = math.exp(margin)
    return float(exp_margin / (1.0 + exp_margin))


def standard_enumeration_fidelity_values(
    *,
    p_full: float,
    p_masked: float,
    necessity: bool,
) -> Tuple[float, float]:
    """Return (reported fidelity, oriented score maximized by the search).

    Necessity uses Fid+ directly. Sufficiency reports Fid- while using -Fid-
    as its oriented score, so maximizing the score remains equivalent to
    maximizing retained sufficiency.
    """
    fidelity_value = float(p_full) - float(p_masked)
    oriented_score = fidelity_value if necessity else -fidelity_value
    return fidelity_value, oriented_score


def enumeration_extreme_fidelity_value(
    *,
    row: Dict[str, Any],
    solution: Any,
    label: str,
) -> Optional[float]:
    """Read exact enumerated fidelity, with a binary-margin legacy fallback."""
    scored_fidelity = to_float_or_none(
        getattr(solution, f"enum_fidelity_at_objective_{label}", None)
    )
    if scored_fidelity is not None:
        return scored_fidelity
    if int(row.get("num_classes") or 0) != 2:
        return None
    p_full = probability_from_binary_margin(row.get("bar_m"))
    masked_margin = getattr(solution, f"enum_sufficiency_raw_at_objective_{label}", None)
    p_masked = probability_from_binary_margin(masked_margin)
    if p_full is None or p_masked is None:
        return None
    return float(p_full - p_masked)


def gt_rank_scope_label(scope: Any) -> str:
    return {
        "full_edge_subsets": "full subsets",
        "automorphism_representatives": "automorphism-reduced reps",
        "root_component_representatives": "v*-component reps",
        "instance_feasible_concrete_edge_subsets": "instance-feasible concrete subsets",
        "instance_feasible_automorphism_representatives": "instance-feasible automorphism reps",
        "instance_feasible_colored_topology_representatives": "instance-feasible colored-topology reps",
        "instance_feasible_root_component_representatives": "instance-feasible v*-component reps",
        "instance_feasible_exact_rooted_colored_component_representatives": (
            "instance-feasible exact rooted-colored v*-component reps"
        ),
        "instance_feasible_exact_rooted_colored_topology_representatives": (
            "instance-feasible exact rooted-colored topology reps"
        ),
    }.get(str(scope), str(scope or "candidates"))





def save_enumeration_extreme_artifacts(
    *,
    row: Dict[str, Any],
    solution: Any,
    config: RunConfig,
    output_dir: Path,
    x: np.ndarray,
    edge_index: np.ndarray,
    gt_edges: Sequence[Tuple[int, int]],
    root_node: Optional[int],
    node_labels: Optional[np.ndarray],
    terminal_log_file: Path,
) -> None:
    if row.get("solver_backend") != "enumeration" or row.get("status") != "optimal":
        return
    if row.get("oracle_objective") not in (
        SUFFICIENCY_OBJECTIVES
        | NECESSITY_OBJECTIVES
        | RANK_OBJECTIVES
    ):
        return

    row_id = int(row["row_id"])
    num_nodes = int(np.asarray(x).shape[0])
    l_hop = int(row["k_hop"]) if row.get("k_hop") is not None else int(config.num_layers)
    dataset = str(row["dataset"])
    target_node = int(root_node) if root_node is not None else None
    metric_label = (
        "necessity"
        if str(row.get("oracle_objective")) in NECESSITY_SCORING_OBJECTIVES
        else "sufficiency"
    )
    primary_extreme = (
        "max"
        if str(row.get("oracle_objective")) in {"max_sufficiency", "max_necessity"}
        else (
            "min"
            if str(row.get("oracle_objective")) in {"min_sufficiency", "min_necessity"}
            else None
        )
    )
    extreme_rows: List[Dict[str, Any]] = []

    for label, attr_name, edge_field, figure_field in (
        ("max", "enum_objective_max_edges", "enum_max_edge_file", "enum_max_figure_file"),
        ("min", "enum_objective_min_edges", "enum_min_edge_file", "enum_min_figure_file"),
    ):
        edges = getattr(solution, attr_name, None)
        if edges is None:
            continue
        normalized_edges = sorted(edge_set(edges))
        metrics = compute_metrics(normalized_edges, gt_edges)
        fidelity_value = enumeration_extreme_fidelity_value(
            row=row,
            solution=solution,
            label=label,
        )
        row[f"enum_fidelity_at_objective_{label}"] = fidelity_value
        rank_position = row.get(
            "gt_rank_max_objective" if label == "max" else "gt_rank_min_objective"
        )
        rank_direction = "descending" if label == "max" else "ascending"
        print_and_log(
            (
                f"    Enum {label} metrics: size={len(normalized_edges)}, "
                f"D={format_float(metrics['D'], 3)}, H={format_float(metrics['H'], 3)}, "
                f"Precision={format_float(metrics['precision'], 3)}, "
                f"Recall={format_float(metrics['recall'], 3)}, F1={format_float(metrics['f1'], 3)}"
            ),
            terminal_log_file,
        )
        seed = int(row.get("nn_seed", 0))
        edge_path = (
            output_dir
            / "edges"
            / f"explanations_{label}"
            / f"oracle_seed{seed}_row{row_id:06d}.csv"
        )
        write_edges_csv(edge_path, normalized_edges)
        row[edge_field] = output_path_string(edge_path)

        figure_file = ""

        extreme_rows.append(
            {
                "extreme": label,
                "metric": metric_label,
                "oracle_objective": str(row.get("oracle_objective")),
                "objective_scale": "oriented_probability_fidelity",
                "objective_value": to_float_or_none(
                    getattr(solution, f"enum_objective_{label}", None)
                ),
                "rho": to_float_or_none(
                    getattr(solution, f"enum_rho_at_objective_{label}", None)
                ),
                "sufficiency_raw": to_float_or_none(
                    getattr(solution, f"enum_sufficiency_raw_at_objective_{label}", None)
                ),
                "necessity_raw": to_float_or_none(
                    getattr(solution, f"enum_necessity_raw_at_objective_{label}", None)
                ),
                "fidelity_metric": "Fid+" if metric_label == "necessity" else "Fid-",
                "fidelity_value": fidelity_value,
                "gt_fidelity_value": to_float_or_none(
                    row.get(
                        "gt_fidelity_plus"
                        if metric_label == "necessity"
                        else "gt_fidelity_minus"
                    )
                ),
                "gt_rank_position": rank_position,
                "gt_rank_total": row.get("gt_rank_total"),
                "gt_rank_tie_count": row.get("gt_rank_tie_count"),
                "gt_rank_direction": rank_direction,
                "gt_rank_scope": row.get("gt_rank_scope"),
                "selected_size": len(normalized_edges),
                "D": metrics["D"],
                "H": metrics["H"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "edge_file": output_path_string(edge_path),
                "figure_file": figure_file,
            }
        )

    if extreme_rows:
        extrema_path = (
            output_dir
            / "edges"
            / "enumeration_extrema"
            / f"extrema_seed{int(row.get('nn_seed', 0))}_row{row_id:06d}.csv"
        )
        write_csv(
            extrema_path,
            extreme_rows,
            [
                "extreme",
                "metric",
                "oracle_objective",
                "objective_scale",
                "objective_value",
                "rho",
                "sufficiency_raw",
                "necessity_raw",
                "fidelity_metric",
                "fidelity_value",
                "gt_fidelity_value",
                "gt_rank_position",
                "gt_rank_total",
                "gt_rank_tie_count",
                "gt_rank_direction",
                "gt_rank_scope",
                "selected_size",
                "D",
                "H",
                "precision",
                "recall",
                "f1",
                "edge_file",
                "figure_file",
            ],
        )
        row["enum_extrema_file"] = output_path_string(extrema_path)


_STDOUT_BROKEN = False


def print_and_log(msg: str, terminal_log_file: Optional[Path] = None) -> None:
    """Print a line and optionally append the same line to a terminal log."""
    if terminal_log_file is None:
        pass
    else:
        try:
            terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
            with terminal_log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    global _STDOUT_BROKEN
    if _STDOUT_BROKEN:
        return
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        _STDOUT_BROKEN = True


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or str(value) == "":
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def format_seconds(value: Any, digits: int = 4) -> str:
    rendered = format_float(value, digits)
    return rendered if rendered == "N/A" else f"{rendered}s"


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def metric_row(row: Dict[str, Any]) -> bool:
    if row.get("status") not in SUCCESS_STATUSES:
        return False
    if row.get("D") is None:
        return False
    if row.get("oracle_objective") in DELTA_OBJECTIVES:
        return row.get("fidelity_delta") is not None
    if row.get("rho_optimal") is None:
        return False
    return True


def mean_or_none(values: Sequence[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and str(v) != "" and math.isfinite(float(v))]
    if not nums:
        return None
    return float(np.mean(nums))


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    include_fn=None,
    sweep_only: bool = False,
) -> List[Dict[str, Any]]:
    filtered = []
    for row in rows:
        if sweep_only and row.get("stage") != "sweep":
            continue
        if include_fn is not None and not include_fn(row):
            continue
        filtered.append(row)

    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    if sweep_only:
        key_fields = [
            "dataset",
            "task",
            "stage",
            "budget_multiplier",
            "budget_k",
            "connectivity",
            "edge_budget_mode",
            "oracle_objective",
            "solver_backend",
            "gt_incumbent",
            "exclude_gt_solution",
            *SUMMARY_ACCEL_FIELDS,
        ]
    else:
        key_fields = [
            "dataset",
            "task",
            "stage",
            "budget_multiplier",
            "budget_k",
            "budget_mode",
            "connectivity",
            "edge_budget_mode",
            "oracle_objective",
            "solver_backend",
            "gt_incumbent",
            "exclude_gt_solution",
            *SUMMARY_ACCEL_FIELDS,
        ]
    for row in filtered:
        key = tuple(row.get(field) for field in key_fields)
        groups.setdefault(key, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple("" if v is None else str(v) for v in item[0])):
        solved = [row for row in group_rows if metric_row(row)]
        statuses = Counter(str(row.get("status")) for row in group_rows)
        summary = {field: value for field, value in zip(key_fields, key)}
        summary.update(
            {
                "total_rows": len(group_rows),
                "metric_rows": len(solved),
                "optimal": statuses.get("optimal", 0),
                "time_limit": statuses.get("time_limit", 0),
                "skipped_incorrect_prediction": statuses.get("skipped_incorrect_prediction", 0),
                "skipped_nonpositive_margin": statuses.get("skipped_nonpositive_margin", 0),
                "skipped_nonbinary_min_sufficiency": statuses.get("skipped_nonbinary_min_sufficiency", 0),
                "skipped_nonbinary_necessity_milp": statuses.get("skipped_nonbinary_necessity_milp", 0),
                "skipped_nonbinary_delta_milp": statuses.get("skipped_nonbinary_delta_milp", 0),
                "skipped_delta_requires_milp": statuses.get("skipped_delta_requires_milp", 0),
                "skipped_delta_requires_exact_budget": statuses.get("skipped_delta_requires_exact_budget", 0),
                "skipped_necessity_graph_only": statuses.get("skipped_necessity_graph_only", 0),
                "skipped_necessity_requires_enumeration": statuses.get("skipped_necessity_requires_enumeration", 0),
                "skipped_no_ground_truth": statuses.get("skipped_no_ground_truth", 0),
                "skipped_empty_candidate_set": statuses.get("skipped_empty_candidate_set", 0),
                "skipped_invalid_budget": statuses.get("skipped_invalid_budget", 0),
                "skipped_enumeration_too_large": statuses.get("skipped_enumeration_too_large", 0),
                "skipped_padding_infeasible": statuses.get("skipped_padding_infeasible", 0),
                "infeasible": statuses.get("infeasible", 0),
                "interrupted_by_user": statuses.get("interrupted_by_user", 0),
                "error": sum(count for status, count in statuses.items() if status.startswith("error")),
                "counterexample_evaluable": sum(
                    1 for row in solved if row.get("counterexample_found") is not None
                ),
                "counterexample_found": sum(
                    1 for row in solved if row.get("counterexample_found") is True
                ),
                "counterexample_exact_tie": sum(
                    1 for row in solved if row.get("counterexample_status") == "exact_tie"
                ),
                "counterexample_gt_better": sum(
                    1 for row in solved if row.get("counterexample_status") == "gt_better"
                ),
                "counterexample_found_rate": (
                    float(
                        sum(1 for row in solved if row.get("counterexample_found") is True)
                        / sum(1 for row in solved if row.get("counterexample_found") is not None)
                    )
                    if any(row.get("counterexample_found") is not None for row in solved)
                    else None
                ),
                "mean_counterexample_advantage": mean_or_none(
                    [row.get("counterexample_advantage") for row in solved]
                ),
                "padding_ok": sum(1 for row in group_rows if row.get("padding_status") == "ok"),
                "padding_insufficient_disconnected_candidates": sum(
                    1 for row in group_rows if row.get("padding_status") == "insufficient_disconnected_candidates"
                ),
                "mean_D": mean_or_none([row.get("D") for row in solved]),
                "mean_H": mean_or_none([row.get("H") for row in solved]),
                "mean_precision": mean_or_none([row.get("precision") for row in solved]),
                "mean_recall": mean_or_none([row.get("recall") for row in solved]),
                "mean_f1": mean_or_none([row.get("f1") for row in solved]),
                "mean_padded_D": mean_or_none([row.get("padded_D") for row in solved]),
                "mean_padded_H": mean_or_none([row.get("padded_H") for row in solved]),
                "mean_padded_precision": mean_or_none([row.get("padded_precision") for row in solved]),
                "mean_padded_recall": mean_or_none([row.get("padded_recall") for row in solved]),
                "mean_padded_f1": mean_or_none([row.get("padded_f1") for row in solved]),
                "mean_padding_needed": mean_or_none([row.get("padding_needed") for row in solved]),
                "mean_padding_candidate_size": mean_or_none([row.get("padding_candidate_size") for row in solved]),
                "mean_padding_added": mean_or_none([row.get("padding_added") for row in solved]),
                "mean_rho_optimal": mean_or_none([row.get("rho_optimal") for row in solved]),
                "mean_sufficiency_raw": mean_or_none([row.get("sufficiency_raw") for row in solved]),
                "mean_necessity_raw": mean_or_none([row.get("necessity_raw") for row in solved]),
                "mean_fidelity_minus": mean_or_none([row.get("fidelity_minus") for row in solved]),
                "mean_fidelity_plus": mean_or_none([row.get("fidelity_plus") for row in solved]),
                "mean_fidelity_delta": mean_or_none([row.get("fidelity_delta") for row in solved]),
                "mean_pwl_approx_error_at_solution": mean_or_none(
                    [row.get("pwl_approx_error_at_solution") for row in solved]
                ),
                "mean_time_s": mean_or_none([row.get("time_s") for row in solved]),
                "mean_enumerated_subgraphs": mean_or_none([row.get("enumerated_subgraphs") for row in solved]),
                "mean_enumerated_signatures": mean_or_none([row.get("enumerated_signatures") for row in solved]),
                "mean_enumeration_pruned": mean_or_none([row.get("enumeration_pruned") for row in solved]),
                "mean_final_mip_gap": mean_or_none([row.get("final_mip_gap") for row in solved]),
                "mean_best_bound": mean_or_none([row.get("best_bound") for row in solved]),
                "mean_incumbent_obj": mean_or_none([row.get("incumbent_obj") for row in solved]),
                "mean_model_size": mean_or_none([row.get("model_size") for row in solved]),
                "mean_root_lp_value": mean_or_none([row.get("root_lp_value") for row in solved]),
                "mean_root_lp_time": mean_or_none([row.get("root_lp_time") for row in solved]),
                "root_lp_callback_rows": sum(
                    1 for row in solved if row.get("root_lp_source") == "mipnode_callback"
                ),
                "root_lp_best_bound_fallback_rows": sum(
                    1 for row in solved if row.get("root_lp_source") == "best_bound_fallback"
                ),
                "mean_callback_time_sec": mean_or_none([row.get("callback_time_sec") for row in solved]),
                "mean_callback_calls": mean_or_none([row.get("callback_calls") for row in solved]),
                "mean_cuts_mipnode": mean_or_none([row.get("cuts_mipnode") for row in solved]),
                "mean_cuts_mipsol_lazy": mean_or_none([row.get("cuts_mipsol_lazy") for row in solved]),
                "mean_anderson_cuts_added": mean_or_none(
                    [row.get("anderson_cuts_added") for row in solved]
                ),
                "mean_connectivity_lazy_cuts_added": mean_or_none(
                    [row.get("connectivity_lazy_cuts_added") for row in solved]
                ),
                "mean_bnb_node_count": mean_or_none([row.get("bnb_node_count") for row in solved]),
            }
        )
        summaries.append(summary)
    return summaries


def summary_fieldnames() -> List[str]:
    return [
        "dataset",
        "task",
        "stage",
        "budget_multiplier",
        "budget_k",
        "budget_mode",
        "connectivity",
        "edge_budget_mode",
        "oracle_objective",
        "solver_backend",
        "gt_incumbent",
        "exclude_gt_solution",
        *SUMMARY_ACCEL_FIELDS,
        "total_rows",
        "metric_rows",
        "optimal",
        "time_limit",
        "skipped_incorrect_prediction",
        "skipped_nonpositive_margin",
        "skipped_nonbinary_min_sufficiency",
        "skipped_nonbinary_necessity_milp",
        "skipped_nonbinary_delta_milp",
        "skipped_delta_requires_milp",
        "skipped_delta_requires_exact_budget",
        "skipped_necessity_graph_only",
        "skipped_necessity_requires_enumeration",
        "skipped_no_ground_truth",
        "skipped_empty_candidate_set",
        "skipped_invalid_budget",
        "skipped_enumeration_too_large",
        "skipped_padding_infeasible",
        "infeasible",
        "interrupted_by_user",
        "error",
        "counterexample_evaluable",
        "counterexample_found",
        "counterexample_exact_tie",
        "counterexample_gt_better",
        "counterexample_found_rate",
        "mean_counterexample_advantage",
        "padding_ok",
        "padding_insufficient_disconnected_candidates",
        "mean_D",
        "mean_H",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_padded_D",
        "mean_padded_H",
        "mean_padded_precision",
        "mean_padded_recall",
        "mean_padded_f1",
        "mean_padding_needed",
        "mean_padding_candidate_size",
        "mean_padding_added",
        "mean_rho_optimal",
        "mean_sufficiency_raw",
        "mean_necessity_raw",
        "mean_fidelity_minus",
        "mean_fidelity_plus",
        "mean_fidelity_delta",
        "mean_pwl_approx_error_at_solution",
        "mean_time_s",
        "mean_enumerated_subgraphs",
        "mean_enumerated_signatures",
        "mean_enumeration_pruned",
        "mean_final_mip_gap",
        "mean_best_bound",
        "mean_incumbent_obj",
        "mean_model_size",
        "mean_root_lp_value",
        "mean_root_lp_time",
        "root_lp_callback_rows",
        "root_lp_best_bound_fallback_rows",
        "mean_callback_time_sec",
        "mean_callback_calls",
        "mean_cuts_mipnode",
        "mean_cuts_mipsol_lazy",
        "mean_anderson_cuts_added",
        "mean_connectivity_lazy_cuts_added",
        "mean_bnb_node_count",
    ]


def sweep_fieldnames() -> List[str]:
    return [
        "dataset",
        "task",
        "stage",
        "budget_multiplier",
        "budget_k",
        "connectivity",
        "edge_budget_mode",
        "oracle_objective",
        "solver_backend",
        "gt_incumbent",
        "exclude_gt_solution",
        *SUMMARY_ACCEL_FIELDS,
        "total_rows",
        "metric_rows",
        "optimal",
        "time_limit",
        "skipped_incorrect_prediction",
        "skipped_nonpositive_margin",
        "skipped_nonbinary_min_sufficiency",
        "skipped_nonbinary_necessity_milp",
        "skipped_nonbinary_delta_milp",
        "skipped_delta_requires_milp",
        "skipped_delta_requires_exact_budget",
        "skipped_necessity_graph_only",
        "skipped_necessity_requires_enumeration",
        "skipped_no_ground_truth",
        "skipped_empty_candidate_set",
        "skipped_invalid_budget",
        "skipped_enumeration_too_large",
        "skipped_padding_infeasible",
        "infeasible",
        "interrupted_by_user",
        "error",
        "counterexample_evaluable",
        "counterexample_found",
        "counterexample_exact_tie",
        "counterexample_gt_better",
        "counterexample_found_rate",
        "mean_counterexample_advantage",
        "padding_ok",
        "padding_insufficient_disconnected_candidates",
        "mean_D",
        "mean_H",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_padded_D",
        "mean_padded_H",
        "mean_padded_precision",
        "mean_padded_recall",
        "mean_padded_f1",
        "mean_padding_needed",
        "mean_padding_candidate_size",
        "mean_padding_added",
        "mean_rho_optimal",
        "mean_sufficiency_raw",
        "mean_necessity_raw",
        "mean_fidelity_minus",
        "mean_fidelity_plus",
        "mean_fidelity_delta",
        "mean_pwl_approx_error_at_solution",
        "mean_time_s",
        "mean_enumerated_subgraphs",
        "mean_enumerated_signatures",
        "mean_enumeration_pruned",
        "mean_final_mip_gap",
        "mean_best_bound",
        "mean_incumbent_obj",
        "mean_model_size",
        "mean_root_lp_value",
        "mean_root_lp_time",
        "root_lp_callback_rows",
        "root_lp_best_bound_fallback_rows",
        "mean_callback_time_sec",
        "mean_callback_calls",
        "mean_cuts_mipnode",
        "mean_cuts_mipsol_lazy",
        "mean_anderson_cuts_added",
        "mean_connectivity_lazy_cuts_added",
        "mean_bnb_node_count",
    ]


def correct_positive_margin(row: Dict[str, Any]) -> bool:
    return bool(row.get("is_correct")) and row.get("bar_m") is not None and float(row["bar_m"]) > 0.0


def write_result_tables(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write per-seed tables using the robust-fidelity seed layout."""
    write_csv(output_dir / "instance_fid.csv", rows, FIELDNAMES)
    write_csv(
        output_dir / "counterexample_results.csv",
        [row for row in rows if row.get("counterexample_status")],
        COUNTEREXAMPLE_RESULT_FIELDS,
    )
    write_csv(
        output_dir / "skipped_targets.csv",
        [row for row in rows if row.get("status") not in SUCCESS_STATUSES],
        FIELDNAMES,
    )


def summarize_rows_by_seed(
    rows: Sequence[Dict[str, Any]],
    *,
    include_fn=None,
    sweep_only: bool = False,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    seeds = sorted(
        {
            int(row["nn_seed"])
            for row in rows
            if row.get("nn_seed") not in (None, "")
        }
    )
    for seed in seeds:
        seed_rows = [
            row
            for row in rows
            if row.get("nn_seed") not in (None, "") and int(row["nn_seed"]) == seed
        ]
        for summary in summarize_rows(seed_rows, include_fn=include_fn, sweep_only=sweep_only):
            summaries.append({"nn_seed": int(seed), **summary})
    return summaries


def counterexample_summary_fields() -> List[str]:
    return [
        "dataset",
        "nn_seed",
        "task",
        "stage",
        "budget_k",
        "oracle_objective",
        "solver_backend",
        "exclude_gt_solution",
        "counterexample_evaluable",
        "counterexample_found",
        "counterexample_exact_tie",
        "counterexample_gt_better",
        "counterexample_found_rate",
        "mean_counterexample_advantage",
    ]


def write_dataset_result_tables(output_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write dataset-run summaries at the same level as robust summaries."""
    fields = summary_fieldnames()
    summary_by_seed_fields = [fields[0], "nn_seed", *fields[1:]]
    summary_by_seed = summarize_rows_by_seed(rows)
    write_csv(output_dir / "summary.csv", summarize_rows(rows), fields)
    write_csv(output_dir / "summary_by_seed.csv", summary_by_seed, summary_by_seed_fields)
    write_csv(
        output_dir / "counterexample_summary_by_seed.csv",
        summary_by_seed,
        counterexample_summary_fields(),
    )
    write_csv(
        output_dir / "summary_correct_positive_margin.csv",
        summarize_rows(rows, include_fn=correct_positive_margin),
        fields,
    )
    write_csv(
        output_dir / "sweep_curves.csv",
        summarize_rows(rows, include_fn=correct_positive_margin, sweep_only=True),
        sweep_fieldnames(),
    )


def seed_result_dir(output_root: Path, run_name: str, dataset: str, nn_seed: int) -> Path:
    return (
        output_root
        / sanitize_run_slug(dataset)
        / dataset_run_name(run_name, dataset)
        / f"seed{int(nn_seed)}"
    )


def dataset_run_name(run_name: str, dataset: str) -> str:
    _, _, task = get_dataset_config(normalize_dataset_name(dataset))
    if task == "node":
        return sanitize_run_slug(run_name)
    parts = [
        part
        for part in sanitize_run_slug(run_name).split("_")
        if not part.startswith("nodebudget-")
    ]
    return sanitize_run_slug("_".join(parts))


def prepare_output_dir(output_dir: Path, save_figures: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "edges").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)


def dataset_task(dataset: str) -> str:
    _, _, task = get_dataset_config(normalize_dataset_name(dataset))
    return str(task)


def effective_seed_config(config: RunConfig, dataset: str) -> Dict[str, Any]:
    seed_config = asdict(config)
    seed_config["mip_time_limit_scope"] = (
        "gurobi_optimize_only_after_preprocessing_and_model_build"
        if config.gt_size_solver == "milp" or "sweep" in config.stages
        else None
    )
    seed_config["gurobi_effective_norel_heur_work"] = effective_gurobi_norel_heur_work(config)
    seed_config["gurobi_norel_hybrid_time_scope"] = (
        "NoRel and subsequent root/B&B share the existing single Gurobi time limit; no model restart"
    )
    seed_config["cap_mode"] = "global"
    if config.connectivity == "none":
        for field in CONNECTIVITY_DISABLED_ACCEL_FIELDS:
            seed_config[field] = False
    return seed_config


def write_dataset_config(config: RunConfig, dataset: str, run_dir: Path) -> None:
    dataset_config = effective_seed_config(config, dataset)
    dataset_config["datasets"] = [dataset]
    dataset_config["run_name"] = run_dir.name
    dataset_config["output_dir"] = str(run_dir.parent)
    dataset_config["counterexample_absolute_tolerance"] = COUNTEREXAMPLE_TOLERANCE
    dataset_config["enumeration_rank_absolute_tolerance"] = ENUMERATION_RANK_TOLERANCE
    dataset_config["enumeration_objective_scale"] = "oriented_probability_fidelity"
    dataset_config["nn_seeds"] = [int(seed) for seed in config.nn_seeds]
    dataset_config["seed_dir_pattern"] = "seed{nn_seed}"
    dataset_config["seed_directories"] = [f"seed{int(seed)}" for seed in config.nn_seeds]
    (run_dir / "config.json").write_text(json.dumps(dataset_config, indent=2), encoding="utf-8")


def write_experiment_record(config: RunConfig, dataset: str, run_dir: Path) -> None:
    """Materialize the prespecified run plan before any dataset evaluation."""
    if config.experiment_template is None:
        return
    template_path = Path(config.experiment_template)
    if not template_path.is_absolute():
        template_path = REPO_DIR / template_path
    if not template_path.is_file():
        raise FileNotFoundError(f"Experiment template does not exist: {template_path}")
    record_path = run_dir / "experiment.md"
    if record_path.exists():
        raise FileExistsError(f"Experiment record already exists: {record_path}")
    template = template_path.read_text(encoding="utf-8").rstrip()
    metadata = (
        "\n\n## Materialized run metadata\n\n"
        f"- Dataset: `{dataset}`\n"
        f"- Run name: `{run_dir.name}`\n"
        f"- Architecture namespace: `{config.model_paper}` (`{config.architecture}`)\n"
        f"- Model shape: `{config.num_layers}L{config.hidden_dim}H`\n"
        f"- Objective: `{config.oracle_objective}`\n"
        f"- Solver: `{config.gt_size_solver}`\n"
        f"- Output directory: `{run_dir}`\n"
    )
    record_path.write_text(template + metadata, encoding="utf-8")


def initialize_seed_output(
    *,
    config: RunConfig,
    dataset: str,
    nn_seed: int,
    seed_dir: Path,
) -> Path:
    prepare_output_dir(seed_dir, config.save_figures)
    if config.use_enumeration_cache:
        Path(config.enumeration_cache_dir).mkdir(parents=True, exist_ok=True)
    seed_config = effective_seed_config(config, dataset)
    run_log_file = seed_dir / "logs" / "run.log"
    run_log_file.write_text("# Main standard-fidelity experiment run log\n", encoding="utf-8")

    print_and_log("=" * 60, run_log_file)
    print_and_log("Main Standard-Fidelity Experiment Configuration", run_log_file)
    print_and_log("=" * 60, run_log_file)
    print_and_log(f"Preset: {config.preset}", run_log_file)
    print_and_log(f"Stages: {config.stages}", run_log_file)
    print_and_log(f"Dataset: {dataset}", run_log_file)
    print_and_log(f"NN seed: {nn_seed}", run_log_file)
    print_and_log(
        f"Architecture: {config.architecture} (checkpoint namespace: {config.model_paper})",
        run_log_file,
    )
    print_and_log(f"Instance filter: {config.instance_filter}", run_log_file)
    print_and_log(
        "Target degree filter: "
        + ("none" if config.min_target_degree is None else f">={config.min_target_degree}"),
        run_log_file,
    )
    print_and_log(f"Budget multipliers: {config.budget_multipliers}", run_log_file)
    print_and_log(f"Time limit: {'inf (unlimited)' if config.time_limit_s is None else f'{config.time_limit_s}s'}", run_log_file)
    print_and_log(
        "MILP time limit scope: Gurobi optimize only (preprocessing, model build and OBBT excluded)",
        run_log_file,
    )
    print_and_log(f"Gurobi parameter overrides: {format_gurobi_param_overrides(config)}", run_log_file)
    print_and_log(
        "Gurobi NoRel hybrid: "
        f"enabled={config.gurobi_norel_hybrid}, "
        f"requested_work={config.gurobi_norel_heur_work}, "
        f"effective_work={effective_gurobi_norel_heur_work(config)}, "
        f"then_continue_same_optimize_to_root_and_B&B={config.gurobi_norel_hybrid}",
        run_log_file,
    )
    if config.metric is not None:
        print_and_log(f"Metric: {config.metric}", run_log_file)
    if config.opt_samples is not None or config.eval_samples is not None:
        print_and_log(
            (
                "Compatibility sampling args ignored by deterministic GT-size/budget-sweep search: "
                f"opt_samples={config.opt_samples}, eval_samples={config.eval_samples}"
            ),
            run_log_file,
        )
    print_and_log(f"Layers: {config.num_layers}, Hidden: {config.hidden_dim}", run_log_file)
    print_and_log(f"K-hop: {'auto(num_layers)' if config.k_hop is None else config.k_hop}", run_log_file)
    print_and_log(f"Connectivity: {config.connectivity}", run_log_file)
    print_and_log(f"GT-size node edge budget mode: {config.node_edge_budget_mode}", run_log_file)
    print_and_log(f"GT-size solver: {config.gt_size_solver}", run_log_file)
    print_and_log(f"Enumeration max subgraphs: {config.enumeration_max_subgraphs}", run_log_file)
    print_and_log(f"Enumeration inference batch size: {config.enumeration_batch_size}", run_log_file)
    print_and_log(
        "Node enumeration inference context: compact candidate-edge support plus root "
        "(global cache/artifact edge IDs preserved)",
        run_log_file,
    )
    print_and_log(f"Enumeration cache: {config.use_enumeration_cache}", run_log_file)
    print_and_log(f"Enumeration cache dir: {config.enumeration_cache_dir}", run_log_file)
    print_and_log(f"Oracle objective: {config.oracle_objective}", run_log_file)
    print_and_log(
        f"Counterexample absolute tolerance: {COUNTEREXAMPLE_TOLERANCE:g}",
        run_log_file,
    )
    if config.gt_size_solver == "enumeration":
        print_and_log(
            "Enumeration output: probability-fidelity max + min + both GT ranks",
            run_log_file,
        )
        print_and_log(
            f"Enumeration rank absolute tolerance: {ENUMERATION_RANK_TOLERANCE:g}",
            run_log_file,
        )
    print_and_log(f"GT incumbent: {config.gt_incumbent}", run_log_file)
    print_and_log(f"Exclude exact GT solution: {config.exclude_gt_solution}", run_log_file)
    print_and_log(f"Acceleration profile: {seed_config['acceleration_profile']}", run_log_file)
    print_and_log(f"O1: {seed_config['use_o1_tightening']}, cap_mode: {seed_config['cap_mode']}", run_log_file)
    print_and_log(
        "O3/Gurobi symmetry: "
        f"{seed_config['use_o3_symmetry_breaking']} "
        f"(Symmetry={2 if seed_config['use_o3_symmetry_breaking'] else 0}, custom hanging-tree O3 disabled)",
        run_log_file,
    )
    print_and_log(f"O4: {seed_config['use_o4_star_constraints']}", run_log_file)
    # O5-O8 are node-specific OR accelerations and are intentionally not used.
    print_and_log(f"O9: {seed_config['use_o9_connectivity_lb']}", run_log_file)
    print_and_log(f"Anderson cuts: {seed_config['use_anderson_cuts']}", run_log_file)
    print_and_log(f"Graph edge branch priority: {config.edge_branch_priority}", run_log_file)
    print_and_log(
        f"Graph OBBT(u): {config.use_obbt_u}, max_layers={config.max_obbt_u_layers}",
        run_log_file,
    )
    print_and_log(f"Graph OBBT two-pass rebuild: {config.obbt_rebuild}", run_log_file)
    print_and_log(f"Graph exact degree-table layer 1: {config.degree_table_layer1}", run_log_file)
    print_and_log(f"Save figures: {config.save_figures}", run_log_file)
    print_and_log(f"Padding seed: {config.padding_seed}", run_log_file)
    print_and_log(f"Verbose/Gurobi console log: {config.verbose}", run_log_file)
    print_and_log(f"Output: {seed_dir}", run_log_file)
    print_and_log("=" * 60, run_log_file)
    return run_log_file


def make_base_row(
    *,
    row_id: int,
    dataset: str,
    task: str,
    nn_seed: int,
    instance_id: int,
    stage: str,
    budget_multiplier: Optional[float],
    budget_k: Optional[int],
    budget_mode: str,
    connectivity: str,
    edge_budget_mode: str,
    k_hop: Optional[int],
    candidate_size: int,
    gt_size: int,
    pred: int,
    label: int,
    c_star: int,
    num_classes: int,
    oracle_objective: str,
    bar_m: float,
) -> Dict[str, Any]:
    return {
        "row_id": row_id,
        "dataset": dataset,
        "task": task,
        "nn_seed": nn_seed,
        "instance_id": instance_id,
        "stage": stage,
        "budget_multiplier": budget_multiplier,
        "budget_k": budget_k,
        "budget_mode": budget_mode,
        "connectivity": connectivity,
        "edge_budget_mode": edge_budget_mode,
        "oracle_objective": oracle_objective,
        "solver_backend": "milp",
        "gt_incumbent": None,
        "exclude_gt_solution": None,
        "gt_solution_representable": None,
        "gt_no_good_cut_added": None,
        "enumerated_subgraphs": None,
        "enumerated_signatures": None,
        "enumeration_pruned": None,
        "enumeration_cache_status": None,
        "enumeration_cache_file": None,
        "k_hop": k_hop,
        "candidate_size": candidate_size,
        "gt_size": gt_size,
        "budget_context_size": max(0, int(candidate_size) - int(gt_size)),
        "status": None,
        "pred": pred,
        "label": label,
        "is_correct": int(pred) == int(label),
        "c_star": c_star,
        "num_classes": num_classes,
        "min_sufficiency_other_class": 1 - int(c_star) if oracle_objective == "min_sufficiency" and int(num_classes) == 2 else None,
        "bar_m": bar_m,
        "rho_optimal": None,
        "sufficiency_raw": None,
        "necessity_raw": None,
        "sufficiency_probability": None,
        "complement_probability": None,
        "fidelity_minus": None,
        "fidelity_plus": None,
        "fidelity_delta": None,
        "objective_value": None,
        "objective_value_true": None,
        "pwl_approx_error_at_solution": None,
        "enum_objective_max": None,
        "enum_objective_min": None,
        "enum_rho_at_objective_max": None,
        "enum_rho_at_objective_min": None,
        "enum_sufficiency_raw_at_objective_max": None,
        "enum_sufficiency_raw_at_objective_min": None,
        "enum_necessity_raw_at_objective_max": None,
        "enum_necessity_raw_at_objective_min": None,
        "enum_fidelity_at_objective_max": None,
        "enum_fidelity_at_objective_min": None,
        "gt_rho": None,
        "gt_sufficiency_raw": None,
        "gt_necessity_raw": None,
        "gt_sufficiency_probability": None,
        "gt_complement_probability": None,
        "gt_fidelity_minus": None,
        "gt_fidelity_plus": None,
        "gt_fidelity_delta": None,
        "gt_objective_value": None,
        "counterexample_advantage": None,
        "counterexample_status": "",
        "counterexample_found": None,
        "counterexample_definition": "",
        "gt_rank_max_objective": None,
        "gt_rank_min_objective": None,
        "gt_rank_position": None,
        "gt_rank_total": None,
        "gt_rank_tie_count": None,
        "gt_rank_direction": "",
        "gt_rank_scope": "",
        "enum_padding_feasible": None,
        "enum_padding_filtered": None,
        "selected_size": None,
        "D": None,
        "H": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "padded_size": None,
        "padded_D": None,
        "padded_H": None,
        "padded_precision": None,
        "padded_recall": None,
        "padded_f1": None,
        "padding_policy": "",
        "padding_seed": None,
        "padding_needed": None,
        "padding_candidate_size": None,
        "padding_added": None,
        "padding_forbidden_node_count": None,
        "padding_status": "",
        "rooted_connected": None,
        "time_s": None,
        "final_mip_gap": None,
        "best_bound": None,
        "incumbent_obj": None,
        "model_size": None,
        "root_lp_value": None,
        "root_lp_time": None,
        "root_lp_source": "",
        "callback_time_sec": None,
        "callback_calls": None,
        "cuts_mipnode": None,
        "cuts_mipsol_lazy": None,
        "anderson_cuts_added": None,
        "connectivity_lazy_cuts_added": None,
        "bnb_node_count": None,
        "gurobi_log": "",
        "terminal_log": "",
        "edge_file": "",
        "gt_edge_file": "",
        "padded_edge_file": "",
        "enum_max_edge_file": "",
        "enum_min_edge_file": "",
        "enum_extrema_file": "",
        "figure_file": "",
        "gt_figure_file": "",
        "enum_max_figure_file": "",
        "enum_min_figure_file": "",
        "error": "",
    }


def acceleration_row_fields(
    config: RunConfig,
    *,
    task: str,
    connectivity: str,
    edge_budget_mode: str,
) -> Dict[str, Any]:
    """Return acceleration/ablation flags stored with each result row."""
    is_node = task == "node"
    uses_connectivity = connectivity != "none"
    node_uses_connectivity = is_node and uses_connectivity
    return {
        "gt_incumbent": bool(config.gt_incumbent),
        "exclude_gt_solution": bool(config.exclude_gt_solution),
        "acceleration_profile": config.acceleration_profile,
        "use_o1_tightening": config.use_o1_tightening,
        "cap_mode": "global",
        "use_o3_symmetry_breaking": config.use_o3_symmetry_breaking,
        "use_o4_star_constraints": (
            config.use_o4_star_constraints if uses_connectivity else False
        ),
        "use_o9_connectivity_lb": (
            config.use_o9_connectivity_lb if uses_connectivity else False
        ),
        "use_anderson_cuts": config.use_anderson_cuts,
        "anderson_max_node_count": config.anderson_max_node_count,
        "anderson_max_cuts": config.anderson_max_cuts,
        "edge_branch_priority": config.edge_branch_priority if task == "graph" else 0,
        "use_obbt_u": bool(config.use_obbt_u) if task == "graph" else False,
        "max_obbt_u_layers": config.max_obbt_u_layers if task == "graph" else None,
        "obbt_rebuild": bool(config.obbt_rebuild) if task == "graph" else False,
        "degree_table_layer1": bool(config.degree_table_layer1) if task == "graph" else False,
        "max_root_sep_rounds": (
            config.max_root_sep_rounds
            if node_uses_connectivity and config.cut_set_scope == "lazy_and_root"
            else 0
        ),
        "cut_set_scope": config.cut_set_scope,
    }


def skip_reason(row: Dict[str, Any], instance_filter: str) -> Optional[str]:
    if row["gt_size"] <= 0:
        return "skipped_no_ground_truth"
    if row["candidate_size"] <= 0:
        return "skipped_empty_candidate_set"
    if instance_filter == "correct" and not bool(row["is_correct"]):
        return "skipped_incorrect_prediction"
    if float(row["bar_m"]) <= 0.0:
        return "skipped_nonpositive_margin"
    if row["budget_k"] is None or int(row["budget_k"]) <= 0:
        return "skipped_invalid_budget"
    if (
        row.get("oracle_objective") == "min_sufficiency"
        and int(row.get("num_classes") or 0) != 2
        and row.get("solver_backend") != "enumeration"
    ):
        return "skipped_nonbinary_min_sufficiency"
    if row.get("oracle_objective") in RANK_OBJECTIVES and row.get("solver_backend") != "enumeration":
        return "skipped_rank_requires_enumeration"
    if row.get("oracle_objective") in DELTA_OBJECTIVES:
        if row.get("solver_backend") != "milp":
            return "skipped_delta_requires_milp"
        if int(row.get("num_classes") or 0) != 2:
            return "skipped_nonbinary_delta_milp"
        if str(row.get("edge_budget_mode") or "") != "eq":
            return "skipped_delta_requires_exact_budget"
    if row.get("oracle_objective") in NECESSITY_OBJECTIVES:
        if row.get("task") == "node" and row.get("solver_backend") != "enumeration":
            # Exact-budget node necessity is encoded on the kept-edge mask,
            # with explicit r=1-z variables carrying the removed explanation.
            # The node MILP now applies cut connectivity and its strengthening
            # constraints to r, so only non-exact budgets still require the
            # enumeration path here.
            if row.get("edge_budget_mode") != "eq":
                return "skipped_necessity_requires_enumeration"
        if row.get("task") not in {"graph", "node"}:
            return "skipped_necessity_graph_only"
        if (
            row.get("solver_backend") == "milp"
            and row.get("oracle_objective") == "max_necessity"
            and int(row.get("num_classes") or 0) != 2
        ):
            return "skipped_nonbinary_necessity_milp"
        if row.get("solver_backend") == "milp" and int(row["budget_k"]) > int(row["candidate_size"]):
            return "skipped_invalid_budget"
    return None


def populate_solution_row(
    *,
    row: Dict[str, Any],
    solution: Any,
    selected_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    elapsed_s: float,
    edge_file: Path,
    gt_edge_file: Path,
    gurobi_log: Path,
    root_node: Optional[int],
) -> None:
    metrics = compute_metrics(selected_edges, gt_edges)
    row["status"] = getattr(solution, "status", "unknown")
    row["rho_optimal"] = to_float_or_none(getattr(solution, "rho_optimal", None))
    row["sufficiency_raw"] = to_float_or_none(getattr(solution, "sufficiency_raw", None))
    if row["sufficiency_raw"] is None:
        row["sufficiency_raw"] = (
            float(row["rho_optimal"]) * float(row["bar_m"]) if row["rho_optimal"] is not None else None
        )
    row["necessity_raw"] = to_float_or_none(getattr(solution, "necessity_raw", None))
    row["objective_value"] = to_float_or_none(getattr(solution, "objective_value", None))
    for field in ENUMERATION_DIAGNOSTIC_FIELDS:
        value = getattr(solution, field, None)
        if field in {"gt_rank_direction", "gt_rank_scope"}:
            row[field] = str(value) if value is not None else ""
        elif field.startswith("gt_rank_") or field.startswith("enum_padding_"):
            row[field] = int(value) if value is not None else None
        else:
            row[field] = to_float_or_none(value)
    row["selected_size"] = len(edge_set(selected_edges))
    row.update(metrics)
    row["time_s"] = float(elapsed_s)
    row["final_mip_gap"] = to_float_or_none(getattr(solution, "final_mip_gap", None))
    row["best_bound"] = to_float_or_none(getattr(solution, "best_bound", None))
    row["incumbent_obj"] = to_float_or_none(getattr(solution, "incumbent_obj", None))
    row["model_size"] = to_float_or_none(getattr(solution, "model_size", None))
    root_lp_value = to_float_or_none(getattr(solution, "root_lp_value", None))
    if root_lp_value is None and row["best_bound"] is not None:
        root_lp_value = row["best_bound"]
        row["root_lp_source"] = "best_bound_fallback"
    elif root_lp_value is not None:
        row["root_lp_source"] = "mipnode_callback"
    else:
        row["root_lp_source"] = ""
    row["root_lp_value"] = root_lp_value
    row["root_lp_time"] = to_float_or_none(getattr(solution, "root_lp_time", None))
    row["callback_time_sec"] = to_float_or_none(getattr(solution, "callback_time_sec", None))
    row["callback_calls"] = to_float_or_none(getattr(solution, "callback_calls", None))
    row["cuts_mipnode"] = to_float_or_none(getattr(solution, "cuts_mipnode", None))
    row["cuts_mipsol_lazy"] = to_float_or_none(getattr(solution, "cuts_mipsol_lazy", None))
    row["anderson_cuts_added"] = to_float_or_none(
        getattr(solution, "anderson_cuts_added", None)
    )
    row["connectivity_lazy_cuts_added"] = to_float_or_none(
        getattr(solution, "connectivity_lazy_cuts_added", None)
    )
    row["bnb_node_count"] = to_float_or_none(getattr(solution, "bnb_node_count", None))
    row["gurobi_log"] = output_path_string(gurobi_log)
    row["edge_file"] = output_path_string(edge_file)
    row["gt_edge_file"] = output_path_string(gt_edge_file)
    if root_node is not None:
        row["rooted_connected"] = rooted_connected(int(root_node), selected_edges)


def populate_gt_baseline_row(
    *,
    row: Dict[str, Any],
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    root_node: Optional[int],
) -> None:
    """Record the GT edge set's objective baseline for MILP and enumeration rows."""
    gt_subset = tuple(sorted(edge_set(gt_edges)))
    if not gt_subset or row.get("bar_m") is None:
        return
    bar_m = float(row["bar_m"])
    if bar_m <= 0.0:
        return

    uses_necessity = str(row.get("oracle_objective")) in NECESSITY_SCORING_OBJECTIVES
    uses_delta = str(row.get("oracle_objective")) in DELTA_OBJECTIVES
    fidelity_metrics = standard_fidelity_components(
        model=model,
        task=str(row["task"]),
        x=x,
        edge_index=edge_index,
        candidate_edges=candidate_edges,
        selected_edges=gt_subset,
        c_star=int(row["c_star"]),
        full_margin=bar_m,
        root_node=root_node,
    )
    row["gt_sufficiency_probability"] = fidelity_metrics["sufficiency_probability"]
    row["gt_complement_probability"] = fidelity_metrics["complement_probability"]
    row["gt_fidelity_minus"] = fidelity_metrics["fidelity_minus"]
    row["gt_fidelity_plus"] = fidelity_metrics["fidelity_plus"]
    row["gt_fidelity_delta"] = fidelity_metrics["fidelity_delta"]
    if uses_delta:
        row["gt_rho"] = None
        row["gt_sufficiency_raw"] = fidelity_metrics["sufficiency_raw"]
        row["gt_necessity_raw"] = fidelity_metrics["necessity_raw"]
        row["gt_objective_value"] = fidelity_metrics["fidelity_delta"]
        return
    already_has_gt = row.get("gt_rho") is not None and row.get("gt_sufficiency_raw") is not None
    if uses_necessity:
        already_has_gt = already_has_gt and row.get("gt_necessity_raw") is not None
    if already_has_gt and row.get("gt_objective_value") is not None:
        return

    if uses_necessity:
        evaluation_edges = tuple(sorted(edge_set(candidate_edges) - edge_set(gt_subset)))
    else:
        evaluation_edges = gt_subset

    margin = selected_margin_for_edges(
        model=model,
        task=str(row["task"]),
        x=x,
        edge_index=edge_index,
        selected_edges=evaluation_edges,
        c_star=int(row["c_star"]),
        root_node=root_node,
    )
    rho_value = float(margin) / bar_m
    row["gt_rho"] = float(rho_value)
    row["gt_sufficiency_raw"] = float(margin)
    if uses_necessity:
        row["gt_necessity_raw"] = float(bar_m - float(margin))
        row["gt_objective_value"] = float(1.0 - rho_value)
    else:
        row["gt_necessity_raw"] = None
        row["gt_objective_value"] = float(rho_value)


def gt_baseline_log_line(row: Dict[str, Any]) -> Optional[str]:
    if str(row.get("oracle_objective")) in DELTA_OBJECTIVES:
        if row.get("gt_fidelity_delta") is None:
            return None
        return (
            f"    GT baseline: p_keep={format_float(row['gt_sufficiency_probability'], 6)}, "
            f"p_complement={format_float(row['gt_complement_probability'], 6)}, "
            f"Fid-={format_float(row['gt_fidelity_minus'], 6)}, "
            f"Fid+={format_float(row['gt_fidelity_plus'], 6)}, "
            f"FidDelta={format_float(row['gt_fidelity_delta'], 6)}"
        )
    if row.get("gt_rho") is None or row.get("gt_sufficiency_raw") is None:
        return None
    if str(row.get("oracle_objective")) in NECESSITY_SCORING_OBJECTIVES:
        return (
            f"    GT baseline: removal_rho={format_float(row['gt_rho'], 4)}, "
            f"kept_margin_raw={format_float(row['gt_sufficiency_raw'], 6)}, "
            f"necessity_raw={format_float(row['gt_necessity_raw'], 6)}, "
            f"Fid+={format_float(row['gt_fidelity_plus'], 6)}, "
            f"objective_value={format_float(row['gt_objective_value'], 6)}"
        )
    return (
        f"    GT baseline: rho={format_float(row['gt_rho'], 4)}, "
        f"sufficiency_raw={format_float(row['gt_sufficiency_raw'], 6)}, "
        f"Fid-={format_float(row['gt_fidelity_minus'], 6)}, "
        f"objective_value={format_float(row['gt_objective_value'], 6)}"
    )


def standard_counterexample_spec(row: Dict[str, Any]) -> Optional[Tuple[str, str, str, float]]:
    """Return (metric label, candidate field, GT field, orientation) for improvement."""
    objective = str(row.get("oracle_objective") or "")
    if objective in {"max_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}:
        return "Fid-", "fidelity_minus", "gt_fidelity_minus", -1.0
    if objective == "min_sufficiency":
        return "Fid-", "fidelity_minus", "gt_fidelity_minus", 1.0
    if objective in {"max_necessity", RANK_NECESSITY_OBJECTIVE}:
        return "Fid+", "fidelity_plus", "gt_fidelity_plus", 1.0
    if objective == "min_necessity":
        return "Fid+", "fidelity_plus", "gt_fidelity_plus", -1.0
    if objective in DELTA_OBJECTIVES:
        return "FidDelta", "fidelity_delta", "gt_fidelity_delta", 1.0
    return None


def attach_standard_counterexample_result(row: Dict[str, Any]) -> None:
    """Compare the optimized explanation with the planted GT deterministically."""
    if str(row.get("status") or "") not in SUCCESS_STATUSES:
        row["counterexample_status"] = "unavailable"
        return
    spec = standard_counterexample_spec(row)
    if spec is None:
        row["counterexample_status"] = "unsupported_objective"
        return
    metric_label, candidate_field, gt_field, orientation = spec
    candidate_value = to_float_or_none(row.get(candidate_field))
    gt_value = to_float_or_none(row.get(gt_field))
    if candidate_value is None or gt_value is None:
        row["counterexample_status"] = "unavailable"
        return

    advantage = float(orientation) * (float(candidate_value) - float(gt_value))
    row["counterexample_advantage"] = advantage
    row["counterexample_definition"] = (
        f"GT_{metric_label}_minus_OPT_{metric_label}"
        if orientation < 0.0
        else f"OPT_{metric_label}_minus_GT_{metric_label}"
    )
    if advantage > COUNTEREXAMPLE_TOLERANCE:
        status = "counterexample_found"
    elif advantage < -COUNTEREXAMPLE_TOLERANCE:
        status = "gt_better"
    else:
        status = "exact_tie"
    row["counterexample_status"] = status
    row["counterexample_found"] = status == "counterexample_found"


def counterexample_log_line(row: Dict[str, Any]) -> str:
    found = row.get("counterexample_found")
    verdict = "YES" if found is True else ("NO" if found is False else "N/A")
    spec = standard_counterexample_spec(row)
    metric_label = spec[0] if spec is not None else "fidelity"
    candidate_field = spec[1] if spec is not None else ""
    gt_field = spec[2] if spec is not None else ""
    return (
        f"    Counterexample: {verdict}, status={row.get('counterexample_status') or 'unavailable'}, "
        f"OPT {metric_label}={format_float(row.get(candidate_field), 6)}, "
        f"GT {metric_label}={format_float(row.get(gt_field), 6)}, "
        f"advantage={format_float(row.get('counterexample_advantage'), 6)}"
    )




def log_solution_summary(
    *,
    row: Dict[str, Any],
    terminal_log_file: Optional[Path],
    elapsed_s: float,
) -> None:
    print_and_log(f"    Status: {row['status']}, Edges: {row['selected_size']}", terminal_log_file)
    print_and_log(
        (
            f"    D={format_float(row['D'], 3)}, H={format_float(row['H'], 3)}, "
            f"Precision={format_float(row['precision'], 3)}, "
            f"Recall={format_float(row['recall'], 3)}, F1={format_float(row['f1'], 3)}"
        ),
        terminal_log_file,
    )
    if row.get("oracle_objective") in DELTA_OBJECTIVES:
        print_and_log(
            (
                f"    p_keep={format_float(row['sufficiency_probability'], 6)}, "
                f"p_complement={format_float(row['complement_probability'], 6)}, "
                f"Fid-={format_float(row['fidelity_minus'], 6)}, "
                f"Fid+={format_float(row['fidelity_plus'], 6)}, "
                f"FidDelta={format_float(row['fidelity_delta'], 6)}, "
                f"PWL_objective={format_float(row['objective_value'], 6)}, "
                f"PWL_error={format_float(row['pwl_approx_error_at_solution'], 6)}"
            ),
            terminal_log_file,
        )
    elif row.get("oracle_objective") in NECESSITY_SCORING_OBJECTIVES:
        print_and_log(
            (
                f"    removal_rho={format_float(row['rho_optimal'], 4)}, "
                f"kept_margin_raw={format_float(row['sufficiency_raw'], 6)}, "
                f"necessity_raw={format_float(row['necessity_raw'], 6)}, "
                f"Fid+={format_float(row['fidelity_plus'], 6)}, "
                f"objective_value={format_float(row['objective_value'], 6)}"
            ),
            terminal_log_file,
        )
        if row.get("rooted_connected") is not None:
            print_and_log(
                f"    removal_rooted_connected={int(bool(row['rooted_connected']))}",
                terminal_log_file,
            )
    else:
        print_and_log(
            (
                f"    rho_optimal={format_float(row['rho_optimal'], 4)}, "
                f"sufficiency_raw={format_float(row['sufficiency_raw'], 6)}, "
                f"Fid-={format_float(row['fidelity_minus'], 6)}"
            ),
            terminal_log_file,
        )
    print_and_log(
        (
            f"    MIPGap={format_float(row['final_mip_gap'], 6)}, "
            f"BestBd={format_float(row['best_bound'], 6)}, "
            f"Incumbent={format_float(row['incumbent_obj'], 6)}"
        ),
        terminal_log_file,
    )
    print_and_log(
        (
            f"    RootLP={format_float(row['root_lp_value'], 6)}, "
            f"RootLPTime={format_seconds(row['root_lp_time'], 4)}, "
            f"RootLPSource={row['root_lp_source'] or 'N/A'}, "
            f"B&BNodes={format_float(row['bnb_node_count'], 0)}"
        ),
        terminal_log_file,
    )
    print_and_log(
        (
            f"    CallbackTime={format_seconds(row['callback_time_sec'], 4)}, "
            f"CallbackCalls={format_float(row['callback_calls'], 0)}, "
            f"CutsMIPNODE={format_float(row['cuts_mipnode'], 0)}, "
            f"CutsMIPSOLLazy={format_float(row['cuts_mipsol_lazy'], 0)}, "
            f"AndersonCutsAdded={format_float(row['anderson_cuts_added'], 0)}, "
            "ConnectivityLazyCutsAdded="
            f"{format_float(row['connectivity_lazy_cuts_added'], 0)}"
        ),
        terminal_log_file,
    )
    print_and_log(f"    Time: {elapsed_s:.2f}s", terminal_log_file)


def run_or_skip_row(
    *,
    row: Dict[str, Any],
    config: RunConfig,
    output_dir: Path,
    explainer: MIPExplainer,
    x: np.ndarray,
    edge_index: np.ndarray,
    gt_edges: Sequence[Tuple[int, int]],
    candidate_edges: Sequence[Tuple[int, int]],
    root_node: Optional[int],
    node_labels: Optional[np.ndarray],
) -> Dict[str, Any]:
    row_id = int(row["row_id"])
    nn_seed = int(row.get("nn_seed", 0))
    terminal_log_file = output_dir / "logs" / f"terminal_seed{nn_seed}_row{row_id:06d}.log"
    terminal_log_file.parent.mkdir(parents=True, exist_ok=True)
    terminal_log_file.write_text(f"# Terminal log for Main oracle row {row_id}\n", encoding="utf-8")
    row["terminal_log"] = output_path_string(terminal_log_file)

    instance_label = "node" if row["task"] == "node" else "graph"
    print_and_log(
        (
            f"\n[row {row_id}] {row['dataset']} seed={row['nn_seed']} "
            f"{instance_label}={row['instance_id']} stage={row['stage']} "
            f"K={row['budget_k']} multiplier={row['budget_multiplier']} "
            f"objective={row['oracle_objective']} pred={row['pred']} "
            f"label={row['label']} margin={format_float(row['bar_m'], 6)}"
        ),
        terminal_log_file,
    )
    print_and_log(
        (
            f"    candidate_edges={row['candidate_size']}, gt_edges={row['gt_size']}, "
            f"context_edges={row['budget_context_size']}, "
            f"connectivity={row['connectivity']}, edge_budget_mode={row['edge_budget_mode']}"
        ),
        terminal_log_file,
    )
    if row.get("oracle_objective") == "min_sufficiency":
        if int(row.get("num_classes") or 0) == 2:
            min_sufficiency_detail = (
                f"min_sufficiency_other_class={row.get('min_sufficiency_other_class')}"
            )
        else:
            min_sufficiency_detail = "min_sufficiency_multiclass_margin=c_star_vs_max_other"
        print_and_log(
            f"    {min_sufficiency_detail}, num_classes={row.get('num_classes')}",
            terminal_log_file,
        )
    if row.get("oracle_objective") == RANK_SUFFICIENCY_OBJECTIVE:
        print_and_log(
            "    rank_mode=sufficiency enumeration diagnostics; primary metrics/figure use GT edges when feasible",
            terminal_log_file,
        )
    if row.get("oracle_objective") == RANK_NECESSITY_OBJECTIVE:
        print_and_log(
            "    rank_necessity_mode=necessity enumeration diagnostics; primary metrics/figure use GT removal edges when feasible",
            terminal_log_file,
        )
    node_necessity = bool(row.get("task") == "node" and row.get("oracle_objective") in NECESSITY_OBJECTIVES)
    graph_necessity = bool(row.get("task") == "graph" and row.get("oracle_objective") in NECESSITY_OBJECTIVES)
    use_enumeration = bool(config.gt_size_solver == "enumeration" and row["stage"] == "gt_size")
    delta_milp = bool(row.get("oracle_objective") in DELTA_OBJECTIVES and not use_enumeration)
    necessity_scoring_objective = bool(row.get("oracle_objective") in NECESSITY_SCORING_OBJECTIVES)
    node_necessity_milp = bool(node_necessity and not use_enumeration)
    graph_necessity_milp = bool(graph_necessity and not use_enumeration)
    if necessity_scoring_objective:
        print_and_log(
            "    necessity_mode=remove selected edges and score the complement graph",
            terminal_log_file,
        )
        if node_necessity_milp or graph_necessity_milp:
            keep_budget = max(0, int(row["candidate_size"]) - int(row["budget_k"]))
            task_prefix = "node" if node_necessity_milp else "graph"
            connectivity_scope = "removal_edges"
            print_and_log(
                f"    {task_prefix}_necessity_milp=kept_edge_budget={keep_budget} "
                f"(candidate_edges - removal_budget), removal_edge_budget={int(row['budget_k'])}, "
                f"connectivity={row['connectivity']}, connectivity_scope={connectivity_scope}, "
                f"edge_budget_mode=eq, cap_mode=global",
                terminal_log_file,
            )
    row["solver_backend"] = "enumeration" if use_enumeration else "milp"
    gt_solution_representable = edge_set(gt_edges).issubset(edge_set(candidate_edges))
    row["gt_solution_representable"] = bool(gt_solution_representable)
    row["gt_no_good_cut_added"] = bool(
        config.exclude_gt_solution and not use_enumeration and gt_solution_representable
    )
    print_and_log(f"    Solver backend: {row['solver_backend']}", terminal_log_file)
    print_and_log(
        (
            f"    GT incumbent={bool(config.gt_incumbent)}, "
            f"exclude_GT={bool(config.exclude_gt_solution)}, "
            f"GT_representable={bool(gt_solution_representable)}, "
            f"GT_no_good_cut_added={bool(row['gt_no_good_cut_added'])}"
        ),
        terminal_log_file,
    )

    reason = skip_reason(row, config.instance_filter)
    if reason is not None:
        row["status"] = reason
        row["selected_size"] = 0
        print_and_log(f"    Status: {reason} (solver not built)", terminal_log_file)
        return row

    log_file = output_dir / "logs" / (
        f"enumeration_seed{nn_seed}_row{row_id:06d}.log"
        if use_enumeration
        else f"gurobi_seed{nn_seed}_row{row_id:06d}.log"
    )
    edge_file = output_dir / "edges" / f"oracle_edges_seed{nn_seed}_row{row_id:06d}.csv"
    gt_edge_file = output_dir / "edges" / f"gt_edges_seed{nn_seed}_row{row_id:06d}.csv"

    start = time.perf_counter()
    try:
        if use_enumeration:
            solution = run_gt_size_enumeration(
                row=row,
                model=explainer.model,
                x=x,
                edge_index=edge_index,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                root_node=root_node,
                max_subgraphs=int(config.enumeration_max_subgraphs),
                enumeration_batch_size=int(config.enumeration_batch_size),
                use_enumeration_cache=bool(config.use_enumeration_cache),
                enumeration_cache_dir=Path(config.enumeration_cache_dir),
            )
            elapsed = time.perf_counter() - start
            row["enumerated_subgraphs"] = int(getattr(solution, "enumerated_subgraphs", 0))
            row["enumerated_signatures"] = int(getattr(solution, "enumerated_signatures", 0))
            row["enumeration_pruned"] = int(getattr(solution, "enumeration_pruned", 0))
            row["enumeration_cache_status"] = getattr(solution, "enumeration_cache_status", None)
            row["enumeration_cache_file"] = getattr(solution, "enumeration_cache_file", None)
            enumeration_log_lines = [
                f"status={solution.status}",
                f"enumerated_subgraphs={row['enumerated_subgraphs']}",
                f"enumerated_signatures={row['enumerated_signatures']}",
                f"enumeration_pruned={row['enumeration_pruned']}",
            ]
            if row["enumeration_cache_status"] is not None:
                enumeration_log_lines.append(f"enumeration_cache_status={row['enumeration_cache_status']}")
            if row["enumeration_cache_file"] is not None:
                enumeration_log_lines.append(f"enumeration_cache_file={row['enumeration_cache_file']}")
            log_file.write_text("\n".join(enumeration_log_lines) + "\n", encoding="utf-8")
            if str(solution.status).startswith("skipped_"):
                row["status"] = str(solution.status)
                row["selected_size"] = 0
                row["time_s"] = float(elapsed)
                row["gurobi_log"] = output_path_string(log_file)
                print_and_log(
                    (
                        f"    Enumeration: subgraphs={row['enumerated_subgraphs']}, "
                        f"signatures={row['enumerated_signatures']}, "
                        f"pruned={row['enumeration_pruned']}"
                    ),
                    terminal_log_file,
                )
                if row.get("enumeration_cache_status") is not None:
                    print_and_log(
                        (
                            f"    Enumeration cache: status={row['enumeration_cache_status']}, "
                            f"file={row.get('enumeration_cache_file')}"
                        ),
                        terminal_log_file,
                    )
                print_and_log(f"    Status: {row['status']} (MILP fallback disabled)", terminal_log_file)
                print_and_log(f"    Time: {elapsed:.2f}s", terminal_log_file)
                return row
        elif delta_milp:
            raise ValueError("FidDelta is not supported in this repository.")
        elif row["task"] == "node":
            mip_objective = str(row["oracle_objective"])
            mip_edge_budget = int(row["budget_k"])
            mip_edge_budget_mode = str(row["edge_budget_mode"])
            mip_connectivity = str(row["connectivity"])
            mip_cap_mode = "global"
            gt_internal_solution = sorted(edge_set(gt_edges))
            if node_necessity_milp:
                mip_objective = (
                    "min_sufficiency"
                    if row["oracle_objective"] == "max_necessity"
                    else "max_sufficiency"
                )
                mip_edge_budget = max(0, int(row["candidate_size"]) - int(row["budget_k"]))
                mip_edge_budget_mode = "eq"
                mip_connectivity = str(row["connectivity"])
                mip_cap_mode = "global"
                gt_internal_solution = sorted(edge_set(candidate_edges) - edge_set(gt_edges))
            mip_initial_solution = gt_internal_solution if config.gt_incumbent else None
            mip_excluded_solutions = (
                [gt_internal_solution] if config.exclude_gt_solution else None
            )
            _, solution = explainer.explain_node(
                node_idx=int(row["instance_id"]),
                x=x,
                edge_index=edge_index,
                rho=0.0,
                edge_budget=mip_edge_budget,
                edge_budget_mode=mip_edge_budget_mode,
                objective=mip_objective,
                connectivity=mip_connectivity,
                explanation_on_complement=bool(node_necessity_milp),
                time_limit_s=config.time_limit_s,
                k_hop=int(row["k_hop"]) if row["k_hop"] is not None else None,
                initial_solution=mip_initial_solution,
                excluded_edge_solutions=mip_excluded_solutions,
                log_file=str(log_file),
                terminal_log_file=str(terminal_log_file),
                gurobi_params=gurobi_param_overrides(config),
                # Nested and back cuts intentionally use their False defaults.
                max_root_sep_rounds=int(row["max_root_sep_rounds"]),
                use_anderson_cuts=bool(row["use_anderson_cuts"]),
                anderson_max_node_count=int(row["anderson_max_node_count"]),
                anderson_max_cuts=int(row["anderson_max_cuts"]),
                use_incident_cap_tightening=bool(row["use_o1_tightening"]),
                cap_mode=mip_cap_mode,
                use_symmetry_breaking=False,
                use_star_constraints=bool(row["use_o4_star_constraints"]),
                use_o9_connectivity_lb=bool(row["use_o9_connectivity_lb"]),
                # O5-O8 and OBBT-u intentionally use their False defaults.
            )
        else:
            mip_objective = str(row["oracle_objective"])
            mip_edge_budget = int(row["budget_k"])
            mip_edge_budget_mode = str(row["edge_budget_mode"])
            mip_connectivity = str(row["connectivity"])
            gt_internal_solution = sorted(edge_set(gt_edges))
            if graph_necessity_milp:
                mip_objective = (
                    "min_sufficiency"
                    if row["oracle_objective"] == "max_necessity"
                    else "max_sufficiency"
                )
                mip_edge_budget = max(0, int(row["candidate_size"]) - int(row["budget_k"]))
                mip_edge_budget_mode = "eq"
                mip_connectivity = str(row["connectivity"])
                gt_internal_solution = sorted(edge_set(candidate_edges) - edge_set(gt_edges))
            mip_initial_solution = gt_internal_solution if config.gt_incumbent else None
            mip_excluded_solutions = (
                [gt_internal_solution] if config.exclude_gt_solution else None
            )
            _, solution = explainer.explain_graph(
                x=x,
                edge_index=edge_index,
                readout=config.readout,
                rho=0.0,
                edge_budget=mip_edge_budget,
                edge_budget_mode=mip_edge_budget_mode,
                objective=mip_objective,
                connectivity=mip_connectivity,
                explanation_on_complement=bool(graph_necessity_milp),
                time_limit_s=config.time_limit_s,
                initial_solution=mip_initial_solution,
                excluded_edge_solutions=mip_excluded_solutions,
                log_file=str(log_file),
                terminal_log_file=str(terminal_log_file),
                gurobi_params=gurobi_param_overrides(config),
                use_incident_cap_tightening=bool(row["use_o1_tightening"]),
                use_symmetry_breaking=False,
                use_star_constraints=bool(row["use_o4_star_constraints"]),
                use_o9_connectivity_lb=bool(row["use_o9_connectivity_lb"]),
                use_bridge_constraints=False,
                use_anderson_cuts=bool(row["use_anderson_cuts"]),
                anderson_max_node_count=int(row["anderson_max_node_count"]),
                anderson_max_cuts=int(row["anderson_max_cuts"]),
                edge_branch_priority=int(row["edge_branch_priority"]),
                use_obbt_u=bool(row["use_obbt_u"]),
                max_obbt_u_layers=row["max_obbt_u_layers"],
                obbt_rebuild=bool(row["obbt_rebuild"]),
                degree_table_layer1=bool(row["degree_table_layer1"]),
            )
        if not use_enumeration:
            elapsed = time.perf_counter() - start
        selected_edges = sorted(edge_set(getattr(solution, "selected_edges", [])))
        if (node_necessity_milp or graph_necessity_milp) and getattr(solution, "z_values", None):
            kept_edges = edge_set(selected_edges)
            selected_edges = sorted(edge_set(candidate_edges) - kept_edges)
        write_edges_csv(edge_file, selected_edges)
        write_edges_csv(gt_edge_file, gt_edges)
        populate_solution_row(
            row=row,
            solution=solution,
            selected_edges=selected_edges,
            gt_edges=gt_edges,
            elapsed_s=elapsed,
            edge_file=edge_file,
            gt_edge_file=gt_edge_file,
            gurobi_log=log_file,
            root_node=root_node,
        )
        if node_necessity_milp or graph_necessity_milp:
            if row.get("sufficiency_raw") is not None:
                row["necessity_raw"] = float(row["bar_m"]) - float(row["sufficiency_raw"])
            if row.get("rho_optimal") is not None:
                row["objective_value"] = 1.0 - float(row["rho_optimal"])
        fidelity_metrics = standard_fidelity_components(
            model=explainer.model,
            task=str(row["task"]),
            x=x,
            edge_index=edge_index,
            candidate_edges=candidate_edges,
            selected_edges=selected_edges,
            c_star=int(row["c_star"]),
            full_margin=float(row["bar_m"]),
            root_node=root_node,
        )
        if delta_milp:
            delta_metrics = fidelity_metrics
            row.update(delta_metrics)
            row["rho_optimal"] = None
            row["objective_value_true"] = float(delta_metrics["fidelity_delta"])
            if row.get("objective_value") is not None:
                row["pwl_approx_error_at_solution"] = abs(
                    float(row["objective_value"]) - float(delta_metrics["fidelity_delta"])
                )
        else:
            for field in (
                "sufficiency_probability",
                "complement_probability",
                "fidelity_minus",
                "fidelity_plus",
                "fidelity_delta",
            ):
                row[field] = fidelity_metrics[field]
        populate_gt_baseline_row(
            row=row,
            model=explainer.model,
            x=x,
            edge_index=edge_index,
            candidate_edges=candidate_edges,
            gt_edges=gt_edges,
            root_node=root_node,
        )
        attach_standard_counterexample_result(row)
        if use_enumeration:
            print_and_log(
                (
                    f"    Enumeration: subgraphs={row['enumerated_subgraphs']}, "
                    f"signatures={row['enumerated_signatures']}, "
                    f"pruned={row['enumeration_pruned']}"
                ),
                terminal_log_file,
            )
            if row.get("enumeration_cache_status") is not None:
                print_and_log(
                    (
                        f"    Enumeration cache: status={row['enumeration_cache_status']}, "
                        f"file={row.get('enumeration_cache_file')}"
                    ),
                    terminal_log_file,
                )
            if row.get("enum_objective_max") is not None and row.get("enum_objective_min") is not None:
                print_and_log(
                    (
                        f"    Enumeration probability-fidelity score range: "
                        f"min={format_float(row['enum_objective_min'], 6)}, "
                        f"max={format_float(row['enum_objective_max'], 6)}"
                    ),
                    terminal_log_file,
                )
            if row.get("oracle_objective") in RANK_OBJECTIVES:
                if row.get("enum_padding_feasible") is not None or row.get("enum_padding_filtered") is not None:
                    print_and_log(
                        (
                            f"    Rank padding filter: feasible={row.get('enum_padding_feasible')}, "
                            f"filtered={row.get('enum_padding_filtered')}"
                        ),
                        terminal_log_file,
                    )
            if row.get("gt_rank_position") is not None:
                print_and_log(
                    (
                        f"    GT probability-fidelity rank: {row['gt_rank_position']}/{row['gt_rank_total']} "
                        f"(ties={row['gt_rank_tie_count']}, direction={row['gt_rank_direction']}, "
                        f"scope={row['gt_rank_scope']}), "
                        f"objective={format_float(row['gt_objective_value'], 6)}, "
                        f"rank_max={row['gt_rank_max_objective']}, "
                        f"rank_min={row['gt_rank_min_objective']}"
                    ),
                    terminal_log_file,
                )
            else:
                print_and_log(
                    "    GT probability-fidelity rank: unavailable because GT is infeasible for this budget/connectivity policy",
                    terminal_log_file,
                )

        gt_line = gt_baseline_log_line(row)
        if gt_line is not None:
            print_and_log(gt_line, terminal_log_file)
        log_solution_summary(row=row, terminal_log_file=terminal_log_file, elapsed_s=elapsed)
        print_and_log(counterexample_log_line(row), terminal_log_file)

        padded_edges: Optional[List[Tuple[int, int]]] = None
        if row["stage"] == "gt_size" and row["task"] == "node":
            padded_edges, padding_info = pad_node_gt_size_edges(
                row=row,
                padding_seed=config.padding_seed,
                root=int(row["instance_id"]),
                selected_edges=selected_edges,
                candidate_edges=candidate_edges,
                target_size=int(row["gt_size"]),
            )
            row.update(padding_info)
            if (
                use_enumeration
                and str(row.get("oracle_objective"))
                in {"max_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}
                and row.get("padding_status") == "insufficient_disconnected_candidates"
            ):
                row["status"] = "skipped_padding_infeasible"
                row["error"] = (
                    "No sufficient node-disjoint padding edges exist to bring the selected sufficiency "
                    "core to GT size without connecting padding back to the core."
                )
                print_and_log(
                    (
                        f"    Padding: policy={row['padding_policy']}, status={row['padding_status']}, "
                        f"needed={row['padding_needed']}, added={row['padding_added']}, "
                        f"candidates={row['padding_candidate_size']}, seed={row['padding_seed']}"
                    ),
                    terminal_log_file,
                )
                print_and_log("    Status: skipped_padding_infeasible (excluded from metric summaries)", terminal_log_file)
                return row
            padded_metrics = compute_metrics(padded_edges, gt_edges)
            padded_file = output_dir / "edges" / f"padded_edges_{row_id:06d}.csv"
            write_edges_csv(padded_file, padded_edges)
            row["padded_size"] = len(edge_set(padded_edges))
            row["padded_D"] = padded_metrics["D"]
            row["padded_H"] = padded_metrics["H"]
            row["padded_precision"] = padded_metrics["precision"]
            row["padded_recall"] = padded_metrics["recall"]
            row["padded_f1"] = padded_metrics["f1"]
            row["padded_edge_file"] = output_path_string(padded_file)
        save_enumeration_extreme_artifacts(
            row=row,
            solution=solution,
            config=config,
            output_dir=output_dir,
            x=x,
            edge_index=edge_index,
            gt_edges=gt_edges,
            root_node=root_node,
            node_labels=node_labels,
            terminal_log_file=terminal_log_file,
        )
        extreme_figure_files = {
            str(path)
            for path in (row.get("enum_max_figure_file"), row.get("enum_min_figure_file"))
            if path
        }
        if row.get("figure_file") and str(row.get("figure_file")) not in extreme_figure_files:
            print_and_log(f"    Figure: {row['figure_file']}", terminal_log_file)
        if row.get("gt_figure_file"):
            print_and_log(f"    GT figure: {row['gt_figure_file']}", terminal_log_file)
        if row.get("enum_max_edge_file"):
            print_and_log(f"    Enum max edges: {row['enum_max_edge_file']}", terminal_log_file)
        if row.get("enum_min_edge_file"):
            print_and_log(f"    Enum min edges: {row['enum_min_edge_file']}", terminal_log_file)
        if row.get("enum_extrema_file"):
            print_and_log(f"    Enum extrema CSV: {row['enum_extrema_file']}", terminal_log_file)
        if row.get("enum_max_figure_file"):
            print_and_log(f"    Enum max figure: {row['enum_max_figure_file']}", terminal_log_file)
        if row.get("enum_min_figure_file"):
            print_and_log(f"    Enum min figure: {row['enum_min_figure_file']}", terminal_log_file)
        if row.get("padding_policy"):
            print_and_log(
                (
                    f"    Padding: policy={row['padding_policy']}, status={row['padding_status']}, "
                    f"needed={row['padding_needed']}, added={row['padding_added']}, "
                    f"candidates={row['padding_candidate_size']}, seed={row['padding_seed']}"
                ),
                terminal_log_file,
            )
        return row
    except CurrentGurobiSolveInterrupted as exc:
        elapsed = time.perf_counter() - start
        row["status"] = "interrupted_by_user"
        row["time_s"] = float(elapsed)
        row["selected_size"] = 0
        row["error"] = str(exc)
        row["gurobi_log"] = output_path_string(log_file)
        print_and_log(f"    Interrupted: {exc}", terminal_log_file)
        print_and_log(f"    Time: {elapsed:.2f}s", terminal_log_file)
        return row
    except Exception as exc:
        elapsed = time.perf_counter() - start
        row["status"] = f"error: {type(exc).__name__}"
        row["time_s"] = float(elapsed)
        row["selected_size"] = 0
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["gurobi_log"] = output_path_string(log_file)
        print_and_log(f"    Error: {row['error']}", terminal_log_file)
        print_and_log(f"    Time: {elapsed:.2f}s", terminal_log_file)
        return row


def planned_rows_for_instance(
    *,
    start_row_id: int,
    dataset: str,
    task: str,
    nn_seed: int,
    instance_id: int,
    pred: int,
    label: int,
    c_star: int,
    num_classes: int,
    bar_m: float,
    gt_edges: Sequence[Tuple[int, int]],
    candidate_edges: Sequence[Tuple[int, int]],
    stages: Sequence[str],
    budget_multipliers: Sequence[float],
    k_hop: Optional[int],
    connectivity: str,
    node_edge_budget_mode: str,
    oracle_objective: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gt_size = len(edge_set(gt_edges))
    candidate_size = len(edge_set(candidate_edges))

    if "gt_size" in stages:
        if task == "graph":
            row_connectivity = connectivity
            edge_budget_mode = "eq"
        elif oracle_objective == RANK_SUFFICIENCY_OBJECTIVE:
            row_connectivity = "cut"
            edge_budget_mode = "le"
        elif oracle_objective == RANK_NECESSITY_OBJECTIVE:
            row_connectivity = "none"
            edge_budget_mode = "eq"
        elif oracle_objective in DELTA_OBJECTIVES:
            row_connectivity = connectivity
            edge_budget_mode = "eq"
        else:
            row_connectivity = connectivity
            edge_budget_mode = node_edge_budget_mode
        rows.append(
            make_base_row(
                row_id=start_row_id + len(rows) + 1,
                dataset=dataset,
                task=task,
                nn_seed=nn_seed,
                instance_id=instance_id,
                stage="gt_size",
                budget_multiplier=None,
                budget_k=gt_size,
                budget_mode="gt_size",
                connectivity=row_connectivity,
                edge_budget_mode=edge_budget_mode,
                k_hop=k_hop,
                candidate_size=candidate_size,
                gt_size=gt_size,
                pred=pred,
                label=label,
                c_star=c_star,
                num_classes=num_classes,
                oracle_objective=oracle_objective,
                bar_m=bar_m,
            )
        )

    if "sweep" in stages:
        for multiplier in budget_multipliers:
            budget_k = int(math.ceil(float(multiplier) * float(gt_size))) if candidate_size > 0 else 0
            budget_k = max(1, min(budget_k, candidate_size)) if candidate_size > 0 else 0
            rows.append(
                make_base_row(
                    row_id=start_row_id + len(rows) + 1,
                    dataset=dataset,
                    task=task,
                    nn_seed=nn_seed,
                    instance_id=instance_id,
                    stage="sweep",
                    budget_multiplier=float(multiplier),
                    budget_k=budget_k,
                    budget_mode="gt_multiplier",
                    connectivity=connectivity,
                    edge_budget_mode="eq",
                    k_hop=k_hop,
                    candidate_size=candidate_size,
                    gt_size=gt_size,
                    pred=pred,
                    label=label,
                    c_star=c_star,
                    num_classes=num_classes,
                    oracle_objective=oracle_objective,
                    bar_m=bar_m,
                )
            )
    return rows


def minimal_node_cache_row(
    *,
    dataset: str,
    node_idx: int,
    budget_k: int,
    connectivity: str,
    edge_budget_mode: str,
    oracle_objective: str,
    exclude_gt_solution: bool,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "task": "node",
        "instance_id": int(node_idx),
        "budget_k": int(budget_k),
        "connectivity": connectivity,
        "edge_budget_mode": edge_budget_mode,
        "oracle_objective": str(oracle_objective),
        "exclude_gt_solution": bool(exclude_gt_solution),
    }


def prewarm_node_sufficiency_cache_for_instance(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    fixed_self_loop_counts: Sequence[int],
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    max_subgraphs: int,
    cache_dir: Path,
) -> str:
    budget_k = int(row["budget_k"])
    if budget_k <= 0:
        return "skipped_empty_gt"
    valid_sizes = (
        list(range(1, budget_k + 1))
        if str(row.get("edge_budget_mode")) == "le"
        else [budget_k]
    )
    node_colors = graph_structural_node_colors(
        x=x,
        fixed_self_loop_counts=fixed_self_loop_counts,
    )
    match_features = True
    metadata = node_enumeration_cache_metadata(
        row=row,
        x=x,
        candidate_edges=candidate_edges,
        gt_edges=gt_edges,
        valid_sizes=valid_sizes,
        match_features=match_features,
        cache_kind="node_sufficiency",
        fixed_self_loop_counts=fixed_self_loop_counts,
    )
    cache_path = node_enumeration_cache_path(
        cache_dir=cache_dir,
        metadata=metadata,
        cache_kind="node_sufficiency",
    )
    status, _, representatives = load_node_enumeration_cache(cache_path=cache_path, metadata=metadata)
    if status == "hit":
        return "hit"

    if str(row.get("connectivity")) == "cut":
        enum_status, enumerated_subgraphs, representatives = node_sufficiency_shape_representatives(
            x=x,
            candidate_edges=candidate_edges,
            node_colors=node_colors,
            gt_edges=gt_edges,
            root_node=int(row["instance_id"]),
            valid_sizes=valid_sizes,
            max_subgraphs=int(max_subgraphs),
            edge_budget_mode=str(row.get("edge_budget_mode")),
            exclude_gt_solution=bool(row.get("exclude_gt_solution")),
        )
    else:
        enum_status, enumerated_subgraphs, representatives = unrestricted_node_edge_representatives(
            node_colors=node_colors,
            candidate_edges=candidate_edges,
            root_node=int(row["instance_id"]),
            valid_sizes=valid_sizes,
            max_subgraphs=int(max_subgraphs),
            excluded_edge_sets=[gt_edges] if bool(row.get("exclude_gt_solution")) else None,
        )
    if enum_status == "skipped_enumeration_too_large":
        return "skipped_too_large"
    if enum_status is not None:
        return f"skipped_{enum_status}"
    if not representatives:
        return "skipped_empty"
    saved = save_node_enumeration_cache(
        cache_path=cache_path,
        metadata=metadata,
        enumerated_subgraphs=enumerated_subgraphs,
        representatives=representatives,
    )
    return f"{status}_saved" if saved else f"{status}_save_failed"


def prewarm_node_necessity_cache_for_instance(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    max_subgraphs: int,
    cache_dir: Path,
) -> str:
    budget_k = int(row["budget_k"])
    if budget_k <= 0:
        return "skipped_empty_gt"
    valid_sizes = [budget_k]
    metadata = node_enumeration_cache_metadata(
        row=row,
        x=x,
        candidate_edges=candidate_edges,
        gt_edges=gt_edges,
        valid_sizes=valid_sizes,
        match_features=True,
        cache_kind="node_necessity",
    )
    cache_path = node_enumeration_cache_path(
        cache_dir=cache_dir,
        metadata=metadata,
        cache_kind="node_necessity",
    )
    status, _ = load_graph_necessity_enumeration_manifest(
        cache_path=cache_path,
        metadata=metadata,
    )
    if status == "hit":
        return "hit"

    eligible_subgraphs = 0

    def count_eligible(subset: Tuple[Tuple[int, int], ...]) -> None:
        nonlocal eligible_subgraphs
        if bool(row.get("exclude_gt_solution")) and edge_set(subset) == edge_set(gt_edges):
            return
        eligible_subgraphs += 1

    started = time.perf_counter()
    concrete_mode = (
        "rooted"
        if str(row.get("connectivity")) in {"cut", "flow"}
        else "all"
    )
    enum_status, enumerated_subgraphs = visit_concrete_edge_subsets_cpp(
        candidate_edges=candidate_edges,
        mode=concrete_mode,
        valid_sizes=valid_sizes,
        root_node=int(row["instance_id"]),
        max_subgraphs=int(max_subgraphs),
        visitor=count_eligible,
    )
    if enum_status == "skipped_enumeration_too_large":
        return "skipped_too_large"
    if eligible_subgraphs <= 0:
        return "skipped_empty"
    saved = save_graph_necessity_enumeration_manifest(
        cache_path=cache_path,
        metadata=metadata,
        enumerated_subgraphs=enumerated_subgraphs,
        eligible_subgraphs=eligible_subgraphs,
        enumeration_time_sec=time.perf_counter() - started,
    )
    return f"{status}_saved" if saved else f"{status}_save_failed"


def prewarm_node_enumeration_cache(
    *,
    config: RunConfig,
    dataset: str,
    candidate_nodes: Sequence[int],
    x: np.ndarray,
    edge_index: np.ndarray,
    num_nodes: int,
    gt_graph: Any,
    gt_edge_labels: Any,
    k_hop: int,
    run_log_file: Optional[Path],
) -> None:
    if (
        not config.use_enumeration_cache
        or config.gt_size_solver != "enumeration"
        or "gt_size" not in config.stages
    ):
        return
    if config.oracle_objective in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}:
        cache_kind = "node_sufficiency"
    elif config.oracle_objective in NECESSITY_SCORING_OBJECTIVES:
        cache_kind = "node_necessity"
    else:
        return

    cache_dir = Path(config.enumeration_cache_dir)
    stats: Counter[str] = Counter()
    print_and_log(
        (
            f"[Main] Prewarming {cache_kind} enumeration cache for "
            f"{len(candidate_nodes)} test node instances"
        ),
        run_log_file,
    )
    fixed_self_loop_counts = fixed_self_loop_counts_from_edge_index(
        edge_index,
        num_nodes=int(num_nodes),
    )
    for node_idx in candidate_nodes:
        gt_edges, _ = extract_node_ground_truth_edges_within_hops(
            node_idx=int(node_idx),
            k_hop=k_hop,
            edge_index=edge_index,
            num_nodes=num_nodes,
            gt_graph=gt_graph,
            gt_edge_labels=gt_edge_labels,
            dataset=dataset,
        )
        candidate_edges = candidate_edges_for_node(
            node_idx=int(node_idx),
            k_hop=k_hop,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        row = minimal_node_cache_row(
            dataset=dataset,
            node_idx=int(node_idx),
            budget_k=len(edge_set(gt_edges)),
            connectivity=str(config.connectivity),
            edge_budget_mode=str(config.node_edge_budget_mode),
            oracle_objective=config.oracle_objective,
            exclude_gt_solution=bool(config.exclude_gt_solution),
        )
        prewarm_started = time.perf_counter()
        if cache_kind == "node_sufficiency":
            status = prewarm_node_sufficiency_cache_for_instance(
                row=row,
                x=x,
                fixed_self_loop_counts=fixed_self_loop_counts,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                max_subgraphs=int(config.enumeration_max_subgraphs),
                cache_dir=cache_dir,
            )
        else:
            status = prewarm_node_necessity_cache_for_instance(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                max_subgraphs=int(config.enumeration_max_subgraphs),
                cache_dir=cache_dir,
            )
        prewarm_elapsed_sec = time.perf_counter() - prewarm_started
        append_enumeration_cache_timing(
            cache_dir=cache_dir,
            dataset=dataset,
            task="node",
            cache_kind=cache_kind,
            instance_id=int(node_idx),
            budget_k=len(edge_set(gt_edges)),
            candidate_edge_count=len(edge_set(candidate_edges)),
            exclude_gt_solution=bool(config.exclude_gt_solution),
            cache_status=str(status),
            elapsed_sec=prewarm_elapsed_sec,
        )
        stats[str(status)] += 1

    summary = ", ".join(f"{key}={stats[key]}" for key in sorted(stats)) if stats else "none"
    print_and_log(f"[Main] Enumeration cache prewarm done: {summary}", run_log_file)


def run_node_dataset_seed(
    *,
    config: RunConfig,
    output_dir: Path,
    dataset: str,
    nn_seed: int,
    rows: List[Dict[str, Any]],
    run_log_file: Optional[Path],
) -> None:
    graph, features, labels, _, _, _ = load_dataset(dataset, normalize_features=config.normalize_features)
    edge_index = np.asarray(graph)
    x = np.asarray(features)
    labels_np = np.asarray(labels)
    num_nodes = int(x.shape[0])
    k_hop = int(config.k_hop if config.k_hop is not None else config.num_layers)

    (gt_graph, gt_edge_labels), gt_test_indices = load_dataset_ground_truth(dataset)
    candidate_nodes = [int(idx) for idx in gt_test_indices]
    explanation_label = NODE_EXPLANATION_LABEL_FILTERS.get(normalize_dataset_name(dataset))
    if explanation_label is not None:
        candidate_nodes = [
            int(idx)
            for idx in candidate_nodes
            if int(idx) < int(labels_np.shape[0]) and scalar_int(labels_np[int(idx)]) == int(explanation_label)
        ]
    if config.min_target_degree is not None:
        node_degrees = node_degrees_from_edge_index(edge_index)
        candidate_nodes = [
            int(idx)
            for idx in candidate_nodes
            if int(node_degrees.get(int(idx), 0)) >= int(config.min_target_degree)
        ]
    prewarm_node_enumeration_cache(
        config=config,
        dataset=dataset,
        candidate_nodes=candidate_nodes,
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        gt_graph=gt_graph,
        gt_edge_labels=gt_edge_labels,
        k_hop=k_hop,
        run_log_file=run_log_file,
    )
    if config.max_instances is not None:
        candidate_nodes = candidate_nodes[: int(config.max_instances)]

    model = model_selector(
        config.model_paper,
        dataset,
        pretrained=True,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        use_jk_concat=config.use_jk_concat,
        normalize_features=config.normalize_features,
        nn_seed=nn_seed,
    )
    model.eval()
    detected_jk = bool(getattr(model, "use_jk_concat", config.use_jk_concat))
    explainer = (
        SimpleNamespace(model=model)
        if config.gt_size_solver == "enumeration"
        else MIPExplainer(model=model, dataset_name=dataset, task="node", use_jk_concat=detected_jk, verbose=config.verbose)
    )

    filter_text_parts: List[str] = []
    if explanation_label is not None:
        filter_text_parts.append(f"label={int(explanation_label)}")
    if config.min_target_degree is not None:
        filter_text_parts.append(f"target_degree>={int(config.min_target_degree)}")
    filter_text = "" if not filter_text_parts else " " + " ".join(filter_text_parts)
    print_and_log(f"[Main] {dataset} seed={nn_seed}: {len(candidate_nodes)} node instances{filter_text}", run_log_file)
    for node_idx in candidate_nodes:
        label = scalar_int(labels_np[node_idx])
        pred, c_star, bar_m, num_classes = node_prediction_and_margin(model, x, edge_index, node_idx)
        gt_edges, _ = extract_node_ground_truth_edges_within_hops(
            node_idx=int(node_idx),
            k_hop=k_hop,
            edge_index=edge_index,
            num_nodes=num_nodes,
            gt_graph=gt_graph,
            gt_edge_labels=gt_edge_labels,
            dataset=dataset,
        )
        candidate_edges = candidate_edges_for_node(
            node_idx=int(node_idx),
            k_hop=k_hop,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        planned = planned_rows_for_instance(
            start_row_id=len(rows),
            dataset=dataset,
            task="node",
            nn_seed=nn_seed,
            instance_id=int(node_idx),
            pred=pred,
            label=label,
            c_star=c_star,
            num_classes=num_classes,
            bar_m=bar_m,
            gt_edges=gt_edges,
            candidate_edges=candidate_edges,
            stages=config.stages,
            budget_multipliers=config.budget_multipliers,
            k_hop=k_hop,
            connectivity=config.connectivity,
            node_edge_budget_mode=config.node_edge_budget_mode,
            oracle_objective=config.oracle_objective,
        )
        for row in planned:
            row.update(
                acceleration_row_fields(
                    config,
                    task=str(row["task"]),
                    connectivity=str(row["connectivity"]),
                    edge_budget_mode=str(row["edge_budget_mode"]),
                )
            )
            if row.get("oracle_objective") in NECESSITY_SCORING_OBJECTIVES:
                row["cap_mode"] = "global"
            rows.append(
                run_or_skip_row(
                    row=row,
                    config=config,
                    output_dir=output_dir,
                    explainer=explainer,
                    x=x,
                    edge_index=edge_index,
                    gt_edges=gt_edges,
                    candidate_edges=candidate_edges,
                    root_node=int(node_idx),
                    node_labels=labels_np,
                )
            )


def graph_gt_edges(
    edge_index: np.ndarray,
    gt_edge_labels: Optional[Sequence[Any]],
) -> List[Tuple[int, int]]:
    if gt_edge_labels is None:
        return []
    gt_edges: set[Tuple[int, int]] = set()
    for col in range(int(edge_index.shape[1])):
        if col < len(gt_edge_labels) and float(gt_edge_labels[col]) > 0.0:
            normalized = normalize_edge((int(edge_index[0, col]), int(edge_index[1, col])))
            if normalized is not None:
                gt_edges.add(normalized)
    return sorted(gt_edges)


def minimal_graph_cache_row(
    *,
    dataset: str,
    instance_id: int,
    budget_k: int,
    connectivity: str,
    oracle_objective: str,
    exclude_gt_solution: bool,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "task": "graph",
        "instance_id": int(instance_id),
        "budget_k": int(budget_k),
        "connectivity": str(connectivity),
        "edge_budget_mode": "eq",
        "oracle_objective": str(oracle_objective),
        "exclude_gt_solution": bool(exclude_gt_solution),
    }


def prewarm_graph_sufficiency_cache_for_instance(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    edge_index: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    max_subgraphs: int,
    cache_dir: Path,
) -> str:
    budget_k = int(row["budget_k"])
    if budget_k <= 0:
        return "skipped_empty_gt"
    fixed_self_loop_counts = fixed_self_loop_counts_from_edge_index(
        edge_index,
        num_nodes=int(np.asarray(x).shape[0]),
    )
    node_colors = graph_structural_node_colors(
        x=x,
        fixed_self_loop_counts=fixed_self_loop_counts,
    )

    if (
        str(row.get("oracle_objective"))
        in {"max_sufficiency", "min_sufficiency", RANK_SUFFICIENCY_OBJECTIVE}
        and homogeneous_node_features(x)
        and uniform_graph_edge_index_convention(
        edge_index,
        num_nodes=int(np.asarray(x).shape[0]),
        ) is not None
    ):
        try:
            patterns = edge_mask_topology_patterns(budget_k)
        except RuntimeError:
            patterns = []
        if patterns:
            if str(row.get("connectivity")) == "cut":
                patterns = [pattern for pattern in patterns if pattern_edges_connected(pattern)]
            if len(patterns) > int(max_subgraphs):
                return "skipped_too_large"

            metadata = graph_sufficiency_pattern_cache_metadata(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                patterns=patterns,
                fixed_self_loop_counts=fixed_self_loop_counts,
            )
            cache_path = graph_sufficiency_pattern_cache_path(cache_dir=cache_dir, metadata=metadata)
            status, representatives = load_graph_sufficiency_pattern_cache(cache_path=cache_path, metadata=metadata)
            if status == "hit":
                return "hit"

            candidate_graph = candidate_nx_graph(x=x, candidate_edges=candidate_edges, root_node=None)
            candidate_node_count = int(candidate_graph.number_of_nodes())
            representatives = []
            excluded_edge_sets = [gt_edges] if bool(row.get("exclude_gt_solution")) else None
            for pattern_edges in patterns:
                pattern_nodes = {node for edge in pattern_edges for node in edge}
                if len(pattern_nodes) > candidate_node_count:
                    continue
                pattern_graph = graph_from_pattern_edges(pattern_edges)
                selected = find_unrooted_monomorphism_edges(
                    candidate_graph=candidate_graph,
                    pattern=pattern_graph,
                    match_features=False,
                    excluded_edge_sets=excluded_edge_sets,
                )
                if selected is None:
                    continue
                representatives.append((tuple(pattern_edges), tuple(sorted(edge_set(selected)))))
            if not representatives:
                return "skipped_empty"
            saved = save_graph_sufficiency_pattern_cache(
                cache_path=cache_path,
                metadata=metadata,
                enumerated_subgraphs=len(patterns),
                representatives=representatives,
            )
            return f"{status}_saved" if saved else f"{status}_save_failed"

    connected_only = str(row.get("connectivity")) == "cut"
    excluded_edge_sets = [gt_edges] if bool(row.get("exclude_gt_solution")) else None
    metadata = graph_sufficiency_colored_cache_metadata(
        row=row,
        x=x,
        candidate_edges=candidate_edges,
        gt_edges=gt_edges,
        valid_sizes=[budget_k],
        fixed_self_loop_counts=fixed_self_loop_counts,
        connected_only=connected_only,
    )
    cache_path = graph_sufficiency_colored_cache_path(cache_dir=cache_dir, metadata=metadata)
    status, _, representatives = load_graph_sufficiency_colored_cache(cache_path=cache_path, metadata=metadata)
    if status == "hit":
        return "hit"

    enum_status, enumerated_subgraphs, representatives = graph_edge_mask_shape_first_representatives(
        x=x,
        candidate_edges=candidate_edges,
        node_colors=node_colors,
        valid_sizes=[budget_k],
        max_subgraphs=int(max_subgraphs),
        connected_only=connected_only,
        excluded_edge_sets=excluded_edge_sets,
    )
    if enum_status == "fallback_connected_topology":
        enum_status, enumerated_subgraphs, representatives = graph_shape_first_representatives(
            x=x,
            candidate_edges=candidate_edges,
            node_colors=node_colors,
            valid_sizes=[budget_k],
            max_subgraphs=int(max_subgraphs),
            excluded_edge_sets=excluded_edge_sets,
        )
    if enum_status == "skipped_enumeration_too_large":
        return "skipped_too_large"
    if enum_status is not None:
        return f"skipped_{enum_status}"
    if not representatives:
        return "skipped_empty"
    saved = save_graph_sufficiency_colored_cache(
        cache_path=cache_path,
        metadata=metadata,
        enumerated_subgraphs=enumerated_subgraphs,
        representatives=representatives,
    )
    return f"{status}_saved" if saved else f"{status}_save_failed"


def prewarm_graph_necessity_cache_for_instance(
    *,
    row: Dict[str, Any],
    x: np.ndarray,
    candidate_edges: Sequence[Tuple[int, int]],
    gt_edges: Sequence[Tuple[int, int]],
    max_subgraphs: int,
    cache_dir: Path,
) -> str:
    budget_k = int(row["budget_k"])
    if budget_k <= 0:
        return "skipped_empty_gt"
    positive_valid_sizes = [budget_k]
    metadata = graph_necessity_enumeration_cache_metadata(
        row=row,
        x=x,
        candidate_edges=candidate_edges,
        gt_edges=gt_edges,
        valid_sizes=positive_valid_sizes,
    )
    cache_path = graph_necessity_enumeration_cache_path(cache_dir=cache_dir, metadata=metadata)
    status, _ = load_graph_necessity_enumeration_manifest(cache_path=cache_path, metadata=metadata)
    if status == "hit":
        return "hit"

    gt_edge_set = edge_set(gt_edges)
    exclude_gt_solution = bool(row.get("exclude_gt_solution"))
    eligible_subgraphs = 0

    def count_eligible(subset: Tuple[Tuple[int, int], ...]) -> None:
        nonlocal eligible_subgraphs
        if exclude_gt_solution and edge_set(subset) == gt_edge_set:
            return
        eligible_subgraphs += 1

    enumeration_started = time.perf_counter()
    concrete_mode = (
        "connected"
        if str(row.get("connectivity")) in {"cut", "flow"}
        else "all"
    )
    enum_status, enumerated_subgraphs = visit_concrete_edge_subsets_cpp(
        candidate_edges=candidate_edges,
        mode=concrete_mode,
        valid_sizes=positive_valid_sizes,
        root_node=None,
        max_subgraphs=int(max_subgraphs),
        visitor=count_eligible,
    )
    if enum_status == "skipped_enumeration_too_large":
        return "skipped_too_large"
    enumeration_time_sec = time.perf_counter() - enumeration_started
    if eligible_subgraphs <= 0:
        return "skipped_empty"
    saved = save_graph_necessity_enumeration_manifest(
        cache_path=cache_path,
        metadata=metadata,
        enumerated_subgraphs=enumerated_subgraphs,
        eligible_subgraphs=eligible_subgraphs,
        enumeration_time_sec=enumeration_time_sec,
    )
    return f"{status}_saved" if saved else f"{status}_save_failed"


def prewarm_graph_enumeration_cache(
    *,
    config: RunConfig,
    dataset: str,
    graph_indices: Sequence[int],
    edge_index_list: Sequence[Any],
    features_list: Sequence[Any],
    gt_edge_labels_list: Sequence[Any],
    run_log_file: Optional[Path],
) -> None:
    if (
        not config.use_enumeration_cache
        or config.gt_size_solver != "enumeration"
        or "gt_size" not in config.stages
    ):
        return
    if config.oracle_objective in {
        "max_sufficiency",
        "min_sufficiency",
        RANK_SUFFICIENCY_OBJECTIVE,
    }:
        cache_kind = "graph_sufficiency"
    elif config.oracle_objective in NECESSITY_SCORING_OBJECTIVES:
        cache_kind = "graph_necessity"
    else:
        return

    cache_dir = Path(config.enumeration_cache_dir)
    stats: Counter[str] = Counter()
    print_and_log(
        (
            f"[Main] Prewarming {cache_kind} enumeration cache for "
            f"{len(graph_indices)} test graph instances"
        ),
        run_log_file,
    )
    for graph_idx in graph_indices:
        x = np.asarray(features_list[int(graph_idx)])
        edge_index = np.asarray(edge_index_list[int(graph_idx)])
        gt_labels = gt_edge_labels_list[int(graph_idx)] if int(graph_idx) < len(gt_edge_labels_list) else None
        gt_edges = graph_gt_edges(edge_index, gt_labels)
        budget_k = len(edge_set(gt_edges))
        candidate_edges = unique_nonloop_edges(edge_index)
        row = minimal_graph_cache_row(
            dataset=dataset,
            instance_id=int(graph_idx),
            budget_k=budget_k,
            connectivity=config.connectivity,
            oracle_objective=config.oracle_objective,
            exclude_gt_solution=config.exclude_gt_solution,
        )
        prewarm_started = time.perf_counter()
        if cache_kind == "graph_sufficiency":
            status = prewarm_graph_sufficiency_cache_for_instance(
                row=row,
                x=x,
                edge_index=edge_index,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                max_subgraphs=int(config.enumeration_max_subgraphs),
                cache_dir=cache_dir,
            )
        else:
            status = prewarm_graph_necessity_cache_for_instance(
                row=row,
                x=x,
                candidate_edges=candidate_edges,
                gt_edges=gt_edges,
                max_subgraphs=int(config.enumeration_max_subgraphs),
                cache_dir=cache_dir,
            )
        prewarm_elapsed_sec = time.perf_counter() - prewarm_started
        append_enumeration_cache_timing(
            cache_dir=cache_dir,
            dataset=dataset,
            task="graph",
            cache_kind=cache_kind,
            instance_id=int(graph_idx),
            budget_k=budget_k,
            candidate_edge_count=len(edge_set(candidate_edges)),
            exclude_gt_solution=bool(config.exclude_gt_solution),
            cache_status=str(status),
            elapsed_sec=prewarm_elapsed_sec,
        )
        stats[str(status)] += 1

    summary = ", ".join(f"{key}={stats[key]}" for key in sorted(stats)) if stats else "none"
    print_and_log(f"[Main] Enumeration cache prewarm done: {summary}", run_log_file)


def run_graph_dataset_seed(
    *,
    config: RunConfig,
    output_dir: Path,
    dataset: str,
    nn_seed: int,
    rows: List[Dict[str, Any]],
    run_log_file: Optional[Path],
) -> None:
    edge_index_list, features_list, labels_array, _, _, test_mask = load_dataset(
        dataset,
        normalize_features=config.normalize_features,
    )
    all_test_indices = [int(idx) for idx in np.where(np.asarray(test_mask))[0]]
    (gt_graphs, gt_edge_labels_list), gt_test_indices = load_dataset_ground_truth(dataset, test_indices=all_test_indices)
    del gt_graphs
    gt_index_set = {int(idx) for idx in gt_test_indices}
    cache_graph_indices = [idx for idx in all_test_indices if idx in gt_index_set]
    prewarm_graph_enumeration_cache(
        config=config,
        dataset=dataset,
        graph_indices=cache_graph_indices,
        edge_index_list=edge_index_list,
        features_list=features_list,
        gt_edge_labels_list=gt_edge_labels_list,
        run_log_file=run_log_file,
    )

    graph_indices = list(cache_graph_indices)
    if config.max_instances is not None:
        graph_indices = graph_indices[: int(config.max_instances)]

    model = model_selector(
        config.model_paper,
        dataset,
        pretrained=True,
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        use_jk_concat=config.use_jk_concat,
        normalize_features=config.normalize_features,
        nn_seed=nn_seed,
    )
    model.eval()
    detected_jk = bool(getattr(model, "use_jk_concat", config.use_jk_concat))
    explainer = (
        SimpleNamespace(model=model)
        if config.gt_size_solver == "enumeration"
        else MIPExplainer(model=model, dataset_name=dataset, task="graph", use_jk_concat=detected_jk, verbose=config.verbose)
    )

    print_and_log(f"[Main] {dataset} seed={nn_seed}: {len(graph_indices)} graph instances", run_log_file)
    for graph_idx in graph_indices:
        x = np.asarray(features_list[graph_idx])
        edge_index = np.asarray(edge_index_list[graph_idx])
        label = scalar_int(labels_array[graph_idx])
        pred, c_star, bar_m, num_classes = graph_prediction_and_margin(model, x, edge_index)
        gt_labels = gt_edge_labels_list[graph_idx] if graph_idx < len(gt_edge_labels_list) else None
        gt_edges = graph_gt_edges(edge_index, gt_labels)
        candidate_edges = unique_nonloop_edges(edge_index)
        planned = planned_rows_for_instance(
            start_row_id=len(rows),
            dataset=dataset,
            task="graph",
            nn_seed=nn_seed,
            instance_id=int(graph_idx),
            pred=pred,
            label=label,
            c_star=c_star,
            num_classes=num_classes,
            bar_m=bar_m,
            gt_edges=gt_edges,
            candidate_edges=candidate_edges,
            stages=config.stages,
            budget_multipliers=config.budget_multipliers,
            k_hop=None,
            connectivity=config.connectivity,
            node_edge_budget_mode=config.node_edge_budget_mode,
            oracle_objective=config.oracle_objective,
        )
        for row in planned:
            row.update(
                acceleration_row_fields(
                    config,
                    task=str(row["task"]),
                    connectivity=str(row["connectivity"]),
                    edge_budget_mode=str(row["edge_budget_mode"]),
                )
            )
            rows.append(
                run_or_skip_row(
                    row=row,
                    config=config,
                    output_dir=output_dir,
                    explainer=explainer,
                    x=x,
                    edge_index=edge_index,
                    gt_edges=gt_edges,
                    candidate_edges=candidate_edges,
                    root_node=None,
                    node_labels=None,
                )
            )


def sanitize_run_slug(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "none"
    chars: List[str] = []
    last_separator = False
    for char in text:
        if char.isascii() and (char.isalnum() or char in {"-", "_", "+", ".", "^"}):
            chars.append(char)
            last_separator = False
        elif not last_separator:
            chars.append("-")
            last_separator = True
    slug = "".join(chars).strip("-_+.")
    return slug or "none"


def compact_int_slug(values: Sequence[int]) -> str:
    unique_values = sorted({int(value) for value in values})
    if not unique_values:
        return "none"
    ranges: List[str] = []
    start = unique_values[0]
    previous = unique_values[0]
    for value in unique_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = value
        previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return "+".join(ranges)


def compact_float_slug(values: Sequence[float]) -> str:
    if not values:
        return "none"
    return "+".join(f"{float(value):g}".replace("-", "m").replace(".", "p") for value in values)


def build_default_run_name(
    *,
    args: argparse.Namespace,
    datasets: Sequence[str],
    seeds: Sequence[int],
    stages: Sequence[str],
    multipliers: Sequence[float],
) -> str:
    has_node_task = any(
        get_dataset_config(normalize_dataset_name(dataset))[2] == "node"
        for dataset in datasets
    )
    fidelity_token = (
        "Fid^-"
        if str(args.oracle_objective) in SUFFICIENCY_OBJECTIVES
        else "Fid^+"
    )
    parts = [
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        sanitize_run_slug(args.gt_size_solver),
        fidelity_token,
        "standard",
        sanitize_run_slug(args.preset),
        sanitize_run_slug("+".join(stages)),
        f"{int(args.num_layers)}L{int(args.hidden_dim)}H{'JK' if bool(args.use_jk_concat) else ''}",
        f"obj-{sanitize_run_slug(args.oracle_objective)}",
    ]
    if "gt_size" in stages:
        parts.append(f"gtsize-{sanitize_run_slug(args.gt_size_solver)}")
    if "gt_size" in stages and has_node_task:
        parts.append(f"nodebudget-{sanitize_run_slug(args.node_edge_budget_mode)}")
    if "sweep" in stages:
        parts.append(f"multipliers-{compact_float_slug(multipliers)}")
    if args.connectivity != "none":
        parts.append(f"conn-{sanitize_run_slug(args.connectivity)}")
    if bool(args.exclude_gt):
        parts.append("exclude-gt")
    if args.instance_filter != "correct":
        parts.append(f"filter-{sanitize_run_slug(args.instance_filter)}")
    if args.min_target_degree is not None:
        parts.append(f"mindeg{int(args.min_target_degree)}")
    if args.max_instances is not None:
        parts.append(f"n{int(args.max_instances)}")
    if args.time_limit is not None:
        parts.append(f"tl{float(args.time_limit):g}s".replace(".", "p"))
    return sanitize_run_slug("_".join(parts))


def build_config(args: argparse.Namespace) -> RunConfig:
    datasets = expand_dataset_selectors(parse_csv_strings(args.datasets))
    seeds = parse_csv_ints(args.nn_seeds)
    stages = parse_csv_strings(args.stages)
    invalid_stages = sorted(set(stages) - {"gt_size", "sweep"})
    if invalid_stages:
        raise ValueError(f"Unknown stages: {invalid_stages}. Expected gt_size,sweep.")
    if args.metric == "necessity" and args.oracle_objective == "max_sufficiency":
        args.oracle_objective = "max_necessity"
    if (
        bool(args.exclude_gt)
        and args.gt_size_solver == "enumeration"
        and "gt_size" in stages
    ):
        supported_objectives = {
            "max_sufficiency",
            "min_sufficiency",
            "max_necessity",
            "min_necessity",
        }
        if args.oracle_objective not in supported_objectives:
            raise ValueError(
                "--exclude_gt with --solver enumeration supports standard "
                "sufficiency/necessity GT-size rows only."
            )
    multipliers = (
        parse_csv_nonnegative_floats(args.budget_multipliers, label="Budget multipliers")
        if "sweep" in stages
        else []
    )
    if args.preset == "smoke":
        if args.max_instances is None:
            args.max_instances = 1
        if "sweep" in stages and args.budget_multipliers == DEFAULT_MULTIPLIERS:
            multipliers = [1.10]
    if args.pilot:
        args.max_instances = PILOT_INSTANCE_COUNT

    output_dir = (
        str(args.output_dir)
        if args.output_dir is not None
        else str(
            RESULTS_DIR
            / (
                "standard_fidelity_enumerations"
                if args.gt_size_solver == "enumeration"
                else "standard_fidelity_mip"
            )
        )
    )

    run_name = args.run_name or build_default_run_name(
        args=args,
        datasets=datasets,
        seeds=seeds,
        stages=stages,
        multipliers=multipliers,
    )
    plain_milp = bool(args.plain_milp)
    acceleration_profile = "plain" if plain_milp else "accelerated"
    architecture = str(args.architecture).lower()
    if architecture not in MODEL_PAPERS:
        raise ValueError(f"Unknown architecture {architecture!r}.")
    if architecture != "gin" and args.gt_size_solver != "enumeration":
        raise ValueError("GCN is supported by the generic enumeration backend only, not the Vanilla MILP encoder.")
    return RunConfig(
        preset=args.preset,
        datasets=datasets,
        nn_seeds=seeds,
        instance_filter=args.instance_filter,
        budget_multipliers=multipliers,
        stages=stages,
        max_instances=args.max_instances,
        min_target_degree=args.min_target_degree,
        time_limit_s=args.time_limit,
        output_dir=output_dir,
        run_name=run_name,
        architecture=architecture,
        model_paper=MODEL_PAPERS[architecture],
        num_layers=args.num_layers,
        hidden_dim=int(args.hidden_dim),
        use_jk_concat=bool(args.use_jk_concat),
        normalize_features=not bool(args.disable_feature_normalization),
        readout=args.readout,
        k_hop=args.k_hop,
        connectivity=args.connectivity,
        node_edge_budget_mode=args.node_edge_budget_mode,
        gt_size_solver=args.gt_size_solver,
        enumeration_max_subgraphs=int(args.enumeration_max_subgraphs),
        enumeration_batch_size=int(args.enumeration_batch_size),
        use_enumeration_cache=bool(args.use_enumeration_cache),
        enumeration_cache_dir=args.enumeration_cache_dir,
        oracle_objective=args.oracle_objective,
        gt_incumbent=bool(args.gt_incumbent),
        exclude_gt_solution=bool(args.exclude_gt),
        acceleration_profile=acceleration_profile,
        use_o1_tightening=False if plain_milp else bool(args.use_o1_tightening),
        cap_mode="global",
        use_o3_symmetry_breaking=False if plain_milp else bool(args.use_o3_symmetry_breaking),
        use_o4_star_constraints=False if plain_milp else bool(args.use_o4_star_constraints),
        # O5-O8 are intentionally excluded from this experiment driver.
        use_o9_connectivity_lb=False if plain_milp else bool(args.use_o9_connectivity_lb),
        use_anderson_cuts=False if plain_milp else bool(args.use_anderson_cuts),
        anderson_max_node_count=int(args.anderson_max_node_count),
        anderson_max_cuts=int(args.anderson_max_cuts),
        edge_branch_priority=int(args.edge_branch_priority),
        use_obbt_u=False if plain_milp else bool(args.use_obbt_u),
        max_obbt_u_layers=args.max_obbt_u_layers,
        obbt_rebuild=False if plain_milp else bool(args.obbt_rebuild),
        degree_table_layer1=False if plain_milp else bool(args.degree_table_layer1),
        # Nested cuts and back cuts are intentionally excluded.
        max_root_sep_rounds=int(args.max_root_sep_rounds),
        cut_set_scope=args.cut_set_scope,
        save_figures=bool(args.save_figures),
        padding_seed=int(args.padding_seed),
        metric=args.metric,
        opt_samples=args.opt_samples,
        eval_samples=args.eval_samples,
        gurobi_method=args.gurobi_method,
        gurobi_threads=args.gurobi_threads,
        gurobi_norel_hybrid=bool(args.gurobi_norel_hybrid),
        gurobi_norel_heur_work=args.gurobi_norel_heur_work,
        gurobi_mem_limit=args.gurobi_mem_limit,
        experiment_template=args.experiment_template,
        verbose=bool(args.verbose),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Main standard-fidelity GT-size and budget experiments.")
    parser.add_argument("--preset", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--datasets", "--dataset", dest="datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--nn_seeds", "--nn_seed", dest="nn_seeds", type=str, default=DEFAULT_SEEDS)
    parser.add_argument("--instance_filter", choices=["correct", "all"], default="correct")
    parser.add_argument(
        "--budget_multipliers",
        "--budget_ratios",
        dest="budget_multipliers",
        type=str,
        default=DEFAULT_MULTIPLIERS,
        help=(
            "Comma-separated budget-sweep multipliers. "
            "Each sweep budget is K=ceil(multiplier * |E_GT|), capped by candidate size. "
            "--budget_ratios is accepted as a deprecated alias."
        ),
    )
    parser.add_argument("--stages", type=str, default=DEFAULT_STAGES)
    parser.add_argument(
        "--max_instances",
        type=int,
        default=DEFAULT_MAX_INSTANCES,
        help=(
            "Per-dataset cap on GT-valid test instances "
            f"(default: {DEFAULT_MAX_INSTANCES})."
        ),
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help=(
            f"Run only the first {PILOT_INSTANCE_COUNT} instances per dataset; "
            f"equivalent to --max_instances {PILOT_INSTANCE_COUNT}."
        ),
    )
    parser.add_argument(
        "--min_target_degree",
        type=int,
        default=None,
        help="For node datasets, keep only target nodes whose full-graph undirected degree is at least this value.",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
        help=(
            f"Gurobi TimeLimit in seconds per MILP instance (default: {DEFAULT_TIME_LIMIT:g}). "
            "Preprocessing, model construction and OBBT are outside this budget."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output root. By default enumeration rows use Main/results/standard_fidelity_enumerations "
            "and MILP rows use Main/results/standard_fidelity_mip."
        ),
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--experiment_template",
        type=str,
        default=None,
        help=(
            "Optional Markdown plan copied to experiment.md in every dataset run "
            "directory before evaluation starts. Dataset and run metadata are appended."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=sorted(MODEL_PAPERS),
        default="gin",
        help="Checkpoint architecture. GCN is available only with --solver enumeration.",
    )
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=FIXED_HIDDEN_DIM,
        help=f"Hidden dimension for the selected checkpoint architecture. Default {FIXED_HIDDEN_DIM} preserves Main's current setting.",
    )
    parser.add_argument("--use_jk_concat", action="store_true", default=False)
    parser.add_argument("--disable_feature_normalization", action="store_true", default=False)
    parser.add_argument("--readout", choices=["sum", "mean"], default="sum")
    parser.add_argument("--k_hop", type=int, default=None)
    parser.add_argument(
        "--connectivity",
        choices=["none", "cut"],
        default="none",
        help=(
            "Connectivity policy for graph rows and node explanations. With none, exact-budget "
            "node masks are unrestricted and need not contain an edge incident to v*. "
            "For node and graph necessity-style objectives, explicit removal variables define the "
            "explanation. With cut, explanation edges must be connected (rooted at v* for node tasks, "
            "with a variable root for graph tasks)."
        ),
    )
    parser.add_argument(
        "--solver",
        dest="gt_size_solver",
        choices=["milp", "enumeration"],
        default="milp",
        help="Solver backend for GT-size rows. Budget-sweep rows always use MILP.",
    )
    parser.add_argument(
        "--enumeration_max_subgraphs",
        type=int,
        default=1_000_000,
        help="Maximum candidate subgraphs or abstract topology patterns considered per enumeration row before skipping.",
    )
    parser.add_argument(
        "--enumeration_batch_size",
        type=int,
        default=DEFAULT_ENUMERATION_BATCH_SIZE,
        help=(
            "Maximum number of feasible standard-enumeration masks scored in one "
            f"node/graph GNN forward pass (default: {DEFAULT_ENUMERATION_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--use_enumeration_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse graph-necessity GT-size enumeration representatives across model seeds and reruns.",
    )
    parser.add_argument(
        "--enumeration_cache_dir",
        type=str,
        default=str(DEFAULT_ENUMERATION_CACHE_DIR),
        help="Directory for reusable GT-size enumeration representative caches.",
    )
    parser.add_argument(
        "--node_edge_budget_mode",
        dest="node_edge_budget_mode",
        choices=["eq"],
        default="eq",
        help=(
            "GT-size node-classification edge budget mode for non-rank objectives. "
            "The CLI exposes exact budget only (#edges == |E_GT|); internal le logic is retained."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["sufficiency", "necessity"],
        default=None,
        help=(
            "Optional family shorthand. --metric necessity changes the default "
            "max_sufficiency objective to max_necessity."
        ),
    )
    parser.add_argument(
        "--opt_samples",
        type=int,
        default=None,
        help="Accepted for command compatibility; deterministic GT-size rows do not sample and ignore this value.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=None,
        help="Accepted for command compatibility; deterministic GT-size rows do not sample and ignore this value.",
    )
    parser.add_argument(
        "--objective",
        dest="oracle_objective",
        choices=[
            "max_sufficiency",
            "min_sufficiency",
            "max_necessity",
            "min_necessity",
        ],
        default="max_sufficiency",
        help=(
            "Requested primary extreme. MILP solves exactly this objective. Enumeration "
            "always saves both max/min candidates, figures, scores, and both GT-rank "
            "directions regardless of this choice."
        ),
    )
    parser.add_argument(
        "--plain_milp",
        action="store_true",
        default=False,
        help="Disable the default acceleration bundle for ablation/plain MILP runs.",
    )
    parser.add_argument(
        "--gt_incumbent",
        "--incumbent",
        dest="gt_incumbent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the GT-derived edge mask as a Gurobi MIP start. Disable with --no-incumbent.",
    )
    parser.add_argument(
        "--exclude_gt",
        "--exclude-gt",
        "--gt-nogood-cut",
        dest="exclude_gt",
        action="store_true",
        default=False,
        help=(
            "Exclude the exact planted GT explanation. MILP uses a Hamming-distance no-good "
            "cut; standard-fidelity enumeration skips that exact edge subset. For MILP "
            "necessity, the equivalent internal kept-edge complement is excluded. MILP rows "
            "require --no-incumbent."
        ),
    )
    parser.add_argument("--use_o1_tightening", action=argparse.BooleanOptionalAction, default=True)
    # Distance-based O1 caps are node-specific and intentionally not exposed.
    # parser.add_argument("--cap_mode", choices=["global", "distance"], default="global")
    parser.add_argument(
        "--use_o3_symmetry_breaking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Gurobi aggressive whole-model symmetry detection (Symmetry=2). "
            "Disabling sets Symmetry=0. The legacy custom hanging-tree lex constraints "
            "are not used by this runner."
        ),
    )
    parser.add_argument("--use_o4_star_constraints", action=argparse.BooleanOptionalAction, default=True)
    # Node-specific OR accelerations are intentionally not exposed.
    # parser.add_argument("--use_o5_forced_edges", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--use_o6_bridge_constraints", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--use_o7_aggregated_bridge_constraints", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--use_o8_hierarchical_bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_o9_connectivity_lb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_anderson_cuts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anderson_max_node_count", type=int, default=0)
    parser.add_argument("--anderson_max_cuts", type=int, default=300)
    parser.add_argument(
        "--edge_branch_priority",
        type=int,
        default=0,
        help=(
            "Gurobi BranchPriority assigned to graph edge-mask variables. "
            "0 preserves the default branching behavior; e.g. 100 requests edge-first branching."
        ),
    )
    parser.add_argument(
        "--use_obbt_u",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run LP-based OBBT on graph ReLU preactivations after exact-budget and "
            "GT no-good constraints are installed."
        ),
    )
    parser.add_argument(
        "--max_obbt_u_layers",
        type=int,
        default=None,
        help="Apply graph OBBT to layers 1..N; default is every model layer.",
    )
    parser.add_argument(
        "--obbt_rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rebuild the graph MILP after OBBT so tightened bounds are embedded "
            "in ReLU and downstream McCormick constraints."
        ),
    )
    parser.add_argument(
        "--degree_table_layer1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "3-layer graph MILP with identical node features: represent layer 1 exactly "
            "by one-hot degree variables and the layer-2 messages by exact binary products. "
            "Exactness-preserving; requires identical node features."
        ),
    )
    # Additional OR-specific accelerations are intentionally not exposed.
    # parser.add_argument("--use_nested_cuts", action=argparse.BooleanOptionalAction, default=False)
    # parser.add_argument("--use_back_cuts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_root_sep_rounds", type=int, default=5)
    parser.add_argument(
        "--cut_set_scope",
        choices=["lazy_only", "lazy_and_root"],
        default="lazy_only",
        help=(
            "Connectivity cut scope. In lazy_only component mode, the callback processes "
            "all components disconnected from the root at each MIPSOL and adds one "
            "cut for every such component. Its representative is the minimum node v "
            "with y_sol[v] > 0.5. The explainer's max_cuts_per_round default does not apply."
        ),
    )
    parser.add_argument("--padding_seed", type=int, default=0)
    parser.add_argument(
        "--gurobi_method",
        type=int,
        default=1,
        help="Gurobi Method parameter applied to GT-size and budget-sweep MILP models (default: 1, dual simplex).",
    )
    parser.add_argument(
        "--gurobi_threads",
        type=int,
        default=1,
        help="Gurobi Threads parameter applied to GT-size and budget-sweep MILP models (default: 1).",
    )
    parser.add_argument(
        "--gurobi_norel_hybrid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the NoRel heuristic before the root relaxation, retain any incumbent it finds, "
            "then continue to root LP and B&B in the same optimize() call and time limit "
            "(default: disabled, so Gurobi starts with the root relaxation and B&B)."
        ),
    )
    parser.add_argument(
        "--gurobi_norel_heur_work",
        type=float,
        default=DEFAULT_GUROBI_NOREL_HEUR_WORK,
        help=(
            "Gurobi NoRelHeurWork budget used only when --gurobi_norel_hybrid is enabled "
            "(default budget: 10 work units). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--gurobi_mem_limit",
        type=float,
        default=None,
        help="Optional Gurobi MemLimit parameter in GB applied to GT-size and budget-sweep MILP models.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print Gurobi solver logs to console. Gurobi logs are saved to file regardless.",
    )
    parser.set_defaults(save_figures=False)
    args = parser.parse_args(argv)
    if args.time_limit is not None and args.time_limit <= 0.0:
        raise ValueError("--time_limit must be positive when provided.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be positive.")
    if args.max_instances is not None and args.max_instances <= 0:
        raise ValueError("--max_instances must be positive when provided.")
    if (
        args.pilot
        and args.max_instances is not None
        and args.max_instances not in {DEFAULT_MAX_INSTANCES, PILOT_INSTANCE_COUNT}
    ):
        raise ValueError(
            f"--pilot sets --max_instances {PILOT_INSTANCE_COUNT}; do not pass a conflicting --max_instances."
        )
    if args.min_target_degree is not None and args.min_target_degree < 0:
        raise ValueError("--min_target_degree must be nonnegative when provided.")
    if args.enumeration_max_subgraphs <= 0:
        raise ValueError("--enumeration_max_subgraphs must be positive.")
    if args.enumeration_batch_size <= 0:
        raise ValueError("--enumeration_batch_size must be positive.")
    if args.use_enumeration_cache and not str(args.enumeration_cache_dir).strip():
        raise ValueError("--enumeration_cache_dir must be nonempty when enumeration cache is enabled.")
    if (
        bool(args.exclude_gt)
        and bool(args.gt_incumbent)
        and (args.gt_size_solver == "milp" or "sweep" in parse_csv_strings(args.stages))
    ):
        raise ValueError("--exclude_gt requires --no-incumbent because the GT solution is cut off.")
    if args.max_obbt_u_layers is not None:
        if args.max_obbt_u_layers < 1 or args.max_obbt_u_layers > args.num_layers:
            raise ValueError("--max_obbt_u_layers must be between 1 and --num_layers.")
    if args.obbt_rebuild and not args.use_obbt_u:
        raise ValueError("--obbt_rebuild requires --use_obbt_u.")
    if args.degree_table_layer1 and args.use_obbt_u:
        raise ValueError("--degree_table_layer1 cannot be combined with --use_obbt_u.")
    if args.anderson_max_node_count < 0:
        raise ValueError("--anderson_max_node_count must be nonnegative.")
    if args.anderson_max_cuts <= 0:
        raise ValueError("--anderson_max_cuts must be positive.")
    if args.edge_branch_priority < 0:
        raise ValueError("--edge_branch_priority must be nonnegative.")
    if args.max_root_sep_rounds < 0:
        raise ValueError("--max_root_sep_rounds must be nonnegative.")
    if args.padding_seed < 0:
        raise ValueError("--padding_seed must be nonnegative.")
    if args.opt_samples is not None and args.opt_samples <= 0:
        raise ValueError("--opt_samples must be positive when provided.")
    if args.eval_samples is not None and args.eval_samples <= 0:
        raise ValueError("--eval_samples must be positive when provided.")
    if args.gurobi_method is not None and args.gurobi_method not in {-1, 0, 1, 2, 3, 4, 5}:
        raise ValueError("--gurobi_method must be one of -1,0,1,2,3,4,5 when provided.")
    if args.gurobi_threads is not None and args.gurobi_threads <= 0:
        raise ValueError("--gurobi_threads must be positive when provided.")
    if args.gurobi_norel_heur_work is not None and args.gurobi_norel_heur_work < 0.0:
        raise ValueError("--gurobi_norel_heur_work must be non-negative when provided.")
    if args.gurobi_mem_limit is not None and args.gurobi_mem_limit <= 0.0:
        raise ValueError("--gurobi_mem_limit must be positive when provided.")
    return args


def force_stages_in_argv(argv: Optional[Sequence[str]], stages: str) -> List[str]:
    incoming = list(sys.argv[1:] if argv is None else argv)
    forced: List[str] = []
    idx = 0
    while idx < len(incoming):
        arg = incoming[idx]
        if arg == "--stages":
            idx += 2
            continue
        if arg.startswith("--stages="):
            idx += 1
            continue
        forced.append(arg)
        idx += 1
    forced.extend(["--stages", stages])
    return forced


def main_with_stages(stages: str, argv: Optional[Sequence[str]] = None) -> int:
    return main(force_stages_in_argv(argv, stages))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    output_root = Path(config.output_dir)
    total_rows = 0

    for dataset in config.datasets:
        canonical = normalize_dataset_name(dataset)
        _, _, task = get_dataset_config(canonical)
        run_dir = (
            output_root
            / sanitize_run_slug(canonical)
            / dataset_run_name(config.run_name, canonical)
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        write_experiment_record(config, canonical, run_dir)
        write_dataset_config(config, canonical, run_dir)
        dataset_rows: List[Dict[str, Any]] = []
        for nn_seed in config.nn_seeds:
            seed_dir = seed_result_dir(output_root, config.run_name, canonical, nn_seed)
            run_log_file = initialize_seed_output(
                config=config,
                dataset=canonical,
                nn_seed=nn_seed,
                seed_dir=seed_dir,
            )
            seed_rows: List[Dict[str, Any]] = []
            if task == "node":
                run_node_dataset_seed(
                    config=config,
                    output_dir=seed_dir,
                    dataset=canonical,
                    nn_seed=nn_seed,
                    rows=seed_rows,
                    run_log_file=run_log_file,
                )
            elif task == "graph":
                run_graph_dataset_seed(
                    config=config,
                    output_dir=seed_dir,
                    dataset=canonical,
                    nn_seed=nn_seed,
                    rows=seed_rows,
                    run_log_file=run_log_file,
                )
            else:
                raise ValueError(f"Unsupported task type for {dataset}: {task}")

            write_result_tables(seed_dir, seed_rows)
            dataset_rows.extend(seed_rows)
            # Flush run-level summaries after every seed so interrupted multi-seed
            # runs retain the same usable aggregate tables as robust experiments.
            write_dataset_result_tables(run_dir, dataset_rows)
            total_rows += len(seed_rows)
            print_and_log(
                f"[Main] Wrote rows: {len(seed_rows)} -> {output_path_string(seed_dir)}",
                run_log_file,
            )

    print(f"[Main] Done. Rows: {total_rows}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
