#!/usr/bin/env python3
"""Run Main robust-fidelity optimization and evaluation.

For each instance, this script separates fixed Bernoulli optimization scenarios
from independent evaluation scenarios. In sufficiency mode it records

    Fid_alpha(S) = p_full(c*) - mean_C p_model(c* | S union C).

In necessity mode it records

    Fid+_alpha(S) = p_full(c*) - mean_D p_model(c* | E_c minus D(S)).

The fixed optimization SAA is solved with the scenario-table MILP, which
precomputes true model probabilities per scenario and does not encode the
network. The runner supports this search only, for GIN checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


MAIN_DIR = Path(__file__).resolve().parent
REPO_DIR = MAIN_DIR.parent
RESULTS_DIR = MAIN_DIR / "results"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from Main.data.dataset_loaders import load_dataset
from Main.data.ground_truth_loaders import load_dataset_ground_truth
from Main.models.model_selector import get_dataset_config, model_selector, normalize_dataset_name
from Main.audit_standard import (
    MODEL_PAPERS,
    NODE_EXPLANATION_LABEL_FILTERS,
    PILOT_INSTANCE_COUNT,
    candidate_edges_for_node,
    edge_set,
    graph_gt_edges,
    graph_prediction_and_margin,
    scalar_int,
    unique_nonloop_edges,
    write_edges_csv,
)
from Main.graph_utils import (
    extract_node_ground_truth_edges_within_hops,
)
from Main.robust_common import (
    CONNECTIVITY_MODES,
    normalize_alpha_metric,
    effective_gurobi_norel_heur_work,
    normalize_connectivity,
    EvalContext,
    edge_col_lists,
    edge_weight_template,
    probabilities_from_weights,
)
from Main.solvers.robust_milp import (
    DEFAULT_SAA_TABLE_MAX_ENTRIES,
    solve_saa_table_milp,
)


DEFAULT_DATASETS = "Tree-Cycles,Infection3-one-hot-binary,BA-2Motifs,BA-HouseANDGrid"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_SUFFICIENCY_ALPHAS = "0.9"
DEFAULT_NECESSITY_ALPHAS = "0.1"
DEFAULT_OPT_SAMPLES = 50
DEFAULT_EVAL_SAMPLES = 1000
DEFAULT_COUNTEREXAMPLE_CI_LEVEL = 0.95
DEFAULT_ANDERSON_MAX_NODE_COUNT = 0
DEFAULT_ANDERSON_MAX_CUTS = 300
DEFAULT_MAX_INSTANCES = 30
DEFAULT_GUROBI_SYMMETRY = 2
DEFAULT_GUROBI_NOREL_HEUR_WORK = 10.0
STANDARD_ENDPOINT_COUNTEREXAMPLE_TOLERANCE = 1e-6
MIP_TIME_LIMIT_SECONDS = 600.0
# Fixed settings of this runner, recorded in config.json as before.
SOLVER = "milp"
ARCHITECTURE = "gin"
SAA_MILP_OBJECTIVE = "prob_table"
SUCCESS_STATUSES = {"optimal", "time_limit", "mem_limit", "counterexample_found"}
INCUMBENT_STATUSES = {"interrupted_by_user"}
EVALUABLE_STATUSES = SUCCESS_STATUSES | INCUMBENT_STATUSES

INSTANCE_FIELDS = [
    "dataset",
    "nn_seed",
    "task",
    "instance_id",
    "row_id",
    "target_kind",
    "enumeration_extreme",
    "metric",
    "alpha",
    "alpha_minus",
    "alpha_plus",
    "status",
    "skip_reason",
    "selected_edge_source",
    "selected_size",
    "candidate_size",
    "connectivity",
    "use_incident_cap_tightening",
    "use_star_constraints",
    "use_connectivity_lower_bound",
    "standard_endpoint_pwl_generator",
    "one_swap_candidate_count",
    "one_swap_feasible_count",
    "one_swap_best_score",
    "one_swap_gt_score",
    "one_swap_advantage",
    "one_swap_counterexample_found",
    "one_swap_eval_time_s",
    "one_swap_edge_file",
    "warm_start_source",
    "gurobi_symmetry",
    "edge_branch_priority",
    "gurobi_mip_focus",
    "optimization_backend",
    "optimization_status",
    "optimization_objective",
    "optimization_objective_value",
    "optimization_objective_scale",
    "optimization_objective_type",
    "attack_objective_type",
    "evaluation_metric_type",
    "gt_score_alpha",
    "paired_advantage_definition",
    "paired_scenario_count",
    "paired_advantage_mean",
    "paired_advantage_std",
    "paired_advantage_se",
    "paired_ci_method",
    "paired_ci_scope",
    "paired_ci_level",
    "paired_ci_lower",
    "paired_ci_upper",
    "paired_t_statistic",
    "paired_pvalue_one_sided",
    "counterexample_status",
    "counterexample_supported",
    "optimization_bound",
    "optimization_gap",
    "optimization_time_s",
    "optimization_samples",
    "evaluation_samples",
    "optimization_candidate_count",
    "optimization_model_size",
    "optimization_root_lp_value",
    "optimization_root_lp_time",
    "optimization_callback_time_sec",
    "optimization_callback_calls",
    "optimization_cuts_mipnode",
    "optimization_cuts_mipsol_lazy",
    "optimization_bnb_node_count",
    "optimization_incumbent_obj",
    "enum_objective_max",
    "enum_objective_min",
    "enum_fidelity_at_objective_max",
    "enum_fidelity_at_objective_min",
    "gt_opt_score_alpha",
    "gt_opt_oriented_score",
    "gt_rank_max_objective",
    "gt_rank_min_objective",
    "gt_rank_position",
    "gt_rank_total",
    "gt_rank_tie_count",
    "gt_rank_direction",
    "gt_rank_scope",
    "gt_rank_tolerance",
    "opt_sampling_seed",
    "opt_num_samples",
    "opt_avg_p",
    "opt_avg_p_true",
    "opt_avg_p_pwl",
    "opt_avg_p_sufficiency",
    "opt_avg_p_necessity",
    "opt_delta_true",
    "opt_delta_pwl",
    "pwl_approx_error_at_solution",
    "opt_mean_sampled_context_size",
    "opt_mean_sampled_drop_size",
    "opt_fid_alpha",
    "opt_nec_alpha",
    "opt_score_alpha",
    "saa_sigmoid_pwl_tail_eps",
    "saa_sigmoid_pwl_points",
    "saa_sigmoid_pwl_core_bound",
    "saa_sigmoid_pwl_tail_mode",
    "saa_table_entries",
    "saa_table_build_time_s",
    "saa_table_solve_time_s",
    "saa_table_max_switches",
    "saa_table_mean_switches",
    "saa_table_relevant_edges",
    "saa_table_inert_edges",
    "saa_table_inert_reduction",
    "saa_table_objective_recheck_abs_error",
    "gurobi_log",
    "optimized_edge_file",
    "enum_max_edge_file",
    "enum_min_edge_file",
    "enum_extrema_file",
    "figure_file",
    "enum_max_figure_file",
    "enum_min_figure_file",
    "gt_figure_file",
    "context_pool_size",
    "k_context",
    "num_combinations",
    "sampling_mode",
    "num_samples",
    "sampling_seed",
    "context_keep_prob",
    "necessity_drop_prob",
    "mean_sampled_context_size",
    "mean_sampled_drop_size",
    "p_full",
    "avg_p",
    "avg_p_with_s",
    "avg_p_without_s",
    "fid_alpha",
    "nec_alpha",
    "score_alpha",
    "c_star",
    "pred",
    "label",
    "is_correct",
    "oracle_run_dir",
    "oracle_instance_results",
    "edge_file_used",
    "tie_objective",
    "time_s",
]

SKIP_FIELDS = [
    "dataset",
    "nn_seed",
    "task",
    "instance_id",
    "row_id",
    "target_kind",
    "metric",
    "status",
    "skip_reason",
    "selected_edge_source",
    "oracle_run_dir",
    "oracle_instance_results",
    "edge_file_used",
]

SUMMARY_BY_SEED_FIELDS = [
    "dataset",
    "nn_seed",
    "metric",
    "target_kind",
    "alpha",
    "alpha_minus",
    "alpha_plus",
    "mean_score_alpha",
    "std_score_alpha",
    "mean_fid_alpha",
    "std_fid_alpha",
    "mean_nec_alpha",
    "std_nec_alpha",
    "n",
]

SUMMARY_FIELDS = [
    "dataset",
    "metric",
    "target_kind",
    "alpha",
    "alpha_minus",
    "alpha_plus",
    "mean_score_alpha",
    "std_score_alpha",
    "mean_fid_alpha",
    "std_fid_alpha",
    "mean_nec_alpha",
    "std_nec_alpha",
    "n",
]

COUNTEREXAMPLE_SUMMARY_BY_SEED_FIELDS = [
    "dataset",
    "nn_seed",
    "metric",
    "alpha",
    "alpha_minus",
    "alpha_plus",
    "analysis_unit",
    "ci_method",
    "ci_level",
    "n_instances",
    "mean_paired_advantage",
    "std_paired_advantage",
    "se_paired_advantage",
    "ci_lower",
    "ci_upper",
    "t_statistic",
    "pvalue_one_sided",
    "seed_mean_status",
    "n_counterexample_supported",
    "n_inconclusive",
    "n_gt_better_supported",
    "n_exact_tie",
    "counterexample_supported_rate",
]


def parse_csv_list(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(part) for part in parse_csv_list(value)]


def parse_float_list(value: str) -> List[float]:
    return [float(part) for part in parse_csv_list(value)]


def saa_mip_backend_name(args: argparse.Namespace) -> str:
    return "saa_mip"


def default_alphas_for_metric(metric: str) -> str:
    return DEFAULT_NECESSITY_ALPHAS if metric == "necessity" else DEFAULT_SUFFICIENCY_ALPHAS


def alpha_pair_for_metric(metric: str, alpha: float) -> Tuple[Optional[float], Optional[float]]:
    """Return (alpha_minus, alpha_plus) using the paper's complementary Delta convention."""
    metric = normalize_alpha_metric(metric)
    alpha = float(alpha)
    if metric == "sufficiency":
        return alpha, None
    if metric == "necessity":
        return None, alpha
    return alpha, float(1.0 - alpha)


def normalize_edge(edge: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    a, b = int(edge[0]), int(edge[1])
    if a == b:
        return None
    return (a, b) if a < b else (b, a)


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def format_metric(value: Any) -> str:
    finite = finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.4g}"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def resolve_repo_path(value: str) -> Path:
    """Resolve current paths and preserve reads of original artifact paths."""
    path = Path(str(value))
    if "Version5" in path.parts:
        tail = path.parts[path.parts.index("Version5") + 1:]
        if tail:
            group = "Explainers" if (tail[0].endswith("_experiments") and tail[0] not in {
                "standard_fidelity_experiments", "robust_fidelity_experiments"
            }) or tail[0] == "baseline_robust_reevaluations" else "Main"
            path = REPO_DIR / group / Path(*tail)
    if "ExplanationEvaluation" in path.parts:
        tail = path.parts[path.parts.index("ExplanationEvaluation") + 1:]
        if tail:
            directories = {"datasets": "data", "models": "models", "explainers": "solvers"}
            if tail[0] in directories:
                return MAIN_DIR / directories[tail[0]] / Path(*tail[1:])
            if tail[:2] == ("configs", "training"):
                return MAIN_DIR / "training" / "configs" / Path(*tail[2:])
            if tail == ("configs", "selector.py"):
                return MAIN_DIR / "training" / "config.py"
            if tail == ("tasks", "training.py"):
                return MAIN_DIR / "training" / "engine.py"
    # Relocate immutable result artifacts from both earlier public layouts.
    audit_families = {
        "standard_fidelity_experiments": "standard_fidelity_mip",
        "robust_fidelity_experiments": "robust_fidelity_mip",
        "standard_fidelity_enumerations": "standard_fidelity_enumerations",
        "robust_fidelity_enumerations": "robust_fidelity_enumerations",
    }
    methods = ("random", "gradcam", "gnnexplainer", "pgexplainer", "flowx_suf", "flowx_nec")
    explainer_families = {method + "_experiments": method for method in methods}
    explainer_families["baseline_robust_reevaluations"] = "robust_reevaluations"
    for group, aliases in [
        ("Main", audit_families),
        ("Explainers", explainer_families),
    ]:
        if group in path.parts:
            tail = path.parts[path.parts.index(group) + 1:]
            if tail and tail[0] in aliases:
                return REPO_DIR / group / "results" / aliases[tail[0]] / Path(*tail[1:])
    names = {"run_standard_fidelity_experiment.py": "audit_standard.py",
             "run_robust_fidelity_experiment.py": "audit_robust.py",
             "systematic_training.py": "train_gnn.py"}
    if path.parent.name == "Main" and path.name in names:
        path = path.with_name(names[path.name])
    return path if path.is_absolute() else REPO_DIR / path


def output_path_string(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_DIR.resolve()))
    except ValueError:
        return str(resolved)


def read_edges_csv(path_value: str) -> List[Tuple[int, int]]:
    path = resolve_repo_path(path_value)
    edges: List[Tuple[int, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"src", "dst"}.issubset(set(reader.fieldnames)):
            for row in reader:
                edge = normalize_edge((int(row["src"]), int(row["dst"])))
                if edge is not None:
                    edges.append(edge)
        else:
            f.seek(0)
            plain = csv.reader(f)
            for row in plain:
                if not row or row[0] == "src":
                    continue
                edge = normalize_edge((int(row[0]), int(row[1])))
                if edge is not None:
                    edges.append(edge)
    return sorted(edge_set(edges))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def choose_device(value: str) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def dataset_task(dataset: str) -> str:
    canonical = normalize_dataset_name(dataset)
    _, _, task = get_dataset_config(canonical)
    return str(task)


@dataclass
class DatasetData:
    dataset: str
    task: str
    normalize_features: bool
    graph: Any = None
    features: Any = None
    labels: Any = None
    edge_index_list: Optional[Sequence[np.ndarray]] = None
    features_list: Optional[Sequence[np.ndarray]] = None
    labels_array: Optional[np.ndarray] = None
    test_mask: Optional[np.ndarray] = None
    gt_edge_labels_list: Optional[Sequence[Any]] = None
    gt_test_indices: Optional[Sequence[int]] = None
    edge_index: Optional[np.ndarray] = None
    num_nodes: Optional[int] = None


def load_dataset_data(dataset: str, normalize_features: bool) -> DatasetData:
    task = dataset_task(dataset)
    if task == "graph":
        edge_index_list, features_list, labels_array, _, _, test_mask = load_dataset(
            dataset,
            normalize_features=normalize_features,
        )
        all_test_indices = [int(idx) for idx in np.where(np.asarray(test_mask))[0]]
        (_, gt_edge_labels_list), gt_test_indices = load_dataset_ground_truth(
            dataset,
            test_indices=all_test_indices,
        )
        return DatasetData(
            dataset=dataset,
            task=task,
            normalize_features=normalize_features,
            edge_index_list=edge_index_list,
            features_list=features_list,
            labels_array=np.asarray(labels_array),
            test_mask=np.asarray(test_mask),
            gt_edge_labels_list=gt_edge_labels_list,
            gt_test_indices=[int(idx) for idx in gt_test_indices],
        )

    graph, features, labels, _, _, _ = load_dataset(dataset, normalize_features=normalize_features)
    if hasattr(graph, "edge_index"):
        edge_index = graph.edge_index.detach().cpu().numpy() if torch.is_tensor(graph.edge_index) else graph.edge_index
    else:
        edge_index = np.asarray(graph)
    x = np.asarray(features, dtype=np.float32)
    return DatasetData(
        dataset=dataset,
        task=task,
        normalize_features=normalize_features,
        graph=graph,
        features=x,
        labels=np.asarray(labels),
        edge_index=np.asarray(edge_index, dtype=np.int64),
        num_nodes=int(x.shape[0]),
    )


def build_edge_column_index(edge_index: np.ndarray) -> Tuple[Dict[Tuple[int, int], List[int]], List[int]]:
    edge_to_cols: Dict[Tuple[int, int], List[int]] = {}
    self_loop_cols: List[int] = []
    edge_index = np.asarray(edge_index, dtype=np.int64)
    for col in range(int(edge_index.shape[1])):
        a, b = int(edge_index[0, col]), int(edge_index[1, col])
        if a == b:
            self_loop_cols.append(col)
            continue
        edge = normalize_edge((a, b))
        if edge is not None:
            edge_to_cols.setdefault(edge, []).append(col)
    return edge_to_cols, self_loop_cols


def build_eval_context(
    *,
    dataset: str,
    data: DatasetData,
    row: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
) -> EvalContext:
    task = str(row["task"])
    if task == "graph":
        graph_idx = int(row["instance_id"])
        if data.edge_index_list is None or data.features_list is None:
            raise ValueError(f"Dataset {dataset} was not loaded as a graph dataset.")
        edge_index = np.asarray(data.edge_index_list[graph_idx], dtype=np.int64)
        x = np.asarray(data.features_list[graph_idx], dtype=np.float32)
        candidate_edges = unique_nonloop_edges(edge_index)
        root_node = None
    else:
        if data.edge_index is None or data.features is None or data.num_nodes is None:
            raise ValueError(f"Dataset {dataset} was not loaded as a node dataset.")
        edge_index = np.asarray(data.edge_index, dtype=np.int64)
        x = np.asarray(data.features, dtype=np.float32)
        k_hop_value = row.get("k_hop")
        if k_hop_value in (None, ""):
            k_hop = int(config.get("num_layers", 3))
        else:
            k_hop = int(float(k_hop_value))
        root_node = int(row["instance_id"])
        candidate_edges = candidate_edges_for_node(
            node_idx=root_node,
            k_hop=k_hop,
            edge_index=edge_index,
            num_nodes=int(data.num_nodes),
        )

    edge_to_cols, self_loop_cols = build_edge_column_index(edge_index)
    return EvalContext(
        dataset=dataset,
        task=task,
        x=x,
        edge_index=edge_index,
        candidate_edges=sorted(edge_set(candidate_edges)),
        c_star=int(float(row["c_star"])),
        root_node=root_node,
        device=device,
        x_tensor=torch.tensor(x, dtype=torch.float32, device=device),
        edge_index_tensor=torch.tensor(edge_index, dtype=torch.long, device=device),
        edge_to_cols=edge_to_cols,
        self_loop_cols=self_loop_cols,
    )


def format_signed_float_for_title(value: Any) -> str:
    finite = finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:+.4g}"


def node_labels_for_context(data: DatasetData, ctx: EvalContext) -> Optional[np.ndarray]:
    if ctx.task != "node" or data.labels is None:
        return None
    return np.asarray(data.labels)


def probability_for_edges(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    selected_edges: Iterable[Tuple[int, int]],
) -> float:
    template = edge_weight_template(ctx, selected_edges)
    probs = probabilities_from_weights(
        model=model,
        ctx=ctx,
        edge_weights=template,
        batch_size=1,
    )
    return float(probs[0].item())


def sampled_context_average_probability(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    base_edges: Sequence[Tuple[int, int]],
    context_edges: Sequence[Tuple[int, int]],
    context_masks: np.ndarray,
    batch_size: int,
    return_probabilities: bool = False,
) -> Tuple[float, int, float, Optional[np.ndarray]]:
    context_edges = sorted(edge_set(context_edges))
    n_context = len(context_edges)
    masks = np.asarray(context_masks, dtype=bool)
    if masks.ndim != 2 or int(masks.shape[1]) != n_context:
        raise ValueError(f"context_masks shape {masks.shape} does not match context size {n_context}.")
    sample_count = int(masks.shape[0])
    if sample_count <= 0:
        raise ValueError("At least one sampled context is required.")

    base_template = edge_weight_template(ctx, base_edges)
    context_cols = edge_col_lists(ctx, context_edges)

    total_prob = 0.0
    total_count = 0
    probability_chunks: List[np.ndarray] = []
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for start in range(0, sample_count, batch_size):
        batch_masks = masks[start : start + batch_size]
        current_size = int(batch_masks.shape[0])
        weights_2d = base_template.repeat(current_size, 1)
        for row_idx, mask in enumerate(batch_masks):
            for context_idx in np.flatnonzero(mask):
                cols = context_cols[int(context_idx)]
                weights_2d[row_idx, torch.tensor(cols, dtype=torch.long, device=ctx.device)] = 1.0
        probs = probabilities_from_weights(
            model=model,
            ctx=ctx,
            edge_weights=weights_2d.reshape(-1),
            batch_size=current_size,
        )
        total_prob += float(probs.sum().item())
        total_count += current_size
        if return_probabilities:
            probability_chunks.append(probs.numpy().astype(np.float64, copy=False))

    if total_count != sample_count:
        raise RuntimeError(f"Expected {sample_count} sampled contexts, evaluated {total_count}.")
    mean_context_size = float(masks.sum(axis=1).mean()) if sample_count else 0.0
    scenario_probabilities = (
        np.concatenate(probability_chunks)
        if return_probabilities
        else None
    )
    if scenario_probabilities is not None and int(scenario_probabilities.size) != sample_count:
        raise RuntimeError(
            f"Expected {sample_count} stored context probabilities, got {scenario_probabilities.size}."
        )
    return float(total_prob / total_count), int(total_count), mean_context_size, scenario_probabilities


def sampled_drop_average_probability(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    full_edges: Sequence[Tuple[int, int]],
    drop_edges: Sequence[Tuple[int, int]],
    drop_masks: np.ndarray,
    batch_size: int,
    return_probabilities: bool = False,
) -> Tuple[float, int, float, Optional[np.ndarray]]:
    drop_edges = sorted(edge_set(drop_edges))
    masks = np.asarray(drop_masks, dtype=bool)
    if masks.ndim != 2 or int(masks.shape[1]) != len(drop_edges):
        raise ValueError(f"drop_masks shape {masks.shape} does not match drop edge size {len(drop_edges)}.")
    sample_count = int(masks.shape[0])
    if sample_count <= 0:
        raise ValueError("At least one sampled drop context is required.")

    full_template = edge_weight_template(ctx, full_edges)
    drop_cols = edge_col_lists(ctx, drop_edges)
    total_prob = 0.0
    total_count = 0
    probability_chunks: List[np.ndarray] = []
    for start in range(0, sample_count, int(batch_size)):
        batch_masks = masks[start : start + int(batch_size)]
        current_size = int(batch_masks.shape[0])
        weights_2d = full_template.repeat(current_size, 1)
        for row_idx, mask in enumerate(batch_masks):
            for drop_idx in np.flatnonzero(mask):
                cols = drop_cols[int(drop_idx)]
                weights_2d[row_idx, torch.tensor(cols, dtype=torch.long, device=ctx.device)] = 0.0
        probs = probabilities_from_weights(
            model=model,
            ctx=ctx,
            edge_weights=weights_2d.reshape(-1),
            batch_size=current_size,
        )
        total_prob += float(probs.sum().item())
        total_count += current_size
        if return_probabilities:
            probability_chunks.append(probs.numpy().astype(np.float64, copy=False))
    mean_drop_size = float(masks.sum(axis=1).mean()) if sample_count else 0.0
    scenario_probabilities = (
        np.concatenate(probability_chunks)
        if return_probabilities
        else None
    )
    if scenario_probabilities is not None and int(scenario_probabilities.size) != sample_count:
        raise RuntimeError(
            f"Expected {sample_count} stored drop probabilities, got {scenario_probabilities.size}."
        )
    return float(total_prob / total_count), int(total_count), mean_drop_size, scenario_probabilities


def deterministic_sampling_seed(
    *,
    base_seed: int,
    row: Dict[str, Any],
    alpha: float,
    purpose: str = "eval",
) -> int:
    """Return the fixed scenario seed shared by all targets for one instance/alpha."""
    parts = [
        str(int(base_seed)),
        str(purpose),
        str(row.get("dataset", "")),
        str(row.get("nn_seed", "")),
        str(row.get("task", "")),
        str(row.get("instance_id", "")),
        str(row.get("row_id", "")),
        f"{float(alpha):.17g}",
    ]
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def sample_bernoulli_context_masks(
    *,
    alpha: float,
    n_context: int,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    num_samples = int(num_samples)
    n_context = int(n_context)
    if num_samples <= 0:
        raise ValueError("number of samples must be positive.")
    if n_context < 0:
        raise ValueError("n_context must be nonnegative.")
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if n_context == 0:
        return np.zeros((num_samples, 0), dtype=bool)
    if alpha <= 0.0:
        return np.zeros((num_samples, n_context), dtype=bool)
    if alpha >= 1.0:
        return np.ones((num_samples, n_context), dtype=bool)
    rng = np.random.default_rng(int(seed))
    return rng.random((num_samples, n_context)) < alpha


def evaluate_alpha_candidate_masks(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    selected_edges: Sequence[Tuple[int, int]],
    metric: str,
    p_full: float,
    alpha: float,
    candidate_masks: np.ndarray,
    necessity_candidate_masks: Optional[np.ndarray] = None,
    sampling_seed: int,
    batch_size: int,
    return_scenario_scores: bool = False,
) -> Dict[str, Any]:
    candidate_set = edge_set(ctx.candidate_edges)
    selected_set = edge_set(selected_edges)
    if not selected_set.issubset(candidate_set):
        outside = sorted(selected_set - candidate_set)
        preview = ";".join(f"{a}-{b}" for a, b in outside[:5])
        raise ValueError(f"selected_edges_outside_candidate:{len(outside)}:{preview}")

    candidate_edges = sorted(candidate_set)
    candidate_edge_to_idx = {edge: idx for idx, edge in enumerate(candidate_edges)}
    masks = np.asarray(candidate_masks, dtype=bool)
    if masks.ndim != 2 or int(masks.shape[1]) != len(candidate_edges):
        raise ValueError(f"candidate_masks shape {masks.shape} does not match candidate size {len(candidate_edges)}.")

    alpha_minus, alpha_plus = alpha_pair_for_metric(metric, alpha)
    mean_drop_size: Optional[float] = None
    scenario_score_alpha: Optional[np.ndarray] = None
    if metric == "necessity":
        context_edges = sorted(selected_set)
        selected_indices = [candidate_edge_to_idx[edge] for edge in context_edges]
        drop_masks = masks[:, selected_indices] if selected_indices else np.zeros((int(masks.shape[0]), 0), dtype=bool)
        avg_p_with_s, evaluated, mean_context_size, scenario_probabilities = sampled_drop_average_probability(
            model=model,
            ctx=ctx,
            full_edges=candidate_edges,
            drop_edges=context_edges,
            drop_masks=drop_masks,
            batch_size=batch_size,
            return_probabilities=return_scenario_scores,
        )
        avg_p_without_s = None
        fid_alpha = float(p_full - avg_p_with_s)
        nec_alpha = fid_alpha
        score_alpha = fid_alpha
        mean_drop_size = float(mean_context_size)
        if scenario_probabilities is not None:
            scenario_score_alpha = float(p_full) - scenario_probabilities
    elif metric == "delta":
        if necessity_candidate_masks is None:
            raise ValueError("FidDelta evaluation requires necessity_candidate_masks.")
        necessity_masks = np.asarray(necessity_candidate_masks, dtype=bool)
        if necessity_masks.shape != masks.shape:
            raise ValueError(
                "FidDelta sufficiency/necessity mask shapes must match, got "
                f"{masks.shape} and {necessity_masks.shape}."
            )

        context_edges = sorted(candidate_set - selected_set)
        context_indices = [candidate_edge_to_idx[edge] for edge in context_edges]
        context_masks = (
            masks[:, context_indices]
            if context_indices
            else np.zeros((int(masks.shape[0]), 0), dtype=bool)
        )
        avg_p_with_s, evaluated, mean_context_size, sufficiency_probabilities = sampled_context_average_probability(
            model=model,
            ctx=ctx,
            base_edges=sorted(selected_set),
            context_edges=context_edges,
            context_masks=context_masks,
            batch_size=batch_size,
            return_probabilities=return_scenario_scores,
        )

        drop_edges = sorted(selected_set)
        selected_indices = [candidate_edge_to_idx[edge] for edge in drop_edges]
        drop_masks = (
            necessity_masks[:, selected_indices]
            if selected_indices
            else np.zeros((int(necessity_masks.shape[0]), 0), dtype=bool)
        )
        avg_p_without_s, necessity_evaluated, mean_drop_size, necessity_probabilities = sampled_drop_average_probability(
            model=model,
            ctx=ctx,
            full_edges=candidate_edges,
            drop_edges=drop_edges,
            drop_masks=drop_masks,
            batch_size=batch_size,
            return_probabilities=return_scenario_scores,
        )
        if int(necessity_evaluated) != int(evaluated):
            raise RuntimeError(
                f"FidDelta evaluated {evaluated} sufficiency and {necessity_evaluated} necessity scenarios."
            )
        fid_alpha = float(p_full - avg_p_with_s)
        nec_alpha = float(p_full - avg_p_without_s)
        score_alpha = float(nec_alpha - fid_alpha)
        if sufficiency_probabilities is not None and necessity_probabilities is not None:
            scenario_score_alpha = sufficiency_probabilities - necessity_probabilities
    else:
        context_edges = sorted(candidate_set - selected_set)
        context_indices = [candidate_edge_to_idx[edge] for edge in context_edges]
        context_masks = masks[:, context_indices] if context_indices else np.zeros((int(masks.shape[0]), 0), dtype=bool)
        avg_p_with_s, evaluated, mean_context_size, scenario_probabilities = sampled_context_average_probability(
            model=model,
            ctx=ctx,
            base_edges=sorted(selected_set),
            context_edges=context_edges,
            context_masks=context_masks,
            batch_size=batch_size,
            return_probabilities=return_scenario_scores,
        )
        avg_p_without_s = None
        fid_alpha = float(p_full - avg_p_with_s)
        nec_alpha = None
        score_alpha = fid_alpha
        if scenario_probabilities is not None:
            scenario_score_alpha = float(p_full) - scenario_probabilities

    result = {
        "selected_size": len(selected_set),
        "candidate_size": len(candidate_set),
        "context_pool_size": len(context_edges),
        "num_combinations": int(evaluated),
        "num_samples": int(evaluated),
        "sampling_seed": int(sampling_seed),
        "context_keep_prob": alpha_minus if alpha_minus is not None else alpha_plus,
        "necessity_drop_prob": alpha_plus,
        "mean_sampled_context_size": float(mean_context_size),
        "mean_sampled_drop_size": mean_drop_size,
        "p_full": float(p_full),
        "avg_p": float(avg_p_with_s),
        "avg_p_with_s": float(avg_p_with_s),
        "avg_p_without_s": None if avg_p_without_s is None else float(avg_p_without_s),
        "fid_alpha": fid_alpha,
        "nec_alpha": nec_alpha,
        "score_alpha": score_alpha,
    }
    if return_scenario_scores:
        if scenario_score_alpha is None:
            raise RuntimeError("Scenario-level scores were requested but not produced.")
        if int(scenario_score_alpha.size) != int(evaluated):
            raise RuntimeError(
                f"Expected {evaluated} scenario scores, got {scenario_score_alpha.size}."
            )
        result["_scenario_score_alpha"] = scenario_score_alpha
    return result


def load_model(dataset: str, seed: int, config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    architecture = str(config.get("architecture", "gin")).lower()
    paper = str(config.get("model_paper", MODEL_PAPERS.get(architecture, "")))
    if architecture not in MODEL_PAPERS or paper != MODEL_PAPERS[architecture]:
        raise ValueError(
            f"Invalid checkpoint architecture metadata: architecture={architecture!r}, paper={paper!r}."
        )
    model = model_selector(
        paper,
        dataset,
        pretrained=True,
        num_layers=int(config.get("num_layers", 3)),
        hidden_dim=int(config.get("hidden_dim", 8)),
        use_jk_concat=bool(config.get("use_jk_concat", False)),
        normalize_features=bool(config.get("normalize_features", True)),
        nn_seed=int(seed),
    )
    model.to(device)
    model.eval()
    return model


def standard_instance_results_path(run_dir: Path) -> Path:
    """Resolve the current standard table name, with legacy compatibility."""
    for filename in ("instance_fid.csv", "instance_results.csv"):
        path = run_dir / filename
        if path.exists():
            return path
    return run_dir / "instance_fid.csv"


SAA_MIP_STATUS_LINE_RE = re.compile(
    r"SAA-MILP status=(?P<status>\S+)\s+"
    r"objective=(?P<objective>\S+)\s+"
    r"bound=(?P<bound>\S+)\s+"
    r"gap=(?P<gap>\S+)\s+"
    r"nodes=(?P<nodes>\S+)\s+"
    r"root_lp=(?P<root_lp>\S+)\s+"
    r"root_lp_time=(?P<root_lp_time>\S+)\s+"
    r"model_size=(?P<model_size>\S+)\s+"
    r"anderson_cuts=(?P<anderson_cuts>\S+)\s+"
    r"(?:connectivity_cuts=(?P<connectivity_cuts>\S+)\s+)?"
    r"time_s=(?P<time_s>\S+)"
)


def append_skipped_alpha_rows(
    *,
    instance_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, Any]],
    row: Dict[str, Any],
    target_kind: str,
    metric: str,
    alphas: Sequence[float],
    skip_reason: str,
    selected_edge_source: str,
    run_dir: Path,
    edge_file_used: str = "",
    optimization_time_s: Optional[float] = None,
) -> None:
    skip_row = {
        "dataset": row.get("dataset"),
        "nn_seed": row.get("nn_seed"),
        "task": row.get("task"),
        "instance_id": row.get("instance_id"),
        "row_id": row.get("row_id"),
        "target_kind": target_kind,
        "metric": metric,
        "status": "skipped",
        "skip_reason": skip_reason,
        "selected_edge_source": selected_edge_source,
        "oracle_run_dir": output_path_string(run_dir),
        "oracle_instance_results": output_path_string(standard_instance_results_path(run_dir)),
        "edge_file_used": edge_file_used,
    }
    if optimization_time_s is not None:
        skip_row["optimization_time_s"] = float(optimization_time_s)
    skipped_rows.append(dict(skip_row))
    for alpha in alphas:
        alpha_minus, alpha_plus = alpha_pair_for_metric(metric, float(alpha))
        alpha_row = dict(skip_row)
        alpha_row.update(
            {
                "alpha": float(alpha),
                "alpha_minus": alpha_minus,
                "alpha_plus": alpha_plus,
                "selected_size": None,
                "candidate_size": row.get("candidate_size"),
                "connectivity": row.get("connectivity", "none"),
                "use_star_constraints": row.get("use_star_constraints", ""),
                "use_connectivity_lower_bound": row.get("use_connectivity_lower_bound", ""),
                "context_pool_size": None,
                "k_context": None,
                "num_combinations": None,
                "sampling_mode": "fixed_candidate_bernoulli",
                "num_samples": None,
                "sampling_seed": None,
                "context_keep_prob": None,
                "necessity_drop_prob": None,
                "mean_sampled_context_size": None,
                "mean_sampled_drop_size": None,
                "p_full": None,
                "avg_p": None,
                "avg_p_with_s": None,
                "avg_p_without_s": None,
                "fid_alpha": None,
                "nec_alpha": None,
                "score_alpha": None,
                "c_star": row.get("c_star"),
                "pred": row.get("pred"),
                "label": row.get("label"),
                "is_correct": row.get("is_correct"),
                "time_s": None,
            }
        )
        instance_rows.append(alpha_row)


def evaluate_target(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    row: Dict[str, Any],
    target_kind: str,
    metric: str,
    selected_edges: Sequence[Tuple[int, int]],
    selected_edge_source: str,
    p_full: float,
    alphas: Sequence[float],
    num_samples: int,
    sampling_seed: int,
    batch_size: int,
    run_dir: Path,
    edge_file_used: str,
    tie_objective: Optional[float],
    include_scenario_scores: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        start = time.perf_counter()
        alpha_minus, alpha_plus = alpha_pair_for_metric(metric, float(alpha))
        sample_seed = deterministic_sampling_seed(
            base_seed=int(sampling_seed),
            row=row,
            alpha=float(alpha),
            purpose="eval",
        )
        candidate_masks = sample_bernoulli_context_masks(
            alpha=float(alpha),
            n_context=len(edge_set(ctx.candidate_edges)),
            num_samples=int(num_samples),
            seed=sample_seed,
        )
        necessity_candidate_masks = (
            sample_bernoulli_context_masks(
                alpha=float(alpha_plus),
                n_context=len(edge_set(ctx.candidate_edges)),
                num_samples=int(num_samples),
                seed=sample_seed,
            )
            if metric == "delta" and alpha_plus is not None
            else None
        )
        metrics = evaluate_alpha_candidate_masks(
            model=model,
            ctx=ctx,
            selected_edges=selected_edges,
            metric=metric,
            p_full=p_full,
            alpha=float(alpha),
            candidate_masks=candidate_masks,
            necessity_candidate_masks=necessity_candidate_masks,
            sampling_seed=sample_seed,
            batch_size=batch_size,
            return_scenario_scores=include_scenario_scores,
        )
        elapsed = time.perf_counter() - start
        result_row = {
            "dataset": row.get("dataset"),
            "nn_seed": int(float(row.get("nn_seed"))),
            "task": row.get("task"),
            "instance_id": int(float(row.get("instance_id"))),
            "row_id": int(float(row.get("row_id"))),
            "target_kind": target_kind,
            "metric": metric,
            "alpha": float(alpha),
            "alpha_minus": alpha_minus,
            "alpha_plus": alpha_plus,
            "status": "ok",
            "skip_reason": "",
            "selected_edge_source": selected_edge_source,
            "selected_size": metrics["selected_size"],
            "candidate_size": metrics["candidate_size"],
            "connectivity": row.get("connectivity", "none"),
            "use_star_constraints": row.get("use_star_constraints", ""),
            "use_connectivity_lower_bound": row.get("use_connectivity_lower_bound", ""),
            "standard_endpoint_pwl_generator": bool(
                row.get("standard_endpoint_pwl_generator", False)
            ),
            "context_pool_size": metrics["context_pool_size"],
            "k_context": None,
            "num_combinations": metrics["num_combinations"],
            "sampling_mode": "fixed_candidate_bernoulli",
            "num_samples": metrics["num_samples"],
            "sampling_seed": metrics["sampling_seed"],
            "context_keep_prob": metrics["context_keep_prob"],
            "necessity_drop_prob": metrics["necessity_drop_prob"],
            "mean_sampled_context_size": metrics["mean_sampled_context_size"],
            "mean_sampled_drop_size": metrics["mean_sampled_drop_size"],
            "p_full": metrics["p_full"],
            "avg_p": metrics["avg_p"],
            "avg_p_with_s": metrics["avg_p_with_s"],
            "avg_p_without_s": metrics["avg_p_without_s"],
            "fid_alpha": metrics["fid_alpha"],
            "nec_alpha": metrics["nec_alpha"],
            "score_alpha": metrics["score_alpha"],
            "c_star": int(float(row.get("c_star"))),
            "pred": row.get("pred"),
            "label": row.get("label"),
            "is_correct": row.get("is_correct"),
            "oracle_run_dir": output_path_string(run_dir),
            "oracle_instance_results": output_path_string(standard_instance_results_path(run_dir)),
            "edge_file_used": edge_file_used,
            "tie_objective": tie_objective,
            "time_s": float(elapsed),
        }
        if include_scenario_scores:
            result_row["_scenario_score_alpha"] = metrics["_scenario_score_alpha"]
        rows.append(result_row)
    return rows


def optimization_candidate_masks_for_alpha(
    *,
    row: Dict[str, Any],
    alpha: float,
    sampling_seed: int,
    candidate_size: int,
    opt_samples: int,
    metric: str = "sufficiency",
) -> Tuple[int, np.ndarray, Optional[np.ndarray]]:
    opt_seed = deterministic_sampling_seed(
        base_seed=int(sampling_seed),
        row=row,
        alpha=float(alpha),
        purpose="opt",
    )
    masks = sample_bernoulli_context_masks(
        alpha=float(alpha),
        n_context=int(candidate_size),
        num_samples=int(opt_samples),
        seed=opt_seed,
    )
    _, alpha_plus = alpha_pair_for_metric(metric, float(alpha))
    necessity_masks = (
        sample_bernoulli_context_masks(
            alpha=float(alpha_plus),
            n_context=int(candidate_size),
            num_samples=int(opt_samples),
            seed=opt_seed,
        )
        if metric == "delta" and alpha_plus is not None
        else None
    )
    return int(opt_seed), masks, necessity_masks


def optimization_scenario_metrics(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    row: Dict[str, Any],
    metric: str,
    selected_edges: Sequence[Tuple[int, int]],
    p_full: float,
    alpha: float,
    args: argparse.Namespace,
    candidate_masks: Optional[np.ndarray] = None,
    necessity_candidate_masks: Optional[np.ndarray] = None,
    opt_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if candidate_masks is None or opt_seed is None:
        opt_seed, candidate_masks, necessity_candidate_masks = optimization_candidate_masks_for_alpha(
            row=row,
            alpha=float(alpha),
            sampling_seed=int(args.sampling_seed),
            candidate_size=len(edge_set(ctx.candidate_edges)),
            opt_samples=int(args.opt_samples),
            metric=metric,
        )
    metrics = evaluate_alpha_candidate_masks(
        model=model,
        ctx=ctx,
        selected_edges=selected_edges,
        metric=metric,
        p_full=p_full,
        alpha=float(alpha),
        candidate_masks=candidate_masks,
        necessity_candidate_masks=necessity_candidate_masks,
        sampling_seed=int(opt_seed),
        batch_size=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
    )
    return {
        "opt_sampling_seed": int(opt_seed),
        "opt_num_samples": int(metrics["num_samples"]),
        "opt_avg_p": float(metrics["avg_p"]),
        "opt_avg_p_true": (
            float(metrics["score_alpha"]) if metric == "delta" else float(metrics["avg_p"])
        ),
        "opt_avg_p_sufficiency": (
            float(metrics["avg_p_with_s"]) if metric == "delta" else None
        ),
        "opt_avg_p_necessity": (
            float(metrics["avg_p_without_s"]) if metric == "delta" else None
        ),
        "opt_delta_true": float(metrics["score_alpha"]) if metric == "delta" else None,
        "opt_mean_sampled_context_size": float(metrics["mean_sampled_context_size"]),
        "opt_mean_sampled_drop_size": metrics["mean_sampled_drop_size"],
        "opt_fid_alpha": metrics["fid_alpha"],
        "opt_nec_alpha": metrics["nec_alpha"],
        "opt_score_alpha": metrics["score_alpha"],
    }


def saa_mip_objective_metadata(
    *,
    args: argparse.Namespace,
    optimization_objective_value: Any,
    optimization_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    objective_type = SAA_MILP_OBJECTIVE
    objective_value = finite_float(optimization_objective_value)
    opt_delta_true = finite_float(optimization_metrics.get("opt_delta_true"))
    opt_avg_p_true = finite_float(optimization_metrics.get("opt_avg_p_true"))
    if opt_avg_p_true is None:
        opt_avg_p_true = finite_float(optimization_metrics.get("opt_avg_p"))
    # The table objective is built from true probabilities, so it must match the
    # independent torch recomputation at the returned mask.
    table_recheck_error = None
    if objective_type == "prob_table" and objective_value is not None and opt_avg_p_true is not None:
        table_recheck_error = abs(float(objective_value) - float(opt_avg_p_true))
    return {
        "optimization_objective_value": objective_value,
        "optimization_objective_type": objective_type,
        "attack_objective_type": objective_type,
        "evaluation_metric_type": "true_probability_alpha",
        "opt_avg_p_true": opt_avg_p_true,
        # Columns of the removed sigmoid-PWL formulation stay empty to keep the schema.
        "opt_avg_p_pwl": None,
        "opt_delta_true": opt_delta_true,
        "opt_delta_pwl": None,
        "pwl_approx_error_at_solution": None,
        "saa_sigmoid_pwl_tail_eps": None,
        "saa_sigmoid_pwl_points": None,
        "saa_sigmoid_pwl_core_bound": None,
        "saa_sigmoid_pwl_tail_mode": None,
        "saa_table_objective_recheck_abs_error": table_recheck_error,
    }


def attach_optimization_scenario_metrics(
    rows: Sequence[Dict[str, Any]],
    metrics_by_alpha: Dict[float, Dict[str, Any]],
) -> None:
    for result_row in rows:
        alpha = float(result_row.get("alpha"))
        metrics = metrics_by_alpha.get(alpha)
        if metrics is None:
            continue
        result_row.update(metrics)


def scenario_metric_line(label: str, prefix: str, metrics: Dict[str, Any]) -> str:
    if prefix == "eval":
        score_key = "score_alpha"
        fid_key = "fid_alpha"
        nec_key = "nec_alpha"
        avg_key = "avg_p"
        num_key = "num_samples"
        seed_key = "sampling_seed"
    else:
        score_key = f"{prefix}_score_alpha"
        fid_key = f"{prefix}_fid_alpha"
        nec_key = f"{prefix}_nec_alpha"
        avg_key = f"{prefix}_avg_p"
        num_key = f"{prefix}_num_samples"
        seed_key = f"{prefix}_sampling_seed"
    return (
        f"{label} {prefix}: "
        f"score_alpha={format_metric(metrics.get(score_key))}, "
        f"fid_alpha={format_metric(metrics.get(fid_key))}, "
        f"nec_alpha={format_metric(metrics.get(nec_key))}, "
        f"avg_p={format_metric(metrics.get(avg_key))}, "
        f"samples={metrics.get(num_key)}, "
        f"seed={metrics.get(seed_key)}"
    )


def paired_t_statistics(values: Sequence[float], confidence_level: float) -> Dict[str, Any]:
    """Return a two-sided paired-t CI and one-sided p-value for mean(values) > 0."""
    from scipy.stats import t as student_t

    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1).")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "se": None,
            "ci_lower": None,
            "ci_upper": None,
            "t_statistic": None,
            "pvalue_one_sided": None,
            "status": "insufficient_data",
        }

    mean_value = float(array.mean())
    if n < 2:
        return {
            "n": n,
            "mean": mean_value,
            "std": None,
            "se": None,
            "ci_lower": None,
            "ci_upper": None,
            "t_statistic": None,
            "pvalue_one_sided": None,
            "status": "insufficient_data",
        }

    std_value = float(array.std(ddof=1))
    se_value = float(std_value / math.sqrt(n))
    if se_value <= 0.0:
        lower = upper = mean_value
        if mean_value > 0.0:
            t_statistic = math.inf
            pvalue_one_sided = 0.0
        elif mean_value < 0.0:
            t_statistic = -math.inf
            pvalue_one_sided = 1.0
        else:
            t_statistic = 0.0
            pvalue_one_sided = 1.0
    else:
        critical = float(student_t.ppf(0.5 + confidence_level / 2.0, df=n - 1))
        half_width = float(critical * se_value)
        lower = float(mean_value - half_width)
        upper = float(mean_value + half_width)
        t_statistic = float(mean_value / se_value)
        pvalue_one_sided = float(student_t.sf(t_statistic, df=n - 1))

    if lower > 0.0:
        status = "counterexample_supported"
    elif upper < 0.0:
        status = "gt_better_supported"
    elif lower == 0.0 and upper == 0.0:
        status = "exact_tie"
    else:
        status = "inconclusive"
    return {
        "n": n,
        "mean": mean_value,
        "std": std_value,
        "se": se_value,
        "ci_lower": lower,
        "ci_upper": upper,
        "t_statistic": t_statistic,
        "pvalue_one_sided": pvalue_one_sided,
        "status": status,
    }


def attach_paired_counterexample_statistics(
    *,
    gt_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    confidence_level: float,
    deterministic_endpoint: bool = False,
    absolute_tolerance: float = STANDARD_ENDPOINT_COUNTEREXAMPLE_TOLERANCE,
) -> None:
    """Attach paired SAA statistics or a deterministic endpoint comparison."""
    gt_by_alpha = {float(row["alpha"]): row for row in gt_rows}
    for candidate_row in candidate_rows:
        alpha = float(candidate_row["alpha"])
        gt_row = gt_by_alpha.get(alpha)
        if gt_row is None:
            raise RuntimeError(f"Missing GT evaluation row for paired alpha={alpha}.")
        if int(candidate_row["sampling_seed"]) != int(gt_row["sampling_seed"]):
            raise RuntimeError(
                "Paired comparison requires a shared evaluation sampling seed, got "
                f"{candidate_row['sampling_seed']} and {gt_row['sampling_seed']}."
            )

        candidate_scores = np.asarray(
            candidate_row.pop("_scenario_score_alpha"), dtype=np.float64
        ).reshape(-1)
        gt_scores = np.asarray(gt_row.get("_scenario_score_alpha"), dtype=np.float64).reshape(-1)
        if candidate_scores.shape != gt_scores.shape:
            raise RuntimeError(
                "Paired GT/candidate scenario-score shapes differ: "
                f"{gt_scores.shape} versus {candidate_scores.shape}."
            )

        metric = normalize_alpha_metric(str(candidate_row.get("metric") or "sufficiency"))
        if metric == "sufficiency":
            advantages = gt_scores - candidate_scores
            advantage_definition = "GT_FidMinus_minus_OPT_FidMinus"
        else:
            advantages = candidate_scores - gt_scores
            advantage_definition = "OPT_score_minus_GT_score"
        if deterministic_endpoint:
            if advantages.size != 1:
                raise RuntimeError(
                    "Deterministic Standard endpoint comparison requires exactly one scenario."
                )
            advantage = float(advantages[0])
            tolerance = float(absolute_tolerance)
            if advantage > tolerance:
                endpoint_status = "counterexample_supported"
            elif advantage < -tolerance:
                endpoint_status = "gt_better_supported"
            else:
                endpoint_status = "exact_tie"
            statistics = {
                "n": 1,
                "mean": advantage,
                "std": 0.0,
                "se": 0.0,
                "ci_lower": advantage,
                "ci_upper": advantage,
                "t_statistic": None,
                "pvalue_one_sided": None,
                "status": endpoint_status,
            }
        else:
            statistics = paired_t_statistics(advantages, confidence_level)
        candidate_row.update(
            {
                "gt_score_alpha": finite_float(gt_row.get("score_alpha")),
                "paired_advantage_definition": advantage_definition,
                "paired_scenario_count": statistics["n"],
                "paired_advantage_mean": statistics["mean"],
                "paired_advantage_std": statistics["std"],
                "paired_advantage_se": statistics["se"],
                "paired_ci_method": (
                    "deterministic_endpoint_abs_tol"
                    if deterministic_endpoint
                    else "paired_t"
                ),
                "paired_ci_scope": (
                    "deterministic_endpoint"
                    if deterministic_endpoint
                    else "per_instance_unadjusted"
                ),
                "paired_ci_level": float(confidence_level),
                "paired_ci_lower": statistics["ci_lower"],
                "paired_ci_upper": statistics["ci_upper"],
                "paired_t_statistic": statistics["t_statistic"],
                "paired_pvalue_one_sided": statistics["pvalue_one_sided"],
                "counterexample_status": statistics["status"],
                "counterexample_supported": statistics["status"] == "counterexample_supported",
            }
        )

    for gt_row in gt_rows:
        gt_row.pop("_scenario_score_alpha", None)


def report_paired_counterexample_statistics(
    *,
    rows: Sequence[Dict[str, Any]],
    terminal_log: Path,
    label: str,
) -> None:
    """Write paired-CI decisions to both stdout and the per-instance terminal log."""
    lines: List[str] = []
    for row in rows:
        deterministic_endpoint = (
            str(row.get("paired_ci_method") or "") == "deterministic_endpoint_abs_tol"
        )
        confidence_pct = 100.0 * float(
            row.get("paired_ci_level") or DEFAULT_COUNTEREXAMPLE_CI_LEVEL
        )
        if deterministic_endpoint:
            line = (
                f"{label} deterministic endpoint: "
                f"advantage={format_metric(row.get('paired_advantage_mean'))} "
                f"abs_tol={STANDARD_ENDPOINT_COUNTEREXAMPLE_TOLERANCE:g} "
                f"status={row.get('counterexample_status')} "
                f"definition={row.get('paired_advantage_definition')}"
            )
        else:
            line = (
                f"{label} paired-CI: advantage={format_metric(row.get('paired_advantage_mean'))} "
                f"{confidence_pct:g}%_CI=[{format_metric(row.get('paired_ci_lower'))},"
                f"{format_metric(row.get('paired_ci_upper'))}] "
                f"status={row.get('counterexample_status')} "
                f"paired_scenarios={row.get('paired_scenario_count')} "
                f"definition={row.get('paired_advantage_definition')}"
            )
        lines.append(line)
        print(
            f"[fidelity-alpha] dataset={row.get('dataset')} seed={row.get('nn_seed')} "
            f"row={row.get('row_id')} alpha={row.get('alpha')} {line}",
            flush=True,
        )
    if lines:
        terminal_log.parent.mkdir(parents=True, exist_ok=True)
        with terminal_log.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], int]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None, None, 0
    if len(finite) == 1:
        return finite[0], 0.0, 1
    return float(np.mean(finite)), float(np.std(finite, ddof=1)), len(finite)


def build_counterexample_summary_by_seed(
    instance_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in instance_rows:
        if row.get("status") != "ok":
            continue
        advantage = finite_float(row.get("paired_advantage_mean"))
        if advantage is None:
            continue
        metric = normalize_alpha_metric(str(row.get("metric") or "sufficiency"))
        key = (
            row.get("dataset"),
            int(float(row.get("nn_seed"))),
            metric,
            float(row.get("alpha")),
            float(row.get("paired_ci_level") or DEFAULT_COUNTEREXAMPLE_CI_LEVEL),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        dataset, seed, metric, alpha, confidence_level = key
        advantages = [float(row["paired_advantage_mean"]) for row in rows]
        statistics = paired_t_statistics(advantages, confidence_level)
        status_counts = Counter(str(row.get("counterexample_status") or "") for row in rows)
        n_instances = len(rows)
        summary_rows.append(
            {
                "dataset": dataset,
                "nn_seed": seed,
                "metric": metric,
                "alpha": alpha,
                "alpha_minus": alpha_pair_for_metric(metric, alpha)[0],
                "alpha_plus": alpha_pair_for_metric(metric, alpha)[1],
                "analysis_unit": "instance",
                "ci_method": "paired_t",
                "ci_level": confidence_level,
                "n_instances": n_instances,
                "mean_paired_advantage": statistics["mean"],
                "std_paired_advantage": statistics["std"],
                "se_paired_advantage": statistics["se"],
                "ci_lower": statistics["ci_lower"],
                "ci_upper": statistics["ci_upper"],
                "t_statistic": statistics["t_statistic"],
                "pvalue_one_sided": statistics["pvalue_one_sided"],
                "seed_mean_status": statistics["status"],
                "n_counterexample_supported": status_counts["counterexample_supported"],
                "n_inconclusive": status_counts["inconclusive"],
                "n_gt_better_supported": status_counts["gt_better_supported"],
                "n_exact_tie": status_counts["exact_tie"],
                "counterexample_supported_rate": (
                    float(status_counts["counterexample_supported"] / n_instances)
                    if n_instances
                    else None
                ),
            }
        )
    return summary_rows


def build_summaries(instance_rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_seed: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    overall: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
    for row in instance_rows:
        if row.get("status") != "ok":
            continue
        metric = normalize_alpha_metric(str(row.get("metric") or "sufficiency"))
        score_value = finite_float(row.get("score_alpha"))
        if score_value is None:
            score_value = finite_float(row.get("nec_alpha" if metric == "necessity" else "fid_alpha"))
        if score_value is None:
            continue
        alpha = float(row["alpha"])
        alpha_minus, alpha_plus = alpha_pair_for_metric(metric, alpha)
        seed_key = (row["dataset"], int(row["nn_seed"]), metric, row["target_kind"], alpha)
        all_key = (row["dataset"], metric, row["target_kind"], alpha)
        for store, key in ((by_seed, seed_key), (overall, all_key)):
            bucket = store.setdefault(key, {"score_alpha": [], "fid_alpha": [], "nec_alpha": []})
            bucket["score_alpha"].append(score_value)
            fid_value = finite_float(row.get("fid_alpha"))
            nec_value = finite_float(row.get("nec_alpha"))
            if fid_value is not None:
                bucket["fid_alpha"].append(fid_value)
            if nec_value is not None:
                bucket["nec_alpha"].append(nec_value)

    summary_by_seed: List[Dict[str, Any]] = []
    for key, buckets in sorted(by_seed.items(), key=lambda item: item[0]):
        dataset, seed, metric, target_kind, alpha = key
        mean_value, std_value, n = mean_std(buckets["score_alpha"])
        mean_fid, std_fid, _ = mean_std(buckets["fid_alpha"])
        mean_nec, std_nec, _ = mean_std(buckets["nec_alpha"])
        summary_by_seed.append(
            {
                "dataset": dataset,
                "nn_seed": seed,
                "metric": metric,
                "target_kind": target_kind,
                "alpha": alpha,
                "alpha_minus": alpha_pair_for_metric(metric, alpha)[0],
                "alpha_plus": alpha_pair_for_metric(metric, alpha)[1],
                "mean_score_alpha": mean_value,
                "std_score_alpha": std_value,
                "mean_fid_alpha": mean_fid,
                "std_fid_alpha": std_fid,
                "mean_nec_alpha": mean_nec,
                "std_nec_alpha": std_nec,
                "n": n,
            }
        )

    summary: List[Dict[str, Any]] = []
    for key, buckets in sorted(overall.items(), key=lambda item: item[0]):
        dataset, metric, target_kind, alpha = key
        mean_value, std_value, n = mean_std(buckets["score_alpha"])
        mean_fid, std_fid, _ = mean_std(buckets["fid_alpha"])
        mean_nec, std_nec, _ = mean_std(buckets["nec_alpha"])
        summary.append(
            {
                "dataset": dataset,
                "metric": metric,
                "target_kind": target_kind,
                "alpha": alpha,
                "alpha_minus": alpha_pair_for_metric(metric, alpha)[0],
                "alpha_plus": alpha_pair_for_metric(metric, alpha)[1],
                "mean_score_alpha": mean_value,
                "std_score_alpha": std_value,
                "mean_fid_alpha": mean_fid,
                "std_fid_alpha": std_fid,
                "mean_nec_alpha": mean_nec,
                "std_nec_alpha": std_nec,
                "n": n,
            }
        )
    return summary_by_seed, summary


def flush_dataset_outputs(
    *,
    output_dir: Path,
    instance_rows: Sequence[Dict[str, Any]],
    skipped_rows: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> None:
    summary_by_seed, summary = build_summaries(instance_rows)
    counterexample_summary_by_seed = build_counterexample_summary_by_seed(instance_rows)
    observed_seeds = {
        int(float(row["nn_seed"]))
        for row in (*instance_rows, *skipped_rows)
        if row.get("nn_seed") not in (None, "")
    }
    for seed in sorted({int(seed) for seed in seeds} | observed_seeds):
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_instance_rows = [
            row
            for row in instance_rows
            if row.get("nn_seed") not in (None, "") and int(float(row["nn_seed"])) == seed
        ]
        seed_skipped_rows = [
            row
            for row in skipped_rows
            if row.get("nn_seed") not in (None, "") and int(float(row["nn_seed"])) == seed
        ]
        # Keep the same canonical per-seed table name as standard fidelity. Alpha and
        # target-kind remain explicit columns, so a filename suffix is unnecessary.
        write_csv(seed_dir / "instance_fid.csv", seed_instance_rows, INSTANCE_FIELDS)
        write_csv(
            seed_dir / "counterexample_results.csv",
            [row for row in seed_instance_rows if row.get("counterexample_status")],
            INSTANCE_FIELDS,
        )
        write_csv(seed_dir / "skipped_targets.csv", seed_skipped_rows, SKIP_FIELDS)
    write_csv(output_dir / "summary_by_seed.csv", summary_by_seed, SUMMARY_BY_SEED_FIELDS)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(
        output_dir / "counterexample_summary_by_seed.csv",
        counterexample_summary_by_seed,
        COUNTEREXAMPLE_SUMMARY_BY_SEED_FIELDS,
    )


def parse_saa_mip_status_from_terminal(terminal_log: Path) -> Optional[Dict[str, Any]]:
    if not terminal_log.exists():
        return None
    try:
        text = terminal_log.read_text(errors="replace")
    except Exception:
        return None
    matches = list(SAA_MIP_STATUS_LINE_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    status = str(match.group("status"))
    if status not in EVALUABLE_STATUSES:
        return None
    return {
        "status": status,
        "objective": finite_float(match.group("objective")),
        "bound": finite_float(match.group("bound")),
        "gap": finite_float(match.group("gap")),
        "nodes": finite_float(match.group("nodes")),
        "root_lp": finite_float(match.group("root_lp")),
        "root_lp_time": finite_float(match.group("root_lp_time")),
        "model_size": finite_float(match.group("model_size")),
        "anderson_cuts": finite_float(match.group("anderson_cuts")),
        "connectivity_cuts": finite_float(match.group("connectivity_cuts")),
        "time_s": finite_float(match.group("time_s")),
    }


def alpha_slug(alphas: Sequence[float]) -> str:
    if not alphas:
        return "alphanone"
    return "alpha" + "+".join(f"{float(alpha):g}" for alpha in alphas)


def create_output_dirs(
    args: argparse.Namespace,
    datasets: Sequence[str],
    alphas: Sequence[float],
    metric: str,
) -> Dict[str, Path]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_DIR / output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metric_token = {
        "sufficiency": "Fid^-_",
        "necessity": "Fid^+_",
        "delta": "Fid^Delta_",
    }[metric]
    solver_token = (
        (f"{str(args.solver)}_")
    )
    sample_token = f"_opt{int(args.opt_samples)}_eval{int(args.eval_samples)}"
    row_token = f"_n{int(args.max_rows)}" if args.max_rows is not None else ""
    alpha_token = alpha_slug(alphas)
    if metric == "delta":
        alpha_token = (
            "alpha-minus"
            + "+".join(f"{float(alpha):g}" for alpha in alphas)
            + "_alpha-plus"
            + "+".join(f"{1.0 - float(alpha):g}" for alpha in alphas)
        )
    run_name = (
        args.run_name.strip()
        if args.run_name
        else f"{timestamp}_{solver_token}{metric_token}{alpha_token}{sample_token}{row_token}"
    )

    output_dirs: Dict[str, Path] = {}
    for dataset in datasets:
        output_dir = output_root / dataset / run_name
        if output_dir.exists() and any(output_dir.iterdir()) and not (args.overwrite or args.resume):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[str(dataset)] = output_dir
    return output_dirs


def write_experiment_records(
    args: argparse.Namespace,
    output_dirs: Mapping[str, Path],
    *,
    metric: str,
    alphas: Sequence[float],
) -> None:
    """Materialize the prespecified run plan before any dataset evaluation."""
    if args.experiment_template is None:
        return
    template_path = Path(args.experiment_template)
    if not template_path.is_absolute():
        template_path = REPO_DIR / template_path
    if not template_path.is_file():
        raise FileNotFoundError(f"Experiment template does not exist: {template_path}")
    template = template_path.read_text(encoding="utf-8").rstrip()
    architecture = str(args.architecture).lower()
    for dataset, output_dir in output_dirs.items():
        record_path = output_dir / "experiment.md"
        if record_path.exists():
            raise FileExistsError(f"Experiment record already exists: {record_path}")
        metadata = (
            "\n\n## Materialized run metadata\n\n"
            f"- Dataset: `{dataset}`\n"
            f"- Run name: `{output_dir.name}`\n"
            f"- Architecture namespace: `{MODEL_PAPERS[architecture]}` (`{architecture}`)\n"
            f"- Model shape: `{int(args.num_layers)}L{int(args.hidden_dim)}H`\n"
            f"- Metric: `{metric}`\n"
            f"- Alpha values: `{','.join(str(float(alpha)) for alpha in alphas)}`\n"
            f"- Solver: `{args.solver}`\n"
            f"- Output directory: `{output_dir}`\n"
        )
        record_path.write_text(template + metadata, encoding="utf-8")


def prepare_seed_output_dir(seed_dir: Path, *, save_figures: bool) -> None:
    """Create the common Standard/Robust per-seed artifact skeleton."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "edges").mkdir(exist_ok=True)
    (seed_dir / "logs").mkdir(exist_ok=True)


def solve_and_evaluate_saa_mip_row(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    row: Dict[str, Any],
    metric: str,
    alphas: Sequence[float],
    p_full: float,
    gt_edges: Sequence[Tuple[int, int]],
    args: argparse.Namespace,
    output_dir: Path,
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    instance_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    budget_k = int(float(row.get("budget_k") or len(edge_set(gt_edges))))
    edge_budget_mode = str(row.get("edge_budget_mode") or "eq")
    if metric in {"necessity", "delta"}:
        edge_budget_mode = "eq"
    connectivity = normalize_connectivity(str(row.get("connectivity") or getattr(args, "connectivity", "none")))
    output_edges_dir = output_dir / "edges"
    output_logs_dir = output_dir / "logs"
    output_edges_dir.mkdir(parents=True, exist_ok=True)
    output_logs_dir.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        alpha_slug_text = str(float(alpha)).replace(".", "p")
        row_id = int(float(row.get("row_id") or 0))
        seed = int(float(row.get("nn_seed") or 0))
        opt_seed = deterministic_sampling_seed(
            base_seed=int(args.sampling_seed),
            row=row,
            alpha=float(alpha),
            purpose="opt",
        )
        scenario_masks = sample_bernoulli_context_masks(
            alpha=float(alpha),
            n_context=len(ctx.candidate_edges),
            num_samples=int(args.opt_samples),
            seed=opt_seed,
        )
        _, alpha_plus = alpha_pair_for_metric(metric, float(alpha))
        necessity_scenario_masks = (
            sample_bernoulli_context_masks(
                alpha=float(alpha_plus),
                n_context=len(ctx.candidate_edges),
                num_samples=int(args.opt_samples),
                seed=opt_seed,
            )
            if metric == "delta" and alpha_plus is not None
            else None
        )
        log_file = output_logs_dir / f"saa_mip_seed{seed}_row{row_id:06d}_alpha{alpha_slug_text}.log"
        terminal_log = output_logs_dir / f"terminal_saa_mip_seed{seed}_row{row_id:06d}_alpha{alpha_slug_text}.log"
        edge_file = output_edges_dir / f"saa_mip_edges_seed{seed}_row{row_id:06d}_alpha{alpha_slug_text}.csv"
        if bool(getattr(args, "resume", False)) and edge_file.exists():
            resume_status = parse_saa_mip_status_from_terminal(terminal_log)
            if resume_status is not None:
                try:
                    selected_edges = sorted(edge_set(read_edges_csv(str(edge_file))))
                except Exception:
                    selected_edges = []
                if selected_edges:
                    print(
                        f"[fidelity-alpha] resume: reusing completed SAA-MILP "
                        f"seed={seed} row={row_id} alpha={float(alpha)} status={resume_status['status']}",
                        flush=True,
                    )
                    eval_rows = evaluate_target(
                        model=model,
                        ctx=ctx,
                        row=row,
                        target_kind="saa_mip",
                        metric=metric,
                        selected_edges=selected_edges,
                        selected_edge_source="saa_mip_fixed_saa",
                        p_full=p_full,
                        alphas=[float(alpha)],
                        num_samples=int(args.eval_samples),
                        sampling_seed=int(args.sampling_seed),
                        batch_size=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
                        run_dir=run_dir,
                        edge_file_used=output_path_string(edge_file),
                        tie_objective=finite_float(resume_status.get("objective")),
                        include_scenario_scores=True,
                    )
                    gt_eval_rows_for_ci = evaluate_target(
                        model=model,
                        ctx=ctx,
                        row=row,
                        target_kind="gt",
                        metric=metric,
                        selected_edges=gt_edges,
                        selected_edge_source="gt_edge_file",
                        p_full=p_full,
                        alphas=[float(alpha)],
                        num_samples=int(args.eval_samples),
                        sampling_seed=int(args.sampling_seed),
                        batch_size=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
                        run_dir=run_dir,
                        edge_file_used=str(row.get("gt_edge_file") or ""),
                        tie_objective=None,
                        include_scenario_scores=True,
                    )
                    attach_paired_counterexample_statistics(
                        gt_rows=gt_eval_rows_for_ci,
                        candidate_rows=eval_rows,
                        confidence_level=float(args.counterexample_ci_level),
                        deterministic_endpoint=False,
                    )
                    report_paired_counterexample_statistics(
                        rows=eval_rows,
                        terminal_log=terminal_log,
                        label="SAA-MILP",
                    )
                    saa_opt_metrics = optimization_scenario_metrics(
                        model=model,
                        ctx=ctx,
                        row=row,
                        metric=metric,
                        selected_edges=selected_edges,
                        p_full=p_full,
                        alpha=float(alpha),
                        args=args,
                        candidate_masks=scenario_masks,
                        necessity_candidate_masks=necessity_scenario_masks,
                        opt_seed=opt_seed,
                    )
                    resume_objective = finite_float(resume_status.get("objective"))
                    for eval_row in eval_rows:
                        eval_row.update(
                            {
                                "optimization_backend": saa_mip_backend_name(args),
                                "optimization_status": resume_status.get("status"),
                                "optimization_objective": resume_objective,
                                "optimization_bound": finite_float(resume_status.get("bound")),
                                "optimization_gap": finite_float(resume_status.get("gap")),
                                "optimization_time_s": finite_float(resume_status.get("time_s")),
                                "optimization_samples": int(args.opt_samples),
                                "evaluation_samples": int(args.eval_samples),
                                "optimization_candidate_count": len(edge_set(ctx.candidate_edges)),
                                "optimization_model_size": finite_float(resume_status.get("model_size")),
                                "optimization_root_lp_value": finite_float(resume_status.get("root_lp")),
                                "optimization_root_lp_time": finite_float(resume_status.get("root_lp_time")),
                                "optimization_callback_time_sec": None,
                                "optimization_callback_calls": None,
                                "optimization_cuts_mipnode": finite_float(resume_status.get("anderson_cuts")),
                                "optimization_cuts_mipsol_lazy": finite_float(resume_status.get("connectivity_cuts")),
                                "optimization_bnb_node_count": finite_float(resume_status.get("nodes")),
                                "optimization_incumbent_obj": finite_float(resume_status.get("objective")),
                                "gurobi_log": output_path_string(log_file),
                                "optimized_edge_file": output_path_string(edge_file),
                            }
                        )
                        eval_row.update(saa_opt_metrics)
                        eval_row.update(
                            saa_mip_objective_metadata(
                                args=args,
                                optimization_objective_value=resume_objective,
                                optimization_metrics=saa_opt_metrics,
                            )
                        )
                    instance_rows.extend(eval_rows)
                    continue
        terminal_log.write_text(
            (
                f"# Main fidelity-alpha SAA-MILP"
                f" standard_endpoint_pwl=False\n"
                f"dataset={row.get('dataset')} seed={seed} row_id={row_id} "
                f"metric={metric} alpha={float(alpha)} connectivity={connectivity} "
                f"incident_cap_tightening=False "
                f"star_constraints={bool(args.use_star_constraints) if connectivity in {'cut', 'flow'} else False} "
                f"connectivity_lower_bound={bool(args.use_connectivity_lower_bound) if connectivity in {'cut', 'flow'} else False} "
                f"opt_samples={int(args.opt_samples)} "
                f"eval_samples={int(args.eval_samples)} objective_type={args.saa_milp_objective} "
                f"pwl_points=None "
                f"pwl_tail_eps=None "
                f"pwl_core_bound=None "
                f"pwl_tail_mode=None "
                f"gt_incumbent={bool(args.gt_incumbent)} "
                f"exclude_gt_solution={bool(args.exclude_gt)} "
                f"gurobi_time_limit_s={float(args.time_limit)} "
                f"gurobi_symmetry={int(args.gurobi_symmetry)} "
                f"edge_branch_priority=0 "
                f"mip_focus=None "
                f"norel_hybrid={bool(args.gurobi_norel_hybrid)} "
                f"norel_requested_work={args.gurobi_norel_heur_work} "
                f"norel_effective_work={effective_gurobi_norel_heur_work(args)} "
                f"norel_then_continue_same_optimize_to_root_and_bnb={bool(args.gurobi_norel_hybrid)}\n"
            ),
            encoding="utf-8",
        )
        gt_opt_metrics = optimization_scenario_metrics(
            model=model,
            ctx=ctx,
            row=row,
            metric=metric,
            selected_edges=gt_edges,
            p_full=p_full,
            alpha=float(alpha),
            args=args,
            candidate_masks=scenario_masks,
            necessity_candidate_masks=necessity_scenario_masks,
            opt_seed=opt_seed,
        )
        gt_eval_rows_for_log = evaluate_target(
            model=model,
            ctx=ctx,
            row=row,
            target_kind="gt",
            metric=metric,
            selected_edges=gt_edges,
            selected_edge_source="gt_edge_file",
            p_full=p_full,
            alphas=[float(alpha)],
            num_samples=int(args.eval_samples),
            sampling_seed=int(args.sampling_seed),
            batch_size=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
            run_dir=run_dir,
            edge_file_used=str(row.get("gt_edge_file") or ""),
            tie_objective=None,
            include_scenario_scores=True,
        )
        with terminal_log.open("a", encoding="utf-8") as f:
            f.write(scenario_metric_line("GT", "opt", gt_opt_metrics) + "\n")
            if gt_eval_rows_for_log:
                f.write(scenario_metric_line("GT", "eval", gt_eval_rows_for_log[0]) + "\n")

        one_swap_metadata: Dict[str, Any] = {
            "one_swap_candidate_count": None,
            "one_swap_feasible_count": None,
            "one_swap_best_score": None,
            "one_swap_gt_score": None,
            "one_swap_advantage": None,
            "one_swap_counterexample_found": None,
            "one_swap_eval_time_s": None,
            "one_swap_edge_file": None,
            "warm_start_source": "gt" if bool(args.gt_incumbent) else "none",
            "use_incident_cap_tightening": False,
            "gurobi_symmetry": int(args.gurobi_symmetry),
            "edge_branch_priority": 0,
            "gurobi_mip_focus": None,
        }
        table_metadata: Dict[str, Any] = {}
        start = time.perf_counter()
        try:
            solution, table_metadata = solve_saa_table_milp(
                model=model,
                ctx=ctx,
                metric=metric,
                alpha=float(alpha),
                scenario_masks=scenario_masks,
                edge_budget=budget_k,
                edge_budget_mode=edge_budget_mode,
                connectivity=connectivity,
                gt_edges=gt_edges,
                args=args,
                opt_seed=opt_seed,
                bar_m=float(row.get("bar_m")),
                log_file=log_file,
                terminal_log=terminal_log,
            )
        except Exception as exc:
            append_skipped_alpha_rows(
                instance_rows=instance_rows,
                skipped_rows=skipped_rows,
                row=row,
                target_kind="saa_mip",
                metric=metric,
                alphas=[float(alpha)],
                skip_reason=f"saa_mip_error:{exc}",
                selected_edge_source="saa_mip",
                run_dir=run_dir,
                optimization_time_s=time.perf_counter() - start,
            )
            continue

        elapsed = time.perf_counter() - start
        if solution.status not in EVALUABLE_STATUSES or not solution.selected_edges:
            append_skipped_alpha_rows(
                instance_rows=instance_rows,
                skipped_rows=skipped_rows,
                row=row,
                target_kind="saa_mip",
                metric=metric,
                alphas=[float(alpha)],
                skip_reason=f"saa_mip_status:{solution.status}",
                selected_edge_source="saa_mip",
                run_dir=run_dir,
                optimization_time_s=elapsed,
            )
            continue

        selected_edges = sorted(edge_set(solution.selected_edges))
        write_edges_csv(edge_file, selected_edges)
        eval_rows = evaluate_target(
            model=model,
            ctx=ctx,
            row=row,
            target_kind="saa_mip",
            metric=metric,
            selected_edges=selected_edges,
            selected_edge_source="saa_mip_fixed_saa",
            p_full=p_full,
            alphas=[float(alpha)],
            num_samples=int(args.eval_samples),
            sampling_seed=int(args.sampling_seed),
            batch_size=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
            run_dir=run_dir,
            edge_file_used=output_path_string(edge_file),
            tie_objective=finite_float(solution.objective_value),
            include_scenario_scores=True,
        )
        attach_paired_counterexample_statistics(
            gt_rows=gt_eval_rows_for_log,
            candidate_rows=eval_rows,
            confidence_level=float(args.counterexample_ci_level),
            deterministic_endpoint=False,
        )
        report_paired_counterexample_statistics(
            rows=eval_rows,
            terminal_log=terminal_log,
            label="SAA-MILP",
        )
        saa_opt_metrics = optimization_scenario_metrics(
            model=model,
            ctx=ctx,
            row=row,
            metric=metric,
            selected_edges=selected_edges,
            p_full=p_full,
            alpha=float(alpha),
            args=args,
            candidate_masks=scenario_masks,
            necessity_candidate_masks=necessity_scenario_masks,
            opt_seed=opt_seed,
        )
        solution_objective = finite_float(solution.objective_value)
        for eval_row in eval_rows:
            eval_row.update(
                {
                    "optimization_backend": saa_mip_backend_name(args),
                    "optimization_status": solution.status,
                    "optimization_objective": solution_objective,
                    "optimization_bound": finite_float(solution.best_bound),
                    "optimization_gap": finite_float(solution.final_mip_gap),
                    "optimization_time_s": float(elapsed),
                    "optimization_samples": int(args.opt_samples),
                    "evaluation_samples": int(args.eval_samples),
                    "optimization_candidate_count": len(edge_set(ctx.candidate_edges)),
                    "optimization_model_size": finite_float(getattr(solution, "model_size", None)),
                    "optimization_root_lp_value": finite_float(getattr(solution, "root_lp_value", None)),
                    "optimization_root_lp_time": finite_float(getattr(solution, "root_lp_time", None)),
                    "optimization_callback_time_sec": finite_float(getattr(solution, "callback_time_sec", None)),
                    "optimization_callback_calls": getattr(solution, "callback_calls", None),
                    "optimization_cuts_mipnode": getattr(solution, "cuts_mipnode", None),
                    "optimization_cuts_mipsol_lazy": getattr(solution, "cuts_mipsol_lazy", None),
                    "optimization_bnb_node_count": finite_float(getattr(solution, "bnb_node_count", None)),
                    "optimization_incumbent_obj": finite_float(getattr(solution, "incumbent_obj", None)),
                    "gurobi_log": output_path_string(log_file),
                    "optimized_edge_file": output_path_string(edge_file),
                }
            )
            eval_row.update(one_swap_metadata)
            eval_row.update(table_metadata)
            eval_row.update(saa_opt_metrics)
            eval_row.update(
                saa_mip_objective_metadata(
                    args=args,
                    optimization_objective_value=solution_objective,
                    optimization_metrics=saa_opt_metrics,
                )
            )
        with terminal_log.open("a", encoding="utf-8") as f:
            f.write(scenario_metric_line("SAA-MILP", "opt", saa_opt_metrics) + "\n")
            if eval_rows:
                f.write(scenario_metric_line("SAA-MILP", "eval", eval_rows[0]) + "\n")
        instance_rows.extend(eval_rows)
    return instance_rows, skipped_rows


def graph_prediction_and_margin_on_device(
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    device: torch.device,
) -> Tuple[int, int, float, int]:
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        edge_tensor = torch.tensor(np.asarray(edge_index), dtype=torch.long, device=device)
        logits = model(x_tensor, edge_tensor).reshape(-1).detach().cpu()
    c_star = int(logits.argmax(dim=-1).item())
    if logits.numel() <= 1:
        margin = 0.0
    else:
        others = torch.cat((logits[:c_star], logits[c_star + 1 :]))
        margin = float(logits[c_star].item() - others.max().item())
    return c_star, c_star, margin, int(logits.numel())


def node_prediction_and_margin_on_device(
    model: torch.nn.Module,
    x: np.ndarray,
    edge_index: np.ndarray,
    node_idx: int,
    device: torch.device,
) -> Tuple[int, int, float, int]:
    with torch.no_grad():
        x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        edge_tensor = torch.tensor(np.asarray(edge_index), dtype=torch.long, device=device)
        logits = model(x_tensor, edge_tensor)
    target_logits = logits[int(node_idx)].reshape(-1).detach().cpu()
    c_star = int(target_logits.argmax(dim=-1).item())
    if target_logits.numel() <= 1:
        margin = 0.0
    else:
        others = torch.cat((target_logits[:c_star], target_logits[c_star + 1 :]))
        margin = float(target_logits[c_star].item() - others.max().item())
    return c_star, c_star, margin, int(target_logits.numel())


def build_direct_graph_milp_rows(
    *,
    dataset: str,
    data: DatasetData,
    model: torch.nn.Module,
    seed: int,
    output_dir: Path,
    max_rows: Optional[int],
    metric: str,
    device: torch.device,
    connectivity: str,
    use_star_constraints: bool,
    use_connectivity_lower_bound: bool,
    row_numbering: str = "eligible",
) -> List[Tuple[Dict[str, Any], List[Tuple[int, int]]]]:
    if data.task != "graph":
        return []
    row_connectivity = normalize_connectivity(connectivity)
    if data.edge_index_list is None or data.features_list is None or data.labels_array is None:
        raise ValueError(f"Dataset {dataset} was not loaded as graph data.")
    if data.gt_edge_labels_list is None or data.gt_test_indices is None:
        raise ValueError(f"Dataset {dataset} is missing graph ground-truth labels.")

    gt_index_set = {int(idx) for idx in data.gt_test_indices}
    graph_indices = [int(idx) for idx in np.where(np.asarray(data.test_mask))[0] if int(idx) in gt_index_set]
    rows: List[Tuple[Dict[str, Any], List[Tuple[int, int]]]] = []
    output_edges_dir = output_dir / "edges"
    output_edges_dir.mkdir(parents=True, exist_ok=True)

    for pool_position, graph_idx in enumerate(graph_indices, start=1):
        x = np.asarray(data.features_list[graph_idx], dtype=np.float32)
        edge_index = np.asarray(data.edge_index_list[graph_idx], dtype=np.int64)
        label = scalar_int(data.labels_array[graph_idx])
        pred, c_star, bar_m, num_classes = graph_prediction_and_margin_on_device(
            model=model,
            x=x,
            edge_index=edge_index,
            device=device,
        )
        gt_labels = data.gt_edge_labels_list[graph_idx] if graph_idx < len(data.gt_edge_labels_list) else None
        gt_edges = graph_gt_edges(edge_index, gt_labels)
        candidate_edges = unique_nonloop_edges(edge_index)
        if pred != label or float(bar_m) <= 0.0 or not gt_edges or not candidate_edges:
            continue

        # "pool" numbers rows by position in the candidate pool, as the standard runner does
        # (misclassified instances keep their position); "eligible" counts kept rows only.
        row_id = int(pool_position) if row_numbering == "pool" else len(rows) + 1
        gt_edge_file = output_edges_dir / f"gt_edges_seed{int(seed)}_row{row_id:06d}.csv"
        write_edges_csv(gt_edge_file, gt_edges)
        row = {
            "row_id": row_id,
            "dataset": dataset,
            "task": "graph",
            "nn_seed": int(seed),
            "instance_id": int(graph_idx),
            "stage": "gt_size",
            "budget_multiplier": None,
            "budget_k": len(edge_set(gt_edges)),
            "budget_mode": "gt_size",
            "connectivity": row_connectivity,
            "edge_budget_mode": "eq",
            "use_star_constraints": bool(use_star_constraints) if row_connectivity in {"cut", "flow"} else False,
            "use_connectivity_lower_bound": bool(use_connectivity_lower_bound) if row_connectivity in {"cut", "flow"} else False,
            "oracle_objective": (
                "max_necessity"
                if metric == "necessity"
                else "max_delta" if metric == "delta" else "max_sufficiency"
            ),
            "solver_backend": "saa_mip",
            "k_hop": None,
            "candidate_size": len(edge_set(candidate_edges)),
            "gt_size": len(edge_set(gt_edges)),
            "budget_context_size": max(0, len(edge_set(candidate_edges)) - len(edge_set(gt_edges))),
            "status": "source",
            "pred": int(pred),
            "label": int(label),
            "is_correct": True,
            "c_star": int(c_star),
            "num_classes": int(num_classes),
            "bar_m": float(bar_m),
            "selected_size": None,
            "edge_file": "",
            "gt_edge_file": output_path_string(gt_edge_file),
            "gurobi_log": "",
            "terminal_log": "",
        }
        rows.append((row, sorted(edge_set(gt_edges))))
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    return rows


def build_direct_node_milp_rows(
    *,
    dataset: str,
    data: DatasetData,
    model: torch.nn.Module,
    seed: int,
    output_dir: Path,
    max_rows: Optional[int],
    metric: str,
    device: torch.device,
    k_hop: int,
    connectivity: str,
    use_star_constraints: bool,
    use_connectivity_lower_bound: bool,
    row_numbering: str = "eligible",
) -> List[Tuple[Dict[str, Any], List[Tuple[int, int]]]]:
    if data.task != "node":
        return []
    row_connectivity = normalize_connectivity(connectivity)
    if data.edge_index is None or data.features is None or data.labels is None or data.num_nodes is None:
        raise ValueError(f"Dataset {dataset} was not loaded as node data.")

    (gt_graph, gt_edge_labels), gt_test_indices = load_dataset_ground_truth(dataset)
    labels_np = np.asarray(data.labels)
    candidate_nodes = [int(idx) for idx in gt_test_indices]
    explanation_label = NODE_EXPLANATION_LABEL_FILTERS.get(normalize_dataset_name(dataset))
    if explanation_label is not None:
        candidate_nodes = [
            int(idx)
            for idx in candidate_nodes
            if int(idx) < int(labels_np.shape[0]) and scalar_int(labels_np[int(idx)]) == int(explanation_label)
        ]

    x = np.asarray(data.features, dtype=np.float32)
    edge_index = np.asarray(data.edge_index, dtype=np.int64)
    rows: List[Tuple[Dict[str, Any], List[Tuple[int, int]]]] = []
    output_edges_dir = output_dir / "edges"
    output_edges_dir.mkdir(parents=True, exist_ok=True)

    for pool_position, node_idx in enumerate(candidate_nodes, start=1):
        label = scalar_int(labels_np[int(node_idx)])
        pred, c_star, bar_m, num_classes = node_prediction_and_margin_on_device(
            model=model,
            x=x,
            edge_index=edge_index,
            node_idx=int(node_idx),
            device=device,
        )
        gt_edges, _ = extract_node_ground_truth_edges_within_hops(
            node_idx=int(node_idx),
            k_hop=int(k_hop),
            edge_index=edge_index,
            num_nodes=int(data.num_nodes),
            gt_graph=gt_graph,
            gt_edge_labels=gt_edge_labels,
            dataset=dataset,
        )
        candidate_edges = candidate_edges_for_node(
            node_idx=int(node_idx),
            k_hop=int(k_hop),
            edge_index=edge_index,
            num_nodes=int(data.num_nodes),
        )
        if pred != label or float(bar_m) <= 0.0 or not gt_edges or not candidate_edges:
            continue

        # "pool" numbers rows by position in the candidate pool, as the standard runner does
        # (misclassified instances keep their position); "eligible" counts kept rows only.
        row_id = int(pool_position) if row_numbering == "pool" else len(rows) + 1
        gt_edge_file = output_edges_dir / f"gt_edges_seed{int(seed)}_row{row_id:06d}.csv"
        write_edges_csv(gt_edge_file, gt_edges)
        row = {
            "row_id": row_id,
            "dataset": dataset,
            "task": "node",
            "nn_seed": int(seed),
            "instance_id": int(node_idx),
            "stage": "gt_size",
            "budget_multiplier": None,
            "budget_k": len(edge_set(gt_edges)),
            "budget_mode": "gt_size",
            "connectivity": row_connectivity,
            "edge_budget_mode": "eq",
            "use_star_constraints": bool(use_star_constraints) if row_connectivity in {"cut", "flow"} else False,
            "use_connectivity_lower_bound": bool(use_connectivity_lower_bound) if row_connectivity in {"cut", "flow"} else False,
            "oracle_objective": (
                "max_necessity"
                if metric == "necessity"
                else "max_delta" if metric == "delta" else "max_sufficiency"
            ),
            "solver_backend": "saa_mip",
            "k_hop": int(k_hop),
            "candidate_size": len(edge_set(candidate_edges)),
            "gt_size": len(edge_set(gt_edges)),
            "budget_context_size": max(0, len(edge_set(candidate_edges)) - len(edge_set(gt_edges))),
            "status": "source",
            "pred": int(pred),
            "label": int(label),
            "is_correct": True,
            "c_star": int(c_star),
            "num_classes": int(num_classes),
            "bar_m": float(bar_m),
            "selected_size": None,
            "edge_file": "",
            "gt_edge_file": output_path_string(gt_edge_file),
            "gurobi_log": "",
            "terminal_log": "",
        }
        rows.append((row, sorted(edge_set(gt_edges))))
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    return rows


def run(args: argparse.Namespace) -> Path:
    datasets = parse_csv_list(args.datasets)
    seeds = parse_int_list(args.nn_seeds)
    requested_row_ids = (
        sorted(set(parse_int_list(args.row_ids)))
        if args.row_ids is not None
        else None
    )
    metric = normalize_alpha_metric(args.metric)
    args.solver, args.architecture, args.saa_milp_objective = SOLVER, ARCHITECTURE, SAA_MILP_OBJECTIVE
    solver, architecture = SOLVER, ARCHITECTURE
    args.requested_time_limit = args.time_limit
    if args.output_root is None:
        args.output_root = str(RESULTS_DIR / "robust_fidelity_mip")
    args.connectivity = normalize_connectivity(args.connectivity)
    if args.connectivity not in {"none", "cut"}:
        raise ValueError("The scenario-table MILP supports --connectivity none or cut.")
    if int(args.saa_table_max_entries) <= 0:
        raise ValueError("--saa_table_max_entries must be positive.")
    if float(args.saa_table_mip_gap) < 0.0:
        raise ValueError("--saa_table_mip_gap must be non-negative.")
    alphas = parse_float_list(args.alphas if args.alphas is not None else default_alphas_for_metric(metric))
    if args.pilot:
        if (
            args.max_rows is not None
            and args.max_rows not in {DEFAULT_MAX_INSTANCES, PILOT_INSTANCE_COUNT}
        ):
            raise ValueError(
                f"--pilot sets --max_rows {PILOT_INSTANCE_COUNT}; do not pass a conflicting --max_rows."
            )
        args.max_rows = PILOT_INSTANCE_COUNT
    if not datasets:
        raise ValueError("--datasets must not be empty.")
    if not seeds:
        raise ValueError("--nn_seeds must not be empty.")
    if requested_row_ids is not None and not requested_row_ids:
        raise ValueError("--row_ids must not be empty when provided.")
    if requested_row_ids is not None and any(row_id <= 0 for row_id in requested_row_ids):
        raise ValueError("--row_ids values must be positive.")
    if not alphas:
        raise ValueError("--alphas must not be empty.")
    if bool(args.exclude_gt) and bool(args.gt_incumbent):
        raise ValueError("--exclude_gt requires --no-incumbent because the GT solution is cut off.")
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("--alphas values must be in [0, 1].")
    if args.graph_batch_size <= 0 or args.node_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max_rows must be positive when provided.")
    if args.opt_samples <= 0:
        raise ValueError("--opt_samples must be positive.")
    if args.eval_samples <= 0:
        raise ValueError("--eval_samples must be positive.")
    if not 0.0 < float(args.counterexample_ci_level) < 1.0:
        raise ValueError("--counterexample_ci_level must be in (0, 1).")
    if args.num_layers <= 0:
        raise ValueError("--num_layers must be positive.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be positive.")
    if args.time_limit is not None and args.time_limit <= 0.0:
        raise ValueError("--time_limit must be positive when provided.")
    if args.anderson_max_node_count < 0:
        raise ValueError("--anderson_max_node_count must be non-negative.")
    if args.anderson_max_cuts <= 0:
        raise ValueError("--anderson_max_cuts must be positive.")
    if args.gurobi_threads is not None and args.gurobi_threads <= 0:
        raise ValueError("--gurobi_threads must be positive when provided.")
    if args.gurobi_method is not None and args.gurobi_method not in {-1, 0, 1, 2, 3, 4, 5}:
        raise ValueError("--gurobi_method must be one of -1,0,1,2,3,4,5 when provided.")
    if args.gurobi_numeric_focus is not None and args.gurobi_numeric_focus not in {0, 1, 2, 3}:
        raise ValueError("--gurobi_numeric_focus must be one of 0,1,2,3 when provided.")
    if args.gurobi_norel_heur_work is not None and args.gurobi_norel_heur_work < 0.0:
        raise ValueError("--gurobi_norel_heur_work must be non-negative when provided.")
    if args.gurobi_nodefile_start is not None and args.gurobi_nodefile_start < 0.0:
        raise ValueError("--gurobi_nodefile_start must be non-negative when provided.")
    if args.gurobi_mem_limit is not None and args.gurobi_mem_limit <= 0.0:
        raise ValueError("--gurobi_mem_limit must be positive when provided.")
    if getattr(args, "gurobi_soft_mem_limit", None) is not None and args.gurobi_soft_mem_limit <= 0.0:
        raise ValueError("--gurobi_soft_mem_limit must be positive when provided.")
    if args.torch_threads is not None:
        torch.set_num_threads(int(args.torch_threads))

    output_dirs = create_output_dirs(args, datasets, alphas, metric)
    write_experiment_records(args, output_dirs, metric=metric, alphas=alphas)
    device = choose_device(args.device)

    for dataset, output_dir in output_dirs.items():
        source_selection_text = (
            "direct dataset/GT rows: SAA-MILP does not require a standard-fidelity source run"
        )
        config_payload = {
            "datasets": [dataset],
            "all_requested_datasets": datasets,
            "nn_seeds": seeds,
            "alphas": alphas,
            "alpha_convention": (
                "For metric=delta, each --alphas value is alpha_minus and alpha_plus=1-alpha_minus."
                if metric == "delta"
                else "The --alphas value is the selected metric's Bernoulli probability."
            ),
            "metric": metric,
            "solver": solver,
            "solver_backend": saa_mip_backend_name(args),
            "requested_time_limit_s": args.requested_time_limit,
            "mip_time_limit_s": float(args.time_limit),
            "mip_time_limit_scope": "gurobi_optimize_only_after_preprocessing_and_model_build",
            "standard_endpoint_pwl_generator": False,
            "certificate_scope": (
                ("fixed_scenario_saa_optimization")
            ),
            "standard_endpoint_counterexample_abs_tolerance": (
                (None)
            ),
            "standard_endpoint_one_edge_swap_precheck": (
                (None)
            ),
            "standard_endpoint_warm_start": (
                (None)
            ),
            "standard_endpoint_use_incident_cap_tightening": False,
            "standard_endpoint_gurobi_symmetry": (
                (None)
            ),
            "standard_endpoint_edge_branch_priority": (
                (None)
            ),
            "standard_endpoint_gurobi_mip_focus": (
                (None)
            ),
            "standard_endpoint_stop_rule": (
                (None)
            ),
            # Keys of the removed robust enumeration stay with None to keep the schema.
            "source_objective": None,
            "effective_source_objective": None,
            "standard_fidelity_root": None,
            "output_dir": output_path_string(output_dir),
            "output_layout": "per_dataset_run_with_seed_subdirectories",
            "seed_dir_pattern": "seed{nn_seed}",
            "seed_directories": [f"seed{int(seed)}" for seed in seeds],
            "graph_batch_size": int(args.graph_batch_size),
            "node_batch_size": int(args.node_batch_size),
            "sampling_mode": "fixed_candidate_bernoulli",
            "architecture": architecture,
            "model_paper": MODEL_PAPERS[architecture],
            "num_layers": int(args.num_layers),
            "hidden_dim": int(args.hidden_dim),
            "connectivity": str(args.connectivity),
            "use_star_constraints": bool(args.use_star_constraints) if args.connectivity in {"cut", "flow"} else False,
            "use_connectivity_lower_bound": bool(args.use_connectivity_lower_bound) if args.connectivity in {"cut", "flow"} else False,
            "optimization_samples": int(args.opt_samples),
            "evaluation_samples": int(args.eval_samples),
            "counterexample_ci_method": (
                ("paired_t")
            ),
            "counterexample_ci_scope": (
                ("per_instance_unadjusted")
            ),
            "counterexample_ci_level": float(args.counterexample_ci_level),
            "counterexample_decision_rule": (
                ("counterexample_supported iff the two-sided paired-t CI for the oriented "
                    "MILP advantage lies strictly above zero; per-instance intervals are unadjusted "
                    "for searching across multiple instances")
            ),
            "saa_milp_objective": str(args.saa_milp_objective),
            "attack_objective_type": str(args.saa_milp_objective),
            "evaluation_metric_type": "true_probability_alpha",
            "saa_sigmoid_pwl_tail_eps": None,
            "saa_sigmoid_pwl_points": None,
            "saa_sigmoid_pwl_core_bound": None,
            "saa_sigmoid_pwl_tail_mode": None,
            "saa_table_max_entries": int(args.saa_table_max_entries),
            "saa_table_inert_reduction": bool(args.saa_table_inert_reduction),
            "saa_table_mip_gap": float(args.saa_table_mip_gap),
            "saa_table_conn_root_cuts": bool(getattr(args, "saa_table_conn_root_cuts", False)),
            "saa_table_conn_root_cut_rounds": int(getattr(args, "saa_table_conn_root_cut_rounds", 50)),
            "direct_row_numbering": str(getattr(args, "direct_row_numbering", "eligible")),
            "saa_table_formulation": (
                "prob_table: scenario s depends on S only through S∩D_s (switch edges); "
                "true torch probabilities p_s(T) for all T⊆D_s with |T|<=K; continuous lambda_{s,T} "
                "with sum_T lambda=1 and z_e=sum_{T∋e} lambda for e in D_s; node tasks drop edges whose "
                "endpoints both lie at distance >= num_layers from the target"
            ),
            "saa_enum_max_candidates": None,
            "saa_enum_candidate_batch_size": None,
            "robust_enumeration_cache_dir": None,
            "robust_enumeration_cache_version": None,
            "saa_enum_candidate_family": None,
            "saa_enum_concrete_mode_policy": None,
            "saa_enum_score_batching": None,
            "saa_enum_objective_scale": None,
            "saa_enum_extrema": None,
            "saa_enum_gt_rank": None,
            "saa_enum_gt_rank_abs_tolerance": None,
            "saa_enum_extrema_evaluation": None,
            "node_enumeration_inference_context": None,
            "independent_evaluation_scope": "GT_and_selected_candidate_only_on_shared_paired_scenarios",
            "use_anderson_cuts": bool(args.use_anderson_cuts),
            "anderson_max_node_count": int(args.anderson_max_node_count),
            "anderson_max_cuts": int(args.anderson_max_cuts),
            "gurobi_threads": args.gurobi_threads,
            "gurobi_method": args.gurobi_method,
            "gurobi_numeric_focus": args.gurobi_numeric_focus,
            "gurobi_symmetry": int(args.gurobi_symmetry),
            "gurobi_norel_hybrid": bool(args.gurobi_norel_hybrid),
            "gurobi_norel_heur_work": args.gurobi_norel_heur_work,
            "gurobi_effective_norel_heur_work": effective_gurobi_norel_heur_work(args),
            "gurobi_norel_hybrid_time_scope": (
                "NoRel and subsequent root/B&B share the existing single Gurobi time limit; no model restart"
            ),
            "gurobi_nodefile_start": args.gurobi_nodefile_start,
            "gurobi_nodefile_dir": args.gurobi_nodefile_dir,
            "gurobi_mem_limit": args.gurobi_mem_limit,
            "gurobi_soft_mem_limit": getattr(args, "gurobi_soft_mem_limit", None),
            "sampling_seed": int(args.sampling_seed),
            "device": str(device),
            "max_rows": args.max_rows,
            "requested_row_ids": requested_row_ids,
            "pilot": bool(args.pilot),
            "resume": bool(args.resume),
            "save_figures": bool(args.save_figures),
            "gt_incumbent": bool(args.gt_incumbent),
            "exclude_gt_solution": bool(args.exclude_gt),
            "gt_no_good_cut_definition": (
                "sum_{e in S_GT}(1-z_e) + sum_{e not in S_GT}z_e >= 1"
                if bool(args.exclude_gt)
                else None
            ),
            "metric_formula": {
                "sufficiency": "fid_minus = p_full(c_star | E_c) - mean p_model(c_star | H_minus)",
                "necessity": "fid_plus = p_full(c_star | E_c) - mean p_model(c_star | H_plus)",
                "delta": (
                    "fid_delta = fid_plus - fid_minus = "
                    "mean p_model(c_star | H_minus) - mean p_model(c_star | H_plus)"
                ),
            }[metric],
            "score_column": "score_alpha",
            "optimization_scenario_score_column": "opt_score_alpha",
            "saa_milp_objective_note": (
                "prob_table builds the SAA objective from true model probabilities of every "
                "scenario table entry and does not encode the network. "
                "All fid_alpha, nec_alpha, and score_alpha values are computed by true torch probabilities."
            ),
            "auxiliary_metrics": (
                "Fid- records p_full - avg_p(H_minus); Fid+ records p_full - avg_p(H_plus); "
                "FidDelta records Fid+ - Fid-."
            ),
            "context_policy": (
                "Fid-: selected edges are always retained and non-selected candidate edges are retained with probability alpha. "
                "Fid+: selected edges are dropped with probability alpha from the candidate graph, so alpha=0.1 retains most edges. "
                "FidDelta combines both on the same selected mask, using alpha_minus=alpha and alpha_plus=1-alpha. "
                "SAA-Enum and SAA-MILP use fixed optimization scenarios; inference uses independently fixed evaluation scenarios."
            ),
            "k_context_rounding": "not_used_for_bernoulli_sampling",
            "sampling": f"opt{int(args.opt_samples)}_eval{int(args.eval_samples)}",
            "source_selection": source_selection_text,
            "experiment_template": (
                output_path_string(Path(args.experiment_template))
                if args.experiment_template is not None
                else None
            ),
        }
        (output_dir / "config.json").write_text(json.dumps(json_safe(config_payload), indent=2, sort_keys=True) + "\n")

    dataset_cache: Dict[Tuple[str, bool], DatasetData] = {}
    instance_rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {dataset: [] for dataset in datasets}
    skipped_rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {dataset: [] for dataset in datasets}

    total_start = time.perf_counter()
    print("[fidelity-alpha] output_dirs:")
    for dataset in datasets:
        print(f"  - {dataset}: {output_path_string(output_dirs[dataset])}")
    print(
        f"[fidelity-alpha] device={device}, metric={metric}, alphas={alphas}, "
        f"solver={solver}, connectivity={args.connectivity}, "
        f"saa_milp_objective={args.saa_milp_objective}, "
        f"opt_samples={int(args.opt_samples)}, eval_samples={int(args.eval_samples)}, "
        f"standard_endpoint_pwl=False",
        flush=True,
    )

    for dataset in datasets:
        output_dir = output_dirs[dataset]
        instance_rows = instance_rows_by_dataset[dataset]
        skipped_rows = skipped_rows_by_dataset[dataset]
        for seed in seeds:
            seed_dir = output_dir / f"seed{int(seed)}"
            prepare_seed_output_dir(seed_dir, save_figures=bool(args.save_figures))
            run_config = {
                "architecture": architecture,
                "model_paper": MODEL_PAPERS[architecture],
                "num_layers": int(args.num_layers),
                "hidden_dim": int(args.hidden_dim),
                "use_jk_concat": False,
                "normalize_features": True,
            }
            data_key = (dataset, True)
            if data_key not in dataset_cache:
                dataset_cache[data_key] = load_dataset_data(dataset, normalize_features=True)
            data = dataset_cache[data_key]
            model = load_model(dataset, seed, run_config, device)
            if data.task == "node":
                direct_rows = build_direct_node_milp_rows(
                    dataset=dataset,
                    data=data,
                    model=model,
                    seed=seed,
                    output_dir=seed_dir,
                    max_rows=args.max_rows,
                    metric=metric,
                    device=device,
                    k_hop=int(run_config["num_layers"]),
                    connectivity=str(args.connectivity),
                    use_star_constraints=bool(args.use_star_constraints),
                    use_connectivity_lower_bound=bool(args.use_connectivity_lower_bound),
                    row_numbering=str(getattr(args, "direct_row_numbering", "eligible")),
                )
            else:
                direct_rows = build_direct_graph_milp_rows(
                    dataset=dataset,
                    data=data,
                    model=model,
                    seed=seed,
                    output_dir=seed_dir,
                    max_rows=args.max_rows,
                    metric=metric,
                    device=device,
                    connectivity=str(args.connectivity),
                    use_star_constraints=bool(args.use_star_constraints),
                    use_connectivity_lower_bound=bool(args.use_connectivity_lower_bound),
                    row_numbering=str(getattr(args, "direct_row_numbering", "eligible")),
                )
            if requested_row_ids is not None:
                requested_row_id_set = set(requested_row_ids)
                direct_rows = [
                    pair
                    for pair in direct_rows
                    if int(float(pair[0].get("row_id") or 0)) in requested_row_id_set
                ]
            print(
                f"[fidelity-alpha] {dataset} seed={seed}: "
                f"direct_saa_mip rows={len(direct_rows)}",
                flush=True,
            )
            for idx, (row, gt_edges) in enumerate(direct_rows, start=1):
                row_start = time.perf_counter()
                try:
                    ctx = build_eval_context(
                        dataset=dataset,
                        data=data,
                        row=row,
                        config=run_config,
                        device=device,
                    )
                    p_full = probability_for_edges(model=model, ctx=ctx, selected_edges=ctx.candidate_edges)
                except Exception as exc:
                    append_skipped_alpha_rows(
                        instance_rows=instance_rows,
                        skipped_rows=skipped_rows,
                        row=row,
                        target_kind="saa_mip",
                        metric=metric,
                        alphas=alphas,
                        skip_reason=f"context_or_p_full_error:{exc}",
                        selected_edge_source="saa_mip",
                        run_dir=seed_dir,
                    )
                    continue

                gt_edge_file = str(row.get("gt_edge_file") or "")
                try:
                    gt_eval_rows = evaluate_target(
                        model=model,
                        ctx=ctx,
                        row=row,
                        target_kind="gt",
                        metric=metric,
                        selected_edges=gt_edges,
                        selected_edge_source="gt_edge_file",
                        p_full=p_full,
                        alphas=alphas,
                        num_samples=int(args.eval_samples),
                        sampling_seed=int(args.sampling_seed),
                        batch_size=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
                        run_dir=seed_dir,
                        edge_file_used=gt_edge_file,
                        tie_objective=None,
                    )
                    gt_opt_metrics_by_alpha = {
                        float(alpha): optimization_scenario_metrics(
                            model=model,
                            ctx=ctx,
                            row=row,
                            metric=metric,
                            selected_edges=gt_edges,
                            p_full=p_full,
                            alpha=float(alpha),
                            args=args,
                        )
                        for alpha in alphas
                    }
                    attach_optimization_scenario_metrics(gt_eval_rows, gt_opt_metrics_by_alpha)
                    instance_rows.extend(gt_eval_rows)
                except Exception as exc:
                    append_skipped_alpha_rows(
                        instance_rows=instance_rows,
                        skipped_rows=skipped_rows,
                        row=row,
                        target_kind="gt",
                        metric=metric,
                        alphas=alphas,
                        skip_reason=f"gt_target_error:{exc}",
                        selected_edge_source="gt_edge_file",
                        run_dir=seed_dir,
                        edge_file_used=gt_edge_file,
                    )

                saa_rows, saa_skips = solve_and_evaluate_saa_mip_row(
                    model=model,
                    ctx=ctx,
                    row=row,
                    metric=metric,
                    alphas=alphas,
                    p_full=p_full,
                    gt_edges=gt_edges,
                    args=args,
                    output_dir=seed_dir,
                    run_dir=seed_dir,
                )
                instance_rows.extend(saa_rows)
                skipped_rows.extend(saa_skips)
                flush_dataset_outputs(
                    output_dir=output_dir,
                    instance_rows=instance_rows,
                    skipped_rows=skipped_rows,
                    seeds=seeds,
                )
                if args.progress_every and idx % int(args.progress_every) == 0:
                    elapsed = time.perf_counter() - row_start
                    print(
                        f"[fidelity-alpha] {dataset} seed={seed}: row {idx}/{len(direct_rows)} "
                        f"last_row_time={format_duration(elapsed)}",
                        flush=True,
                    )

    total_instance_rows = 0
    total_skipped_rows = 0
    for dataset in datasets:
        output_dir = output_dirs[dataset]
        instance_rows = instance_rows_by_dataset[dataset]
        skipped_rows = skipped_rows_by_dataset[dataset]
        flush_dataset_outputs(
            output_dir=output_dir,
            instance_rows=instance_rows,
            skipped_rows=skipped_rows,
            seeds=seeds,
        )
        total_instance_rows += len(instance_rows)
        total_skipped_rows += len(skipped_rows)
        print(
            f"[fidelity-alpha] {dataset}: wrote {len(instance_rows)} instance alpha rows, "
            f"skipped targets={len(skipped_rows)}",
            flush=True,
        )

    print(f"[fidelity-alpha] wrote {total_instance_rows} total instance alpha rows")
    print(f"[fidelity-alpha] skipped targets={total_skipped_rows}")
    print(f"[fidelity-alpha] done in {format_duration(time.perf_counter() - total_start)}")
    return output_dirs[datasets[0]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Main fidelity-alpha SAA optimization and inference.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated datasets.")
    parser.add_argument("--dataset", dest="datasets", help="Alias for --datasets.")
    parser.add_argument("--nn_seeds", default=DEFAULT_SEEDS, help="Comma-separated neural-network seeds.")
    parser.add_argument("--nn_seed", dest="nn_seeds", help="Alias for --nn_seeds.")
    parser.add_argument(
        "--alphas",
        default=None,
        help=(
            "Comma-separated alpha values in [0, 1]. Defaults to 0.9 for --metric sufficiency "
            "and 0.1 for --metric necessity."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["necessity", "sufficiency"],
        default="sufficiency",
        help=(
            "Alpha metric to compute. 'sufficiency' is Fid-alpha-: selected edges plus retained context. "
            "'necessity' is Fid-alpha+: selected edges are probabilistically dropped from the candidate graph."
        ),
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Root directory for per-dataset fidelity-alpha outputs (default: Main/results/robust_fidelity_mip)."
        ),
    )
    parser.add_argument(
        "--run_name",
        default="",
        help="Optional output subdirectory name under each dataset. Defaults to timestamp_solver_Fid^-/Fid^+_alpha...",
    )
    parser.add_argument(
        "--experiment_template",
        default=None,
        help=(
            "Optional Markdown plan copied to experiment.md in every dataset run "
            "directory before evaluation starts. Dataset and run metadata are appended."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing output directory. Completed SAA-MILP rows with terminal status and edge CSV "
            "are reused instead of solved again, and CSV summaries are flushed incrementally."
        ),
    )
    parser.add_argument(
        "--max_rows",
        "--max_instances",
        dest="max_rows",
        type=int,
        default=DEFAULT_MAX_INSTANCES,
        help=(
            "Per-dataset/seed cap on evaluable test instances "
            f"(default: {DEFAULT_MAX_INSTANCES}); --max_rows is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--direct_row_numbering",
        choices=["eligible", "pool"],
        default="eligible",
        help=(
            "Row IDs of direct SAA-MILP rows. 'eligible' (default, previous behavior) counts only "
            "correctly classified instances. 'pool' uses the one-based position in the candidate pool, "
            "as the standard runner does, so misclassified instances keep their position. Row IDs "
            "enter the scenario seeds, so compare runs only under the same numbering."
        ),
    )
    parser.add_argument(
        "--row_ids",
        default=None,
        help=(
            "Optional comma-separated one-based row IDs to evaluate after constructing the usual "
            "per-dataset/seed cohort and applying --max_rows. This is intended for targeted recovery "
            "runs and does not renumber rows."
        ),
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help=(
            f"Evaluate only the first {PILOT_INSTANCE_COUNT} rows per dataset/seed; "
            f"equivalent to --max_rows {PILOT_INSTANCE_COUNT}."
        ),
    )
    parser.add_argument(
        "--opt_samples",
        type=int,
        default=DEFAULT_OPT_SAMPLES,
        help="Number of fixed Bernoulli(alpha) scenarios used for SAA optimization.",
    )
    parser.add_argument(
        "--saa_table_max_entries",
        type=int,
        default=DEFAULT_SAA_TABLE_MAX_ENTRIES,
        help=(
            "Maximum number of scenario-table entries (true-probability forward passes) for "
            f"the scenario-table MILP. Default {DEFAULT_SAA_TABLE_MAX_ENTRIES} = N_opt * 10^6. "
            "Rows above the cap are skipped with a recorded reason."
        ),
    )
    parser.add_argument(
        "--saa_table_mip_gap",
        type=float,
        default=0.0,
        help=(
            "Gurobi MIPGap for the scenario-table MILP (default 0: prove the table optimum). "
            "Gurobi's default 1e-4 is of the same order as the audit tolerance."
        ),
    )
    parser.add_argument(
        "--saa_table_conn_root_cuts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "prob_table with --connectivity cut: also separate the cut-set inequalities "
            "y_t <= z(delta(C)) [+ r(C) + eps_empty] at fractional root LP points (MIPNODE, node count 0) "
            "by max-flow, as Gurobi user cuts (PreCrush=1). The lazy cuts at integer solutions are unchanged."
        ),
    )
    parser.add_argument(
        "--saa_table_conn_root_cut_rounds",
        type=int,
        default=50,
        help="Maximum number of root MIPNODE callback calls that run the fractional connectivity separation.",
    )
    parser.add_argument(
        "--saa_table_inert_reduction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For node tasks, drop switch edges whose endpoints both lie at distance >= --num_layers "
            "from the target. Disable only for validation."
        ),
    )
    parser.add_argument(
        "--eval_samples",
        "--sampling",
        dest="eval_samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help=(
            "Number of independent Bernoulli(alpha) scenarios used for post-optimization "
            f"true-probability inference (default: {DEFAULT_EVAL_SAMPLES}). "
            "--sampling is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--counterexample_ci_level",
        type=float,
        default=DEFAULT_COUNTEREXAMPLE_CI_LEVEL,
        help=(
            "Confidence level for the per-instance paired-t interval comparing the MILP "
            f"explanation with GT on shared evaluation scenarios (default: {DEFAULT_COUNTEREXAMPLE_CI_LEVEL})."
        ),
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=0,
        help="Base seed for deterministic Bernoulli context sampling.",
    )
    parser.add_argument("--num_layers", type=int, default=3, help="Number of message-passing layers in the selected checkpoint.")
    parser.add_argument("--hidden_dim", type=int, default=8, help="Hidden dimension in the selected checkpoint.")
    parser.add_argument("--graph_batch_size", type=int, default=8192, help="Batched graph copies per forward pass.")
    parser.add_argument("--node_batch_size", type=int, default=128, help="Batched node-graph copies per forward pass.")
    parser.add_argument(
        "--time_limit",
        type=float,
        default=MIP_TIME_LIMIT_SECONDS,
        help=(
            "Gurobi TimeLimit in seconds for each SAA-MILP instance "
            f"(default: {MIP_TIME_LIMIT_SECONDS:g}). Swap-batch preprocessing "
            "and model construction are outside this budget."
        ),
    )
    parser.add_argument("--readout", choices=["sum", "mean"], default="sum", help="Graph readout for SAA-MILP.")
    parser.add_argument(
        "--connectivity",
        "--saa_mip_connectivity",
        dest="connectivity",
        choices=sorted(CONNECTIVITY_MODES),
        default="none",
        help=(
            "Connectivity policy for direct SAA-MILP selected edges. "
            "'none' leaves exact-budget node masks unrestricted and does not require "
            "a target-incident explanation edge; "
            "'cut' enforces connected explanations with lazy cuts; "
            "'flow' enforces connected explanations with single-commodity flow constraints; "
            "node rows are rooted at the target only for cut/flow."
        ),
    )
    parser.add_argument(
        "--use_star_constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable star inequalities for SAA-MILP connectivity helper node variables.",
    )
    parser.add_argument(
        "--use_connectivity_lower_bound",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the connectivity lower bound sum(z) >= sum(y)-1 when SAA-MILP connectivity is enabled.",
    )
    parser.add_argument(
        "--use_anderson_cuts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add Anderson ReLU user cuts in the SAA-MILP MIPNODE callback. "
            "Use --no-use_anderson_cuts to disable."
        ),
    )
    parser.add_argument(
        "--anderson_max_node_count",
        type=int,
        default=DEFAULT_ANDERSON_MAX_NODE_COUNT,
        help="Largest Gurobi MIP node count at which alpha SAA-MILP Anderson separation is attempted.",
    )
    parser.add_argument(
        "--anderson_max_cuts",
        type=int,
        default=DEFAULT_ANDERSON_MAX_CUTS,
        help="Maximum Anderson user cuts added per SAA-MILP instance.",
    )
    parser.add_argument(
        "--gurobi_threads",
        type=int,
        default=1,
        help="Gurobi Threads parameter for SAA-MILP (default: 1). Lower values reduce memory pressure.",
    )
    parser.add_argument(
        "--gurobi_method",
        type=int,
        default=1,
        help=(
            "Gurobi Method parameter for SAA-MILP (default: 1, dual simplex) to avoid "
            "concurrent/barrier root LP memory spikes."
        ),
    )
    parser.add_argument(
        "--gurobi_numeric_focus",
        type=int,
        default=3,
        help=(
            "Gurobi NumericFocus parameter for the shared-scenario SAA-MILP (default: 3). "
            "The conservative setting prevents false presolve infeasibility caused by accumulated "
            "floating-point error across many bounded GNN forward blocks."
        ),
    )
    parser.add_argument(
        "--gurobi_symmetry",
        type=int,
        choices=[-1, 0, 1, 2],
        default=DEFAULT_GUROBI_SYMMETRY,
        help=(
            "Gurobi Symmetry parameter for every SAA-MILP "
            "(default: 2, aggressive whole-model symmetry detection)."
        ),
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
        "--gurobi_nodefile_start",
        type=float,
        default=None,
        help="Optional Gurobi NodefileStart in GB for spilling B&B node storage to disk.",
    )
    parser.add_argument(
        "--gurobi_nodefile_dir",
        default=None,
        help="Optional Gurobi NodefileDir for node files.",
    )
    parser.add_argument(
        "--gurobi_mem_limit",
        type=float,
        default=None,
        help="Optional Gurobi MemLimit in GB; makes Gurobi stop before the OS kills the process.",
    )
    parser.add_argument(
        "--gurobi_soft_mem_limit",
        type=float,
        default=None,
        help=(
            "Optional Gurobi SoftMemLimit in GB for the prob_table SAA-MILP. Gurobi then ends with "
            "status MEM_LIMIT (recorded as mem_limit, incumbent kept) instead of an out-of-memory error."
        ),
    )
    parser.add_argument(
        "--gt_incumbent",
        "--incumbent",
        dest="gt_incumbent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the GT-derived incumbent as a Gurobi MIP start. Disable with --no-incumbent.",
    )
    parser.add_argument(
        "--exclude_gt",
        "--exclude-gt",
        "--gt-nogood-cut",
        dest="exclude_gt",
        action="store_true",
        default=False,
        help=(
            "Exclude the exact S_GT edge mask with a Hamming-distance no-good cut. "
            "Requires --no-incumbent."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch device: cpu, cuda, or auto.")
    parser.add_argument("--torch_threads", type=int, default=None, help="Optional torch.set_num_threads value.")
    parser.add_argument("--progress_every", type=int, default=10, help="Print progress every N oracle rows; 0 disables.")
    parser.set_defaults(save_figures=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
