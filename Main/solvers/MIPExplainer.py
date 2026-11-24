"""
MILP-based Explainer for Vanilla GNN (GIN-style).

This module provides a high-level interface for explaining GNN predictions
using Mixed Integer Linear Programming (MILP).

Usage:
    from Main.solvers.MIPExplainer import MIPExplainer

    explainer = MIPExplainer(model, dataset_name)
    edge_mask, result = explainer.explain_node(node_idx, data)
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union, Set

import numpy as np
import torch
import torch.nn as nn

from .mip_gnn_vanilla import (
    VanillaNodeGnnWeights,
    VanillaNodeGnnWeights4L,
    VanillaNodeGnnWeights2L,
    VanillaGraphGnnWeights,
    VanillaGraphGnnWeights4L,
    VanillaGraphGnnWeights2L,
    build_vanilla_nodegnn_milp,
    build_vanilla_nodegnn_milp_4l,
    build_vanilla_nodegnn_milp_2l,
    build_vanilla_graphgnn_milp,
    build_vanilla_graphgnn_milp_4l,
    build_vanilla_graphgnn_milp_2l,
    optimize_and_extract,
    optimize_and_extract_graph,
    compute_adjacency,
    forward_vanilla_nodegnn_full,
    forward_vanilla_nodegnn_full_4l,
    forward_vanilla_nodegnn_full_2l,
    forward_vanilla_graphgnn_full,
    forward_vanilla_graphgnn_full_2l,
    MilpSolution,
    ObjectiveMode,
    ConnectivityMode,
    _apply_obbt_u_bounds,
    extract_u_bounds_from_model,
)


def _edge_sequence_with_self_loops(
    edge_index: np.ndarray,
    num_nodes: int,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Return canonical raw edge sequence plus per-node self-loop counts."""
    raw_edges: List[Tuple[int, int]] = []
    self_loop_counts = np.zeros(num_nodes, dtype=float)
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i == j:
            raw_edges.append((i, i))
            self_loop_counts[i] += 1.0
        else:
            a, b = (i, j) if i < j else (j, i)
            raw_edges.append((a, b))
    return raw_edges, self_loop_counts


def _compute_original_node_reference_prediction(
    x: np.ndarray,
    edge_index: np.ndarray,
    node_idx: int,
    weights: Union[VanillaNodeGnnWeights2L, VanillaNodeGnnWeights, VanillaNodeGnnWeights4L],
) -> Tuple[int, float]:
    """Return (c_star, bar_m) from the original full graph before any k-hop truncation."""
    full_edge_sequence, _ = _edge_sequence_with_self_loops(edge_index, int(x.shape[0]))
    full_adj = compute_adjacency(int(x.shape[0]), full_edge_sequence)
    use_jk_concat = getattr(weights, "use_jk_concat", True)

    if isinstance(weights, VanillaNodeGnnWeights2L):
        h1_full, h2_full = forward_vanilla_nodegnn_full_2l(x, full_adj, weights)
        v_embed = np.concatenate([h1_full[node_idx], h2_full[node_idx]], axis=0) if use_jk_concat else h2_full[node_idx]
    elif isinstance(weights, VanillaNodeGnnWeights4L):
        h1_full, h2_full, h3_full, h4_full = forward_vanilla_nodegnn_full_4l(x, full_adj, weights)
        v_embed = (
            np.concatenate([h1_full[node_idx], h2_full[node_idx], h3_full[node_idx], h4_full[node_idx]], axis=0)
            if use_jk_concat
            else h4_full[node_idx]
        )
    else:
        h1_full, h2_full, h3_full = forward_vanilla_nodegnn_full(x, full_adj, weights)
        v_embed = (
            np.concatenate([h1_full[node_idx], h2_full[node_idx], h3_full[node_idx]], axis=0)
            if use_jk_concat
            else h3_full[node_idx]
        )

    bar_Z = v_embed @ weights.W_out.T + weights.b_out
    c_star = int(np.argmax(bar_Z))
    num_classes = int(weights.num_classes)
    bar_m = float(bar_Z[c_star] - np.max(np.delete(bar_Z, c_star))) if num_classes >= 2 else 0.0
    return c_star, bar_m


def extract_k_hop_subgraph(
    node_idx: int,
    k: int,
    edge_index: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Extract K-hop subgraph around the target node.

    Args:
        node_idx: Target node index
        k: Number of hops
        edge_index: Edge index (2, num_edges) in COO format
        num_nodes: Total number of nodes

    Returns:
        sub_edge_index: Edge index for subgraph (2, num_sub_edges)
        sub_nodes: Array of node indices in subgraph (sorted)
        node_to_sub: Mapping from original node index to subgraph node index
        sub_to_node: Mapping from subgraph node index to original node index
    """
    # Build adjacency list
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    for k_idx in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k_idx]), int(edge_index[1, k_idx])
        neighbors[i].add(j)
        neighbors[j].add(i)

    # BFS to find k-hop neighbors
    visited: Set[int] = {node_idx}
    frontier: Set[int] = {node_idx}

    for _ in range(k):
        new_frontier: Set[int] = set()
        for node in frontier:
            for neighbor in neighbors[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_frontier.add(neighbor)
        frontier = new_frontier
        if not frontier:
            break

    # Sort nodes for consistent ordering
    sub_nodes = np.array(sorted(visited))

    # Create mappings
    node_to_sub = {node: idx for idx, node in enumerate(sub_nodes)}
    sub_to_node = {idx: node for idx, node in enumerate(sub_nodes)}

    # Filter edges to subgraph
    sub_edges = []
    for k_idx in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k_idx]), int(edge_index[1, k_idx])
        if i in visited and j in visited:
            sub_edges.append((node_to_sub[i], node_to_sub[j]))

    if sub_edges:
        sub_edge_index = np.array(sub_edges, dtype=np.int64).T
    else:
        sub_edge_index = np.zeros((2, 0), dtype=np.int64)

    return sub_edge_index, sub_nodes, node_to_sub, sub_to_node


def _append_log_line(log_path: Optional[str], msg: str) -> None:
    """Append one line to log file (best-effort)."""
    if not log_path:
        return
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _print_and_log(msg: str, *, verbose: bool, terminal_log_file: Optional[str]) -> None:
    """Mirror verbose terminal messages into terminal log file."""
    if not verbose:
        return
    print(msg)
    _append_log_line(terminal_log_file, msg)


def _apply_gurobi_param_overrides(
    model: object,
    gurobi_params: Optional[Dict[str, object]],
    *,
    verbose: bool,
    terminal_log_file: Optional[str],
) -> None:
    """Apply optional Gurobi parameter overrides before optimize()."""
    if not gurobi_params:
        return
    applied: List[str] = []
    for name, value in gurobi_params.items():
        if value is None:
            continue
        try:
            if hasattr(model, "setParam"):
                model.setParam(str(name), value)
            else:
                setattr(model.Params, str(name), value)
        except Exception as exc:
            raise ValueError(f"Failed to set Gurobi parameter {name}={value!r}") from exc
        applied.append(f"{name}={value}")
    if applied:
        _print_and_log(
            "[MIPExplainer] Gurobi parameter overrides: " + ", ".join(applied),
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )


def _set_model_time_limit(model: object, time_limit_s: Optional[float]) -> None:
    """Apply the requested optimize limit without charging preprocessing time."""
    if time_limit_s is None:
        return
    model.Params.TimeLimit = max(1e-3, float(time_limit_s))


def _maybe_relax_edge_mask_variables(
    *,
    model: object,
    z_vars: Dict[Tuple[int, int], object],
    relax_edge_mask: bool,
    verbose: bool,
    terminal_log_file: Optional[str],
) -> None:
    """Relax only the edge-mask variables z to [0, 1] continuous."""
    if not relax_edge_mask:
        return

    try:
        from gurobipy import GRB
    except Exception as exc:
        raise RuntimeError("Gurobi is required to relax edge-mask variables.") from exc

    for var in z_vars.values():
        var.VType = GRB.CONTINUOUS
        var.LB = 0.0
        var.UB = 1.0
    model.update()

    _print_and_log(
        "[MIPExplainer] Relaxed edge-mask variables z to CONTINUOUS in [0, 1]",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )


def _maybe_enforce_edge_budget_equality(
    *,
    model: object,
    z_vars: Dict[Tuple[int, int], object],
    edges: List[Tuple[int, int]],
    objective: ObjectiveMode,
    edge_budget: Optional[int],
    edge_budget_mode: Literal["le", "eq"],
    verbose: bool,
    terminal_log_file: Optional[str],
) -> None:
    """
    Enforce optional equality edge-budget mode for fidelity objectives:
      - "le": keep default builder constraint (#edges <= K)
      - "eq": add lower bound (#edges >= K), resulting in (#edges == K)
    """
    if edge_budget_mode not in ("le", "eq"):
        raise ValueError(f"Unknown edge_budget_mode='{edge_budget_mode}'. Expected 'le' or 'eq'.")
    if edge_budget_mode == "eq" and objective in ("max_sufficiency", "min_sufficiency") and edge_budget is not None:
        model.addConstr(
            sum(z_vars[e] for e in edges) >= int(edge_budget),
            name="edge_budget_eq_lb",
        )
        _print_and_log(
            f"[MIPExplainer] Enforcing exact edge budget: #edges == {int(edge_budget)}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )


def _maybe_enforce_node_budget(
    *,
    model: object,
    y_vars: Optional[Dict[int, object]],
    num_nodes: int,
    node_budget: Optional[int],
    connectivity: ConnectivityMode,
    verbose: bool,
    terminal_log_file: Optional[str],
) -> None:
    """Enforce optional node-budget upper bound on connectivity helper variables."""
    if node_budget is None:
        return
    if connectivity == "none" or y_vars is None:
        raise ValueError("--node_budget is only supported when connectivity is 'flow' or 'cut'.")
    effective_budget = min(int(node_budget), int(num_nodes))
    model.addConstr(
        sum(y_vars[i] for i in range(num_nodes)) <= effective_budget,
        name="node_budget",
    )
    _print_and_log(
        f"[MIPExplainer] Enforcing node budget: #nodes <= {effective_budget}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )


def _maybe_enforce_node_induced(
    *,
    model: object,
    z_vars: Dict[Tuple[int, int], object],
    y_vars: Optional[Dict[int, object]],
    node_induced: bool,
    connectivity: ConnectivityMode,
    verbose: bool,
    terminal_log_file: Optional[str],
) -> None:
    """Enforce node-induced subgraph: z_ij >= y_i + y_j - 1 for all edges.

    When active, edges are fully determined by node selection:
    both endpoints selected => edge must be included.  Combined with
    the existing z_ij <= y_i and z_ij <= y_j, this yields z_ij = y_i AND y_j.
    """
    if not node_induced:
        return
    if connectivity == "none" or y_vars is None:
        raise ValueError("--node_induced requires connectivity in {'flow', 'cut'}.")
    count = 0
    for (i, j), zij in z_vars.items():
        model.addConstr(zij >= y_vars[i] + y_vars[j] - 1.0, name=f"node_induced_{i}_{j}")
        count += 1
    _print_and_log(
        f"[MIPExplainer] Enforcing node-induced subgraph: {count} constraints (z_ij >= y_i + y_j - 1)",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )


def _canonicalize_edge_solution(
    edges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    canonical: Set[Tuple[int, int]] = set()
    for i, j in edges:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        canonical.add((a, b) if a < b else (b, a))
    return sorted(canonical)


def _map_edge_solutions_to_working_subgraph(
    edge_solutions: Optional[Sequence[Sequence[Tuple[int, int]]]],
    node_to_sub: Optional[Dict[int, int]],
) -> List[List[Tuple[int, int]]]:
    if not edge_solutions:
        return []
    if node_to_sub is None:
        return [_canonicalize_edge_solution(sol) for sol in edge_solutions]

    mapped_solutions: List[List[Tuple[int, int]]] = []
    for sol in edge_solutions:
        mapped: Set[Tuple[int, int]] = set()
        for i, j in sol:
            a = int(i)
            b = int(j)
            if a not in node_to_sub or b not in node_to_sub:
                continue
            u = int(node_to_sub[a])
            v = int(node_to_sub[b])
            if u == v:
                continue
            mapped.add((u, v) if u < v else (v, u))
        mapped_solutions.append(sorted(mapped))
    return mapped_solutions


def _maybe_add_no_good_edge_solution_cuts(
    *,
    model: object,
    z_vars: Dict[Tuple[int, int], object],
    edges: List[Tuple[int, int]],
    excluded_edge_solutions: Optional[Sequence[Sequence[Tuple[int, int]]]],
    verbose: bool,
    terminal_log_file: Optional[str],
) -> None:
    """Exclude previously found binary edge selections via Hamming-distance no-good cuts."""
    if not excluded_edge_solutions:
        return

    try:
        import gurobipy as gp
    except Exception as exc:
        raise RuntimeError("gurobipy is required to add no-good edge cuts.") from exc

    edge_set = set(edges)
    added = 0
    for sol_idx, forbidden_solution in enumerate(excluded_edge_solutions):
        forbidden = [e for e in _canonicalize_edge_solution(forbidden_solution) if e in edge_set]
        forbidden_set = set(forbidden)
        lhs = gp.LinExpr()
        rhs = 1 - len(forbidden_set)
        for e in edges:
            coeff = -1.0 if e in forbidden_set else 1.0
            lhs.addTerms(coeff, z_vars[e])
        model.addConstr(lhs >= rhs, name=f"nogood_edge_solution_{sol_idx}")
        added += 1
        _print_and_log(
            f"[MIPExplainer] Added no-good cut #{sol_idx + 1}: exclude exact edge set of size {len(forbidden_set)}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    if added > 0:
        _print_and_log(
            f"[MIPExplainer] Added {added} no-good edge-solution cut(s)",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )


class MIPExplainer:
    """
    MILP-based Explainer for Vanilla GNN models.

    Finds a minimal subgraph that preserves the model's prediction
    by encoding the GNN forward pass as MILP constraints.

    Attributes:
        model: Trained PyTorch GNN model (NodeGNN or GraphGNN)
        dataset_name: Name of the dataset for logging
        task: "node" or "graph" classification
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_name: str,
        task: Literal["node", "graph"] = "node",
        use_jk_concat: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize MIPExplainer.

        Args:
            model: Trained PyTorch GNN model
            dataset_name: Name of the dataset
            task: "node" or "graph" classification
            use_jk_concat: Whether the model uses JK-concat
            verbose: Print debug information
        """
        self.model = model
        self.dataset_name = dataset_name
        self.task = task
        self.use_jk_concat = use_jk_concat
        self.verbose = verbose

        # Extract weights from model
        self.weights = self._extract_weights()

        # Verification context (set after explain_node/explain_graph)
        self._last_working_x = None
        self._last_working_edges = None
        self._last_self_loop_counts = None
        self._last_solution = None

    def _extract_weights(self) -> Union[
        VanillaNodeGnnWeights,
        VanillaNodeGnnWeights4L,
        VanillaNodeGnnWeights2L,
        VanillaGraphGnnWeights,
        VanillaGraphGnnWeights4L,
        VanillaGraphGnnWeights2L,
    ]:
        """Extract numpy weights from PyTorch model (Vanilla.py structure)."""
        model = self.model
        model.eval()

        # Determine number of layers
        num_layers = getattr(model, 'num_layers', 3)

        def get_param(name: str) -> np.ndarray:
            """Get parameter by name from model."""
            for n, p in model.named_parameters():
                if n == name:
                    return p.detach().cpu().numpy()
            raise ValueError(f"Parameter {name} not found in model")

        def get_param_optional(name: str, default: np.ndarray) -> np.ndarray:
            """Get parameter by name, return default if not found."""
            for n, p in model.named_parameters():
                if n == name:
                    return p.detach().cpu().numpy()
            return default

        param_names = [n for n, _ in model.named_parameters()]
        if self.verbose:
            print(f"[MIPExplainer] Model parameters: {param_names}")

        # Pattern for Vanilla.py: conv1.mlp.fc1.weight, conv1.mlp.fc2.weight, conv1.eps
        try:
            # Layer 1: conv1.mlp.fc1, conv1.mlp.fc2, conv1.eps
            eps1 = get_param_optional("conv1.eps", np.array(0.0))
            W1_1 = get_param("conv1.mlp.fc1.weight").T  # (in, hidden)
            b1_1 = get_param("conv1.mlp.fc1.bias")
            W1_2 = get_param("conv1.mlp.fc2.weight").T  # (hidden, out)
            b1_2 = get_param("conv1.mlp.fc2.bias")

            # Layer 2: conv2.*
            eps2 = get_param_optional("conv2.eps", np.array(0.0))
            W2_1 = get_param("conv2.mlp.fc1.weight").T
            b2_1 = get_param("conv2.mlp.fc1.bias")
            W2_2 = get_param("conv2.mlp.fc2.weight").T
            b2_2 = get_param("conv2.mlp.fc2.bias")

            if num_layers >= 3:
                eps3 = get_param_optional("conv3.eps", np.array(0.0))
                W3_1 = get_param("conv3.mlp.fc1.weight").T
                b3_1 = get_param("conv3.mlp.fc1.bias")
                W3_2 = get_param("conv3.mlp.fc2.weight").T
                b3_2 = get_param("conv3.mlp.fc2.bias")
            if num_layers >= 4:
                eps4 = get_param_optional("conv4.eps", np.array(0.0))
                W4_1 = get_param("conv4.mlp.fc1.weight").T
                b4_1 = get_param("conv4.mlp.fc1.bias")
                W4_2 = get_param("conv4.mlp.fc2.weight").T
                b4_2 = get_param("conv4.mlp.fc2.bias")

            # Classifier: lin.weight, lin.bias
            W_out = get_param("lin.weight")  # (num_classes, embedding_size)
            b_out = get_param("lin.bias")

        except ValueError as e:
            raise ValueError(f"Could not extract weights. Model structure not recognized. "
                           f"Parameters: {param_names}") from e

        if num_layers == 2:
            if self.task == "node":
                return VanillaNodeGnnWeights2L(
                    eps1=float(eps1), W1_1=W1_1, b1_1=b1_1, W1_2=W1_2, b1_2=b1_2,
                    eps2=float(eps2), W2_1=W2_1, b2_1=b2_1, W2_2=W2_2, b2_2=b2_2,
                    W_out=W_out, b_out=b_out, use_jk_concat=self.use_jk_concat,
                )
            else:
                return VanillaGraphGnnWeights2L(
                    eps1=float(eps1), W1_1=W1_1, b1_1=b1_1, W1_2=W1_2, b1_2=b1_2,
                    eps2=float(eps2), W2_1=W2_1, b2_1=b2_1, W2_2=W2_2, b2_2=b2_2,
                    W_out=W_out, b_out=b_out,
                )
        elif num_layers == 3:
            if self.task == "node":
                return VanillaNodeGnnWeights(
                    eps1=float(eps1), W1_1=W1_1, b1_1=b1_1, W1_2=W1_2, b1_2=b1_2,
                    eps2=float(eps2), W2_1=W2_1, b2_1=b2_1, W2_2=W2_2, b2_2=b2_2,
                    eps3=float(eps3), W3_1=W3_1, b3_1=b3_1, W3_2=W3_2, b3_2=b3_2,
                    W_out=W_out, b_out=b_out, use_jk_concat=self.use_jk_concat,
                )
            else:
                return VanillaGraphGnnWeights(
                    eps1=float(eps1), W1_1=W1_1, b1_1=b1_1, W1_2=W1_2, b1_2=b1_2,
                    eps2=float(eps2), W2_1=W2_1, b2_1=b2_1, W2_2=W2_2, b2_2=b2_2,
                    eps3=float(eps3), W3_1=W3_1, b3_1=b3_1, W3_2=W3_2, b3_2=b3_2,
                    W_out=W_out, b_out=b_out,
                )
        elif num_layers == 4:
            if self.task == "node":
                return VanillaNodeGnnWeights4L(
                    eps1=float(eps1), W1_1=W1_1, b1_1=b1_1, W1_2=W1_2, b1_2=b1_2,
                    eps2=float(eps2), W2_1=W2_1, b2_1=b2_1, W2_2=W2_2, b2_2=b2_2,
                    eps3=float(eps3), W3_1=W3_1, b3_1=b3_1, W3_2=W3_2, b3_2=b3_2,
                    eps4=float(eps4), W4_1=W4_1, b4_1=b4_1, W4_2=W4_2, b4_2=b4_2,
                    W_out=W_out, b_out=b_out, use_jk_concat=self.use_jk_concat,
                )
            # Graph 4-layer extraction kept for future graph-4L MILP support.
            return VanillaGraphGnnWeights4L(
                eps1=float(eps1), W1_1=W1_1, b1_1=b1_1, W1_2=W1_2, b1_2=b1_2,
                eps2=float(eps2), W2_1=W2_1, b2_1=b2_1, W2_2=W2_2, b2_2=b2_2,
                eps3=float(eps3), W3_1=W3_1, b3_1=b3_1, W3_2=W3_2, b3_2=b3_2,
                eps4=float(eps4), W4_1=W4_1, b4_1=b4_1, W4_2=W4_2, b4_2=b4_2,
                W_out=W_out, b_out=b_out,
            )
        else:
            raise ValueError(f"Unsupported num_layers={num_layers}; expected 2, 3, or 4.")

    def explain_node(
        self,
        node_idx: int,
        x: np.ndarray,
        edge_index: np.ndarray,
        *,
        rho: float = 0.0,
        edge_budget: Optional[int] = None,
        node_budget: Optional[int] = None,
        edge_budget_mode: Literal["le", "eq"] = "le",
        min_edges: Optional[int] = None,
        objective: ObjectiveMode = "min_edges",
        connectivity: ConnectivityMode = "none",
        explanation_on_complement: bool = False,
        use_nested_cuts: bool = False,
        use_back_cuts: bool = False,
        max_cuts_per_round: int = 100,
        max_root_sep_rounds: int = 10,
        use_anderson_cuts: bool = False,
        anderson_max_node_count: int = 0,
        anderson_max_cuts: int = 100,
        use_incident_cap_tightening: bool = False,
        cap_mode: str = "global",
        use_symmetry_breaking: bool = False,
        use_star_constraints: bool = True,
        use_connectivity_lower_bound: bool = False,
        use_forced_edges: bool = False,
        use_bridge_constraints: bool = False,
        use_aggregated_bridge_constraints: bool = False,
        use_hierarchical_bridge: bool = False,
        use_obbt_u: bool = False,
        max_obbt_u_layers: Optional[int] = None,
        node_induced: bool = False,
        relax_edge_mask: bool = False,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
        seed: int = 0,
        initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
        k_hop: Optional[int] = None,
        excluded_edge_solutions: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
        log_file: Optional[str] = None,
        terminal_log_file: Optional[str] = None,
        gurobi_params: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, MilpSolution]:
        """
        Explain prediction for a single node.

        Args:
            node_idx: Index of node to explain
            x: Node features (n, d)
            edge_index: Edge index (2, num_edges) in COO format
            rho: Margin preservation ratio (0 to 1)
            edge_budget: Maximum number of edges
            node_budget: Maximum number of selected nodes (connectivity='flow'/'cut' only)
            edge_budget_mode: "le" => #edges <= K, "eq" => #edges == K (fidelity objectives only)
            min_edges: Minimum number of edges
            objective: "min_edges", "max_sufficiency", "min_sufficiency", or "feasible"
            connectivity: "none", "flow", or "cut"
            explanation_on_complement: If True, z remains the kept/message-passing
                mask while explicit r=1-z removal variables define the explanation
                connectivity, star inequalities, the connectivity lower bound, and lazy connectivity cuts. With connectivity
                enabled for a node task, that removal explanation is rooted at the
                target node.
            use_anderson_cuts: Enable Anderson ReLU user cuts at MIPNODE
            anderson_max_node_count: Maximum B&B node count for Anderson cuts (0=root only)
            anderson_max_cuts: Maximum Anderson cuts added per callback
            use_incident_cap_tightening: Enable Top-κ incident cap tightening
            cap_mode: "global" (κ_i = K) or "distance" (κ_i = max(0, K - dist(i)))
            use_symmetry_breaking: Enable leaf symmetry breaking
            use_star_constraints: Enable star inequalities on the explanation-edge
                variables when connectivity is enabled.
            use_connectivity_lower_bound: Enable Σq >= Σy - 1 on explanation-edge
                variables q (q=z normally, q=r for complement explanations).
            use_forced_edges: Enable forced edge detection and constraints
            use_bridge_constraints: Enable bridge-based implication constraints
            use_aggregated_bridge_constraints: Enable aggregated bridge constraints
            use_hierarchical_bridge: Enable bridge-tree hierarchical constraints
            use_obbt_u: Enable LP-based OBBT tightening for u bounds (connectivity='none' only).
            max_obbt_u_layers: Apply u-OBBT to layers [1..max_obbt_u_layers]. None = all layers.
            node_induced: If True, enforce node-induced subgraph (z_ij = y_i AND y_j).
            relax_edge_mask: If True, relax only edge-mask variables z to [0,1] and
                extract the explanation by taking the top-K edges by z value.
            time_limit_s: Gurobi optimize time limit, excluding preprocessing, build and OBBT.
            mip_gap: MIP optimality gap
            seed: Random seed
            initial_solution: Optional edge set used as a Gurobi MIP start.
            k_hop: If provided (including 0), extract K-hop subgraph around target node.
                   Reduces problem size significantly for large graphs.
                   Recommended: k_hop = num_layers (e.g., 3 for 3-layer GNN).
            excluded_edge_solutions: Previously found edge selections to exclude via no-good cuts.
            log_file: Path for Gurobi log file
            terminal_log_file: Path to mirror verbose terminal messages
            gurobi_params: Optional Gurobi model parameter overrides, e.g. Method/Threads/MemLimit.

        Returns:
            edge_mask: Binary mask over edges (1 = selected)
            solution: MilpSolution with detailed results (edges in original indices)
        """
        if self.task != "node":
            raise ValueError("explain_node requires task='node'")
        if node_budget is not None and connectivity == "none":
            raise ValueError("node_budget requires connectivity in {'flow', 'cut'}.")
        if node_budget is not None and edge_budget is not None:
            raise ValueError("Use either edge_budget or node_budget, not both.")
        if node_induced and connectivity == "none":
            raise ValueError("node_induced requires connectivity in {'flow', 'cut'}.")
        if relax_edge_mask and node_budget is not None:
            raise ValueError("relax_edge_mask currently supports edge budgets only, not node_budget.")
        if relax_edge_mask and edge_budget is None:
            raise ValueError("relax_edge_mask requires edge_budget so top-K extraction knows K.")
        if explanation_on_complement:
            if edge_budget is None or edge_budget_mode != "eq":
                raise ValueError(
                    "explanation_on_complement requires an exact kept-edge budget "
                    "(edge_budget is not None and edge_budget_mode='eq')."
                )
            if node_budget is not None:
                raise ValueError("explanation_on_complement does not support node_budget.")
            if relax_edge_mask:
                raise ValueError("explanation_on_complement requires binary edge-mask variables.")

        edge_index = np.asarray(edge_index)
        x = np.asarray(x, dtype=float)
        reference_c_star, reference_bar_m = _compute_original_node_reference_prediction(
            x=x,
            edge_index=edge_index,
            node_idx=node_idx,
            weights=self.weights,
        )
        _print_and_log(
            f"[MIPExplainer] Original full-graph reference: class={reference_c_star}, margin={reference_bar_m:.4f}",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )

        # Optionally extract K-hop subgraph for efficiency
        if k_hop is not None and k_hop >= 0:
            sub_edge_index, sub_nodes, node_to_sub, sub_to_node = extract_k_hop_subgraph(
                node_idx=node_idx,
                k=k_hop,
                edge_index=edge_index,
                num_nodes=x.shape[0],
            )

            # Map target node to subgraph index
            sub_node_idx = node_to_sub[node_idx]
            sub_x = x[sub_nodes]

            _print_and_log(
                f"[MIPExplainer] Extracted {k_hop}-hop subgraph: {len(sub_nodes)} nodes, {sub_edge_index.shape[1]//2} edges",
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )

            # Use subgraph for MIP
            working_x = sub_x
            working_edge_index = sub_edge_index
            working_node_idx = sub_node_idx
        else:
            working_x = x
            working_edge_index = edge_index
            working_node_idx = node_idx
            sub_to_node = None

        mapped_initial_solution = None
        if initial_solution is not None:
            mapped_solutions = _map_edge_solutions_to_working_subgraph(
                [initial_solution],
                node_to_sub if sub_to_node is not None else None,
            )
            mapped_initial_solution = mapped_solutions[0] if mapped_solutions else []

        # Convert edge_index to canonical edge sequence while retaining self-loops as fixed messages.
        milp_edge_sequence, self_loop_counts = _edge_sequence_with_self_loops(working_edge_index, int(working_x.shape[0]))
        undirected_edges = []
        edge_to_idx = {}  # map (a, b) to list of original edge indices
        for k in range(working_edge_index.shape[1]):
            i, j = int(working_edge_index[0, k]), int(working_edge_index[1, k])
            if i != j:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in edge_to_idx:
                    edge_to_idx[(a, b)] = []
                    undirected_edges.append((a, b))
                edge_to_idx[(a, b)].append(k)

        _print_and_log(
            f"[MIPExplainer] Explaining node {node_idx} (working: {working_node_idx})",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIPExplainer] Working graph: {working_x.shape[0]} nodes, {len(undirected_edges)} edges",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )

        # Build MILP - select appropriate builder based on weights type
        # Preprocessing/build/OBBT do not consume the final solver time limit.
        start_time = time.time()
        if isinstance(self.weights, VanillaNodeGnnWeights2L):
            # Use 2-layer builder
            build = build_vanilla_nodegnn_milp_2l(
                X=working_x,
                undirected_edges=milp_edge_sequence,
                v_star=working_node_idx,
                weights=self.weights,
                reference_c_star=reference_c_star,
                reference_bar_m=reference_bar_m,
                rho=rho,
                edge_budget=edge_budget,
                node_budget=node_budget,
                min_edges=min_edges,
                objective=objective,
                connectivity=connectivity,
                connectivity_on_complement=explanation_on_complement,
                use_maxflow_separation=False,
                use_nested_cuts=use_nested_cuts,
                use_back_cuts=use_back_cuts,
                max_cuts_per_round=max_cuts_per_round,
                max_root_sep_rounds=max_root_sep_rounds,
                use_anderson_cuts=use_anderson_cuts,
                anderson_max_node_count=anderson_max_node_count,
                anderson_max_cuts=anderson_max_cuts,
                use_incident_cap_tightening=use_incident_cap_tightening,
                cap_mode=cap_mode,
                use_symmetry_breaking=use_symmetry_breaking,
                use_star_constraints=use_star_constraints,
                use_connectivity_lower_bound=use_connectivity_lower_bound,
                use_forced_edges=use_forced_edges,
                use_bridge_constraints=use_bridge_constraints,
                use_aggregated_bridge_constraints=use_aggregated_bridge_constraints,
                use_hierarchical_bridge=use_hierarchical_bridge,
                use_obbt_u=use_obbt_u,
                max_obbt_u_layers=max_obbt_u_layers,
                verbose=self.verbose,
                time_limit_s=time_limit_s,
                wallclock_deadline_s=None,
                mip_gap=mip_gap,
                seed=seed,
                initial_solution=mapped_initial_solution,
                log_file=log_file,
                terminal_log_file=terminal_log_file,
            )
        else:
            # Use 3-layer or 4-layer builder
            if isinstance(self.weights, VanillaNodeGnnWeights4L):
                build = build_vanilla_nodegnn_milp_4l(
                    X=working_x,
                    undirected_edges=milp_edge_sequence,
                    v_star=working_node_idx,
                    weights=self.weights,
                    reference_c_star=reference_c_star,
                    reference_bar_m=reference_bar_m,
                    rho=rho,
                    edge_budget=edge_budget,
                    node_budget=node_budget,
                    min_edges=min_edges,
                    objective=objective,
                    connectivity=connectivity,
                    connectivity_on_complement=explanation_on_complement,
                    use_maxflow_separation=False,
                    use_nested_cuts=use_nested_cuts,
                    use_back_cuts=use_back_cuts,
                    max_cuts_per_round=max_cuts_per_round,
                    max_root_sep_rounds=max_root_sep_rounds,
                    use_anderson_cuts=use_anderson_cuts,
                    anderson_max_node_count=anderson_max_node_count,
                    anderson_max_cuts=anderson_max_cuts,
                    use_incident_cap_tightening=use_incident_cap_tightening,
                    cap_mode=cap_mode,
                    use_symmetry_breaking=use_symmetry_breaking,
                    use_star_constraints=use_star_constraints,
                    use_connectivity_lower_bound=use_connectivity_lower_bound,
                    use_forced_edges=use_forced_edges,
                    use_bridge_constraints=use_bridge_constraints,
                    use_aggregated_bridge_constraints=use_aggregated_bridge_constraints,
                    use_hierarchical_bridge=use_hierarchical_bridge,
                    use_obbt_u=use_obbt_u,
                    max_obbt_u_layers=max_obbt_u_layers,
                    verbose=self.verbose,
                    time_limit_s=time_limit_s,
                    wallclock_deadline_s=None,
                    mip_gap=mip_gap,
                    seed=seed,
                    initial_solution=mapped_initial_solution,
                    log_file=log_file,
                    terminal_log_file=terminal_log_file,
                )
            else:
                build = build_vanilla_nodegnn_milp(
                    X=working_x,
                    undirected_edges=milp_edge_sequence,
                    v_star=working_node_idx,
                    weights=self.weights,
                    reference_c_star=reference_c_star,
                    reference_bar_m=reference_bar_m,
                    rho=rho,
                    edge_budget=edge_budget,
                    node_budget=node_budget,
                    min_edges=min_edges,
                    objective=objective,
                    connectivity=connectivity,
                    connectivity_on_complement=explanation_on_complement,
                    use_maxflow_separation=False,
                    use_nested_cuts=use_nested_cuts,
                    use_back_cuts=use_back_cuts,
                    max_cuts_per_round=max_cuts_per_round,
                    max_root_sep_rounds=max_root_sep_rounds,
                    use_anderson_cuts=use_anderson_cuts,
                    anderson_max_node_count=anderson_max_node_count,
                    anderson_max_cuts=anderson_max_cuts,
                    use_incident_cap_tightening=use_incident_cap_tightening,
                    cap_mode=cap_mode,
                    use_symmetry_breaking=use_symmetry_breaking,
                    use_star_constraints=use_star_constraints,
                    use_connectivity_lower_bound=use_connectivity_lower_bound,
                    use_forced_edges=use_forced_edges,
                    use_bridge_constraints=use_bridge_constraints,
                    use_aggregated_bridge_constraints=use_aggregated_bridge_constraints,
                    use_hierarchical_bridge=use_hierarchical_bridge,
                    use_obbt_u=use_obbt_u,
                    max_obbt_u_layers=max_obbt_u_layers,
                    verbose=self.verbose,
                    time_limit_s=time_limit_s,
                    wallclock_deadline_s=None,
                    mip_gap=mip_gap,
                    seed=seed,
                    initial_solution=mapped_initial_solution,
                    log_file=log_file,
                    terminal_log_file=terminal_log_file,
                )
        build_time = time.time() - start_time
        _apply_gurobi_param_overrides(
            build.model,
            gurobi_params,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )

        # Optimize
        opt_start = time.time()
        _maybe_enforce_edge_budget_equality(
            model=build.model,
            z_vars=build.z,
            edges=build.edges,
            objective=objective,
            edge_budget=edge_budget,
            edge_budget_mode=edge_budget_mode,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        if explanation_on_complement:
            if build.removal_z is None:
                raise RuntimeError("Complement explanation requested without removal-edge variables.")
            removal_budget = len(build.edges) - int(edge_budget)
            build.model.addConstr(
                sum(build.removal_z[e] for e in build.edges) == removal_budget,
                name="removal_edge_budget_eq",
            )
            _print_and_log(
                f"[MIPExplainer] Enforcing exact removal budget: #removed_edges == {removal_budget}",
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )
        _maybe_enforce_node_budget(
            model=build.model,
            y_vars=build.y,
            num_nodes=build.n,
            node_budget=node_budget,
            connectivity=connectivity,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _maybe_enforce_node_induced(
            model=build.model,
            z_vars=build.z,
            y_vars=build.y,
            node_induced=node_induced,
            connectivity=connectivity,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _maybe_add_no_good_edge_solution_cuts(
            model=build.model,
            z_vars=build.z,
            edges=build.edges,
            excluded_edge_solutions=_map_edge_solutions_to_working_subgraph(
                excluded_edge_solutions,
                node_to_sub if sub_to_node is not None else None,
            ),
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _maybe_relax_edge_mask_variables(
            model=build.model,
            z_vars=build.z,
            relax_edge_mask=relax_edge_mask,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _set_model_time_limit(build.model, time_limit_s)
        solution = optimize_and_extract(
            build,
            relax_edge_mask=relax_edge_mask,
            selection_edge_budget=edge_budget,
        )
        opt_time = time.time() - opt_start

        # Store verification context (subgraph data for later verification)
        self._last_working_x = working_x
        self._last_working_edges = milp_edge_sequence
        self._last_self_loop_counts = self_loop_counts
        self._last_solution = solution  # This has subgraph indices

        _print_and_log(
            f"[MIPExplainer] Build time: {build_time:.2f}s, Opt time: {opt_time:.2f}s",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIPExplainer] Status: {solution.status}, Edges: {len(solution.selected_edges)}",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )

        # Convert selected edges back to original graph indices
        if sub_to_node is not None:
            original_selected = []
            for (a, b) in solution.selected_edges:
                orig_a = sub_to_node[a]
                orig_b = sub_to_node[b]
                oa, ob = (orig_a, orig_b) if orig_a < orig_b else (orig_b, orig_a)
                original_selected.append((oa, ob))
            original_z_values = {}
            for (a, b), z_value in solution.z_values.items():
                orig_a = sub_to_node[a]
                orig_b = sub_to_node[b]
                oa, ob = (orig_a, orig_b) if orig_a < orig_b else (orig_b, orig_a)
                original_z_values[(oa, ob)] = float(z_value)
            # Update solution with original indices
            # v_star needs to be converted back to original index
            original_v_star = sub_to_node[solution.v_star] if solution.v_star is not None else node_idx
            solution = MilpSolution(
                status=solution.status,
                objective_value=solution.objective_value,
                selected_edges=original_selected,
                z_values=original_z_values,
                rho_optimal=getattr(solution, 'rho_optimal', None),
                bar_m=getattr(solution, 'bar_m', None),
                v_star=original_v_star,
                c_star=getattr(solution, 'c_star', None),
                final_mip_gap=getattr(solution, 'final_mip_gap', None),
                best_bound=getattr(solution, 'best_bound', None),
                incumbent_obj=getattr(solution, 'incumbent_obj', None),
                model_size=getattr(solution, 'model_size', None),
                root_lp_value=getattr(solution, 'root_lp_value', None),
                root_lp_time=getattr(solution, 'root_lp_time', None),
                callback_time_sec=getattr(solution, 'callback_time_sec', None),
                callback_calls=getattr(solution, 'callback_calls', None),
                cuts_mipnode=getattr(solution, 'cuts_mipnode', None),
                cuts_mipsol_lazy=getattr(solution, 'cuts_mipsol_lazy', None),
                anderson_cuts_added=getattr(solution, 'anderson_cuts_added', None),
                connectivity_lazy_cuts_added=getattr(
                    solution, 'connectivity_lazy_cuts_added', None
                ),
                bnb_node_count=getattr(solution, 'bnb_node_count', None),
            )

        # Create edge mask (on original edge_index)
        edge_mask = np.zeros(edge_index.shape[1], dtype=float)
        selected_set = set(solution.selected_edges)
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in selected_set:
                edge_mask[k] = 1.0

        return edge_mask, solution

    def explain_graph(
        self,
        x: np.ndarray,
        edge_index: np.ndarray,
        *,
        readout: str = "sum",
        connectivity: ConnectivityMode = "none",
        explanation_on_complement: bool = False,
        rho: float = 0.0,
        edge_budget: Optional[int] = None,
        node_budget: Optional[int] = None,
        edge_budget_mode: Literal["le", "eq"] = "le",
        min_edges: Optional[int] = None,
        objective: ObjectiveMode = "min_edges",
        use_incident_cap_tightening: bool = False,
        use_symmetry_breaking: bool = False,
        use_star_constraints: bool = True,
        use_connectivity_lower_bound: bool = False,
        use_bridge_constraints: bool = False,
        use_anderson_cuts: bool = False,
        anderson_max_node_count: int = 0,
        anderson_max_cuts: int = 50,
        edge_branch_priority: int = 0,
        use_obbt_u: bool = False,
        max_obbt_u_layers: Optional[int] = None,
        obbt_rebuild: bool = False,
        degree_table_layer1: bool = False,
        node_induced: bool = False,
        relax_edge_mask: bool = False,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
        seed: int = 0,
        initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
        excluded_edge_solutions: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
        log_file: Optional[str] = None,
        terminal_log_file: Optional[str] = None,
        gurobi_params: Optional[Dict[str, object]] = None,
    ) -> Tuple[np.ndarray, MilpSolution]:
        """
        Explain prediction for a graph.

        Args:
            x: Node features (n, d)
            edge_index: Edge index (2, num_edges) in COO format
            readout: Pooling method ("sum" or "mean")
            connectivity: "none", "flow", or "cut"
            explanation_on_complement: If True, z remains the kept/message-passing
                mask while explicit removal variables 1-z define connectivity,
                star inequalities, the connectivity lower bound, and lazy connectivity cuts.
            rho: Margin preservation ratio
            edge_budget: Maximum number of edges
            node_budget: Maximum number of selected nodes (connectivity='flow'/'cut' only)
            edge_budget_mode: "le" => #edges <= K, "eq" => #edges == K (fidelity objectives only)
            min_edges: Minimum number of edges
            objective: "min_edges", "max_sufficiency", "min_sufficiency", or "feasible"
            use_incident_cap_tightening: Enable Top-κ incident cap tightening
            use_symmetry_breaking: Enable leaf symmetry breaking
            use_star_constraints: Enable star inequalities when connectivity is enabled
            use_connectivity_lower_bound: Enable connectivity lower bound Σz >= Σy - 1 when connectivity is enabled
            use_bridge_constraints: Kept for API compatibility (not applied in graph mode)
            use_anderson_cuts: Enable Anderson ReLU user cuts at MIPNODE
            anderson_max_node_count: Highest B&B node count where Anderson separation may run
            anderson_max_cuts: Maximum Anderson cuts in one separation pass
            edge_branch_priority: Positive values prioritize graph edge-mask
                variables over default-priority integer variables when branching.
            use_obbt_u: Enable LP-based OBBT tightening for u bounds.
            max_obbt_u_layers: Apply u-OBBT to layers [1..max_obbt_u_layers]. None = all layers.
            obbt_rebuild: Rebuild after OBBT so tightened u bounds enter ReLU
                big-M, downstream interval, and McCormick constraints.
            degree_table_layer1: For the 3-layer graph MILP with identical node
                features, represent layer 1 exactly by one-hot degree variables.
            node_induced: If True, enforce node-induced subgraph (z_ij = y_i AND y_j).
            relax_edge_mask: If True, relax only edge-mask variables z to [0,1] and
                extract the explanation by taking the top-K edges by z value.
            time_limit_s: Gurobi optimize time limit, excluding preprocessing, build and OBBT.
            mip_gap: MIP optimality gap
            seed: Random seed
            initial_solution: Optional edge set used as a Gurobi MIP start.
            excluded_edge_solutions: Previously found edge selections to exclude via no-good cuts.
            log_file: Path for Gurobi log file
            terminal_log_file: Path to mirror verbose terminal messages
            gurobi_params: Optional Gurobi model parameter overrides, e.g. Method/Threads/MemLimit.

        Returns:
            edge_mask: Binary mask over edges
            solution: MilpSolution with detailed results
        """
        if self.task != "graph":
            raise ValueError("explain_graph requires task='graph'")
        if connectivity not in ("none", "flow", "cut"):
            raise ValueError(
                f"Unsupported connectivity for graph explanation: {connectivity}. "
                "Use 'none', 'flow', or 'cut'."
            )
        if node_budget is not None and connectivity == "none":
            raise ValueError("node_budget requires connectivity in {'flow', 'cut'}.")
        if node_budget is not None and edge_budget is not None:
            raise ValueError("Use either edge_budget or node_budget, not both.")
        if node_induced and connectivity == "none":
            raise ValueError("node_induced requires connectivity in {'flow', 'cut'}.")
        if relax_edge_mask and node_budget is not None:
            raise ValueError("relax_edge_mask currently supports edge budgets only, not node_budget.")
        if relax_edge_mask and edge_budget is None:
            raise ValueError("relax_edge_mask requires edge_budget so top-K extraction knows K.")
        if explanation_on_complement:
            if edge_budget is None or edge_budget_mode != "eq":
                raise ValueError(
                    "explanation_on_complement requires an exact kept-edge budget "
                    "(edge_budget is not None and edge_budget_mode='eq')."
                )
            if node_budget is not None:
                raise ValueError("explanation_on_complement does not support node_budget.")
            if relax_edge_mask:
                raise ValueError("explanation_on_complement requires binary edge-mask variables.")

        edge_index = np.asarray(edge_index)
        milp_edge_sequence, self_loop_counts = _edge_sequence_with_self_loops(edge_index, int(x.shape[0]))
        undirected_edges = []
        seen_nonloop = set()
        for edge in milp_edge_sequence:
            if edge[0] == edge[1]:
                continue
            if edge not in seen_nonloop:
                seen_nonloop.add(edge)
                undirected_edges.append(edge)

        x = np.asarray(x, dtype=float)

        _print_and_log(
            "[MIPExplainer] Explaining graph",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIPExplainer] Graph: {x.shape[0]} nodes, {len(undirected_edges)} edges",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )

        # Preprocessing/build/OBBT do not consume the final solver time limit.
        start_time = time.time()
        if isinstance(self.weights, VanillaGraphGnnWeights2L):
            graph_builder = build_vanilla_graphgnn_milp_2l
            graph_obbt_layers = (1, 2)
            graph_obbt_prefix = "[MIP-Graph-2L]"
        elif isinstance(self.weights, VanillaGraphGnnWeights4L):
            graph_builder = build_vanilla_graphgnn_milp_4l
            graph_obbt_layers = (1, 2, 3, 4)
            graph_obbt_prefix = "[MIP-Graph-4L]"
        else:
            graph_builder = build_vanilla_graphgnn_milp
            graph_obbt_layers = (1, 2, 3)
            graph_obbt_prefix = "[MIP-Graph]"
        if degree_table_layer1 and graph_builder is not build_vanilla_graphgnn_milp:
            raise ValueError("degree_table_layer1 is implemented for the 3-layer graph MILP only.")
        degree_table_kwargs = {"use_degree_table_layer1": True} if degree_table_layer1 else {}

        def make_graph_build(
            u_bound_overrides: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        ):
            return graph_builder(
                X=x,
                undirected_edges=milp_edge_sequence,
                weights=self.weights,
                readout=readout,
                rho=rho,
                edge_budget=edge_budget,
                node_budget=node_budget,
                min_edges=min_edges,
                objective=objective,
                connectivity=connectivity,
                connectivity_on_complement=explanation_on_complement,
                use_incident_cap_tightening=use_incident_cap_tightening,
                cap_mode="global",
                use_symmetry_breaking=use_symmetry_breaking,
                use_star_constraints=use_star_constraints,
                use_connectivity_lower_bound=use_connectivity_lower_bound,
                use_bridge_constraints=use_bridge_constraints,
                use_anderson_cuts=use_anderson_cuts,
                anderson_max_node_count=anderson_max_node_count,
                anderson_max_cuts=anderson_max_cuts,
                edge_branch_priority=edge_branch_priority,
                use_obbt_u=use_obbt_u,
                max_obbt_u_layers=max_obbt_u_layers,
                defer_obbt_u=True,
                u_bound_overrides=u_bound_overrides,
                verbose=self.verbose,
                time_limit_s=time_limit_s,
                wallclock_deadline_s=None,
                mip_gap=mip_gap,
                seed=seed,
                initial_solution=initial_solution,
                log_file=log_file,
                terminal_log_file=terminal_log_file,
                **degree_table_kwargs,
            )

        def install_graph_constraints(current_build) -> None:
            _apply_gurobi_param_overrides(
                current_build.model,
                gurobi_params,
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )
            _maybe_enforce_edge_budget_equality(
                model=current_build.model,
                z_vars=current_build.z,
                edges=current_build.edges,
                objective=objective,
                edge_budget=edge_budget,
                edge_budget_mode=edge_budget_mode,
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )
            if explanation_on_complement:
                if current_build.removal_z is None:
                    raise RuntimeError(
                        "Complement graph explanation requested without removal-edge variables."
                    )
                removal_budget = len(current_build.edges) - int(edge_budget)
                current_build.model.addConstr(
                    sum(current_build.removal_z[e] for e in current_build.edges)
                    == removal_budget,
                    name="removal_edge_budget_eq",
                )
            _maybe_enforce_node_budget(
                model=current_build.model,
                y_vars=current_build.y,
                num_nodes=current_build.n,
                node_budget=node_budget,
                connectivity=connectivity,
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )
            _maybe_enforce_node_induced(
                model=current_build.model,
                z_vars=current_build.z,
                y_vars=current_build.y,
                node_induced=node_induced,
                connectivity=connectivity,
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )
            _maybe_add_no_good_edge_solution_cuts(
                model=current_build.model,
                z_vars=current_build.z,
                edges=current_build.edges,
                excluded_edge_solutions=excluded_edge_solutions,
                verbose=self.verbose,
                terminal_log_file=terminal_log_file,
            )

        build = make_graph_build()
        install_graph_constraints(build)
        opt_start = time.time()
        _apply_obbt_u_bounds(
            build.model,
            use_obbt_u=use_obbt_u,
            max_obbt_u_layers=max_obbt_u_layers,
            available_layers=graph_obbt_layers,
            connectivity=connectivity,
            wallclock_deadline_s=None,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
            log_prefix=graph_obbt_prefix,
        )
        if use_obbt_u and obbt_rebuild:
            target_obbt_layers = tuple(
                layer
                for layer in graph_obbt_layers
                if max_obbt_u_layers is None or layer <= int(max_obbt_u_layers)
            )
            tightened_u_bounds = extract_u_bounds_from_model(
                build.model,
                layers=target_obbt_layers,
            )
            old_model = build.model
            build = make_graph_build(tightened_u_bounds)
            install_graph_constraints(build)
            try:
                old_model.dispose()
            except Exception:
                pass
        build_time = time.time() - start_time
        _maybe_relax_edge_mask_variables(
            model=build.model,
            z_vars=build.z,
            relax_edge_mask=relax_edge_mask,
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _set_model_time_limit(build.model, time_limit_s)
        solution = optimize_and_extract_graph(
            build,
            relax_edge_mask=relax_edge_mask,
            selection_edge_budget=edge_budget,
        )
        opt_time = time.time() - opt_start

        # Store verification context
        self._last_working_x = x
        self._last_working_edges = milp_edge_sequence
        self._last_self_loop_counts = self_loop_counts
        self._last_solution = solution

        _print_and_log(
            f"[MIPExplainer] Build time: {build_time:.2f}s, Opt time: {opt_time:.2f}s",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIPExplainer] Status: {solution.status}, Edges: {len(solution.selected_edges)}",
            verbose=self.verbose,
            terminal_log_file=terminal_log_file,
        )

        edge_mask = np.zeros(edge_index.shape[1], dtype=float)
        selected_set = set(solution.selected_edges)
        for k in range(edge_index.shape[1]):
            i, j = int(edge_index[0, k]), int(edge_index[1, k])
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in selected_set:
                edge_mask[k] = 1.0

        return edge_mask, solution

    def verify_last_solution(
        self,
        rho: Optional[float] = None,
        tol: float = 1e-4,
        verbose: bool = False,
    ) -> Tuple[bool, str]:
        """
        Verify the last MILP solution by running forward pass with selected edges.

        This uses the subgraph data that was used for MILP solving, ensuring
        correct verification even when k_hop subgraph extraction was used.

        Args:
            rho: Margin preservation ratio (uses rho_optimal from solution if available)
            tol: Tolerance for margin comparison
            verbose: Print debug info

        Returns:
            (is_valid, message): Tuple of validity flag and message
        """
        if self._last_solution is None:
            return False, "No solution to verify (run explain_node/explain_graph first)"

        sol = self._last_solution

        if not sol.selected_edges:
            return False, "No edges selected in solution"

        # Determine rho to use
        if rho is None:
            rho = sol.rho_optimal if sol.rho_optimal is not None else 0.0

        # Note: rho can be negative when MILP finds that prediction cannot be preserved
        # with the given edge budget. We still verify that rho_actual matches rho_optimal.

        if self.task == "node":
            from Main.solvers.mip_gnn_vanilla import verify_node_solution
            return verify_node_solution(
                X=self._last_working_x,
                selected_edges=sol.selected_edges,
                weights=self.weights,
                v_star=sol.v_star,
                c_star=sol.c_star,
                bar_m=sol.bar_m,
                rho=rho,
                self_loop_counts=self._last_self_loop_counts,
                tol=tol,
                verbose=verbose,
            )
        else:
            from Main.solvers.mip_gnn_vanilla import verify_graph_solution
            return verify_graph_solution(
                X=self._last_working_x,
                selected_edges=sol.selected_edges,
                weights=self.weights,
                c_star=sol.c_star,
                bar_m=sol.bar_m,
                rho=rho,
                readout="sum",
                self_loop_counts=self._last_self_loop_counts,
                tol=tol,
                verbose=verbose,
            )


def save_explanation_result(
    result_dir: str,
    dataset_name: str,
    node_idx: int,
    solution: MilpSolution,
    edge_mask: np.ndarray,
    metadata: Optional[Dict] = None,
):
    """
    Save MIP explanation results to file.

    Args:
        result_dir: Directory to save results
        dataset_name: Name of the dataset
        node_idx: Index of explained node (or graph index)
        solution: MilpSolution object
        edge_mask: Binary edge mask
        metadata: Additional metadata to save
    """
    os.makedirs(result_dir, exist_ok=True)

    result = {
        "dataset": dataset_name,
        "node_idx": node_idx,
        "status": solution.status,
        "objective_value": solution.objective_value,
        "num_selected_edges": len(solution.selected_edges),
        "selected_edges": solution.selected_edges,
        "rho_optimal": solution.rho_optimal,
    }

    if metadata:
        result.update(metadata)

    # Save JSON
    json_path = os.path.join(result_dir, f"node_{node_idx}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save edge mask
    mask_path = os.path.join(result_dir, f"node_{node_idx}_mask.npy")
    np.save(mask_path, edge_mask)

    return json_path


def load_explanation_result(result_dir: str, node_idx: int) -> Tuple[Dict, np.ndarray]:
    """Load saved explanation results."""
    json_path = os.path.join(result_dir, f"node_{node_idx}.json")
    mask_path = os.path.join(result_dir, f"node_{node_idx}_mask.npy")

    with open(json_path, "r") as f:
        result = json.load(f)

    edge_mask = np.load(mask_path)
    return result, edge_mask
