"""Scenario-table SAA-MILP for robust fidelity.

Before solving, the target-class probability is computed with the trained model for
every relevant edge subset in each fixed optimization scenario. The MILP then selects
the exact-K edge mask with the best scenario average. It does not encode the network.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from Main.solvers.gin_milp import (
    MilpSolution,
    _configure_gurobi_logging,
    _optimize_gurobi_model,
    _require_gurobi,
    _set_time_limit_from_deadline,
    _selected_connected_components,
)
from Main.audit_standard import edge_set
from Main.robust_common import (
    normalize_alpha_metric,
    effective_gurobi_norel_heur_work,
    normalize_connectivity,
    EvalContext,
    edge_col_lists,
    edge_weight_template,
    probabilities_from_weights,
)

# N_opt * 10^6: the evaluation budget implied by the manuscript's enumeration rule.
DEFAULT_SAA_TABLE_MAX_ENTRIES = 50_000_000


def inert_candidate_edge_mask(ctx: EvalContext, num_layers: int) -> np.ndarray:
    """Mark candidate edges that cannot influence the target logit.

    For a ``num_layers``-layer message-passing GIN whose target logit is read
    from the last layer, an edge whose two endpoints both lie at distance at
    least ``num_layers`` from the target never enters the target's
    computation.  Masking only removes edges, so distances in every masked
    input are at least those in the unmasked graph and the edge stays inert.
    Graph classification reads out every node, so no edge is inert there.
    """
    candidate_edges = list(ctx.candidate_edges)
    if ctx.task != "node":
        return np.zeros(len(candidate_edges), dtype=bool)
    if ctx.root_node is None:
        raise ValueError("Node task requires root_node.")
    neighbors: Dict[int, set] = {}
    for a, b in np.asarray(ctx.edge_index, dtype=np.int64).T:
        if int(a) == int(b):
            continue
        neighbors.setdefault(int(a), set()).add(int(b))
        neighbors.setdefault(int(b), set()).add(int(a))
    root = int(ctx.root_node)
    distance = {root: 0}
    queue: deque = deque([root])
    while queue:
        node = queue.popleft()
        if distance[node] >= int(num_layers):
            continue
        for neighbor in neighbors.get(node, ()):
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    far = int(num_layers)
    return np.asarray(
        [min(distance.get(int(a), far), distance.get(int(b), far)) >= far for a, b in candidate_edges],
        dtype=bool,
    )


def saa_table_switch_masks(metric: str, scenario_masks: np.ndarray) -> np.ndarray:
    """Return, per scenario, the candidate edges whose state depends on S.

    Sufficiency draws keep flags for non-selected edges, so an edge switches
    when its keep flag is false.  Necessity draws drop flags for selected
    edges, so an edge switches when its drop flag is true.
    """
    masks = np.asarray(scenario_masks, dtype=bool)
    if metric == "sufficiency":
        return ~masks
    if metric == "necessity":
        return masks.copy()
    raise ValueError(f"prob_table supports sufficiency and necessity, got {metric!r}.")


def saa_table_entry_count(switch_counts: Iterable[int], max_subset_size: int) -> int:
    return int(
        sum(
            sum(math.comb(int(d), j) for j in range(min(int(d), int(max_subset_size)) + 1))
            for d in switch_counts
        )
    )


def subset_membership_rows(num_switches: int, max_subset_size: int) -> np.ndarray:
    """Rows are all subsets of range(num_switches) with at most max_subset_size members."""
    import itertools

    num_switches = int(num_switches)
    sizes = range(min(num_switches, int(max_subset_size)) + 1)
    total = sum(math.comb(num_switches, j) for j in sizes)
    rows = np.zeros((total, num_switches), dtype=bool)
    row = 0
    for size in sizes:
        for subset in itertools.combinations(range(num_switches), size):
            if subset:
                rows[row, list(subset)] = True
            row += 1
    return rows


@dataclass
class SaaScenarioTable:
    """True-probability table p_s(T) for every scenario s and T ⊆ relevant switches."""

    metric: str
    inert: np.ndarray
    inert_reduction: bool
    relevant_idx: List[np.ndarray]
    membership: List[np.ndarray]
    probabilities: List[np.ndarray]
    build_time_s: float

    @property
    def entry_count(self) -> int:
        return int(sum(int(p.size) for p in self.probabilities))

    def score(self, selected: np.ndarray) -> float:
        """SAA average probability of a boolean candidate-edge selection."""
        selected = np.asarray(selected, dtype=bool)
        total = 0.0
        for relevant_idx, membership, probs in zip(self.relevant_idx, self.membership, self.probabilities):
            target = selected[relevant_idx]
            matches = np.flatnonzero((membership == target[None, :]).all(axis=1))
            if matches.size != 1:
                raise ValueError("Selection is not represented in the scenario table.")
            total += float(probs[int(matches[0])])
        return total / float(len(self.probabilities))


def build_saa_scenario_table(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    metric: str,
    scenario_masks: np.ndarray,
    max_subset_size: int,
    num_layers: int,
    inert_reduction: bool,
    inference_batch: int,
    max_entries: int,
) -> SaaScenarioTable:
    t_build = time.perf_counter()
    candidate_edges = list(ctx.candidate_edges)
    m = len(candidate_edges)
    switches = saa_table_switch_masks(metric, scenario_masks)
    inert = inert_candidate_edge_mask(ctx, int(num_layers)) if inert_reduction else np.zeros(m, dtype=bool)
    relevant_switches = switches & ~inert[None, :]
    entry_count = saa_table_entry_count(relevant_switches.sum(axis=1), max_subset_size)
    if entry_count > int(max_entries):
        raise ValueError(
            f"prob_table needs {entry_count} table entries, above --saa_table_max_entries={int(max_entries)}."
        )

    template = edge_weight_template(ctx, candidate_edges)
    candidate_cols = edge_col_lists(ctx, candidate_edges)
    col_index = torch.tensor([col for cols in candidate_cols for col in cols], dtype=torch.long, device=ctx.device)
    col_owner = torch.tensor(
        [edge_idx for edge_idx, cols in enumerate(candidate_cols) for _ in cols],
        dtype=torch.long,
        device=ctx.device,
    )

    def keep_probabilities(keep: np.ndarray) -> np.ndarray:
        out = np.empty(int(keep.shape[0]), dtype=np.float64)
        keep_tensor = torch.from_numpy(np.ascontiguousarray(keep)).to(ctx.device)
        for start in range(0, int(keep.shape[0]), int(inference_batch)):
            chunk = keep_tensor[start : start + int(inference_batch)]
            size = int(chunk.shape[0])
            weights = template.repeat(size, 1)
            weights[:, col_index] = chunk.index_select(1, col_owner).to(weights.dtype)
            probs = probabilities_from_weights(
                model=model,
                ctx=ctx,
                edge_weights=weights.reshape(-1),
                batch_size=size,
            )
            out[start : start + size] = probs.numpy().astype(np.float64)
        return out

    relevant_list: List[np.ndarray] = []
    membership_list: List[np.ndarray] = []
    probability_list: List[np.ndarray] = []
    keep_chunk_rows = 16384
    for s in range(int(switches.shape[0])):
        relevant_idx = np.flatnonzero(relevant_switches[s])
        switch_idx = np.flatnonzero(switches[s])
        membership = subset_membership_rows(len(relevant_idx), max_subset_size)
        chunks: List[np.ndarray] = []
        for start in range(0, int(membership.shape[0]), keep_chunk_rows):
            chunk = membership[start : start + keep_chunk_rows]
            keep = np.ones((int(chunk.shape[0]), m), dtype=bool)
            if metric == "sufficiency":
                # Non-selected switches are dropped; inert switches stay dropped.
                keep[:, switch_idx] = False
                keep[:, relevant_idx] = chunk
            else:
                # Selected switches are dropped; inert switches stay present.
                keep[:, relevant_idx] = ~chunk
            chunks.append(keep_probabilities(keep))
        relevant_list.append(relevant_idx)
        membership_list.append(membership)
        probability_list.append(np.concatenate(chunks))
    table = SaaScenarioTable(
        metric=metric,
        inert=inert,
        inert_reduction=bool(inert_reduction),
        relevant_idx=relevant_list,
        membership=membership_list,
        probabilities=probability_list,
        build_time_s=time.perf_counter() - t_build,
    )
    if table.entry_count != entry_count:
        raise RuntimeError(f"Table entry count mismatch: built {table.entry_count}, expected {entry_count}.")
    return table


def saa_table_matrices(table: SaaScenarioTable, num_candidate_edges: int) -> Dict[str, Any]:
    """Sparse constraint blocks of the table formulation over x = (z, λ).

    conv @ λ = 1 picks one row per scenario, and link_lambda @ λ - link_z @ z = 0
    ties the picked row to the selected switch edges.
    """
    import scipy.sparse as sp

    conv_sizes: List[int] = []
    link_rows: List[np.ndarray] = []
    link_cols: List[np.ndarray] = []
    link_z_rows: List[int] = []
    link_z_cols: List[int] = []
    offset = 0
    link_row = 0
    for relevant_idx, membership in zip(table.relevant_idx, table.membership):
        n_entries = int(membership.shape[0])
        conv_sizes.append(n_entries)
        entry_idx, switch_pos = np.nonzero(membership)
        link_rows.append((link_row + switch_pos).astype(np.int32))
        link_cols.append((offset + entry_idx).astype(np.int32))
        for pos, edge_idx in enumerate(relevant_idx):
            link_z_rows.append(link_row + pos)
            link_z_cols.append(int(edge_idx))
        link_row += len(relevant_idx)
        offset += n_entries
    num_scenarios = len(conv_sizes)
    conv = sp.csr_matrix(
        (
            np.ones(offset),
            (
                np.repeat(np.arange(num_scenarios, dtype=np.int32), conv_sizes),
                np.arange(offset, dtype=np.int32),
            ),
        ),
        shape=(num_scenarios, offset),
    )
    link_row_index = np.concatenate(link_rows) if link_rows else np.zeros(0, dtype=np.int32)
    link_col_index = np.concatenate(link_cols) if link_cols else np.zeros(0, dtype=np.int32)
    del link_rows, link_cols
    link_lambda = sp.csr_matrix(
        (np.ones(int(link_row_index.size)), (link_row_index, link_col_index)),
        shape=(link_row, offset),
    )
    del link_row_index, link_col_index
    link_z = sp.csr_matrix(
        (
            np.ones(len(link_z_rows)),
            (np.asarray(link_z_rows, dtype=np.int64), np.asarray(link_z_cols, dtype=np.int64)),
        ),
        shape=(link_row, int(num_candidate_edges)),
    )
    return {
        "p_flat": np.concatenate(table.probabilities),
        "conv": conv,
        "link_lambda": link_lambda,
        "link_z": link_z,
    }


def solve_saa_table_milp(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    metric: str,
    alpha: float,
    scenario_masks: np.ndarray,
    edge_budget: int,
    edge_budget_mode: str,
    connectivity: str,
    gt_edges: Sequence[Tuple[int, int]],
    args: argparse.Namespace,
    opt_seed: int,
    bar_m: float,
    log_file: Path,
    terminal_log: Path,
) -> Tuple[MilpSolution, Dict[str, Any]]:
    """Solve the fixed-scenario SAA exactly with true-probability scenario tables.

    Scenario s depends on S only through S ∩ D_s, where D_s are the switches
    of ``saa_table_switch_masks``.  For every T ⊆ D_s the true model
    probability p_s(T) is precomputed.  Continuous λ_{s,T} with
    Σ_T λ_{s,T} = 1 and z_e = Σ_{T∋e} λ_{s,T} select the table row, which
    integral z forces to T = S ∩ D_s.  The objective is the SAA average of
    true probabilities, maximized for sufficiency and minimized for necessity.
    """
    gp, GRB = _require_gurobi()

    metric = normalize_alpha_metric(metric)
    if metric not in {"sufficiency", "necessity"}:
        raise ValueError("prob_table supports --metric sufficiency or necessity.")
    connectivity = normalize_connectivity(connectivity)
    if connectivity not in {"none", "cut"}:
        raise ValueError("prob_table supports --connectivity none or cut.")

    candidate_edges = list(ctx.candidate_edges)
    m = len(candidate_edges)
    masks = np.asarray(scenario_masks, dtype=bool)
    if masks.ndim != 2 or int(masks.shape[1]) != m:
        raise ValueError(f"scenario mask shape {masks.shape} does not match {m} candidate edges.")
    num_scenarios = int(masks.shape[0])
    effective_budget = max(0, min(int(edge_budget), m))

    table = build_saa_scenario_table(
        model=model,
        ctx=ctx,
        metric=metric,
        scenario_masks=masks,
        max_subset_size=effective_budget,
        num_layers=int(args.num_layers),
        inert_reduction=bool(getattr(args, "saa_table_inert_reduction", True)),
        inference_batch=int(args.graph_batch_size if ctx.task == "graph" else args.node_batch_size),
        max_entries=int(getattr(args, "saa_table_max_entries", DEFAULT_SAA_TABLE_MAX_ENTRIES)),
    )
    inert = table.inert
    use_reduction = table.inert_reduction
    switch_counts = np.asarray([len(idx) for idx in table.relevant_idx], dtype=np.int64)
    build_time = table.build_time_s
    matrices = saa_table_matrices(table, m)
    del table
    p_flat = matrices["p_flat"]
    conv_matrix = matrices["conv"]
    link_lambda = matrices["link_lambda"]
    link_z = matrices["link_z"]
    offset = int(p_flat.size)
    link_row = int(link_lambda.shape[0])

    gurobi_model = gp.Model("Main_SAA_Table_MILP")
    _configure_gurobi_logging(gurobi_model, verbose=True, log_file=str(log_file))
    gurobi_model.Params.Seed = int(opt_seed) % 2_000_000_000
    _set_time_limit_from_deadline(
        gurobi_model,
        wallclock_deadline_s=None,
        fallback_time_limit_s=None if args.time_limit is None else float(args.time_limit),
    )
    if args.gurobi_threads is not None:
        gurobi_model.Params.Threads = max(1, int(args.gurobi_threads))
    if args.gurobi_method is not None:
        gurobi_model.Params.Method = int(args.gurobi_method)
    if args.gurobi_numeric_focus is not None:
        gurobi_model.Params.NumericFocus = int(args.gurobi_numeric_focus)
    gurobi_model.Params.NoRelHeurWork = max(0.0, float(effective_gurobi_norel_heur_work(args)))
    if args.gurobi_nodefile_start is not None:
        gurobi_model.Params.NodefileStart = max(0.0, float(args.gurobi_nodefile_start))
    if args.gurobi_nodefile_dir:
        gurobi_model.Params.NodefileDir = str(args.gurobi_nodefile_dir)
    if args.gurobi_mem_limit is not None:
        gurobi_model.Params.MemLimit = max(0.0, float(args.gurobi_mem_limit))
    if getattr(args, "gurobi_soft_mem_limit", None) is not None:
        gurobi_model.Params.SoftMemLimit = max(0.0, float(args.gurobi_soft_mem_limit))
    if args.gurobi_symmetry is not None:
        gurobi_model.Params.Symmetry = int(args.gurobi_symmetry)
    # The default relative MIPGap (1e-4) is of the same order as the audit tolerance,
    # so the table solve is run to a proven optimum unless requested otherwise.
    gurobi_model.Params.MIPGap = max(0.0, float(getattr(args, "saa_table_mip_gap", 0.0)))

    # One variable vector x = (z, λ) and CSR blocks passed through addMConstr keep
    # memory close to the matrix size for tables with millions of entries.
    import scipy.sparse as sp

    num_vars = m + offset
    vtypes = np.full(num_vars, GRB.CONTINUOUS)
    vtypes[:m] = GRB.BINARY
    objective = np.zeros(num_vars)
    objective[m:] = p_flat / float(num_scenarios)
    x = gurobi_model.addMVar(num_vars, lb=0.0, ub=1.0, obj=objective, vtype=vtypes, name="x")
    del vtypes, objective, p_flat
    gurobi_model.ModelSense = GRB.MAXIMIZE if metric == "sufficiency" else GRB.MINIMIZE

    def z_row(coefficients: np.ndarray) -> Any:
        return sp.csr_matrix(
            (np.asarray(coefficients, dtype=float), (np.zeros(m, dtype=np.int64), np.arange(m))),
            shape=(1, num_vars),
        )

    budget_sense = GRB.EQUAL if str(edge_budget_mode) == "eq" else GRB.LESS_EQUAL
    gurobi_model.addMConstr(z_row(np.ones(m)), x, budget_sense, np.asarray([float(effective_budget)]), name="edge_budget")

    gt_edge_set = set(edge_set(gt_edges))
    candidate_lookup = set(candidate_edges)
    gt_solution_representable = gt_edge_set.issubset(candidate_lookup)
    gt_no_good_cut_added = False
    if bool(args.exclude_gt) and gt_solution_representable:
        in_gt = np.asarray([edge in gt_edge_set for edge in candidate_edges], dtype=bool)
        gurobi_model.addMConstr(
            z_row(np.where(in_gt, -1.0, 1.0)),
            x,
            GRB.GREATER_EQUAL,
            np.asarray([1.0 - float(in_gt.sum())]),
            name="nogood_exact_gt",
        )
        gt_no_good_cut_added = True
    if bool(args.gt_incumbent):
        x[:m].Start = np.asarray([1.0 if edge in gt_edge_set else 0.0 for edge in candidate_edges])

    conv_block = sp.hstack([sp.csr_matrix((num_scenarios, m)), conv_matrix], format="csr")
    del conv_matrix
    gurobi_model.addMConstr(conv_block, x, GRB.EQUAL, np.ones(num_scenarios), name="table_convexity")
    del conv_block
    if link_row > 0:
        link_block = sp.hstack([-link_z, link_lambda], format="csr")
        del link_lambda, link_z
        gurobi_model.addMConstr(link_block, x, GRB.EQUAL, np.zeros(link_row), name="table_link")
        del link_block
    del matrices
    gurobi_model.update()

    # Connectivity acts on z only. The block adds node variables y, the root (the target for node tasks,
    # a chosen root r for graph tasks), star and connectivity lower-bound inequalities, and lazy cuts
    # y_t <= z(delta(C)) [+ r(C) + empty] for selected components C without the root.
    connectivity_cut_count = [0]
    connectivity_callback = None
    root_cut_state: Dict[str, Any] = {
        "enabled": False, "calls": 0, "cuts": 0, "first_root_bound": None, "last_root_bound": None,
    }
    if connectivity == "cut":
        gurobi_model.Params.LazyConstraints = 1
        z_vars = x[:m].tolist()
        edge_var = {edge: z_vars[idx] for idx, edge in enumerate(candidate_edges)}
        node_list = sorted({int(node) for edge in candidate_edges for node in edge})
        if ctx.task == "node":
            if ctx.root_node is None:
                raise ValueError("Node prob_table connectivity requires root_node.")
            root_node = int(ctx.root_node)
            if root_node not in node_list:
                node_list = sorted(set(node_list) | {root_node})
        else:
            root_node = None
        y = {i: gurobi_model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in node_list}
        incident: Dict[int, List[Any]] = {i: [] for i in node_list}
        for (i, j), zvar in edge_var.items():
            incident[int(i)].append(zvar)
            incident[int(j)].append(zvar)
        gurobi_model.update()
        for (i, j), zvar in edge_var.items():
            gurobi_model.addConstr(zvar <= y[int(i)], name=f"edge_node1_{i}_{j}")
            gurobi_model.addConstr(zvar <= y[int(j)], name=f"edge_node2_{i}_{j}")
        r: Optional[Dict[int, Any]] = None
        eps_empty: Any = None
        use_star = bool(getattr(args, "use_star_constraints", False))
        if ctx.task == "node":
            gurobi_model.addConstr(y[root_node] == 1.0, name="root_selected")
            if use_star:
                for i in node_list:
                    if i == root_node:
                        continue
                    if incident[i]:
                        gurobi_model.addConstr(y[i] <= gp.quicksum(incident[i]), name=f"star_{i}")
                    else:
                        gurobi_model.addConstr(y[i] == 0.0, name=f"isolated_{i}")
        else:
            r = {i: gurobi_model.addVar(vtype=GRB.BINARY, name=f"r_{i}") for i in node_list}
            eps_empty = gurobi_model.addVar(vtype=GRB.BINARY, name="eps_empty")
            gurobi_model.update()
            gurobi_model.addConstr(eps_empty + gp.quicksum(r.values()) == 1.0, name="root_or_empty")
            for i in node_list:
                gurobi_model.addConstr(y[i] <= 1.0 - eps_empty, name=f"y_empty_{i}")
                gurobi_model.addConstr(r[i] <= y[i], name=f"root_implies_selected_{i}")
                if use_star:
                    if incident[i]:
                        gurobi_model.addConstr(
                            y[i] <= gp.quicksum(incident[i]) + r[i] + eps_empty, name=f"star_or_root_{i}"
                        )
                    else:
                        gurobi_model.addConstr(y[i] <= r[i] + eps_empty, name=f"isolated_or_root_{i}")
        if bool(getattr(args, "use_connectivity_lower_bound", False)):
            gurobi_model.addConstr(
                gp.quicksum(z_vars) >= gp.quicksum(y.values()) - 1.0, name="connectivity_lower_bound"
            )
        gurobi_model.update()

        # Optional fractional separation at the root (--saa_table_conn_root_cuts). The inequalities are
        # the same cut-set inequalities as the lazy cuts below, so they are valid for every connected mask.
        root_cut_state["enabled"] = bool(getattr(args, "saa_table_conn_root_cuts", False))
        root_cut_rounds = max(0, int(getattr(args, "saa_table_conn_root_cut_rounds", 50)))
        if root_cut_state["enabled"]:
            import networkx as nx

            gurobi_model.Params.PreCrush = 1
        root_cut_tol = 1e-5

        def _separate_root_connectivity_cuts(cb_model: Any) -> None:
            if root_cut_state["calls"] >= root_cut_rounds:
                return
            if int(cb_model.cbGet(GRB.Callback.MIPNODE_NODCNT)) != 0:
                return
            if int(cb_model.cbGet(GRB.Callback.MIPNODE_STATUS)) != GRB.OPTIMAL:
                return
            bound = float(cb_model.cbGet(GRB.Callback.MIPNODE_OBJBND))
            if root_cut_state["first_root_bound"] is None:
                root_cut_state["first_root_bound"] = bound
            root_cut_state["last_root_bound"] = bound
            root_cut_state["calls"] += 1
            z_rel = cb_model.cbGetNodeRel(z_vars)
            y_rel = cb_model.cbGetNodeRel([y[i] for i in node_list])
            flow_graph = nx.DiGraph()
            flow_graph.add_nodes_from(node_list)
            for k, (a, b) in enumerate(candidate_edges):
                cap = float(z_rel[k])
                if cap <= 1e-9:
                    continue
                for u, v in ((int(a), int(b)), (int(b), int(a))):
                    if flow_graph.has_edge(u, v):
                        flow_graph[u][v]["capacity"] += cap
                    else:
                        flow_graph.add_edge(u, v, capacity=cap)
            if ctx.task == "node":
                source: Any = root_node
                constant = 0.0
            else:
                r_rel = cb_model.cbGetNodeRel([r[i] for i in node_list])
                source = "__root_source__"
                flow_graph.add_node(source)
                for k, i in enumerate(node_list):
                    if float(r_rel[k]) > 1e-9:
                        flow_graph.add_edge(source, i, capacity=float(r_rel[k]))
                constant = float(cb_model.cbGetNodeRel(eps_empty))
            for k, t in enumerate(node_list):
                y_t = float(y_rel[k])
                if y_t <= root_cut_tol or (ctx.task == "node" and t == root_node):
                    continue
                cut_value, (_, sink_side) = nx.minimum_cut(flow_graph, source, t, capacity="capacity")
                if float(cut_value) + constant >= y_t - root_cut_tol:
                    continue
                component = {int(v) for v in sink_side if v != source}
                rhs = gp.quicksum(
                    zvar for (a, b), zvar in edge_var.items() if (int(a) in component) != (int(b) in component)
                )
                if ctx.task != "node":
                    rhs = rhs + gp.quicksum(r[i] for i in component) + eps_empty
                cb_model.cbCut(y[t] <= rhs)
                root_cut_state["cuts"] += 1

        def connectivity_callback(cb_model: Any, where: int) -> None:
            if where == GRB.Callback.MIPNODE:
                if root_cut_state["enabled"]:
                    _separate_root_connectivity_cuts(cb_model)
                return
            if where != GRB.Callback.MIPSOL:
                return
            z_sol = cb_model.cbGetSolution(z_vars)
            y_sol = cb_model.cbGetSolution([y[i] for i in node_list])
            selected_nodes = {node_list[k] for k, value in enumerate(y_sol) if value > 0.5}
            if not selected_nodes:
                return
            selected = [candidate_edges[k] for k, value in enumerate(z_sol) if value > 0.5]
            components = _selected_connected_components(selected_nodes, selected)
            if len(components) <= 1:
                return
            if ctx.task == "node":
                root_component = next((c for c in components if root_node in c), None)
            else:
                if float(cb_model.cbGetSolution(eps_empty)) > 0.5:
                    return
                r_sol = cb_model.cbGetSolution([r[i] for i in node_list])
                chosen_root = node_list[int(np.argmax(r_sol))]
                root_component = next((c for c in components if chosen_root in c), None)
            if root_component is None:
                return
            for component in components:
                if component == root_component:
                    continue
                t = min(component)
                cut_edges = [
                    zvar for (a, b), zvar in edge_var.items() if (int(a) in component) != (int(b) in component)
                ]
                rhs = gp.quicksum(cut_edges)
                if ctx.task != "node":
                    rhs = rhs + gp.quicksum(r[i] for i in component) + eps_empty
                cb_model.cbLazy(y[t] <= rhs)
                connectivity_cut_count[0] += 1

    table_metadata = {
        "saa_table_entries": int(offset),
        "saa_table_build_time_s": float(build_time),
        "saa_table_max_switches": int(switch_counts.max()) if num_scenarios else 0,
        "saa_table_mean_switches": float(switch_counts.mean()) if num_scenarios else 0.0,
        "saa_table_relevant_edges": int(m - int(inert.sum())),
        "saa_table_inert_edges": int(inert.sum()),
        "saa_table_inert_reduction": bool(use_reduction),
    }
    terminal_log.parent.mkdir(parents=True, exist_ok=True)
    with terminal_log.open("a", encoding="utf-8") as f:
        f.write(
            f"SAA-Table task={ctx.task} metric={metric} alpha={alpha} scenarios={num_scenarios} "
            f"budget={effective_budget} mode={edge_budget_mode} candidate_edges={m} "
            f"inert_edges={int(inert.sum())} inert_reduction={bool(use_reduction)} "
            f"table_entries={offset} max_switches={table_metadata['saa_table_max_switches']} "
            f"mean_switches={table_metadata['saa_table_mean_switches']:.6g} "
            f"build_time_s={build_time:.3f} mip_gap={float(getattr(args, 'saa_table_mip_gap', 0.0))} "
            f"connectivity={connectivity} "
            f"star_constraints={bool(getattr(args, 'use_star_constraints', False)) if connectivity == 'cut' else False} "
            f"connectivity_lower_bound={bool(getattr(args, 'use_connectivity_lower_bound', False)) if connectivity == 'cut' else False} "
            f"exclude_gt_solution={bool(args.exclude_gt)} "
            f"gt_no_good_cut_added={gt_no_good_cut_added} objective_type=prob_table\n"
        )

    solve_t0 = time.perf_counter()
    interrupted = _optimize_gurobi_model(
        gurobi_model, callback=connectivity_callback, terminal_log_file=str(terminal_log)
    )
    elapsed = time.perf_counter() - solve_t0
    table_metadata["saa_table_solve_time_s"] = float(elapsed)
    if root_cut_state["enabled"]:
        table_metadata["saa_table_conn_root_cut_calls"] = int(root_cut_state["calls"])
        table_metadata["saa_table_conn_root_cuts_added"] = int(root_cut_state["cuts"])
        table_metadata["saa_table_conn_first_root_bound"] = root_cut_state["first_root_bound"]
        table_metadata["saa_table_conn_last_root_bound"] = root_cut_state["last_root_bound"]
        with terminal_log.open("a", encoding="utf-8") as f:
            f.write(
                f"SAA-Table root connectivity cuts: calls={root_cut_state['calls']} "
                f"cuts={root_cut_state['cuts']} first_root_bound={root_cut_state['first_root_bound']} "
                f"last_root_bound={root_cut_state['last_root_bound']}\n"
            )

    def safe_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    status_code = int(gurobi_model.Status)
    if interrupted or status_code == getattr(GRB, "INTERRUPTED", 11):
        status = "interrupted_by_user"
    elif status_code == GRB.OPTIMAL:
        status = "optimal"
    elif status_code == GRB.TIME_LIMIT:
        status = "time_limit"
    elif status_code == getattr(GRB, "MEM_LIMIT", -999):
        status = "mem_limit"
    elif status_code == GRB.INFEASIBLE:
        status = "infeasible"
    else:
        status = f"status_{status_code}"

    has_solution = getattr(gurobi_model, "SolCount", 0) > 0
    objective_value = safe_float(gurobi_model.ObjVal) if has_solution else None
    z_array = np.asarray(x.X[:m], dtype=float) if has_solution else np.zeros(m)
    z_values = {edge: float(z_array[idx]) for idx, edge in enumerate(candidate_edges)} if has_solution else {}
    selected_edges = [edge for idx, edge in enumerate(candidate_edges) if z_array[idx] > 0.5] if has_solution else []
    with terminal_log.open("a", encoding="utf-8") as f:
        f.write(
            f"SAA-MILP status={status} objective={objective_value} "
            f"bound={safe_float(getattr(gurobi_model, 'ObjBound', None))} "
            f"gap={safe_float(getattr(gurobi_model, 'MIPGap', None))} "
            f"nodes={safe_float(getattr(gurobi_model, 'NodeCount', None))} "
            f"root_lp=None root_lp_time=None "
            f"model_size={safe_float(getattr(gurobi_model, 'NumNZs', None))} "
            f"anderson_cuts=0 connectivity_cuts={connectivity_cut_count[0]} "
            f"time_s={elapsed:.3f} "
            f"objective_type=prob_table\n"
        )
    solution = MilpSolution(
        status=status,
        objective_value=objective_value,
        selected_edges=selected_edges,
        z_values=z_values,
        rho_optimal=objective_value,
        c_star=int(ctx.c_star),
        bar_m=float(bar_m),
        final_mip_gap=safe_float(getattr(gurobi_model, "MIPGap", None)),
        best_bound=safe_float(getattr(gurobi_model, "ObjBound", None)),
        incumbent_obj=objective_value,
        model_size=safe_float(getattr(gurobi_model, "NumNZs", None)),
        callback_time_sec=0.0,
        callback_calls=0,
        cuts_mipnode=0,
        cuts_mipsol_lazy=0,
        anderson_cuts_added=0,
        connectivity_lazy_cuts_added=int(connectivity_cut_count[0]),
        bnb_node_count=safe_float(getattr(gurobi_model, "NodeCount", None)),
    )
    gurobi_model.dispose()
    return solution, table_metadata
