"""
MIP formulation for the MILP-friendly GIN NodeGNN model.

Implements the MILP formulation of the trained GIN given in the manuscript appendix.

Model (node classification):
  Let h0 = X.

  For layer l in {1,2,3,4}:
    s_i^(l) = (1 + eps_l) * h_i^(l-1) + sum_{j in N(i)} z_ij * h_j^(l-1)
    u_i^(l) = s_i^(l) @ W1^(l) + b1^(l)
    a_i^(l) = ReLU(u_i^(l))
    h_i^(l) = a_i^(l) @ W2^(l) + b2^(l)

  Output logits for target node v_star:
    Z = [h1, h2, h3, h4][v_star] @ W_out.T + b_out  (JK-concat)
    or Z = hL[v_star] @ W_out.T + b_out             (last-layer-only)

Where z_ij are binary edge-selection variables.
This is an exact MILP (only ReLU introduces integer variables).

Note: Do NOT use addGenConstrIndicator (not suitable for research purposes).
"""

from __future__ import annotations

from collections import deque
import signal
from dataclasses import dataclass
import os
import threading
import time
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union, cast

import numpy as np

# Try to import scipy for accelerated max-flow computation
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    _HAS_SCIPY_MAXFLOW = True
except ImportError:
    _HAS_SCIPY_MAXFLOW = False

ObjectiveMode = Literal["min_edges", "max_sufficiency", "min_sufficiency", "feasible"]
ConnectivityMode = Literal["none", "flow", "cut"]


class CurrentGurobiSolveInterrupted(Exception):
    """Raised when the user aborts only the current Gurobi solve via Ctrl+\."""


def _require_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "gurobipy is required for MIPExplainer. "
            "Install Gurobi + gurobipy and ensure you have a valid license."
        ) from e
    return gp, GRB


def _validate_objective_mode(objective: str, rho: float) -> None:
    if objective not in ("min_edges", "max_sufficiency", "min_sufficiency", "feasible"):
        raise ValueError("Unknown objective mode")
    if objective in ("min_edges", "feasible") and not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0,1] for min_edges/feasible")


def _add_margin_constraints(
    *,
    model: object,
    Z: Dict[int, object],
    c_star: int,
    num_classes: int,
    bar_m: float,
    objective: str,
    rho: float,
) -> Optional[object]:
    gp, GRB = _require_gurobi()
    del gp

    rho_var = None
    if objective == "max_sufficiency":
        # Allow negative rho (prediction may flip if subgraph cannot preserve margin).
        rho_var = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho")
        model.update()
        for c in range(num_classes):
            if c == c_star:
                continue
            model.addConstr(Z[c_star] - Z[c] - float(bar_m) * rho_var >= 0.0, name=f"margin_{c}")
    elif objective == "min_sufficiency":
        if int(num_classes) != 2:
            raise ValueError("objective='min_sufficiency' is only supported for binary classification")
        if float(bar_m) <= 0.0:
            raise ValueError("objective='min_sufficiency' requires a positive reference margin")
        other_c = 1 - int(c_star)
        rho_var = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="rho")
        model.update()
        model.addConstr(
            Z[c_star] - Z[other_c] - float(bar_m) * rho_var == 0.0,
            name=f"margin_{other_c}",
        )
    else:
        for c in range(num_classes):
            if c == c_star:
                continue
            model.addConstr(Z[c_star] - Z[c] >= float(rho * bar_m), name=f"margin_{c}")
    return rho_var


def _set_selection_objective(
    *,
    model: object,
    z: Dict[Tuple[int, int], object],
    edges: Sequence[Tuple[int, int]],
    objective: str,
    rho_var: Optional[object],
    edge_budget: Optional[int],
    node_budget: Optional[int],
) -> None:
    gp, GRB = _require_gurobi()
    if objective == "min_edges":
        model.ModelSense = GRB.MINIMIZE
        model.setObjective(gp.quicksum(z[e] for e in edges))
    elif objective == "max_sufficiency":
        if edge_budget is None and node_budget is None:
            raise ValueError("objective='max_sufficiency' requires edge_budget or node_budget")
        model.ModelSense = GRB.MAXIMIZE
        model.setObjective(rho_var)
    elif objective == "min_sufficiency":
        if edge_budget is None and node_budget is None:
            raise ValueError("objective='min_sufficiency' requires edge_budget or node_budget")
        model.ModelSense = GRB.MINIMIZE
        model.setObjective(rho_var)
    else:
        if edge_budget is None and node_budget is None:
            raise ValueError("objective='feasible' requires edge_budget or node_budget")
        model.ModelSense = GRB.MINIMIZE
        model.setObjective(0.0)


def _set_edge_branch_priority(
    z: Dict[Tuple[int, int], object],
    edge_branch_priority: int,
) -> None:
    """Prefer branching on explanation-edge variables when requested."""
    priority = int(edge_branch_priority)
    if priority < 0:
        raise ValueError("edge_branch_priority must be nonnegative")
    if priority == 0:
        return
    for z_var in z.values():
        z_var.BranchPriority = priority


def _intersect_u_bound_override(
    L_u: np.ndarray,
    U_u: np.ndarray,
    *,
    layer: int,
    u_bound_overrides: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect analytic bounds with certified LP-OBBT bounds for one layer."""
    if not u_bound_overrides or int(layer) not in u_bound_overrides:
        return L_u, U_u
    override_L, override_U = u_bound_overrides[int(layer)]
    override_L = np.asarray(override_L, dtype=float)
    override_U = np.asarray(override_U, dtype=float)
    if override_L.shape != L_u.shape or override_U.shape != U_u.shape:
        raise ValueError(
            f"OBBT u-bound shape mismatch at layer {layer}: "
            f"expected {L_u.shape}, got {override_L.shape}/{override_U.shape}"
        )
    tightened_L = np.maximum(np.asarray(L_u, dtype=float), override_L)
    tightened_U = np.minimum(np.asarray(U_u, dtype=float), override_U)
    if np.any(tightened_L > tightened_U + 1e-6):
        raise ValueError(f"Inconsistent OBBT u bounds at layer {layer}")
    # Absorb harmless LP/numeric crossings at equality.
    crossing = tightened_L > tightened_U
    if np.any(crossing):
        midpoint = 0.5 * (tightened_L[crossing] + tightened_U[crossing])
        tightened_L[crossing] = midpoint
        tightened_U[crossing] = midpoint
    return tightened_L, tightened_U


def extract_u_bounds_from_model(
    model: object,
    *,
    layers: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Extract current u-variable bounds for a certified two-pass MILP rebuild."""
    requested = {int(layer) for layer in layers}
    entries: Dict[int, List[Tuple[int, int, float, float]]] = {
        layer: [] for layer in requested
    }
    for var in model.getVars():
        parts = str(var.VarName).split("_")
        if len(parts) != 3 or not parts[0].startswith("u"):
            continue
        try:
            layer = int(parts[0][1:])
            node = int(parts[1])
            channel = int(parts[2])
        except ValueError:
            continue
        if layer in entries:
            entries[layer].append((node, channel, float(var.LB), float(var.UB)))

    result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for layer in sorted(requested):
        values = entries[layer]
        if not values:
            raise ValueError(f"No u variables found for requested OBBT layer {layer}")
        n = max(node for node, _, _, _ in values) + 1
        width = max(channel for _, channel, _, _ in values) + 1
        L_u = np.full((n, width), np.nan, dtype=float)
        U_u = np.full((n, width), np.nan, dtype=float)
        for node, channel, lb, ub in values:
            L_u[node, channel] = lb
            U_u[node, channel] = ub
        if np.isnan(L_u).any() or np.isnan(U_u).any():
            raise ValueError(f"Incomplete u-variable grid for OBBT layer {layer}")
        result[layer] = (L_u, U_u)
    return result


def _unique_undirected_edges(edges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Convert edges to unique undirected edge list."""
    undirected = set()
    for i, j in edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        undirected.add((a, b))
    return sorted(undirected)


@dataclass(frozen=True)
class Adjacency:
    """Adjacency list representation (excludes self-loops)."""
    n: int
    neighbors: List[List[int]]
    self_loop_counts: np.ndarray


def compute_adjacency(n: int, undirected_edges: Sequence[Tuple[int, int]]) -> Adjacency:
    """Compute adjacency list from edge list while tracking fixed self-loops."""
    if n <= 0:
        raise ValueError("n must be positive")
    neighbors: List[List[int]] = [[] for _ in range(n)]
    self_loop_counts = np.zeros(n, dtype=float)
    edge_set = set()
    for i, j in undirected_edges:
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"Edge ({i},{j}) out of bounds for n={n}")
        if i == j:
            self_loop_counts[i] += 1.0
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edge_set:
            continue
        edge_set.add((a, b))
        neighbors[a].append(b)
        neighbors[b].append(a)
    return Adjacency(n=n, neighbors=neighbors, self_loop_counts=self_loop_counts)


def _selected_connected_components(
    selected_nodes: Set[int],
    selected_edges: Sequence[Tuple[int, int]],
) -> List[Set[int]]:
    """Return the components induced by selected nodes and selected edges."""
    if not selected_nodes:
        return []

    adjacency: Dict[int, Set[int]] = {
        int(node): set() for node in selected_nodes
    }
    for i, j in selected_edges:
        i = int(i)
        j = int(j)
        if i not in adjacency or j not in adjacency:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)

    components: List[Set[int]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        component = {start}
        queue = deque([start])
        unseen.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def _node_self_coefficients(adj: Adjacency, alpha: float) -> np.ndarray:
    """Return per-node coefficient for the always-on self message."""
    return float(alpha) + adj.self_loop_counts


def _selected_self_loop_counts(
    n: int,
    self_loop_counts: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    if self_loop_counts is None:
        return np.zeros(n, dtype=float)
    arr = np.asarray(self_loop_counts, dtype=float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"Expected self_loop_counts length {n}, got {arr.shape[0]}")
    return arr


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


def _configure_gurobi_logging(model, *, verbose: bool, log_file: Optional[str]) -> None:
    """Configure Gurobi logging so LogFile is populated even when console is quiet."""
    has_log_file = bool(log_file)
    model.Params.OutputFlag = 1 if (verbose or has_log_file) else 0
    if has_log_file:
        parent = os.path.dirname(str(log_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        model.Params.LogFile = str(log_file)
        if not verbose:
            try:
                model.Params.LogToConsole = 0
            except Exception:
                pass


def _optimize_gurobi_model(
    model,
    *,
    callback=None,
    terminal_log_file: Optional[str] = None,
) -> bool:
    """
    Run `model.optimize(...)` while treating Ctrl+\ as "terminate current solve only".

    Returns:
        True iff the current solve was interrupted by the user.
    """
    interrupted = False
    notified = False

    def _mark_interrupted() -> None:
        nonlocal interrupted, notified
        interrupted = True
        if not notified:
            msg = "[MIP] Ctrl+\\ received: terminating current Gurobi solve and continuing to the next instance."
            print(msg)
            _append_log_line(terminal_log_file, msg)
            notified = True
        try:
            model.terminate()
        except Exception:
            pass

    previous_sigquit = None
    install_handler = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGQUIT")
    if install_handler:
        previous_sigquit = signal.getsignal(signal.SIGQUIT)

        def _sigquit_handler(signum, frame):
            del signum, frame
            _mark_interrupted()

        signal.signal(signal.SIGQUIT, _sigquit_handler)

    try:
        if callback is None:
            model.optimize()
        else:
            model.optimize(callback)
    finally:
        if install_handler and previous_sigquit is not None:
            signal.signal(signal.SIGQUIT, previous_sigquit)

    return interrupted


def _remaining_wallclock_time(wallclock_deadline_s: Optional[float]) -> Optional[float]:
    """Return remaining wall-clock budget in seconds, or None when unlimited."""
    if wallclock_deadline_s is None:
        return None
    return max(0.0, float(wallclock_deadline_s) - time.perf_counter())


def _mask_budget_value(
    edge_budget: Optional[int],
    node_budget: Optional[int],
) -> Optional[int]:
    """Return the active budget used by budget-aware strengthening rules."""
    if edge_budget is not None and node_budget is not None:
        raise ValueError("Use either edge_budget or node_budget, not both.")
    if edge_budget is not None:
        return int(edge_budget)
    if node_budget is not None:
        return int(node_budget)
    return None


def _mask_budget_edge_bonus(
    edge_budget: Optional[int],
    node_budget: Optional[int],
) -> int:
    """
    Return the edge-budget path bonus for rooted residual formulas.

    Seminar guidance:
    - edge budget: K - d + 1
    - node budget: K - d
    """
    _mask_budget_value(edge_budget, node_budget)
    return 1 if edge_budget is not None else 0


def _flow_cap_upper_bound(
    num_nodes: int,
    edge_budget: Optional[int],
    node_budget: Optional[int],
) -> float:
    """Upper bound on helper-flow variables under the active mask budget."""
    natural_cap = max(int(num_nodes) - 1, 0)
    if edge_budget is not None:
        return float(min(natural_cap, int(edge_budget)))
    if node_budget is not None:
        return float(min(natural_cap, max(int(node_budget) - 1, 0)))
    return float(natural_cap)


def _set_time_limit_from_deadline(
    model: object,
    *,
    wallclock_deadline_s: Optional[float],
    fallback_time_limit_s: Optional[float] = None,
    min_positive_s: float = 1e-3,
) -> Optional[float]:
    """
    Set model TimeLimit from remaining wall-clock budget (optionally capped by fallback).

    Returns the effective time limit that was set, or None if no limit is set.
    """
    remaining = _remaining_wallclock_time(wallclock_deadline_s)
    effective = None
    if remaining is not None and fallback_time_limit_s is not None:
        effective = min(float(remaining), float(fallback_time_limit_s))
    elif remaining is not None:
        effective = float(remaining)
    elif fallback_time_limit_s is not None:
        effective = float(fallback_time_limit_s)

    if effective is None:
        return None
    model.Params.TimeLimit = max(min_positive_s, effective)
    return float(model.Params.TimeLimit)


# =============================================================================
# Weight Classes
# =============================================================================

@dataclass(frozen=True)
class GINNodeGnnWeights:
    """
    Weights for 3-layer GIN NodeGNN (node classification).

    Each GNN layer is a GIN-style update with a 2-layer MLP:
    - s = (1+eps) * h + sum of neighbor h
    - u = s @ W1 + b1
    - a = ReLU(u)
    - h_new = a @ W2 + b2
    """
    eps1: float
    W1_1: np.ndarray  # (d0, M1)
    b1_1: np.ndarray  # (M1,)
    W1_2: np.ndarray  # (M1, H)
    b1_2: np.ndarray  # (H,)

    eps2: float
    W2_1: np.ndarray  # (H, M2)
    b2_1: np.ndarray  # (M2,)
    W2_2: np.ndarray  # (M2, H)
    b2_2: np.ndarray  # (H,)

    eps3: float
    W3_1: np.ndarray  # (H, M3)
    b3_1: np.ndarray  # (M3,)
    W3_2: np.ndarray  # (M3, H)
    b3_2: np.ndarray  # (H,)

    W_out: np.ndarray  # (C, 3H) for JK-concat or (C, H) for last-layer-only
    b_out: np.ndarray  # (C,)

    use_jk_concat: bool = True  # True = JK-concat, False = last-layer-only

    @property
    def hidden_dim(self) -> int:
        return int(self.W1_2.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.W1_1.shape[0])

    @property
    def mlp_hidden1(self) -> int:
        return int(self.W1_1.shape[1])

    @property
    def mlp_hidden2(self) -> int:
        return int(self.W2_1.shape[1])

    @property
    def mlp_hidden3(self) -> int:
        return int(self.W3_1.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.W_out.shape[0])


@dataclass(frozen=True)
class GINNodeGnnWeights4L:
    """Weights for 4-layer GIN NodeGNN (node classification)."""
    eps1: float
    W1_1: np.ndarray
    b1_1: np.ndarray
    W1_2: np.ndarray
    b1_2: np.ndarray

    eps2: float
    W2_1: np.ndarray
    b2_1: np.ndarray
    W2_2: np.ndarray
    b2_2: np.ndarray

    eps3: float
    W3_1: np.ndarray
    b3_1: np.ndarray
    W3_2: np.ndarray
    b3_2: np.ndarray

    eps4: float
    W4_1: np.ndarray
    b4_1: np.ndarray
    W4_2: np.ndarray
    b4_2: np.ndarray

    W_out: np.ndarray  # (C, 4H) for JK-concat or (C, H) for last-layer-only
    b_out: np.ndarray

    use_jk_concat: bool = True

    @property
    def hidden_dim(self) -> int:
        return int(self.W1_2.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.W1_1.shape[0])

    @property
    def mlp_hidden1(self) -> int:
        return int(self.W1_1.shape[1])

    @property
    def mlp_hidden2(self) -> int:
        return int(self.W2_1.shape[1])

    @property
    def mlp_hidden3(self) -> int:
        return int(self.W3_1.shape[1])

    @property
    def mlp_hidden4(self) -> int:
        return int(self.W4_1.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.W_out.shape[0])


@dataclass(frozen=True)
class GINNodeGnnWeights2L:
    """Weights for 2-layer GIN NodeGNN (node classification)."""
    eps1: float
    W1_1: np.ndarray
    b1_1: np.ndarray
    W1_2: np.ndarray
    b1_2: np.ndarray

    eps2: float
    W2_1: np.ndarray
    b2_1: np.ndarray
    W2_2: np.ndarray
    b2_2: np.ndarray

    W_out: np.ndarray  # (C, 2H) for JK-concat
    b_out: np.ndarray

    use_jk_concat: bool = True

    @property
    def hidden_dim(self) -> int:
        return int(self.W1_2.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.W1_1.shape[0])

    @property
    def mlp_hidden1(self) -> int:
        return int(self.W1_1.shape[1])

    @property
    def mlp_hidden2(self) -> int:
        return int(self.W2_1.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.W_out.shape[0])


@dataclass(frozen=True)
class GINGraphGnnWeights:
    """
    Weights for 3-layer GIN GraphGNN (graph classification).
    Uses global pooling on h3 only (not JK-concat).
    """
    eps1: float
    W1_1: np.ndarray
    b1_1: np.ndarray
    W1_2: np.ndarray
    b1_2: np.ndarray

    eps2: float
    W2_1: np.ndarray
    b2_1: np.ndarray
    W2_2: np.ndarray
    b2_2: np.ndarray

    eps3: float
    W3_1: np.ndarray
    b3_1: np.ndarray
    W3_2: np.ndarray
    b3_2: np.ndarray

    W_out: np.ndarray  # (C, H) - only h3 after pooling
    b_out: np.ndarray

    @property
    def hidden_dim(self) -> int:
        return int(self.W1_2.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.W1_1.shape[0])

    @property
    def mlp_hidden1(self) -> int:
        return int(self.W1_1.shape[1])

    @property
    def mlp_hidden2(self) -> int:
        return int(self.W2_1.shape[1])

    @property
    def mlp_hidden3(self) -> int:
        return int(self.W3_1.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.W_out.shape[0])


@dataclass(frozen=True)
class GINGraphGnnWeights4L:
    """Weights for 4-layer GIN GraphGNN (graph classification)."""
    eps1: float
    W1_1: np.ndarray
    b1_1: np.ndarray
    W1_2: np.ndarray
    b1_2: np.ndarray

    eps2: float
    W2_1: np.ndarray
    b2_1: np.ndarray
    W2_2: np.ndarray
    b2_2: np.ndarray

    eps3: float
    W3_1: np.ndarray
    b3_1: np.ndarray
    W3_2: np.ndarray
    b3_2: np.ndarray

    eps4: float
    W4_1: np.ndarray
    b4_1: np.ndarray
    W4_2: np.ndarray
    b4_2: np.ndarray

    W_out: np.ndarray  # (C, H) - last-layer pooled embedding only
    b_out: np.ndarray

    @property
    def hidden_dim(self) -> int:
        return int(self.W1_2.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.W1_1.shape[0])

    @property
    def mlp_hidden1(self) -> int:
        return int(self.W1_1.shape[1])

    @property
    def mlp_hidden2(self) -> int:
        return int(self.W2_1.shape[1])

    @property
    def mlp_hidden3(self) -> int:
        return int(self.W3_1.shape[1])

    @property
    def mlp_hidden4(self) -> int:
        return int(self.W4_1.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.W_out.shape[0])


@dataclass(frozen=True)
class GINGraphGnnWeights2L:
    """Weights for 2-layer GIN GraphGNN (graph classification)."""
    eps1: float
    W1_1: np.ndarray
    b1_1: np.ndarray
    W1_2: np.ndarray
    b1_2: np.ndarray

    eps2: float
    W2_1: np.ndarray
    b2_1: np.ndarray
    W2_2: np.ndarray
    b2_2: np.ndarray

    W_out: np.ndarray  # (C, H) - only h2 after pooling
    b_out: np.ndarray

    @property
    def hidden_dim(self) -> int:
        return int(self.W1_2.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.W1_1.shape[0])

    @property
    def mlp_hidden1(self) -> int:
        return int(self.W1_1.shape[1])

    @property
    def mlp_hidden2(self) -> int:
        return int(self.W2_1.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.W_out.shape[0])


# =============================================================================
# MILP Build / Solution Dataclasses
# =============================================================================


@dataclass(frozen=True)
class AndersonReluSpec:
    """Data needed to separate one Anderson ReLU cut family."""
    x_vars: Tuple[object, ...]
    y_var: object
    delta_var: object
    w: np.ndarray
    L: np.ndarray
    U: np.ndarray
    b: float
    tag: str = ""


@dataclass(frozen=True)
class MilpBuild:
    """MILP build result for node classification."""
    model: object
    z: Dict[Tuple[int, int], object]
    y: Optional[Dict[int, object]]
    edges: List[Tuple[int, int]]
    n: int
    v_star: int
    tilde_a: Adjacency
    c_star: int
    rho_var: Optional[object]
    use_maxflow_separation: bool
    connectivity_z: Optional[Dict[Tuple[int, int], object]] = None
    removal_z: Optional[Dict[Tuple[int, int], object]] = None
    bar_m: float = 0.0  # Original margin for verification
    # Nested/back cuts parameters
    use_nested_cuts: bool = False  # Enable nested cut separation (multiple cuts per terminal)
    use_back_cuts: bool = False    # Enable back cut separation (sink-side min-cut)
    max_cuts_per_round: int = 50  # Maximum cuts per separation round
    max_root_sep_rounds: int = 1   # Maximum separation rounds at root node (MIPNODE)
    # Anderson ReLU user cuts
    use_anderson_cuts: bool = False
    anderson_max_node_count: int = 0  # 0 = root only; single pass at first eligible MIPNODE
    anderson_max_cuts: int = 50  # Max cuts in that single Anderson pass
    anderson_relus: Tuple[AndersonReluSpec, ...] = ()
    log_file: Optional[str] = None  # Gurobi solver log file path
    terminal_log_file: Optional[str] = None  # Terminal print log path


@dataclass(frozen=True)
class GraphMilpBuild:
    """MILP build result for graph classification.
    """
    model: object
    z: Dict[Tuple[int, int], object]
    y: Optional[Dict[int, object]]
    r: Optional[Dict[int, object]]
    edges: List[Tuple[int, int]]
    n: int
    root_node: int
    tilde_a: Adjacency
    c_star: int
    connectivity: Literal["none", "flow", "cut"] = "none"
    rho_var: Optional[object] = None
    bar_m: float = 0.0  # Original margin for verification
    eps_empty: Optional[object] = None  # Empty explanation indicator (graph task + connectivity)
    connectivity_z: Optional[Dict[Tuple[int, int], object]] = None
    removal_z: Optional[Dict[Tuple[int, int], object]] = None
    use_anderson_cuts: bool = False
    anderson_max_node_count: int = 0
    anderson_max_cuts: int = 50
    anderson_relus: Tuple[AndersonReluSpec, ...] = ()
    terminal_log_file: Optional[str] = None


@dataclass(frozen=True)
class MilpSolution:
    """Solution from MILP optimization."""
    status: str
    objective_value: Optional[float]
    selected_edges: List[Tuple[int, int]]
    z_values: Dict[Tuple[int, int], float]
    rho_optimal: Optional[float] = None
    bar_m: Optional[float] = None  # Original margin for verification
    v_star: Optional[int] = None   # Target node for node classification
    c_star: Optional[int] = None   # Original predicted class
    final_mip_gap: Optional[float] = None  # Final reported MIP gap
    best_bound: Optional[float] = None  # Best dual bound at termination
    incumbent_obj: Optional[float] = None  # Best incumbent objective at termination
    model_size: Optional[float] = None  # Model size proxy (NumNZs)
    root_lp_value: Optional[float] = None  # Root LP objective/bound value at MIPNODE root
    root_lp_time: Optional[float] = None  # Runtime (sec) when root LP value captured
    callback_time_sec: Optional[float] = None  # Total callback wall time (sec)
    callback_calls: Optional[int] = None  # Total callback invocations
    cuts_mipnode: Optional[int] = None  # Total cuts added at MIPNODE (cbCut)
    cuts_mipsol_lazy: Optional[int] = None  # Total lazy cuts added at MIPSOL (cbLazy)
    anderson_cuts_added: Optional[int] = None  # Anderson user cuts added at MIPNODE
    connectivity_lazy_cuts_added: Optional[int] = None  # Connectivity cuts added via cbLazy
    bnb_node_count: Optional[float] = None  # Gurobi explored branch-and-bound nodes


# =============================================================================
# Forward Pass Functions
# =============================================================================

def _neighbor_sum_all(H: np.ndarray, adj: Adjacency) -> np.ndarray:
    """Sum neighbor representations over all edges (full graph)."""
    n, d = H.shape
    out = np.zeros((n, d), dtype=float)
    for i in range(n):
        for j in adj.neighbors[i]:
            out[i, :] += H[j, :]
    return out


def _neighbor_sum_selected(
    H: np.ndarray,
    n: int,
    selected_edges: List[Tuple[int, int]],
) -> np.ndarray:
    """Sum neighbor representations over selected edges only."""
    d = H.shape[1]
    out = np.zeros((n, d), dtype=float)
    for (i, j) in selected_edges:
        # Edge (i, j) means i aggregates from j and j aggregates from i
        out[i, :] += H[j, :]
        out[j, :] += H[i, :]
    return out


def forward_gin_nodegnn_with_mask(
    X: np.ndarray,
    selected_edges: List[Tuple[int, int]],
    weights: Union[GINNodeGnnWeights2L, GINNodeGnnWeights, GINNodeGnnWeights4L],
    v_star: int,
    self_loop_counts: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    """
    Forward pass with selected edges only, returns logits for v_star.

    Args:
        X: Node features (n, d0)
        selected_edges: List of selected undirected edges (canonical form: i < j)
        weights: Model weights
        v_star: Target node

    Returns:
        Z: Logits for v_star (C,)
    """
    n = X.shape[0]
    use_jk_concat = getattr(weights, 'use_jk_concat', True)
    self_coeff = _selected_self_loop_counts(n, self_loop_counts)

    alpha1 = 1.0 + float(weights.eps1)
    neigh1 = _neighbor_sum_selected(X, n, selected_edges)
    pre1 = (alpha1 + self_coeff).reshape(-1, 1) * X + neigh1
    u1 = pre1 @ weights.W1_1 + weights.b1_1.reshape(1, -1)
    a1 = np.maximum(u1, 0.0)
    h1 = a1 @ weights.W1_2 + weights.b1_2.reshape(1, -1)

    alpha2 = 1.0 + float(weights.eps2)
    neigh2 = _neighbor_sum_selected(h1, n, selected_edges)
    pre2 = (alpha2 + self_coeff).reshape(-1, 1) * h1 + neigh2
    u2 = pre2 @ weights.W2_1 + weights.b2_1.reshape(1, -1)
    a2 = np.maximum(u2, 0.0)
    h2 = a2 @ weights.W2_2 + weights.b2_2.reshape(1, -1)

    has_layer3 = hasattr(weights, "eps3")
    has_layer4 = hasattr(weights, "eps4")
    h3 = None
    h4 = None

    if has_layer3:
        alpha3 = 1.0 + float(weights.eps3)
        neigh3 = _neighbor_sum_selected(h2, n, selected_edges)
        pre3 = (alpha3 + self_coeff).reshape(-1, 1) * h2 + neigh3
        u3 = pre3 @ weights.W3_1 + weights.b3_1.reshape(1, -1)
        a3 = np.maximum(u3, 0.0)
        h3 = a3 @ weights.W3_2 + weights.b3_2.reshape(1, -1)

    if has_layer4:
        alpha4 = 1.0 + float(weights.eps4)
        neigh4 = _neighbor_sum_selected(h3, n, selected_edges)
        pre4 = (alpha4 + self_coeff).reshape(-1, 1) * h3 + neigh4
        u4 = pre4 @ weights.W4_1 + weights.b4_1.reshape(1, -1)
        a4 = np.maximum(u4, 0.0)
        h4 = a4 @ weights.W4_2 + weights.b4_2.reshape(1, -1)

    if use_jk_concat:
        if has_layer4:
            v_embed = np.concatenate([h1[v_star], h2[v_star], h3[v_star], h4[v_star]], axis=0)
        elif has_layer3:
            v_embed = np.concatenate([h1[v_star], h2[v_star], h3[v_star]], axis=0)
        else:
            v_embed = np.concatenate([h1[v_star], h2[v_star]], axis=0)
    else:
        if has_layer4:
            v_embed = h4[v_star]
        elif has_layer3:
            v_embed = h3[v_star]
        else:
            v_embed = h2[v_star]

    Z = v_embed @ weights.W_out.T + weights.b_out
    return Z


def forward_gin_graphgnn_with_mask(
    X: np.ndarray,
    selected_edges: List[Tuple[int, int]],
    weights: Union[GINGraphGnnWeights2L, GINGraphGnnWeights, GINGraphGnnWeights4L],
    readout: str = "sum",
    self_loop_counts: Optional[Union[np.ndarray, Sequence[float]]] = None,
) -> np.ndarray:
    """
    Forward pass for graph classification with selected edges only.

    Args:
        X: Node features (n, d0)
        selected_edges: List of selected undirected edges
        weights: Model weights
        readout: "sum" or "mean"

    Returns:
        Z: Logits (C,)
    """
    n = X.shape[0]
    self_coeff = _selected_self_loop_counts(n, self_loop_counts)

    alpha1 = 1.0 + float(weights.eps1)
    neigh1 = _neighbor_sum_selected(X, n, selected_edges)
    pre1 = (alpha1 + self_coeff).reshape(-1, 1) * X + neigh1
    u1 = pre1 @ weights.W1_1 + weights.b1_1.reshape(1, -1)
    a1 = np.maximum(u1, 0.0)
    h1 = a1 @ weights.W1_2 + weights.b1_2.reshape(1, -1)

    alpha2 = 1.0 + float(weights.eps2)
    neigh2 = _neighbor_sum_selected(h1, n, selected_edges)
    pre2 = (alpha2 + self_coeff).reshape(-1, 1) * h1 + neigh2
    u2 = pre2 @ weights.W2_1 + weights.b2_1.reshape(1, -1)
    a2 = np.maximum(u2, 0.0)
    h2 = a2 @ weights.W2_2 + weights.b2_2.reshape(1, -1)

    has_layer3 = hasattr(weights, "eps3")
    has_layer4 = hasattr(weights, "eps4")
    h_last = h2

    if has_layer3:
        alpha3 = 1.0 + float(weights.eps3)
        neigh3 = _neighbor_sum_selected(h2, n, selected_edges)
        pre3 = (alpha3 + self_coeff).reshape(-1, 1) * h2 + neigh3
        u3 = pre3 @ weights.W3_1 + weights.b3_1.reshape(1, -1)
        a3 = np.maximum(u3, 0.0)
        h3 = a3 @ weights.W3_2 + weights.b3_2.reshape(1, -1)
        h_last = h3

    if has_layer4:
        alpha4 = 1.0 + float(weights.eps4)
        neigh4 = _neighbor_sum_selected(h3, n, selected_edges)
        pre4 = (alpha4 + self_coeff).reshape(-1, 1) * h3 + neigh4
        u4 = pre4 @ weights.W4_1 + weights.b4_1.reshape(1, -1)
        a4 = np.maximum(u4, 0.0)
        h4 = a4 @ weights.W4_2 + weights.b4_2.reshape(1, -1)
        h_last = h4

    if readout == "sum":
        pooled = h_last.sum(axis=0)
    else:
        pooled = h_last.mean(axis=0)

    Z = pooled @ weights.W_out.T + weights.b_out
    return Z


def verify_node_solution(
    X: np.ndarray,
    selected_edges: List[Tuple[int, int]],
    weights: Union[GINNodeGnnWeights2L, GINNodeGnnWeights, GINNodeGnnWeights4L],
    v_star: int,
    c_star: int,
    bar_m: float,
    rho: float = 0.0,
    self_loop_counts: Optional[Union[np.ndarray, Sequence[float]]] = None,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Verify that the MILP solution satisfies the margin constraint.

    This computes the actual margin from forward pass and compares with MILP's rho.

    Args:
        X: Node features
        selected_edges: Selected edges from MILP solution
        weights: Model weights
        v_star: Target node
        c_star: Original predicted class
        bar_m: Original margin (from full graph)
        rho: MILP's rho_optimal (margin ratio to verify)
        tol: Tolerance for comparison
        verbose: Print debug info

    Returns:
        (is_valid, message): Tuple of validity flag and message
    """
    # Forward pass with selected edges
    Z = forward_gin_nodegnn_with_mask(X, selected_edges, weights, v_star, self_loop_counts=self_loop_counts)

    # Check prediction
    pred_class = int(np.argmax(Z))

    # Compute actual margin
    actual_margin = float(Z[c_star] - np.max(np.delete(Z, c_star)))

    # Compute rho_actual = actual_margin / bar_m
    if abs(bar_m) > 1e-10:
        rho_actual = actual_margin / bar_m
    else:
        rho_actual = 0.0 if abs(actual_margin) < 1e-10 else float('inf')

    if verbose:
        print(f"[Verify] Logits: {Z}")
        print(f"[Verify] Predicted class: {pred_class}, Original: {c_star}")
        print(f"[Verify] Actual margin: {actual_margin:.6f}, bar_m: {bar_m:.6f}")
        print(f"[Verify] rho_actual: {rho_actual:.6f}, rho_optimal: {rho:.6f}")

    # Check if rho_actual matches rho_optimal (within tolerance)
    rho_diff = abs(rho_actual - rho)
    if rho_diff > tol:
        return False, f"rho mismatch: actual={rho_actual:.6f}, optimal={rho:.6f}, diff={rho_diff:.6f}"

    return True, f"rho matches: actual={rho_actual:.6f}, optimal={rho:.6f}"


def verify_graph_solution(
    X: np.ndarray,
    selected_edges: List[Tuple[int, int]],
    weights: Union[GINGraphGnnWeights2L, GINGraphGnnWeights, GINGraphGnnWeights4L],
    c_star: int,
    bar_m: float,
    rho: float = 0.0,
    readout: str = "sum",
    self_loop_counts: Optional[Union[np.ndarray, Sequence[float]]] = None,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Verify that the MILP solution's rho matches actual forward pass.

    Computes rho_actual = actual_margin / bar_m and compares with rho_optimal.
    """
    Z = forward_gin_graphgnn_with_mask(
        X,
        selected_edges,
        weights,
        readout,
        self_loop_counts=self_loop_counts,
    )

    # Compute actual margin from forward pass
    actual_margin = float(Z[c_star] - np.max(np.delete(Z, c_star)))

    # Compute rho_actual = actual_margin / bar_m
    rho_actual = actual_margin / bar_m

    if verbose:
        print(f"[Verify] Logits: {Z}")
        print(f"[Verify] c_star={c_star}, bar_m={bar_m:.6f}")
        print(f"[Verify] actual_margin={actual_margin:.6f}")
        print(f"[Verify] rho_actual={rho_actual:.6f}, rho_optimal={rho:.6f}")

    # Compare rho values
    rho_diff = abs(rho_actual - rho)

    if rho_diff > tol:
        return False, f"rho mismatch: actual={rho_actual:.6f}, optimal={rho:.6f}, diff={rho_diff:.6f}"

    return True, f"rho matches: actual={rho_actual:.6f}, optimal={rho:.6f}, diff={rho_diff:.6f}"


def forward_gin_nodegnn_full(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINNodeGnnWeights,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass with all edges selected. Returns (h1, h2, h3)."""
    alpha1 = _node_self_coefficients(adj, 1.0 + float(weights.eps1))
    neigh1 = _neighbor_sum_all(X, adj)
    pre1 = alpha1.reshape(-1, 1) * X + neigh1
    u1 = pre1 @ weights.W1_1 + weights.b1_1.reshape(1, -1)
    a1 = np.maximum(u1, 0.0)
    h1 = a1 @ weights.W1_2 + weights.b1_2.reshape(1, -1)

    alpha2 = _node_self_coefficients(adj, 1.0 + float(weights.eps2))
    neigh2 = _neighbor_sum_all(h1, adj)
    pre2 = alpha2.reshape(-1, 1) * h1 + neigh2
    u2 = pre2 @ weights.W2_1 + weights.b2_1.reshape(1, -1)
    a2 = np.maximum(u2, 0.0)
    h2 = a2 @ weights.W2_2 + weights.b2_2.reshape(1, -1)

    alpha3 = _node_self_coefficients(adj, 1.0 + float(weights.eps3))
    neigh3 = _neighbor_sum_all(h2, adj)
    pre3 = alpha3.reshape(-1, 1) * h2 + neigh3
    u3 = pre3 @ weights.W3_1 + weights.b3_1.reshape(1, -1)
    a3 = np.maximum(u3, 0.0)
    h3 = a3 @ weights.W3_2 + weights.b3_2.reshape(1, -1)

    return h1, h2, h3


def forward_gin_nodegnn_full_4l(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINNodeGnnWeights4L,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass for 4-layer NodeGNN. Returns (h1, h2, h3, h4)."""
    h1, h2, h3 = forward_gin_nodegnn_full(X, adj, weights)

    alpha4 = _node_self_coefficients(adj, 1.0 + float(weights.eps4))
    neigh4 = _neighbor_sum_all(h3, adj)
    pre4 = alpha4.reshape(-1, 1) * h3 + neigh4
    u4 = pre4 @ weights.W4_1 + weights.b4_1.reshape(1, -1)
    a4 = np.maximum(u4, 0.0)
    h4 = a4 @ weights.W4_2 + weights.b4_2.reshape(1, -1)
    return h1, h2, h3, h4


def forward_gin_nodegnn_full_2l(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINNodeGnnWeights2L,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass for 2-layer NodeGNN. Returns (h1, h2)."""
    alpha1 = _node_self_coefficients(adj, 1.0 + float(weights.eps1))
    neigh1 = _neighbor_sum_all(X, adj)
    pre1 = alpha1.reshape(-1, 1) * X + neigh1
    u1 = pre1 @ weights.W1_1 + weights.b1_1.reshape(1, -1)
    a1 = np.maximum(u1, 0.0)
    h1 = a1 @ weights.W1_2 + weights.b1_2.reshape(1, -1)

    alpha2 = _node_self_coefficients(adj, 1.0 + float(weights.eps2))
    neigh2 = _neighbor_sum_all(h1, adj)
    pre2 = alpha2.reshape(-1, 1) * h1 + neigh2
    u2 = pre2 @ weights.W2_1 + weights.b2_1.reshape(1, -1)
    a2 = np.maximum(u2, 0.0)
    h2 = a2 @ weights.W2_2 + weights.b2_2.reshape(1, -1)

    return h1, h2


def forward_gin_graphgnn_full(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINGraphGnnWeights,
    readout: str = "sum",
) -> np.ndarray:
    """Forward pass for graph classification. Returns logits Z."""
    h1, h2, h3 = forward_gin_nodegnn_full(X, adj, weights)
    if readout == "sum":
        pool3 = h3.sum(axis=0)
    else:  # mean
        pool3 = h3.mean(axis=0)
    Z = pool3 @ weights.W_out.T + weights.b_out
    return Z


def forward_gin_graphgnn_full_4l(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINGraphGnnWeights4L,
    readout: str = "sum",
) -> np.ndarray:
    """Forward pass for 4-layer graph classification."""
    _, _, _, h4 = forward_gin_nodegnn_full_4l(X, adj, weights)
    if readout == "sum":
        pool4 = h4.sum(axis=0)
    else:
        pool4 = h4.mean(axis=0)
    Z = pool4 @ weights.W_out.T + weights.b_out
    return Z


def forward_gin_graphgnn_full_2l(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINGraphGnnWeights2L,
    readout: str = "sum",
) -> np.ndarray:
    """Forward pass for 2-layer graph classification."""
    h1, h2 = forward_gin_nodegnn_full_2l(X, adj, weights)
    if readout == "sum":
        pool2 = h2.sum(axis=0)
    else:
        pool2 = h2.mean(axis=0)
    Z = pool2 @ weights.W_out.T + weights.b_out
    return Z


# =============================================================================
# Bound Computation (Interval Propagation)
# =============================================================================

def _linear_bounds(
    L_x: np.ndarray, U_x: np.ndarray, W: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interval bounds for Y = X @ W + b.
    L_x/U_x: (n, d_in), W: (d_in, d_out), b: (d_out,).
    """
    w_pos = np.maximum(W, 0.0)
    w_neg = np.minimum(W, 0.0)
    U_y = L_x @ w_neg + U_x @ w_pos + b.reshape(1, -1)
    L_y = L_x @ w_pos + U_x @ w_neg + b.reshape(1, -1)
    return L_y, U_y


def _pre_bounds_layer1(
    adj: Adjacency, X: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds for pre = alpha * X + sum_{j in N(i)} z_ij * X_j.
    z_ij in {0,1}, so neighbor contribution is in [min(0,X_j), max(0,X_j)].

    NOTE: This function is kept for compatibility but _u1_bounds_direct is preferred.
    """
    n, _ = X.shape
    base = _node_self_coefficients(adj, alpha).reshape(-1, 1) * X
    L_pre = base.copy().astype(float)
    U_pre = base.copy().astype(float)
    for i in range(n):
        for j in adj.neighbors[i]:
            L_pre[i, :] += np.minimum(0.0, X[j, :])
            U_pre[i, :] += np.maximum(0.0, X[j, :])
    return L_pre, U_pre


def _u1_bounds_direct(
    adj: Adjacency,
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    alpha: float,
    kappa: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct bounds for u^(1) as in paper Eq (bound_u1_L), (bound_u1_U).

    u_{i,m}^{(1)} = α(XW)_{i,m} + b_m + Σ_{j∈N(i)} z_{ij}(XW)_{j,m}

    Let d_{j,m} = (XW)_{j,m}, c_{i,m} = α(XW)_{i,m} + b_m.
    Since z_{ij} ∈ {0,1}:
        L_{i,m} = c_{i,m} + Σ_{j∈N(i)} min{0, d_{j,m}}
        U_{i,m} = c_{i,m} + Σ_{j∈N(i)} max{0, d_{j,m}}

    This is tighter than computing bounds on pre first, then applying linear transform.

    Top-κ incident cap tightening (when kappa is provided):
        Instead of summing over ALL neighbors, only sum the κ_i most extreme:
        - For L: sum the κ_i most negative values (bottom-k)
        - For U: sum the κ_i most positive values (top-k)

    Args:
        adj: Adjacency structure
        X: Node features (n, d0)
        W: Weight matrix W1_1 (d0, M1)
        b: Bias vector b1_1 (M1,)
        alpha: 1 + eps
        kappa: Per-node incident caps (n,). If None, sum over all neighbors.

    Returns:
        L_u1, U_u1: Bounds on u^(1) of shape (n, M1)
    """
    n = X.shape[0]
    M1 = W.shape[1]
    XW = X @ W  # (n, M1) - precompute once
    c = _node_self_coefficients(adj, alpha).reshape(-1, 1) * XW + b.reshape(1, -1)  # (n, M1)

    L_u = c.copy().astype(float)
    U_u = c.copy().astype(float)

    for i in range(n):
        neighbors = adj.neighbors[i]

        if kappa is None:
            # Default: sum over all neighbors
            for j in neighbors:
                L_u[i, :] += np.minimum(0.0, XW[j, :])
                U_u[i, :] += np.maximum(0.0, XW[j, :])
        else:
            # Top-κ incident cap tightening
            k_i = int(kappa[i])

            if k_i == 0:
                # No edges can be selected incident to i
                continue

            if k_i >= len(neighbors):
                # No tightening: sum over all neighbors
                for j in neighbors:
                    L_u[i, :] += np.minimum(0.0, XW[j, :])
                    U_u[i, :] += np.maximum(0.0, XW[j, :])
            else:
                # Top-κ tightening: select k_i most extreme contributors per dimension
                for m in range(M1):
                    # For upper bound: sort max(0, XW[j,m]) descending, take top-k
                    neighbor_pos = [max(0.0, XW[j, m]) for j in neighbors]
                    neighbor_pos_sorted = sorted(neighbor_pos, reverse=True)
                    U_u[i, m] += sum(neighbor_pos_sorted[:k_i])

                    # For lower bound: sort min(0, XW[j,m]) ascending, take top-k negative
                    neighbor_neg = [min(0.0, XW[j, m]) for j in neighbors]
                    neighbor_neg_sorted = sorted(neighbor_neg)  # ascending
                    L_u[i, m] += sum(neighbor_neg_sorted[:k_i])

    return L_u, U_u


def _pre_bounds_selectable(
    adj: Adjacency, L_h: np.ndarray, U_h: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds for pre = alpha * h_i + sum_{j in N(i)} z_ij * h_j.
    h_j in [L_h, U_h], z_ij in {0,1}.
    """
    n, d = L_h.shape
    alpha_eff = _node_self_coefficients(adj, alpha).reshape(-1, 1)
    self_L = np.minimum(alpha_eff * L_h, alpha_eff * U_h)
    self_U = np.maximum(alpha_eff * L_h, alpha_eff * U_h)
    L_pre = self_L.copy().astype(float)
    U_pre = self_U.copy().astype(float)
    for i in range(n):
        for j in adj.neighbors[i]:
            L_pre[i, :] += np.minimum(0.0, L_h[j, :])
            U_pre[i, :] += np.maximum(0.0, U_h[j, :])
    return L_pre, U_pre


CapMode = Literal["global", "distance"]


def _resolve_obbt_layers(
    max_obbt_u_layers: Optional[int],
    *,
    available_layers: Sequence[int],
) -> List[int]:
    """Validate and normalize requested OBBT layers from max layer index."""
    available = sorted({int(l) for l in available_layers})
    if not available:
        return []
    if max_obbt_u_layers is None:
        return available
    max_layer = int(max_obbt_u_layers)
    if max_layer < 1:
        raise ValueError("max_obbt_u_layers must be >= 1 when provided.")
    if max_layer > max(available):
        raise ValueError(
            f"Invalid max_obbt_u_layers={max_layer}; available layers are {available}."
        )
    return [l for l in available if l <= max_layer]


def _collect_u_vars_by_layer(
    model: object,
    target_layers: Sequence[int],
) -> Dict[int, List[object]]:
    """Collect u-layer variables by name pattern u{layer}_*."""
    out: Dict[int, List[object]] = {int(layer): [] for layer in target_layers}
    prefixes = {int(layer): f"u{int(layer)}_" for layer in target_layers}
    for var in model.getVars():
        name = str(var.VarName)
        for layer, prefix in prefixes.items():
            if name.startswith(prefix):
                out[layer].append(var)
                break
    return out


def _apply_obbt_u_bounds(
    model: object,
    *,
    use_obbt_u: bool,
    max_obbt_u_layers: Optional[int],
    available_layers: Sequence[int],
    connectivity: ConnectivityMode = "none",
    wallclock_deadline_s: Optional[float] = None,
    verbose: bool = False,
    terminal_log_file: Optional[str] = None,
    log_prefix: str = "[MIP]",
) -> None:
    """
    Tighten bounds of u-variables using LP-based OBBT.

    For each selected u variable, solve LP twice:
    1) minimize u
    2) maximize u
    and apply tightened [lb, ub] back to the original MILP variables.
    """
    if not use_obbt_u:
        return

    if connectivity != "none":
        _print_and_log(
            f"{log_prefix} OBBT(u) skipped: connectivity='{connectivity}' (currently only connectivity='none' is supported).",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        return

    target_layers = _resolve_obbt_layers(
        max_obbt_u_layers,
        available_layers=available_layers,
    )
    if not target_layers:
        _print_and_log(
            f"{log_prefix} OBBT(u) skipped: no target layers.",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        return

    _, GRB = _require_gurobi()
    model.update()
    u_by_layer = _collect_u_vars_by_layer(model, target_layers)
    total_u_vars = sum(len(vs) for vs in u_by_layer.values())
    if total_u_vars == 0:
        _print_and_log(
            f"{log_prefix} OBBT(u) skipped: no u variables found for layers {target_layers}.",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        return

    _print_and_log(
        f"{log_prefix} OBBT(u) start: layers={target_layers}, vars={total_u_vars}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    obbt_total_t0 = time.perf_counter()

    lp = model.relax()
    lp.Params.OutputFlag = 0
    # For max_sufficiency models rho is typically free below; give a very loose finite LB
    # to avoid LP unboundedness in OBBT subproblems.
    rho_lp = lp.getVarByName("rho")
    if rho_lp is not None:
        try:
            if not np.isfinite(float(rho_lp.LB)):
                rho_lp.LB = -1e9
        except Exception:
            rho_lp.LB = -1e9
    lp.update()

    tol = 1e-6
    tightened_count = 0
    lp_fail_count = 0
    numeric_skip_count = 0
    deadline_reached = False

    for layer in target_layers:
        for orig_u_var in u_by_layer.get(layer, []):
            remaining = _remaining_wallclock_time(wallclock_deadline_s)
            if remaining is not None and remaining <= 0.0:
                deadline_reached = True
                break
            if remaining is not None:
                lp.Params.TimeLimit = max(1e-3, float(remaining))

            lp_u_var = lp.getVarByName(orig_u_var.VarName)
            if lp_u_var is None:
                lp_fail_count += 2
                continue

            old_lb = float(orig_u_var.LB)
            old_ub = float(orig_u_var.UB)
            new_lb = old_lb
            new_ub = old_ub

            lp.setObjective(lp_u_var, GRB.MINIMIZE)
            if _optimize_gurobi_model(lp, terminal_log_file=terminal_log_file):
                raise CurrentGurobiSolveInterrupted("User interrupted current Gurobi solve during OBBT(u).")
            if lp.Status == GRB.OPTIMAL and np.isfinite(lp.ObjVal):
                new_lb = max(new_lb, float(lp.ObjVal))
            else:
                lp_fail_count += 1

            lp.setObjective(lp_u_var, GRB.MAXIMIZE)
            if _optimize_gurobi_model(lp, terminal_log_file=terminal_log_file):
                raise CurrentGurobiSolveInterrupted("User interrupted current Gurobi solve during OBBT(u).")
            if lp.Status == GRB.OPTIMAL and np.isfinite(lp.ObjVal):
                new_ub = min(new_ub, float(lp.ObjVal))
            else:
                lp_fail_count += 1

            if new_lb > new_ub + tol:
                numeric_skip_count += 1
                continue

            tightened = False
            if new_lb > old_lb + tol:
                orig_u_var.LB = new_lb
                tightened = True
            if new_ub < old_ub - tol:
                orig_u_var.UB = new_ub
                tightened = True
            if tightened:
                tightened_count += 1
        if deadline_reached:
            break

    model.update()
    _print_and_log(
        f"{log_prefix} OBBT(u) done: tightened={tightened_count}/{total_u_vars}, lp_failures={lp_fail_count}, numeric_skips={numeric_skip_count}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    obbt_total_time = time.perf_counter() - obbt_total_t0
    _print_and_log(
        f"{log_prefix} OBBT(u) total_time_sec={obbt_total_time:.4f}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    if deadline_reached:
        _print_and_log(
            f"{log_prefix} OBBT(u) stopped early due to wall-clock time limit.",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )


def _obbt_tighten_u_layer_node_lp(
    *,
    X: np.ndarray,
    adj: Adjacency,
    edges: Sequence[Tuple[int, int]],
    target_layer: int,
    alpha_list: Sequence[float],
    W_in_list: Sequence[np.ndarray],
    b_in_list: Sequence[np.ndarray],
    W_out_list: Sequence[np.ndarray],
    b_out_list: Sequence[np.ndarray],
    L_pre_list: Sequence[np.ndarray],
    U_pre_list: Sequence[np.ndarray],
    L_u_list: Sequence[np.ndarray],
    U_u_list: Sequence[np.ndarray],
    L_a_list: Sequence[np.ndarray],
    U_a_list: Sequence[np.ndarray],
    L_h_list: Sequence[np.ndarray],
    U_h_list: Sequence[np.ndarray],
    edge_budget: Optional[int],
    min_edges: Optional[int],
    wallclock_deadline_s: Optional[float] = None,
    verbose: bool = False,
    terminal_log_file: Optional[str] = None,
    log_prefix: str = "[MIP]",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LP-OBBT for node-model u^(target_layer) with a prefix relaxation up to that layer.

    Prefix logic:
    - For layers < target_layer: include full linear + ReLU LP relaxation + h constraints.
    - For layer == target_layer: include only p/u definition (no ReLU needed).
    """
    gp, GRB = _require_gurobi()

    if target_layer < 1:
        raise ValueError("target_layer must be >= 1")
    if not (
        len(alpha_list) >= target_layer
        and len(W_in_list) >= target_layer
        and len(b_in_list) >= target_layer
        and len(L_pre_list) >= target_layer
        and len(U_pre_list) >= target_layer
        and len(L_u_list) >= target_layer
        and len(U_u_list) >= target_layer
    ):
        raise ValueError("Insufficient layer data for target-layer OBBT.")

    n, d0 = X.shape
    h_dim = int(W_out_list[0].shape[1]) if W_out_list else int(W_in_list[1].shape[0]) if target_layer > 1 else d0

    model = gp.Model(f"OBBT_u_layer{target_layer}")
    model.Params.OutputFlag = 0

    z = {(i, j): model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"z_{i}_{j}") for (i, j) in edges}

    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    h_vars_by_layer: Dict[int, Dict[Tuple[int, int], object]] = {}
    u_target: Dict[Tuple[int, int], object] = {}

    for layer_idx in range(1, target_layer + 1):
        alpha = float(alpha_list[layer_idx - 1])
        W_in = W_in_list[layer_idx - 1]
        b_in = b_in_list[layer_idx - 1]

        L_pre = L_pre_list[layer_idx - 1]
        U_pre = U_pre_list[layer_idx - 1]
        L_u = L_u_list[layer_idx - 1]
        U_u = U_u_list[layer_idx - 1]

        m_dim = int(W_in.shape[1])

        if layer_idx == 1:
            p = {
                (i, f): model.addVar(lb=float(L_pre[i, f]), ub=float(U_pre[i, f]), name=f"p1_{i}_{f}")
                for i in range(n)
                for f in range(d0)
            }
            u = {
                (i, m): model.addVar(lb=float(L_u[i, m]), ub=float(U_u[i, m]), name=f"u1_{i}_{m}")
                for i in range(n)
                for m in range(m_dim)
            }
            model.update()

            for i in range(n):
                for f in range(d0):
                    expr = gp.LinExpr()
                    expr.addConstant(float((alpha + adj.self_loop_counts[i]) * X[i, f]))
                    for j in adj.neighbors[i]:
                        expr.addTerms(float(X[j, f]), z_ij(i, j))
                    model.addConstr(p[(i, f)] == expr, name=f"p1def_{i}_{f}")

            for i in range(n):
                for m in range(m_dim):
                    expr = gp.LinExpr()
                    expr.addConstant(float(b_in[m]))
                    for f in range(d0):
                        wval = float(W_in[f, m])
                        if wval != 0.0:
                            expr.addTerms(wval, p[(i, f)])
                    model.addConstr(u[(i, m)] == expr, name=f"u1def_{i}_{m}")

        else:
            prev_h = h_vars_by_layer[layer_idx - 1]
            L_prev_h = L_h_list[layer_idx - 2]
            U_prev_h = U_h_list[layer_idx - 2]

            t = {}
            for (u_idx, v_idx) in edges:
                for k in range(h_dim):
                    lb_v = float(min(0.0, L_prev_h[v_idx, k]))
                    ub_v = float(max(0.0, U_prev_h[v_idx, k]))
                    t[(u_idx, v_idx, k)] = model.addVar(lb=lb_v, ub=ub_v, name=f"t{layer_idx}_{u_idx}_{v_idx}_{k}")
                    lb_u = float(min(0.0, L_prev_h[u_idx, k]))
                    ub_u = float(max(0.0, U_prev_h[u_idx, k]))
                    t[(v_idx, u_idx, k)] = model.addVar(lb=lb_u, ub=ub_u, name=f"t{layer_idx}_{v_idx}_{u_idx}_{k}")

            p = {
                (i, k): model.addVar(lb=float(L_pre[i, k]), ub=float(U_pre[i, k]), name=f"p{layer_idx}_{i}_{k}")
                for i in range(n)
                for k in range(h_dim)
            }
            u = {
                (i, m): model.addVar(lb=float(L_u[i, m]), ub=float(U_u[i, m]), name=f"u{layer_idx}_{i}_{m}")
                for i in range(n)
                for m in range(m_dim)
            }
            model.update()

            for (u_idx, v_idx) in edges:
                zuv = z[(u_idx, v_idx)]
                for k in range(h_dim):
                    hv = prev_h[(v_idx, k)]
                    tv = t[(u_idx, v_idx, k)]
                    Lb, Ub = float(L_prev_h[v_idx, k]), float(U_prev_h[v_idx, k])
                    model.addConstr(tv <= hv - Lb * (1 - zuv))
                    model.addConstr(tv >= hv - Ub * (1 - zuv))
                    model.addConstr(tv <= Ub * zuv)
                    model.addConstr(tv >= Lb * zuv)

                    hu = prev_h[(u_idx, k)]
                    tu = t[(v_idx, u_idx, k)]
                    Lb, Ub = float(L_prev_h[u_idx, k]), float(U_prev_h[u_idx, k])
                    model.addConstr(tu <= hu - Lb * (1 - zuv))
                    model.addConstr(tu >= hu - Ub * (1 - zuv))
                    model.addConstr(tu <= Ub * zuv)
                    model.addConstr(tu >= Lb * zuv)

            for i in range(n):
                for k in range(h_dim):
                    expr = gp.LinExpr()
                    expr.addTerms(float(alpha + adj.self_loop_counts[i]), prev_h[(i, k)])
                    for j in adj.neighbors[i]:
                        expr.addTerms(1.0, t[(i, j, k)])
                    model.addConstr(p[(i, k)] == expr, name=f"p{layer_idx}def_{i}_{k}")

            for i in range(n):
                for m in range(m_dim):
                    expr = gp.LinExpr()
                    expr.addConstant(float(b_in[m]))
                    for k in range(h_dim):
                        wval = float(W_in[k, m])
                        if wval != 0.0:
                            expr.addTerms(wval, p[(i, k)])
                    model.addConstr(u[(i, m)] == expr, name=f"u{layer_idx}def_{i}_{m}")

        if layer_idx == target_layer:
            u_target = u
            continue

        W_out = W_out_list[layer_idx - 1]
        b_out = b_out_list[layer_idx - 1]
        L_u_cur = L_u_list[layer_idx - 1]
        U_u_cur = U_u_list[layer_idx - 1]
        L_a_cur = L_a_list[layer_idx - 1]
        U_a_cur = U_a_list[layer_idx - 1]
        L_h_cur = L_h_list[layer_idx - 1]
        U_h_cur = U_h_list[layer_idx - 1]

        a = {
            (i, m): model.addVar(lb=float(L_a_cur[i, m]), ub=float(U_a_cur[i, m]), name=f"a{layer_idx}_{i}_{m}")
            for i in range(n)
            for m in range(m_dim)
        }
        h_cur = {
            (i, k): model.addVar(lb=float(L_h_cur[i, k]), ub=float(U_h_cur[i, k]), name=f"h{layer_idx}_{i}_{k}")
            for i in range(n)
            for k in range(h_dim)
        }
        d = {}
        model.update()

        for i in range(n):
            for m in range(m_dim):
                L_val, U_val = float(L_u_cur[i, m]), float(U_u_cur[i, m])
                uvar = u[(i, m)]
                avar = a[(i, m)]
                if U_val <= 0.0:
                    model.addConstr(avar == 0.0)
                elif L_val >= 0.0:
                    model.addConstr(avar == uvar)
                else:
                    d[(i, m)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"d{layer_idx}_{i}_{m}")
                    model.addConstr(avar >= uvar)
                    model.addConstr(avar >= 0.0)
                    model.addConstr(avar <= uvar - L_val * (1 - d[(i, m)]))
                    model.addConstr(avar <= U_val * d[(i, m)])

        model.update()

        for i in range(n):
            for k in range(h_dim):
                expr = gp.LinExpr()
                expr.addConstant(float(b_out[k]))
                for m in range(m_dim):
                    wval = float(W_out[m, k])
                    if wval != 0.0:
                        expr.addTerms(wval, a[(i, m)])
                model.addConstr(h_cur[(i, k)] == expr, name=f"h{layer_idx}def_{i}_{k}")

        h_vars_by_layer[layer_idx] = h_cur

    model.update()
    old_L = L_u_list[target_layer - 1]
    old_U = U_u_list[target_layer - 1]
    new_L = old_L.copy()
    new_U = old_U.copy()

    tol = 1e-6
    tightened_count = 0
    fail_count = 0
    deadline_reached = False

    for i in range(n):
        for m in range(old_L.shape[1]):
            remaining = _remaining_wallclock_time(wallclock_deadline_s)
            if remaining is not None and remaining <= 0.0:
                deadline_reached = True
                break
            if remaining is not None:
                model.Params.TimeLimit = max(1e-3, float(remaining))

            uvar = u_target[(i, m)]
            cur_lb = float(new_L[i, m])
            cur_ub = float(new_U[i, m])

            model.setObjective(uvar, GRB.MINIMIZE)
            if _optimize_gurobi_model(model, terminal_log_file=terminal_log_file):
                raise CurrentGurobiSolveInterrupted("User interrupted current Gurobi solve during OBBT(u).")
            if model.Status == GRB.OPTIMAL and np.isfinite(model.ObjVal):
                cur_lb = max(cur_lb, float(model.ObjVal))
            else:
                fail_count += 1

            model.setObjective(uvar, GRB.MAXIMIZE)
            if _optimize_gurobi_model(model, terminal_log_file=terminal_log_file):
                raise CurrentGurobiSolveInterrupted("User interrupted current Gurobi solve during OBBT(u).")
            if model.Status == GRB.OPTIMAL and np.isfinite(model.ObjVal):
                cur_ub = min(cur_ub, float(model.ObjVal))
            else:
                fail_count += 1

            if cur_lb > cur_ub + tol:
                continue

            if cur_lb > old_L[i, m] + tol or cur_ub < old_U[i, m] - tol:
                tightened_count += 1

            new_L[i, m] = cur_lb
            new_U[i, m] = cur_ub
        if deadline_reached:
            break

    _print_and_log(
        f"{log_prefix} OBBT(u^{target_layer}) done: tightened={tightened_count}/{old_L.size}, lp_failures={fail_count}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    if deadline_reached:
        _print_and_log(
            f"{log_prefix} OBBT(u^{target_layer}) stopped early due to wall-clock time limit.",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    return new_L, new_U


def _compute_node_bounds_with_sequential_obbt(
    *,
    X: np.ndarray,
    adj: Adjacency,
    edges: Sequence[Tuple[int, int]],
    alpha_list: Sequence[float],
    W_in_list: Sequence[np.ndarray],
    b_in_list: Sequence[np.ndarray],
    W_out_list: Sequence[np.ndarray],
    b_out_list: Sequence[np.ndarray],
    use_incident_cap_tightening: bool,
    kappa: Optional[np.ndarray],
    use_obbt_u: bool,
    max_obbt_u_layers: Optional[int],
    connectivity: ConnectivityMode,
    edge_budget: Optional[int],
    min_edges: Optional[int],
    wallclock_deadline_s: Optional[float],
    verbose: bool,
    terminal_log_file: Optional[str],
    log_prefix: str,
) -> Tuple[
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
    List[np.ndarray], List[np.ndarray],
]:
    """
    Compute node-layer bounds with optional sequential OBBT.

    Flow:
    1) Build baseline IBP/incident-cap bounds for all layers.
    2) For selected OBBT layers (deep -> shallow), tighten u^(l) by LP.
    3) After each u-tightening, refresh downstream bounds in the order
       u -> a -> p(next layer) -> ... while keeping previously tightened
       u-bounds as hard intersections.
    """
    num_layers = len(alpha_list)
    if not (
        len(W_in_list) == num_layers
        and len(b_in_list) == num_layers
        and len(W_out_list) == num_layers
        and len(b_out_list) == num_layers
    ):
        raise ValueError("Layer parameter lengths must match alpha_list.")

    L_pre_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    U_pre_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    L_u_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    U_u_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    L_a_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    U_a_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    L_h_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]
    U_h_list: List[np.ndarray] = [np.empty((0, 0), dtype=float) for _ in range(num_layers)]

    def _compute_pre_u(layer_idx: int) -> None:
        idx = layer_idx - 1
        alpha = float(alpha_list[idx])
        if layer_idx == 1:
            L_pre, U_pre = _pre_bounds_layer1(adj, X, alpha)
            L_u, U_u = _u1_bounds_direct(
                adj,
                X,
                W_in_list[idx],
                b_in_list[idx],
                alpha,
                kappa if use_incident_cap_tightening else None,
            )
        else:
            if use_incident_cap_tightening and kappa is not None:
                L_pre, U_pre = _pre_bounds_selectable_topk(
                    adj,
                    L_h_list[idx - 1],
                    U_h_list[idx - 1],
                    alpha,
                    kappa,
                )
            else:
                L_pre, U_pre = _pre_bounds_selectable(
                    adj,
                    L_h_list[idx - 1],
                    U_h_list[idx - 1],
                    alpha,
                )
            L_u, U_u = _linear_bounds(L_pre, U_pre, W_in_list[idx], b_in_list[idx])

        L_pre_list[idx], U_pre_list[idx] = L_pre, U_pre
        L_u_list[idx], U_u_list[idx] = L_u, U_u

    def _refresh_from_layer(
        start_layer: int,
        fixed_u_bounds: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for layer_idx in range(start_layer, num_layers + 1):
            idx = layer_idx - 1
            if layer_idx == start_layer:
                if L_u_list[idx].size == 0 or U_u_list[idx].size == 0:
                    _compute_pre_u(layer_idx)
            else:
                _compute_pre_u(layer_idx)

            if layer_idx in fixed_u_bounds:
                fixed_L, fixed_U = fixed_u_bounds[layer_idx]
                cur_L = np.maximum(L_u_list[idx], fixed_L)
                cur_U = np.minimum(U_u_list[idx], fixed_U)
                invalid = cur_L > cur_U
                if np.any(invalid):
                    cur_L[invalid] = fixed_L[invalid]
                    cur_U[invalid] = fixed_U[invalid]
                L_u_list[idx], U_u_list[idx] = cur_L, cur_U
                fixed_u_bounds[layer_idx] = (cur_L.copy(), cur_U.copy())

            L_a = np.maximum(L_u_list[idx], 0.0)
            U_a = np.maximum(U_u_list[idx], 0.0)
            L_h, U_h = _linear_bounds(L_a, U_a, W_out_list[idx], b_out_list[idx])
            L_a_list[idx], U_a_list[idx] = L_a, U_a
            L_h_list[idx], U_h_list[idx] = L_h, U_h

    # Initial pass: baseline bounds with currently selected tightening options.
    _refresh_from_layer(1, fixed_u_bounds={})

    if use_obbt_u:
        if connectivity == "none":
            obbt_layers = _resolve_obbt_layers(
                max_obbt_u_layers,
                available_layers=tuple(range(1, num_layers + 1)),
            )
            obbt_schedule = sorted(obbt_layers, reverse=True)
            _print_and_log(
                f"{log_prefix} OBBT(u) enabled on layers={obbt_layers}, schedule={obbt_schedule}",
                verbose=verbose,
                terminal_log_file=terminal_log_file,
            )
        else:
            obbt_layers = []
            obbt_schedule = []
            _print_and_log(
                f"{log_prefix} OBBT(u) skipped: connectivity='{connectivity}' (currently only connectivity='none' is supported).",
                verbose=verbose,
                terminal_log_file=terminal_log_file,
            )
    else:
        obbt_layers = []
        obbt_schedule = []

    fixed_u_bounds: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    obbt_total_t0 = time.perf_counter() if obbt_schedule else None
    for target_layer in obbt_schedule:
        remaining = _remaining_wallclock_time(wallclock_deadline_s)
        if remaining is not None and remaining <= 0.0:
            _print_and_log(
                f"{log_prefix} OBBT(u) stopped before layer {target_layer} due to wall-clock time limit.",
                verbose=verbose,
                terminal_log_file=terminal_log_file,
            )
            break
        tightened_L, tightened_U = _obbt_tighten_u_layer_node_lp(
            X=X,
            adj=adj,
            edges=edges,
            target_layer=target_layer,
            alpha_list=alpha_list,
            W_in_list=W_in_list,
            b_in_list=b_in_list,
            W_out_list=W_out_list,
            b_out_list=b_out_list,
            L_pre_list=L_pre_list,
            U_pre_list=U_pre_list,
            L_u_list=L_u_list,
            U_u_list=U_u_list,
            L_a_list=L_a_list[: max(0, target_layer - 1)],
            U_a_list=U_a_list[: max(0, target_layer - 1)],
            L_h_list=L_h_list[: max(0, target_layer - 1)],
            U_h_list=U_h_list[: max(0, target_layer - 1)],
            edge_budget=edge_budget,
            min_edges=min_edges,
            wallclock_deadline_s=wallclock_deadline_s,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
            log_prefix=log_prefix,
        )
        idx = target_layer - 1
        L_u_list[idx], U_u_list[idx] = tightened_L, tightened_U
        fixed_u_bounds[target_layer] = (tightened_L.copy(), tightened_U.copy())

        # Sequential propagation: u^(l) -> a^(l) -> p^(l+1) -> ...
        _refresh_from_layer(target_layer, fixed_u_bounds=fixed_u_bounds)

    if obbt_total_t0 is not None:
        obbt_total_time = time.perf_counter() - obbt_total_t0
        _print_and_log(
            f"{log_prefix} OBBT(u) total_time_sec={obbt_total_time:.4f}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    return (
        L_pre_list, U_pre_list,
        L_u_list, U_u_list,
        L_a_list, U_a_list,
        L_h_list, U_h_list,
    )


def compute_bfs_distances(adj: Adjacency, source: int) -> np.ndarray:
    """
    Compute BFS distances from source node to all other nodes.
    Returns array of distances (inf for unreachable nodes).
    """
    n = adj.n
    dist = np.full(n, np.inf)
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.neighbors[u]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def compute_incident_caps(
    adj: Adjacency,
    edge_budget: Optional[int],
    cap_mode: CapMode = "global",
    v_star: Optional[int] = None,
    forced_edges: Optional[List[Tuple[int, int]]] = None,
    node_budget: Optional[int] = None,
) -> np.ndarray:
    """
    Compute per-node incident edge caps κ_i.

    Args:
        adj: Adjacency structure
        edge_budget: Global edge budget K
        cap_mode: "global" or "distance"
        v_star: Target node (required for "distance" mode)
        forced_edges: List of forced edges (v*, j) to consider for additional tightening
        node_budget: Global node budget K for connectivity-constrained subgraph mode

    Returns:
        kappa: array of shape (n,) with κ_i for each node
    """
    n = adj.n

    K = _mask_budget_value(edge_budget, node_budget)
    if K is None:
        # No budget constraint: κ_i = |N(i)|
        return np.array([len(adj.neighbors[i]) for i in range(n)])
    edge_bonus = _mask_budget_edge_bonus(edge_budget, node_budget)

    if cap_mode == "global":
        # κ_i = min(|N(i)|, K)
        kappa = np.array([min(len(adj.neighbors[i]), K) for i in range(n)])
    elif cap_mode == "distance":
        if v_star is None:
            raise ValueError("v_star is required for distance cap mode")
        # Edge budgets use K - dist(i) + 1 along rooted paths; node budgets use K - dist(i).
        dist_from_vstar = compute_bfs_distances(adj, v_star)
        kappa = np.zeros(n, dtype=int)
        for i in range(n):
            if np.isinf(dist_from_vstar[i]):
                kappa[i] = 0
            else:
                d = int(dist_from_vstar[i])
                if d == 0:
                    residual = max(0, K - 1 + edge_bonus)
                else:
                    residual = max(0, K - d + edge_bonus)
                kappa[i] = min(len(adj.neighbors[i]), residual)

        # NOTE: forced_edges tightening was removed as it could cause over-tight bounds.
        # The heuristic "edge (v*, j) doesn't help reach i → decrement κ_i" is unsafe
        # because it compounds across multiple forced edges and can exclude valid solutions.
    else:
        raise ValueError(f"Unknown cap_mode: {cap_mode}")

    return kappa


def _quantized_feature_key(x: np.ndarray, tol: float) -> Tuple[int, ...]:
    """Convert a feature vector to a stable hashable key."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    return tuple(np.round(np.asarray(x, dtype=float) / tol).astype(np.int64).tolist())


def _component_internal_edge_count(
    component_nodes: Set[int],
    subgraph_neighbors: Dict[int, Set[int]],
) -> int:
    """Count undirected edges inside a node set."""
    edge_twice = 0
    for i in component_nodes:
        edge_twice += sum(1 for j in subgraph_neighbors[i] if j in component_nodes)
    return edge_twice // 2


def _rooted_tree_signature(
    node: int,
    parent: int,
    subgraph_neighbors: Dict[int, Set[int]],
    X: np.ndarray,
    self_loop_counts: np.ndarray,
    tol: float,
) -> Tuple[Tuple[int, ...], int, Tuple[object, ...]]:
    """Canonical rooted-tree signature using features, self-loops, and child multiset."""
    child_sigs = [
        _rooted_tree_signature(child, node, subgraph_neighbors, X, self_loop_counts, tol)
        for child in subgraph_neighbors[node]
        if child != parent
    ]
    child_sigs.sort()
    return (
        _quantized_feature_key(X[node], tol),
        int(round(float(self_loop_counts[node]))),
        tuple(child_sigs),
    )


def find_symmetric_branch_groups(
    adj: Adjacency,
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    tol: float = 1e-8,
) -> Dict[int, List[List[int]]]:
    """
    Find symmetric sibling branch groups for symmetry breaking.

    For each parent node u, consider each incident direction u-v. Remove u from the
    current subgraph and look at the connected component reachable from v. If that
    component is:

    1. attached to u through exactly one neighbor v, and
    2. a tree,

    then it defines a rooted branch hanging off u. Two such branches are grouped
    together when their rooted-tree signatures match exactly, including node
    features, fixed self-loop multiplicities, and child structure. This strictly
    generalizes the old leaf-only rule to depth-2, depth-3, ... symmetric trees.

    Args:
        adj: Adjacency structure of the current subgraph
        X: Node features (n, d0)
        undirected_edges: List of undirected edges in the subgraph
        tol: Tolerance for feature comparison

    Returns:
        Dictionary mapping parent node u -> list of symmetric sibling groups.
        Each group is a sorted list of branch-root node indices adjacent to u.
        Only groups with size >= 2 are included.
    """
    n = adj.n

    subgraph_neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, j in undirected_edges:
        if i == j:
            continue
        subgraph_neighbors[i].add(j)
        subgraph_neighbors[j].add(i)

    symmetric_groups: Dict[int, List[List[int]]] = {}

    for u in range(n):
        if len(subgraph_neighbors[u]) < 2:
            continue

        parent_neighbors = set(subgraph_neighbors[u])
        visited: Set[int] = set()
        signature_groups: Dict[Tuple[object, ...], List[int]] = {}

        for v in sorted(parent_neighbors):
            if v in visited:
                continue

            component_nodes: Set[int] = set()
            queue = deque([v])
            visited.add(v)
            component_nodes.add(v)

            while queue:
                cur = queue.popleft()
                for nxt in subgraph_neighbors[cur]:
                    if nxt == u or nxt in visited:
                        continue
                    visited.add(nxt)
                    component_nodes.add(nxt)
                    queue.append(nxt)

            branch_roots = sorted(component_nodes & parent_neighbors)
            if len(branch_roots) != 1 or branch_roots[0] != v:
                continue

            if _component_internal_edge_count(component_nodes, subgraph_neighbors) != len(component_nodes) - 1:
                continue

            signature = _rooted_tree_signature(
                node=v,
                parent=u,
                subgraph_neighbors=subgraph_neighbors,
                X=X,
                self_loop_counts=adj.self_loop_counts,
                tol=tol,
            )
            signature_groups.setdefault(signature, []).append(v)

        groups = [sorted(group) for group in signature_groups.values() if len(group) >= 2]
        if groups:
            symmetric_groups[u] = groups

    return symmetric_groups


def add_symmetry_breaking_constraints(
    model,
    z: Dict[Tuple[int, int], "gp.Var"],
    symmetric_groups: Dict[int, List[List[int]]],
    verbose: bool = False,
    terminal_log_file: Optional[str] = None,
) -> int:
    """
    Add symmetry breaking constraints for symmetric sibling branches.

    For each parent node u with symmetric branch roots {v1, v2, ...},
    add lexicographic ordering constraints on the root-adjacent edges:
        z_{u,v1} >= z_{u,v2} >= z_{u,v3} >= ...

    This eliminates k! equivalent solutions for each group of k symmetric branches.

    Args:
        model: Gurobi model
        z: Edge selection variables (keyed by canonical (i,j) with i < j)
        symmetric_groups: Output from find_symmetric_branch_groups
        verbose: Print debug info
        terminal_log_file: Path to mirror verbose terminal messages

    Returns:
        Number of symmetry breaking constraints added
    """
    gp, GRB = _require_gurobi()

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z.get((a, b), None)

    num_constraints = 0
    total_permutations_eliminated = 1

    for u, groups in symmetric_groups.items():
        for group in groups:
            # Add ordering: z_{u,v[0]} >= z_{u,v[1]} >= ... >= z_{u,v[k-1]}
            for idx in range(len(group) - 1):
                v1, v2 = group[idx], group[idx + 1]
                z1 = z_ij(u, v1)
                z2 = z_ij(u, v2)
                if z1 is not None and z2 is not None:
                    model.addConstr(z1 >= z2, name=f"sym_break_{u}_{v1}_{v2}")
                    num_constraints += 1

            # Count eliminated permutations
            k = len(group)
            factorial = 1
            for i in range(2, k + 1):
                factorial *= i
            total_permutations_eliminated *= factorial

    if num_constraints > 0:
        _print_and_log(
            f"[MIP] Symmetry breaking edges: {num_constraints} constraints, "
            f"eliminated {total_permutations_eliminated}x equivalent solutions",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    return num_constraints


def add_symmetry_breaking_constraints_nodes(
    model,
    y: Dict[int, "gp.Var"],
    symmetric_groups: Dict[int, List[List[int]]],
    verbose: bool = False,
    terminal_log_file: Optional[str] = None,
) -> int:
    """
    Add symmetry breaking constraints for symmetric branch roots on y variables.

    For each parent node u with symmetric branch roots {v1, v2, ...},
    add lexicographic ordering constraints on those root nodes:
        y_{v1} >= y_{v2} >= y_{v3} >= ...

    This complements edge-level symmetry breaking by also ordering node selections.

    Args:
        model: Gurobi model
        y: Node selection variables (keyed by node index)
        symmetric_groups: Output from find_symmetric_branch_groups
        verbose: Print debug info
        terminal_log_file: Path to mirror verbose terminal messages

    Returns:
        Number of symmetry breaking constraints added
    """
    gp, GRB = _require_gurobi()

    num_constraints = 0
    total_permutations_eliminated = 1

    for u, groups in symmetric_groups.items():
        for group in groups:
            # Add ordering: y_{v[0]} >= y_{v[1]} >= ... >= y_{v[k-1]}
            for idx in range(len(group) - 1):
                v1, v2 = group[idx], group[idx + 1]
                if v1 in y and v2 in y:
                    model.addConstr(y[v1] >= y[v2], name=f"sym_break_node_{v1}_{v2}")
                    num_constraints += 1

            # Count eliminated permutations
            k = len(group)
            factorial = 1
            for i in range(2, k + 1):
                factorial *= i
            total_permutations_eliminated *= factorial

    if num_constraints > 0:
        _print_and_log(
            f"[MIP] Symmetry breaking nodes: {num_constraints} constraints, "
            f"eliminated {total_permutations_eliminated}x equivalent solutions",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    return num_constraints


# ========== Bridge-based Constraints ==========

def find_bridges_tarjan(
    n: int,
    adj: List[List[int]],
) -> List[Tuple[int, int]]:
    """
    Find all bridges (cut-edges) in an undirected graph using Tarjan's DFS algorithm.

    A bridge is an edge whose removal disconnects the graph.
    Time complexity: O(V + E)

    Args:
        n: Number of nodes
        adj: Adjacency list (undirected)

    Returns:
        List of bridges as (a, b) tuples with a < b
    """
    disc = [-1] * n  # Discovery time
    low = [-1] * n   # Lowest reachable discovery time
    parent = [-1] * n
    bridges: List[Tuple[int, int]] = []
    timer = [0]  # Mutable counter

    def dfs(u: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1

        for v in adj[u]:
            if disc[v] == -1:  # Not visited
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # If lowest reachable from v is below v's discovery time,
                # then (u, v) is a bridge
                if low[v] > disc[u]:
                    a, b = (u, v) if u < v else (v, u)
                    bridges.append((a, b))
            elif v != parent[u]:  # Back edge (not to parent)
                low[u] = min(low[u], disc[v])

    # Handle disconnected components
    for i in range(n):
        if disc[i] == -1:
            dfs(i)

    return sorted(bridges)


def find_2ec_components(
    n: int,
    adj: List[List[int]],
    bridges: List[Tuple[int, int]],
) -> List[int]:
    """
    Find 2-edge-connected components by removing bridges.

    Two nodes are in the same 2EC component if they remain connected
    after removing all bridges.

    Args:
        n: Number of nodes
        adj: Adjacency list (undirected)
        bridges: List of bridge edges

    Returns:
        Component ID for each node (0-indexed)
    """
    bridge_set = set(bridges) | {(b, a) for (a, b) in bridges}

    comp_id = [-1] * n
    current_comp = 0

    for start in range(n):
        if comp_id[start] != -1:
            continue

        # BFS without using bridge edges
        queue = deque([start])
        comp_id[start] = current_comp

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if comp_id[v] == -1:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) not in bridge_set:
                        comp_id[v] = current_comp
                        queue.append(v)

        current_comp += 1

    return comp_id


def build_bridge_tree(
    n: int,
    bridges: List[Tuple[int, int]],
    comp_id: List[int],
) -> Tuple[int, Dict[int, List[Tuple[int, Tuple[int, int]]]]]:
    """
    Build the bridge-tree (block-cut tree of 2EC components).

    Each node in the tree represents a 2EC component.
    Each edge in the tree corresponds to a bridge in the original graph.

    Args:
        n: Number of nodes in original graph
        bridges: List of bridge edges
        comp_id: Component ID for each node

    Returns:
        Tuple of (num_components, tree_adj)
        tree_adj: Dict mapping component ID to list of (neighbor_comp, bridge_edge)
    """
    num_comps = max(comp_id) + 1 if comp_id else 0

    # Build tree adjacency with bridge info
    tree_adj: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {c: [] for c in range(num_comps)}

    for (a, b) in bridges:
        ca, cb = comp_id[a], comp_id[b]
        if ca != cb:  # Should always be true for bridges
            tree_adj[ca].append((cb, (a, b)))
            tree_adj[cb].append((ca, (a, b)))

    return num_comps, tree_adj


def compute_bridge_subtree_nodes(
    n: int,
    adj: List[List[int]],
    v_star: int,
    bridges: List[Tuple[int, int]],
    comp_id: List[int],
) -> Dict[Tuple[int, int], List[int]]:
    """
    For each bridge, compute the set of nodes in the subtree opposite to v*.

    When the bridge-tree is rooted at v*'s component, each bridge defines
    a subtree. S_e contains all original nodes in that subtree.

    Args:
        n: Number of nodes
        adj: Adjacency list (undirected)
        v_star: Target node
        bridges: List of bridge edges
        comp_id: Component ID for each node

    Returns:
        S_map: Dict mapping bridge (a, b) to sorted list of nodes in S_e
    """
    if not bridges:
        return {}

    num_comps, tree_adj = build_bridge_tree(n, bridges, comp_id)
    root_comp = comp_id[v_star]

    # Find nodes in each component
    comp_nodes: Dict[int, List[int]] = {c: [] for c in range(num_comps)}
    for i in range(n):
        comp_nodes[comp_id[i]].append(i)

    # Root the tree at root_comp and compute subtree sizes via DFS
    # For each bridge, we need the subtree on the side away from root

    parent_comp = [-1] * num_comps
    parent_bridge: Dict[int, Optional[Tuple[int, int]]] = {c: None for c in range(num_comps)}
    children: Dict[int, List[int]] = {c: [] for c in range(num_comps)}

    # BFS to establish parent-child relationships
    visited = [False] * num_comps
    queue = deque([root_comp])
    visited[root_comp] = True

    while queue:
        c = queue.popleft()
        for (nc, bridge) in tree_adj[c]:
            if not visited[nc]:
                visited[nc] = True
                parent_comp[nc] = c
                parent_bridge[nc] = bridge
                children[c].append(nc)
                queue.append(nc)

    # Compute subtree node count via post-order DFS
    subtree_nodes: Dict[int, List[int]] = {}  # comp -> all nodes in subtree rooted at comp

    def get_subtree_nodes(c: int) -> List[int]:
        """Get all original nodes in subtree rooted at component c."""
        if c in subtree_nodes:
            return subtree_nodes[c]

        nodes = list(comp_nodes[c])
        for child in children[c]:
            nodes.extend(get_subtree_nodes(child))

        subtree_nodes[c] = nodes
        return nodes

    # Compute S_e for each bridge
    S_map: Dict[Tuple[int, int], List[int]] = {}

    for c in range(num_comps):
        if parent_bridge[c] is not None:
            bridge = parent_bridge[c]
            # Nodes in subtree rooted at c (away from root_comp)
            S_e = get_subtree_nodes(c)
            # Normalize bridge to (a, b) with a < b
            a, b = bridge if bridge[0] < bridge[1] else (bridge[1], bridge[0])
            S_map[(a, b)] = sorted(S_e)

    return S_map


def compute_bridge_constraints(
    n: int,
    adj: List[List[int]],
    v_star: int,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], List[int]], List[Tuple[int, Tuple[int, int]]]]:
    """
    Compute bridge-based implications for node classification.

    For each bridge e and each node k in S_e (subtree away from v*):
        y_k <= z_e  (if node k is selected, bridge e must be selected)

    Args:
        n: Number of nodes
        adj: Adjacency list (undirected, 0-indexed)
        v_star: Target node index

    Returns:
        bridges: List of bridge edges (a, b) with a < b
        S_map: Dict mapping bridge to list of nodes in opposite subtree
        implications: List of (k, e) tuples for constraint "y[k] <= z[e]"
    """
    bridges = find_bridges_tarjan(n, adj)

    if not bridges:
        return [], {}, []

    comp_id = find_2ec_components(n, adj, bridges)
    S_map = compute_bridge_subtree_nodes(n, adj, v_star, bridges, comp_id)

    # Generate implications
    implications: List[Tuple[int, Tuple[int, int]]] = []
    for bridge, nodes in S_map.items():
        for k in nodes:
            implications.append((k, bridge))

    return bridges, S_map, implications


def compute_bridge_parent_child_relations(
    n: int,
    adj: List[List[int]],
    v_star: int,
    bridges: List[Tuple[int, int]],
    comp_id: List[int],
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Compute parent-child bridge relations on the rooted bridge-tree.

    Returns pairs (parent_bridge, child_bridge) where selecting child bridge
    should imply selecting parent bridge.
    """
    if not bridges:
        return []

    num_comps, tree_adj = build_bridge_tree(n, bridges, comp_id)
    root_comp = comp_id[v_star]

    parent_comp = [-1] * num_comps
    parent_bridge: Dict[int, Optional[Tuple[int, int]]] = {c: None for c in range(num_comps)}
    children: Dict[int, List[int]] = {c: [] for c in range(num_comps)}

    visited = [False] * num_comps
    queue = deque([root_comp])
    visited[root_comp] = True

    while queue:
        c = queue.popleft()
        for nc, bridge in tree_adj[c]:
            if not visited[nc]:
                visited[nc] = True
                parent_comp[nc] = c
                parent_bridge[nc] = bridge
                children[c].append(nc)
                queue.append(nc)

    relations: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for c in range(num_comps):
        p_bridge = parent_bridge[c]
        if p_bridge is None:
            continue
        pa, pb = p_bridge if p_bridge[0] < p_bridge[1] else (p_bridge[1], p_bridge[0])
        for ch in children[c]:
            c_bridge = parent_bridge[ch]
            if c_bridge is None:
                continue
            ca, cb = c_bridge if c_bridge[0] < c_bridge[1] else (c_bridge[1], c_bridge[0])
            relations.append(((pa, pb), (ca, cb)))

    # Deduplicate while preserving deterministic order
    seen: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    unique_relations: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for rel in relations:
        if rel not in seen:
            seen.add(rel)
            unique_relations.append(rel)
    return unique_relations


def add_hierarchical_bridge_constraints(
    model,
    z: Dict[Tuple[int, int], "gp.Var"],
    n: int,
    adj: List[List[int]],
    v_star: int,
    verbose: bool = False,
    terminal_log_file: Optional[str] = None,
) -> int:
    """
    Bridge-tree hierarchical constraints.

    For each parent-child bridge pair (e_parent, e_child) on the rooted bridge-tree:
        z_{e_child} <= z_{e_parent}
    """
    bridges = find_bridges_tarjan(n, adj)
    if not bridges:
        return 0

    comp_id = find_2ec_components(n, adj, bridges)
    relations = compute_bridge_parent_child_relations(n, adj, v_star, bridges, comp_id)
    if not relations:
        return 0

    num_constraints = 0
    for parent_bridge, child_bridge in relations:
        if parent_bridge not in z or child_bridge not in z:
            continue
        model.addConstr(
            z[child_bridge] <= z[parent_bridge],
            name=f"bridge_hier_{child_bridge[0]}_{child_bridge[1]}_to_{parent_bridge[0]}_{parent_bridge[1]}",
        )
        num_constraints += 1

    if num_constraints > 0:
        _print_and_log(
            f"[MIP] Hierarchical bridge constraints: {num_constraints} constraints added",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    return num_constraints


def add_bridge_constraints(
    model,
    y: Dict[int, "gp.Var"],
    z: Dict[Tuple[int, int], "gp.Var"],
    n: int,
    adj: List[List[int]],
    v_star: int,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    use_individual: bool = True,
    use_aggregated: bool = False,
    verbose: bool = False,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
) -> int:
    """
    Add bridge-based constraints to the MILP model.

    For each bridge e = (u, v) where u is on v*'s side and v is on the opposite side S:

    1. Individual constraints (if use_individual=True): y_k <= z_e for each k ∈ S
    2. Aggregated constraint (if use_aggregated=True):
       Σ_{k ∈ S} y_k <= max{0, min(|S|, K - d + 1[edge budget])} * z_e

    where:
    - S is the set of nodes in the subtree opposite to v*
    - K is the active mask budget
    - d is the distance from v* to the bridge (number of edges)

    Args:
        model: Gurobi model
        y: Node selection variables
        z: Edge selection variables
        n: Number of nodes
        adj: Adjacency list (undirected)
        v_star: Target node
        edge_budget: Edge budget K
        node_budget: Node budget K for connectivity-constrained subgraph mode
        use_individual: Add individual constraints y_k <= z_e
        use_aggregated: Add aggregated budget-aware constraint
        verbose: Print debug info
        log_file: Deprecated (kept for backward compatibility; not used)
        terminal_log_file: Path to mirror verbose terminal messages

    Returns:
        Number of constraints added
    """
    gp, GRB = _require_gurobi()

    def log_msg(msg: str):
        """Print to terminal and append only to terminal log."""
        _print_and_log(msg, verbose=verbose, terminal_log_file=terminal_log_file)

    bridges, S_map, _ = compute_bridge_constraints(n, adj, v_star)

    if not bridges:
        return 0

    # Compute BFS distances from v* to all nodes
    dist = [-1] * n
    dist[v_star] = 0
    queue = deque([v_star])
    while queue:
        u = queue.popleft()
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = dist[u] + 1
                queue.append(w)

    log_msg(f"[MIP] Bridge analysis: {len(bridges)} bridges found")

    num_constraints = 0
    mask_budget = _mask_budget_value(edge_budget, node_budget)
    edge_bonus = _mask_budget_edge_bonus(edge_budget, node_budget)

    for bridge, S_nodes in S_map.items():
        if bridge not in z:
            continue

        # Get valid y variables for nodes in S
        valid_nodes = [k for k in S_nodes if k in y]
        if not valid_nodes:
            continue

        S_size = len(valid_nodes)

        # Compute distance to bridge: min distance to either endpoint
        # The bridge connects u (v* side) to v (S side)
        # d = dist to the closer endpoint of the bridge
        d = min(dist[bridge[0]], dist[bridge[1]]) + 1  # +1 because we need to cross the bridge

        # Compute RHS coefficient using the active budget semantics.
        if mask_budget is not None:
            remaining_budget = mask_budget - d + edge_bonus
            coeff = max(0, min(S_size, remaining_budget))
        else:
            coeff = S_size  # Without budget info, use |S|

        # Add individual and/or aggregated constraints based on flags
        # 1. Individual constraints: y_k <= z_e for each k in S
        #    - Enforces that selecting any node in S requires the bridge
        # 2. Aggregated constraint: Σ_{k ∈ S} y_k <= coeff * z_e
        #    - Strengthens LP relaxation with budget-aware coefficient

        if coeff == 0:
            # If coefficient is 0, no nodes in S can be selected (unreachable given budget)
            for k in valid_nodes:
                model.addConstr(y[k] == 0, name=f"bridge_unreachable_{k}")
                num_constraints += 1
        else:
            # Individual constraints: y_k <= z_e
            if use_individual:
                for k in valid_nodes:
                    model.addConstr(y[k] <= z[bridge],
                                   name=f"bridge_ind_{bridge[0]}_{bridge[1]}_{k}")
                    num_constraints += 1

            # Aggregated constraint: Σ_{k ∈ S} y_k <= coeff * z_e
            # Only add if use_aggregated=True AND coeff < S_size
            if use_aggregated and coeff < S_size:
                lhs = gp.quicksum(y[k] for k in valid_nodes)
                model.addConstr(lhs <= coeff * z[bridge],
                               name=f"bridge_agg_{bridge[0]}_{bridge[1]}")
                num_constraints += 1

        if verbose:
            ind_added = "yes" if (use_individual and coeff > 0) else "no"
            agg_added = "yes" if (use_aggregated and coeff > 0 and coeff < S_size) else "no"
            log_msg(f"      Bridge {bridge}: |S|={S_size}, d={d}, coeff={coeff}, ind={ind_added}, agg={agg_added}")

    if num_constraints > 0:
        log_msg(f"[MIP] Bridge constraints: {num_constraints} constraints added")

    return num_constraints


def _test_bridge_constraints():
    """
    Unit tests for bridge detection and S_e computation.

    Test cases:
    1. Chain: 0 - 1 - 2 - 3 (all edges are bridges)
    2. Triangle + tail: 0-1-2-0 with 2-3 tail (only 2-3 is bridge)
    3. Cycle: 0-1-2-3-0 (no bridges)
    4. Disconnected: 0-1, 2-3 (edges 0-1, 2-3 are bridges within components)
    """
    print("Running bridge constraint unit tests...")

    # Test 1: Chain 0 - 1 - 2 - 3
    # v* = 0: bridges = [(0,1), (1,2), (2,3)]
    # S_{0,1} = [1,2,3], S_{1,2} = [2,3], S_{2,3} = [3]
    adj1 = [[1], [0, 2], [1, 3], [2]]
    bridges1, S_map1, impl1 = compute_bridge_constraints(4, adj1, v_star=0)
    assert set(bridges1) == {(0, 1), (1, 2), (2, 3)}, f"Chain bridges: {bridges1}"
    assert set(S_map1[(0, 1)]) == {1, 2, 3}, f"Chain S_(0,1): {S_map1.get((0,1))}"
    assert set(S_map1[(1, 2)]) == {2, 3}, f"Chain S_(1,2): {S_map1.get((1,2))}"
    assert set(S_map1[(2, 3)]) == {3}, f"Chain S_(2,3): {S_map1.get((2,3))}"
    print("  ✓ Test 1 (Chain): PASSED")

    # Test 2: Triangle + tail: 0-1-2-0, 2-3
    # v* = 0: only (2,3) is a bridge
    # S_{2,3} = [3]
    adj2 = [[1, 2], [0, 2], [0, 1, 3], [2]]
    bridges2, S_map2, impl2 = compute_bridge_constraints(4, adj2, v_star=0)
    assert bridges2 == [(2, 3)], f"Triangle+tail bridges: {bridges2}"
    assert set(S_map2[(2, 3)]) == {3}, f"Triangle+tail S_(2,3): {S_map2.get((2,3))}"
    print("  ✓ Test 2 (Triangle + tail): PASSED")

    # Test 3: Cycle 0-1-2-3-0 (no bridges)
    adj3 = [[1, 3], [0, 2], [1, 3], [2, 0]]
    bridges3, S_map3, impl3 = compute_bridge_constraints(4, adj3, v_star=0)
    assert bridges3 == [], f"Cycle bridges: {bridges3}"
    assert S_map3 == {}, f"Cycle S_map: {S_map3}"
    assert impl3 == [], f"Cycle implications: {impl3}"
    print("  ✓ Test 3 (Cycle): PASSED")

    # Test 4: Two components 0-1, 2-3 (disconnected)
    # Each component has one bridge, but since disconnected,
    # we only care about v*'s component
    adj4 = [[1], [0], [3], [2]]
    bridges4, S_map4, impl4 = compute_bridge_constraints(4, adj4, v_star=0)
    # Bridge (0,1) is in v*'s component, S_{0,1} = [1]
    # Bridge (2,3) is in other component, but comp_id different from v*
    assert (0, 1) in bridges4, f"Disconnected bridges: {bridges4}"
    # S_map should have S_{0,1} = [1] (nodes reachable through bridge away from v*=0)
    if (0, 1) in S_map4:
        assert set(S_map4[(0, 1)]) == {1}, f"Disconnected S_(0,1): {S_map4.get((0,1))}"
    print("  ✓ Test 4 (Disconnected): PASSED")

    print("All bridge constraint unit tests PASSED!")


def compute_forced_edges(
    adj: Adjacency,
    v_star: int,
    edge_budget: Optional[int],
    undirected_edges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Compute edges incident to v* that must be selected (forced to 1).

    Idea: If v* has t incident edges, and excluding any single neighbor j,
    the remaining (t-1) neighbors' reachable subgraph has fewer than K edges,
    then the edge (v*, j) MUST be selected to reach K edges.

    More precisely: For each neighbor j of v*, compute the maximum number of
    edges reachable from v* without using edge (v*, j). If this count < K,
    then z_{v*,j} = 1 is forced.

    Args:
        adj: Adjacency structure
        v_star: Target node
        edge_budget: Edge budget K (if None, no forcing)
        undirected_edges: List of undirected edges in the subgraph

    Returns:
        List of forced edges (as canonical tuples with i < j)
    """
    if edge_budget is None:
        return []

    K = edge_budget
    neighbors_of_vstar = adj.neighbors[v_star]
    t = len(neighbors_of_vstar)

    if t <= 1:
        # Only 0 or 1 neighbor, no forcing possible
        return []

    # Build edge set for quick lookup
    edge_set = set()
    for (i, j) in undirected_edges:
        a, b = (i, j) if i < j else (j, i)
        edge_set.add((a, b))

    # Build full adjacency for BFS
    n = adj.n
    full_neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for (i, j) in undirected_edges:
        full_neighbors[i].add(j)
        full_neighbors[j].add(i)

    forced_edges = []

    for excluded_neighbor in neighbors_of_vstar:
        # BFS from v* without using edge (v*, excluded_neighbor)
        visited = {v_star}
        queue = deque([v_star])
        reachable_edges = set()

        while queue:
            node = queue.popleft()
            for neighbor in full_neighbors[node]:
                # Skip the excluded edge
                if node == v_star and neighbor == excluded_neighbor:
                    continue
                if neighbor == v_star and node == excluded_neighbor:
                    continue

                # Add this edge to reachable set
                a, b = (node, neighbor) if node < neighbor else (neighbor, node)
                if (a, b) in edge_set:
                    reachable_edges.add((a, b))

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # If reachable edges < K, the excluded edge must be selected
        if len(reachable_edges) < K:
            a, b = (v_star, excluded_neighbor) if v_star < excluded_neighbor else (excluded_neighbor, v_star)
            forced_edges.append((a, b))

    return forced_edges


def add_forced_edge_constraints(
    model,
    z: Dict[Tuple[int, int], "gp.Var"],
    forced_edges: List[Tuple[int, int]],
    verbose: bool = False,
    terminal_log_file: Optional[str] = None,
) -> int:
    """
    Add constraints to force certain edges to be selected.

    Args:
        model: Gurobi model
        z: Edge selection variables
        forced_edges: List of edges that must be selected
        verbose: Print debug info

    Returns:
        Number of constraints added
    """
    num_constraints = 0
    for (i, j) in forced_edges:
        if (i, j) in z:
            model.addConstr(z[(i, j)] == 1, name=f"forced_{i}_{j}")
            num_constraints += 1

    if num_constraints > 0:
        _print_and_log(
            f"[MIP] Forced edges: {num_constraints} edges fixed to 1",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    return num_constraints


def _pre_bounds_selectable_topk(
    adj: Adjacency,
    L_h: np.ndarray,
    U_h: np.ndarray,
    alpha: float,
    kappa: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-κ tightened bounds for pre = alpha * h_i + sum_{j in N(i)} z_ij * h_j.

    Instead of summing over all neighbors, we only consider the κ_i most
    extreme contributors (sorted by their bounds).

    Args:
        adj: Adjacency structure
        L_h, U_h: Bounds on h (n, d)
        alpha: Self-loop coefficient (1 + eps)
        kappa: Per-node incident edge caps (n,)

    Returns:
        L_pre, U_pre: Tightened bounds (n, d)
    """
    n, d = L_h.shape

    # Self contribution bounds
    alpha_eff = _node_self_coefficients(adj, alpha).reshape(-1, 1)
    self_L = np.minimum(alpha_eff * L_h, alpha_eff * U_h)
    self_U = np.maximum(alpha_eff * L_h, alpha_eff * U_h)

    L_pre = self_L.copy().astype(float)
    U_pre = self_U.copy().astype(float)

    for i in range(n):
        neighbors = adj.neighbors[i]
        k_i = int(kappa[i])

        if k_i == 0:
            # No edges can be selected incident to i
            continue

        if k_i >= len(neighbors):
            # Default: sum over all neighbors (no tightening)
            for j in neighbors:
                L_pre[i, :] += np.minimum(0.0, L_h[j, :])
                U_pre[i, :] += np.maximum(0.0, U_h[j, :])
        else:
            # Top-κ tightening: select k_i most extreme contributors
            for dim in range(d):
                # For upper bound: sort U_h[j, dim] in decreasing order, take top-k positive ones
                neighbor_U = [max(0.0, U_h[j, dim]) for j in neighbors]
                neighbor_U_sorted = sorted(neighbor_U, reverse=True)
                U_pre[i, dim] += sum(neighbor_U_sorted[:k_i])

                # For lower bound: sort L_h[j, dim] in increasing order, take top-k negative ones
                neighbor_L = [min(0.0, L_h[j, dim]) for j in neighbors]
                neighbor_L_sorted = sorted(neighbor_L)  # ascending
                L_pre[i, dim] += sum(neighbor_L_sorted[:k_i])

    return L_pre, U_pre


# =============================================================================
# Node-explanation edge scopes
# =============================================================================

def _add_complement_edge_variables(
    model,
    z: Dict[Tuple[int, int], object],
    edges: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], object]:
    """Create explicit removal variables r_e = 1 - z_e."""
    gp, GRB = _require_gurobi()
    removal_z = {
        edge: model.addVar(vtype=GRB.BINARY, name=f"r_edge_{edge[0]}_{edge[1]}")
        for edge in edges
    }
    model.update()
    for edge in edges:
        model.addConstr(
            removal_z[edge] + z[edge] == 1.0,
            name=f"edge_complement_{edge[0]}_{edge[1]}",
        )
    return removal_z


# =============================================================================
# MILP Building: 3-Layer Node Classification
# =============================================================================

def build_gin_nodegnn_milp(
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    *,
    v_star: int,
    weights: GINNodeGnnWeights,
    reference_c_star: Optional[int] = None,
    reference_bar_m: Optional[float] = None,
    rho: float = 0.0,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    min_edges: Optional[int] = None,
    objective: ObjectiveMode = "min_edges",
    connectivity: ConnectivityMode = "none",
    connectivity_on_complement: bool = False,
    use_maxflow_separation: bool = False,
    use_nested_cuts: bool = False,
    use_back_cuts: bool = False,
    max_cuts_per_round: int = 25,
    max_root_sep_rounds: int = 1,
    use_anderson_cuts: bool = False,
    anderson_max_node_count: int = 0,
    anderson_max_cuts: int = 50,
    use_incident_cap_tightening: bool = False,
    cap_mode: CapMode = "global",
    use_symmetry_breaking: bool = False,
    use_star_constraints: bool = True,
    use_connectivity_lower_bound: bool = False,
    use_forced_edges: bool = False,
    use_bridge_constraints: bool = False,
    use_aggregated_bridge_constraints: bool = False,
    use_hierarchical_bridge: bool = False,
    use_obbt_u: bool = False,
    max_obbt_u_layers: Optional[int] = None,
    verbose: bool = False,
    time_limit_s: Optional[float] = None,
    wallclock_deadline_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    seed: int = 0,
    initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
) -> MilpBuild:
    """
    Build MILP for 3-layer GIN NodeGNN explanation.

    Args:
        X: Node features (n, d0)
        undirected_edges: List of undirected edges
        v_star: Target node to explain
        weights: Pre-extracted model weights
        rho: Margin preservation ratio (default 0.0)
        edge_budget: Maximum number of edges (edge-budget formulation)
        node_budget: Maximum number of selected nodes (connectivity-constrained subgraph mode)
        min_edges: Minimum number of edges
        objective: "min_edges", "max_sufficiency", "min_sufficiency", or "feasible"
        connectivity: "none", "flow", or "cut"
        use_maxflow_separation: Use max-flow for cut separation
        use_incident_cap_tightening: Enable Top-κ incident cap tightening
        cap_mode: "global" (κ_i = K) or "distance" (κ_i = max(0, K - dist(i)))
        use_symmetry_breaking: Enable symmetric-branch symmetry breaking
        use_connectivity_lower_bound: Enable connectivity lower bound Σz >= Σy - 1 when connectivity is enabled
        use_forced_edges: Enable forced edge detection and constraints
        use_bridge_constraints: Enable bridge-based implication constraints
        use_aggregated_bridge_constraints: Enable aggregated bridge constraints
        use_hierarchical_bridge: Enable bridge-tree hierarchical constraints
        use_obbt_u: Enable LP-based OBBT tightening for u-variable bounds.
            Currently applied only when connectivity='none'.
        max_obbt_u_layers: Apply u-OBBT to layers [1..max_obbt_u_layers]. None = all layers.
        verbose: Print progress
        time_limit_s: Solver-side limit (further capped by remaining wall-clock budget when provided)
        mip_gap: Gurobi MIP gap
        seed: Random seed
        initial_solution: Warm start edges
        log_file: Path for Gurobi log file
        terminal_log_file: Path to mirror verbose terminal messages

    Returns:
        MilpBuild object with Gurobi model and variables
    """
    _validate_objective_mode(objective, rho)
    if connectivity not in ("none", "flow", "cut"):
        raise ValueError("Unknown connectivity mode")

    gp, GRB = _require_gurobi()

    n, d0 = X.shape
    if not (0 <= v_star < n):
        raise ValueError("v_star out of range")

    h = weights.hidden_dim
    m1, m2, m3 = weights.mlp_hidden1, weights.mlp_hidden2, weights.mlp_hidden3
    num_classes = weights.num_classes
    use_jk_concat = getattr(weights, 'use_jk_concat', True)

    adj = compute_adjacency(n, undirected_edges)
    edges = _unique_undirected_edges(undirected_edges)

    # Full forward pass for reference prediction
    h1_full, h2_full, h3_full = forward_gin_nodegnn_full(X, adj, weights)
    if use_jk_concat:
        v_embed = np.concatenate([h1_full[v_star], h2_full[v_star], h3_full[v_star]], axis=0)
    else:
        v_embed = h3_full[v_star]
    bar_Z = v_embed @ weights.W_out.T + weights.b_out
    local_c_star = int(np.argmax(bar_Z))
    local_bar_m = float(bar_Z[local_c_star] - np.max(np.delete(bar_Z, local_c_star))) if num_classes >= 2 else 0.0
    if (reference_c_star is None) != (reference_bar_m is None):
        raise ValueError("reference_c_star and reference_bar_m must be provided together.")
    if reference_c_star is None:
        c_star = local_c_star
        bar_m = local_bar_m
        ref_msg = f"[MIP] Reference prediction (working graph): class={c_star}, margin={bar_m:.4f}"
    else:
        c_star = int(reference_c_star)
        if not (0 <= c_star < num_classes):
            raise ValueError(f"reference_c_star={c_star} out of range for num_classes={num_classes}")
        bar_m = float(reference_bar_m)
        ref_msg = f"[MIP] Reference prediction (original full graph): class={c_star}, margin={bar_m:.4f}"
        _print_and_log(
            f"[MIP] Working-graph full prediction: class={local_c_star}, margin={local_bar_m:.4f}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _print_and_log(
        ref_msg,
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    _print_and_log(
        f"[MIP] Graph: {n} nodes, {len(edges)} edges",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )

    # ========== Forced Edges - computed early for incident cap tightening ==========
    forced_edges: List[Tuple[int, int]] = []
    if use_forced_edges:
        forced_edges = compute_forced_edges(adj, v_star, edge_budget, edges)
        if forced_edges:
            _print_and_log(
                f"[MIP] Forced edges: {len(forced_edges)} edges",
                verbose=verbose,
                terminal_log_file=terminal_log_file,
            )

    # ========== Bound propagation ==========
    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)
    alpha3 = 1.0 + float(weights.eps3)

    # Compute incident caps for Top-κ tightening
    # Pass forced_edges for additional tightening in distance mode
    if use_incident_cap_tightening:
        kappa = compute_incident_caps(
            adj,
            edge_budget,
            cap_mode,
            v_star,
            forced_edges,
            node_budget=node_budget,
        )
        _print_and_log(
            f"[MIP] Top-κ tightening enabled, cap_mode={cap_mode}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIP] kappa range: [{kappa.min()}, {kappa.max()}]",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        kappa = None

    (
        L_pre_list, U_pre_list,
        L_u_list, U_u_list,
        L_a_list, U_a_list,
        L_h_list, U_h_list,
    ) = _compute_node_bounds_with_sequential_obbt(
        X=X,
        adj=adj,
        edges=edges,
        alpha_list=[alpha1, alpha2, alpha3],
        W_in_list=[weights.W1_1, weights.W2_1, weights.W3_1],
        b_in_list=[weights.b1_1, weights.b2_1, weights.b3_1],
        W_out_list=[weights.W1_2, weights.W2_2, weights.W3_2],
        b_out_list=[weights.b1_2, weights.b2_2, weights.b3_2],
        use_incident_cap_tightening=use_incident_cap_tightening,
        kappa=kappa,
        use_obbt_u=use_obbt_u,
        max_obbt_u_layers=max_obbt_u_layers,
        connectivity=connectivity,
        edge_budget=edge_budget,
        min_edges=min_edges,
        wallclock_deadline_s=wallclock_deadline_s,
        verbose=verbose,
        terminal_log_file=terminal_log_file,
        log_prefix="[MIP]",
    )

    L_pre1, L_pre2, L_pre3 = L_pre_list
    U_pre1, U_pre2, U_pre3 = U_pre_list
    L_u1, L_u2, L_u3 = L_u_list
    U_u1, U_u2, U_u3 = U_u_list
    L_a1, L_a2, L_a3 = L_a_list
    U_a1, U_a2, U_a3 = U_a_list
    L_h1, L_h2, L_h3 = L_h_list
    U_h1, U_h2, U_h3 = U_h_list

    # ========== Create Gurobi model ==========
    model = gp.Model("GINNodeGNN_MILP")
    _configure_gurobi_logging(model, verbose=verbose, log_file=log_file)
    model.Params.Seed = int(seed)
    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if use_anderson_cuts:
        # Required for robust user-cut mapping in callback.
        model.Params.PreCrush = 1

    # ========== Edge selection variables z ==========
    z: Dict[Tuple[int, int], object] = {}
    for i, j in edges:
        z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
    model.update()

    removal_z = _add_complement_edge_variables(model, z, edges) if connectivity_on_complement else None
    explanation_z = removal_z if removal_z is not None else z
    connectivity_edge_budget = (
        max(0, len(edges) - int(edge_budget))
        if connectivity_on_complement and edge_budget is not None
        else edge_budget
    )

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    # ========== Symmetry Breaking ==========
    # Create y variables early if symmetry breaking is enabled (needed for node symmetry breaking)
    # Bridge constraints reuse these variables if also enabled
    y: Optional[Dict[int, "gp.Var"]] = None
    symmetric_groups: Optional[Dict[int, List[List[int]]]] = None
    if use_symmetry_breaking:
        symmetric_groups = find_symmetric_branch_groups(adj, X, edges)
        # Edge-level symmetry breaking
        add_symmetry_breaking_constraints(
            model,
            explanation_z,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        # Create y variables for node-level symmetry breaking
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
        model.update()
        # y[i] <= sum of incident z edges
        for i in range(n):
            incident_edges = [z_ij(i, j) for j in adj.neighbors[i]]
            if incident_edges:
                model.addConstr(y[i] <= gp.quicksum(incident_edges), name=f"y_upper_{i}")
            else:
                model.addConstr(y[i] == 0, name=f"y_isolated_{i}")
        # Node-level symmetry breaking
        add_symmetry_breaking_constraints_nodes(
            model,
            y,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Forced Edges Constraints ==========
    # forced_edges was already computed above for incident cap tightening
    if use_forced_edges:
        add_forced_edge_constraints(
            model,
            z,
            forced_edges,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Hierarchical Bridge Constraints ==========
    if use_hierarchical_bridge:
        adj_list: List[List[int]] = [[] for _ in range(n)]
        for (a, b) in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        add_hierarchical_bridge_constraints(
            model,
            z,
            n,
            adj_list,
            v_star,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Bridge Constraints - Node Classification ==========
    # Create y variables if not already created by symmetry breaking
    if use_bridge_constraints or use_aggregated_bridge_constraints:
        if y is None:
            # Create y variables for node selection (y[i] = 1 if node i has at least one incident selected edge)
            y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
            model.update()
            # y[i] <= sum of incident z edges
            for i in range(n):
                incident_edges = [z_ij(i, j) for j in adj.neighbors[i]]
                if incident_edges:
                    model.addConstr(y[i] <= gp.quicksum(incident_edges), name=f"y_upper_{i}")
                else:
                    model.addConstr(y[i] == 0, name=f"y_isolated_{i}")

        # Build adjacency list for bridge detection
        adj_list: List[List[int]] = [[] for _ in range(n)]
        for (a, b) in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)

        add_bridge_constraints(model, y, z, n, adj_list, v_star,
                               edge_budget=edge_budget,
                               node_budget=node_budget,
                               use_individual=use_bridge_constraints,
                               use_aggregated=use_aggregated_bridge_constraints,
                               verbose=verbose,
                               log_file=log_file,
                               terminal_log_file=terminal_log_file)

    # ========== Layer 1 Variables ==========
    anderson_relus: List[AndersonReluSpec] = []
    p1 = {(i, f): model.addVar(lb=float(L_pre1[i, f]), ub=float(U_pre1[i, f]),
                                name=f"p1_{i}_{f}") for i in range(n) for f in range(d0)}
    u1 = {(i, m): model.addVar(lb=float(L_u1[i, m]), ub=float(U_u1[i, m]),
                                name=f"u1_{i}_{m}") for i in range(n) for m in range(m1)}
    a1 = {(i, m): model.addVar(lb=float(L_a1[i, m]), ub=float(U_a1[i, m]),
                                name=f"a1_{i}_{m}") for i in range(n) for m in range(m1)}
    d1 = {}  # ReLU indicator (only for ambiguous neurons)
    h1 = {(i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]),
                                name=f"h1_{i}_{k}") for i in range(n) for k in range(h)}
    model.update()

    # Layer 1: p1 = alpha1 * X + sum_j z_ij * X_j
    for i in range(n):
        for f in range(d0):
            expr = gp.LinExpr()
            expr.addConstant(float((alpha1 + adj.self_loop_counts[i]) * X[i, f]))
            for j in adj.neighbors[i]:
                expr.addTerms(float(X[j, f]), z_ij(i, j))
            model.addConstr(p1[(i, f)] == expr, name=f"p1def_{i}_{f}")

    # Layer 1: u1 = p1 @ W1_1 + b1_1
    for i in range(n):
        for m in range(m1):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_1[m]))
            for f in range(d0):
                wval = float(weights.W1_1[f, m])
                if wval != 0.0:
                    expr.addTerms(wval, p1[(i, f)])
            model.addConstr(u1[(i, m)] == expr, name=f"u1def_{i}_{m}")

    # Layer 1: ReLU a1 = max(0, u1) using Big-M
    for i in range(n):
        for m in range(m1):
            lb_u, ub_u = float(L_u1[i, m]), float(U_u1[i, m])
            uvar, avar = u1[(i, m)], a1[(i, m)]
            if ub_u <= 0.0:
                model.addConstr(avar == 0.0, name=f"relu1_off_{i}_{m}")
            elif lb_u >= 0.0:
                model.addConstr(avar == uvar, name=f"relu1_on_{i}_{m}")
            else:
                dvar = model.addVar(vtype=GRB.BINARY, name=f"d1_{i}_{m}")
                d1[(i, m)] = dvar
                model.addConstr(avar >= uvar, name=f"relu1_1_{i}_{m}")
                model.addConstr(avar >= 0.0, name=f"relu1_2_{i}_{m}")
                model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu1_3_{i}_{m}")
                model.addConstr(avar <= ub_u * dvar, name=f"relu1_4_{i}_{m}")
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p1[(i, f)] for f in range(d0)),
                            y_var=avar,
                            delta_var=dvar,
                            w=np.asarray(weights.W1_1[:, m], dtype=float).copy(),
                            L=np.asarray(L_pre1[i, :], dtype=float).copy(),
                            U=np.asarray(U_pre1[i, :], dtype=float).copy(),
                            b=float(weights.b1_1[m]),
                            tag=f"relu1_{i}_{m}",
                        )
                    )
    # Layer 1: h1 = a1 @ W1_2 + b1_2
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_2[k]))
            for m in range(m1):
                wval = float(weights.W1_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a1[(i, m)])
            model.addConstr(h1[(i, k)] == expr, name=f"h1def_{i}_{k}")

    model.update()

    # ========== Layer 2 Variables ==========
    t2 = {}  # McCormick linearization: t2 = z * h1
    for (u, v) in edges:
        for k in range(h):
            lb_v = float(min(0.0, L_h1[v, k]))
            ub_v = float(max(0.0, U_h1[v, k]))
            t2[(u, v, k)] = model.addVar(lb=lb_v, ub=ub_v, name=f"t2_{u}_{v}_{k}")
            lb_u = float(min(0.0, L_h1[u, k]))
            ub_u = float(max(0.0, U_h1[u, k]))
            t2[(v, u, k)] = model.addVar(lb=lb_u, ub=ub_u, name=f"t2_{v}_{u}_{k}")

    p2 = {(i, k): model.addVar(lb=float(L_pre2[i, k]), ub=float(U_pre2[i, k]),
                                name=f"p2_{i}_{k}") for i in range(n) for k in range(h)}
    u2 = {(i, m): model.addVar(lb=float(L_u2[i, m]), ub=float(U_u2[i, m]),
                                name=f"u2_{i}_{m}") for i in range(n) for m in range(m2)}
    a2 = {(i, m): model.addVar(lb=float(L_a2[i, m]), ub=float(U_a2[i, m]),
                                name=f"a2_{i}_{m}") for i in range(n) for m in range(m2)}
    d2 = {}
    h2 = {(i, k): model.addVar(lb=float(L_h2[i, k]), ub=float(U_h2[i, k]),
                                name=f"h2_{i}_{k}") for i in range(n) for k in range(h)}
    model.update()

    # Layer 2: McCormick constraints for t2 = z * h1
    for (u, v) in edges:
        zuv = z[(u, v)]
        for k in range(h):
            # t2[(u,v,k)] = z * h1[v,k]
            hv = h1[(v, k)]
            tv = t2[(u, v, k)]
            L, U = float(L_h1[v, k]), float(U_h1[v, k])
            model.addConstr(tv <= hv - L * (1 - zuv), name=f"t2mc1_{u}_{v}_{k}")
            model.addConstr(tv >= hv - U * (1 - zuv), name=f"t2mc2_{u}_{v}_{k}")
            model.addConstr(tv <= U * zuv, name=f"t2mc3_{u}_{v}_{k}")
            model.addConstr(tv >= L * zuv, name=f"t2mc4_{u}_{v}_{k}")

            # t2[(v,u,k)] = z * h1[u,k]
            hu = h1[(u, k)]
            tu = t2[(v, u, k)]
            L, U = float(L_h1[u, k]), float(U_h1[u, k])
            model.addConstr(tu <= hu - L * (1 - zuv), name=f"t2mc1_{v}_{u}_{k}")
            model.addConstr(tu >= hu - U * (1 - zuv), name=f"t2mc2_{v}_{u}_{k}")
            model.addConstr(tu <= U * zuv, name=f"t2mc3_{v}_{u}_{k}")
            model.addConstr(tu >= L * zuv, name=f"t2mc4_{v}_{u}_{k}")

    # Layer 2: p2 = alpha2 * h1 + sum_j t2
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addTerms(float(alpha2 + adj.self_loop_counts[i]), h1[(i, k)])
            for j in adj.neighbors[i]:
                expr.addTerms(1.0, t2[(i, j, k)])
            model.addConstr(p2[(i, k)] == expr, name=f"p2def_{i}_{k}")

    # Layer 2: u2 = p2 @ W2_1 + b2_1
    for i in range(n):
        for m in range(m2):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_1[m]))
            for k in range(h):
                wval = float(weights.W2_1[k, m])
                if wval != 0.0:
                    expr.addTerms(wval, p2[(i, k)])
            model.addConstr(u2[(i, m)] == expr, name=f"u2def_{i}_{m}")

    # Layer 2: ReLU
    for i in range(n):
        for m in range(m2):
            lb_u, ub_u = float(L_u2[i, m]), float(U_u2[i, m])
            uvar, avar = u2[(i, m)], a2[(i, m)]
            if ub_u <= 0.0:
                model.addConstr(avar == 0.0, name=f"relu2_off_{i}_{m}")
            elif lb_u >= 0.0:
                model.addConstr(avar == uvar, name=f"relu2_on_{i}_{m}")
            else:
                dvar = model.addVar(vtype=GRB.BINARY, name=f"d2_{i}_{m}")
                d2[(i, m)] = dvar
                model.addConstr(avar >= uvar, name=f"relu2_1_{i}_{m}")
                model.addConstr(avar >= 0.0, name=f"relu2_2_{i}_{m}")
                model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu2_3_{i}_{m}")
                model.addConstr(avar <= ub_u * dvar, name=f"relu2_4_{i}_{m}")
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p2[(i, k)] for k in range(h)),
                            y_var=avar,
                            delta_var=dvar,
                            w=np.asarray(weights.W2_1[:, m], dtype=float).copy(),
                            L=np.asarray(L_pre2[i, :], dtype=float).copy(),
                            U=np.asarray(U_pre2[i, :], dtype=float).copy(),
                            b=float(weights.b2_1[m]),
                            tag=f"relu2_{i}_{m}",
                        )
                    )
    # Layer 2: h2 = a2 @ W2_2 + b2_2
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_2[k]))
            for m in range(m2):
                wval = float(weights.W2_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a2[(i, m)])
            model.addConstr(h2[(i, k)] == expr, name=f"h2def_{i}_{k}")

    model.update()

    # ========== Layer 3 Variables ==========
    t3 = {}
    for (u, v) in edges:
        for k in range(h):
            lb_v = float(min(0.0, L_h2[v, k]))
            ub_v = float(max(0.0, U_h2[v, k]))
            t3[(u, v, k)] = model.addVar(lb=lb_v, ub=ub_v, name=f"t3_{u}_{v}_{k}")
            lb_u = float(min(0.0, L_h2[u, k]))
            ub_u = float(max(0.0, U_h2[u, k]))
            t3[(v, u, k)] = model.addVar(lb=lb_u, ub=ub_u, name=f"t3_{v}_{u}_{k}")

    p3 = {(i, k): model.addVar(lb=float(L_pre3[i, k]), ub=float(U_pre3[i, k]),
                                name=f"p3_{i}_{k}") for i in range(n) for k in range(h)}
    u3 = {(i, m): model.addVar(lb=float(L_u3[i, m]), ub=float(U_u3[i, m]),
                                name=f"u3_{i}_{m}") for i in range(n) for m in range(m3)}
    a3 = {(i, m): model.addVar(lb=float(L_a3[i, m]), ub=float(U_a3[i, m]),
                                name=f"a3_{i}_{m}") for i in range(n) for m in range(m3)}
    d3 = {}
    h3 = {(i, k): model.addVar(lb=float(L_h3[i, k]), ub=float(U_h3[i, k]),
                                name=f"h3_{i}_{k}") for i in range(n) for k in range(h)}
    model.update()

    # Layer 3: McCormick constraints
    for (u, v) in edges:
        zuv = z[(u, v)]
        for k in range(h):
            hv = h2[(v, k)]
            tv = t3[(u, v, k)]
            L, U = float(L_h2[v, k]), float(U_h2[v, k])
            model.addConstr(tv <= hv - L * (1 - zuv), name=f"t3mc1_{u}_{v}_{k}")
            model.addConstr(tv >= hv - U * (1 - zuv), name=f"t3mc2_{u}_{v}_{k}")
            model.addConstr(tv <= U * zuv, name=f"t3mc3_{u}_{v}_{k}")
            model.addConstr(tv >= L * zuv, name=f"t3mc4_{u}_{v}_{k}")

            hu = h2[(u, k)]
            tu = t3[(v, u, k)]
            L, U = float(L_h2[u, k]), float(U_h2[u, k])
            model.addConstr(tu <= hu - L * (1 - zuv), name=f"t3mc1_{v}_{u}_{k}")
            model.addConstr(tu >= hu - U * (1 - zuv), name=f"t3mc2_{v}_{u}_{k}")
            model.addConstr(tu <= U * zuv, name=f"t3mc3_{v}_{u}_{k}")
            model.addConstr(tu >= L * zuv, name=f"t3mc4_{v}_{u}_{k}")

    # Layer 3: p3 = alpha3 * h2 + sum_j t3
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addTerms(float(alpha3 + adj.self_loop_counts[i]), h2[(i, k)])
            for j in adj.neighbors[i]:
                expr.addTerms(1.0, t3[(i, j, k)])
            model.addConstr(p3[(i, k)] == expr, name=f"p3def_{i}_{k}")

    # Layer 3: u3, ReLU, h3
    for i in range(n):
        for m in range(m3):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b3_1[m]))
            for k in range(h):
                wval = float(weights.W3_1[k, m])
                if wval != 0.0:
                    expr.addTerms(wval, p3[(i, k)])
            model.addConstr(u3[(i, m)] == expr, name=f"u3def_{i}_{m}")

    for i in range(n):
        for m in range(m3):
            lb_u, ub_u = float(L_u3[i, m]), float(U_u3[i, m])
            uvar, avar = u3[(i, m)], a3[(i, m)]
            if ub_u <= 0.0:
                model.addConstr(avar == 0.0, name=f"relu3_off_{i}_{m}")
            elif lb_u >= 0.0:
                model.addConstr(avar == uvar, name=f"relu3_on_{i}_{m}")
            else:
                dvar = model.addVar(vtype=GRB.BINARY, name=f"d3_{i}_{m}")
                d3[(i, m)] = dvar
                model.addConstr(avar >= uvar, name=f"relu3_1_{i}_{m}")
                model.addConstr(avar >= 0.0, name=f"relu3_2_{i}_{m}")
                model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu3_3_{i}_{m}")
                model.addConstr(avar <= ub_u * dvar, name=f"relu3_4_{i}_{m}")
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p3[(i, k)] for k in range(h)),
                            y_var=avar,
                            delta_var=dvar,
                            w=np.asarray(weights.W3_1[:, m], dtype=float).copy(),
                            L=np.asarray(L_pre3[i, :], dtype=float).copy(),
                            U=np.asarray(U_pre3[i, :], dtype=float).copy(),
                            b=float(weights.b3_1[m]),
                            tag=f"relu3_{i}_{m}",
                        )
                    )

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b3_2[k]))
            for m in range(m3):
                wval = float(weights.W3_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a3[(i, m)])
            model.addConstr(h3[(i, k)] == expr, name=f"h3def_{i}_{k}")

    model.update()

    # ========== Output logits Z for v_star ==========
    Z = {c: model.addVar(lb=-GRB.INFINITY, name=f"Z_{c}") for c in range(num_classes)}
    model.update()

    for c in range(num_classes):
        expr = gp.LinExpr()
        expr.addConstant(float(weights.b_out[c]))
        if use_jk_concat:
            for k in range(h):
                wk = float(weights.W_out[c, k])
                if wk != 0.0:
                    expr.addTerms(wk, h1[(v_star, k)])
            for k in range(h):
                wk = float(weights.W_out[c, h + k])
                if wk != 0.0:
                    expr.addTerms(wk, h2[(v_star, k)])
            for k in range(h):
                wk = float(weights.W_out[c, 2 * h + k])
                if wk != 0.0:
                    expr.addTerms(wk, h3[(v_star, k)])
        else:
            for k in range(h):
                wk = float(weights.W_out[c, k])
                if wk != 0.0:
                    expr.addTerms(wk, h3[(v_star, k)])
        model.addConstr(Z[c] == expr, name=f"Zdef_{c}")

    # ========== Margin constraints ==========
    rho_var = _add_margin_constraints(
        model=model,
        Z=Z,
        c_star=c_star,
        num_classes=num_classes,
        bar_m=bar_m,
        objective=objective,
        rho=rho,
    )

    # ========== Edge budget constraints ==========
    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    # ========== Connectivity constraints ==========
    y = None
    if connectivity != "none":
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
        model.update()
        model.addConstr(y[v_star] == 1.0, name="root_selected")
        for (i, j), zij in explanation_z.items():
            model.addConstr(zij <= y[i], name=f"edge_node1_{i}_{j}")
            model.addConstr(zij <= y[j], name=f"edge_node2_{i}_{j}")
        if use_connectivity_lower_bound:
            model.addConstr(
                gp.quicksum(explanation_z[e] for e in edges) >= gp.quicksum(y[i] for i in range(n)) - 1.0,
                name="connectivity_lower_bound",
            )

        # Optional star inequalities
        if use_star_constraints:
            for i in range(n):
                if i == v_star:
                    continue
                incident = [explanation_z[(a, b)] for (a, b) in edges if a == i or b == i]
                if incident:
                    model.addConstr(y[i] <= gp.quicksum(incident), name=f"star_{i}")

        if connectivity == "flow":
            f = {}
            cap = _flow_cap_upper_bound(n, connectivity_edge_budget, node_budget)
            for (i, j) in edges:
                f[(i, j)] = model.addVar(lb=0.0, ub=cap, name=f"f_{i}_{j}")
                f[(j, i)] = model.addVar(lb=0.0, ub=cap, name=f"f_{j}_{i}")
            model.update()

            for i in range(n):
                out_arcs = [f[(i, j)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                in_arcs = [f[(j, i)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                if i == v_star:
                    model.addConstr(
                        gp.quicksum(out_arcs) - gp.quicksum(in_arcs)
                        == gp.quicksum(y[j] for j in range(n) if j != v_star),
                        name=f"flow_root_{i}",
                    )
                else:
                    model.addConstr(
                        gp.quicksum(in_arcs) - gp.quicksum(out_arcs) == y[i],
                        name=f"flow_balance_{i}",
                    )
            for (i, j), zij in explanation_z.items():
                model.addConstr(f[(i, j)] + f[(j, i)] <= cap * zij, name=f"flow_cap_{i}_{j}")

        elif connectivity == "cut":
            model.Params.LazyConstraints = 1

    # ========== Objective ==========
    _set_selection_objective(
        model=model,
        z=z,
        edges=edges,
        objective=objective,
        rho_var=rho_var,
        edge_budget=edge_budget,
        node_budget=node_budget,
    )

    # ========== Warm start ==========
    if initial_solution is not None:
        initial_set = set(_unique_undirected_edges(initial_solution))
        for e, zvar in z.items():
            zvar.Start = 1.0 if e in initial_set else 0.0
        _print_and_log(
            f"[MIP] Warm start with {len(initial_set)} edges",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )

    return MilpBuild(
        model=model,
        z=z,
        y=y,
        edges=edges,
        n=n,
        v_star=v_star,
        tilde_a=adj,
        c_star=c_star,
        rho_var=rho_var,
        use_maxflow_separation=use_maxflow_separation,
        connectivity_z=explanation_z,
        removal_z=removal_z,
        bar_m=bar_m,
        use_nested_cuts=use_nested_cuts,
        use_back_cuts=use_back_cuts,
        max_cuts_per_round=max_cuts_per_round,
        max_root_sep_rounds=max_root_sep_rounds,
        use_anderson_cuts=use_anderson_cuts,
        anderson_max_node_count=anderson_max_node_count,
        anderson_max_cuts=anderson_max_cuts,
        anderson_relus=tuple(anderson_relus),
        log_file=log_file,
        terminal_log_file=terminal_log_file,
    )


# =============================================================================
# MILP Building: 4-Layer Node Classification
# =============================================================================

def build_gin_nodegnn_milp_4l(
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    *,
    v_star: int,
    weights: GINNodeGnnWeights4L,
    reference_c_star: Optional[int] = None,
    reference_bar_m: Optional[float] = None,
    rho: float = 0.0,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    min_edges: Optional[int] = None,
    objective: ObjectiveMode = "min_edges",
    connectivity: ConnectivityMode = "none",
    connectivity_on_complement: bool = False,
    use_maxflow_separation: bool = False,
    use_nested_cuts: bool = False,
    use_back_cuts: bool = False,
    max_cuts_per_round: int = 25,
    max_root_sep_rounds: int = 1,
    use_anderson_cuts: bool = False,
    anderson_max_node_count: int = 0,
    anderson_max_cuts: int = 50,
    use_incident_cap_tightening: bool = False,
    cap_mode: CapMode = "global",
    use_symmetry_breaking: bool = False,
    use_star_constraints: bool = True,
    use_connectivity_lower_bound: bool = False,
    use_forced_edges: bool = False,
    use_bridge_constraints: bool = False,
    use_aggregated_bridge_constraints: bool = False,
    use_hierarchical_bridge: bool = False,
    use_obbt_u: bool = False,
    max_obbt_u_layers: Optional[int] = None,
    verbose: bool = False,
    time_limit_s: Optional[float] = None,
    wallclock_deadline_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    seed: int = 0,
    initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
) -> MilpBuild:
    """Build MILP for 4-layer GIN NodeGNN explanation."""
    _validate_objective_mode(objective, rho)
    if connectivity not in ("none", "flow", "cut"):
        raise ValueError("Unknown connectivity mode")

    gp, GRB = _require_gurobi()

    n, d0 = X.shape
    if not (0 <= v_star < n):
        raise ValueError("v_star out of range")

    h = weights.hidden_dim
    m1, m2, m3, m4 = (
        weights.mlp_hidden1,
        weights.mlp_hidden2,
        weights.mlp_hidden3,
        weights.mlp_hidden4,
    )
    num_classes = weights.num_classes
    use_jk_concat = getattr(weights, "use_jk_concat", True)

    adj = compute_adjacency(n, undirected_edges)
    edges = _unique_undirected_edges(undirected_edges)

    # Full forward pass for reference prediction
    h1_full, h2_full, h3_full, h4_full = forward_gin_nodegnn_full_4l(X, adj, weights)
    if use_jk_concat:
        v_embed = np.concatenate([h1_full[v_star], h2_full[v_star], h3_full[v_star], h4_full[v_star]], axis=0)
    else:
        v_embed = h4_full[v_star]
    bar_Z = v_embed @ weights.W_out.T + weights.b_out
    local_c_star = int(np.argmax(bar_Z))
    local_bar_m = float(bar_Z[local_c_star] - np.max(np.delete(bar_Z, local_c_star))) if num_classes >= 2 else 0.0
    if (reference_c_star is None) != (reference_bar_m is None):
        raise ValueError("reference_c_star and reference_bar_m must be provided together.")
    if reference_c_star is None:
        c_star = local_c_star
        bar_m = local_bar_m
        ref_msg = f"[MIP-4L] Reference prediction (working graph): class={c_star}, margin={bar_m:.4f}"
    else:
        c_star = int(reference_c_star)
        if not (0 <= c_star < num_classes):
            raise ValueError(f"reference_c_star={c_star} out of range for num_classes={num_classes}")
        bar_m = float(reference_bar_m)
        ref_msg = f"[MIP-4L] Reference prediction (original full graph): class={c_star}, margin={bar_m:.4f}"
        _print_and_log(
            f"[MIP-4L] Working-graph full prediction: class={local_c_star}, margin={local_bar_m:.4f}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _print_and_log(
        ref_msg,
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    _print_and_log(
        f"[MIP-4L] Graph: {n} nodes, {len(edges)} edges",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )

    # ========== Forced Edges - computed early for incident cap tightening ==========
    forced_edges: List[Tuple[int, int]] = []
    if use_forced_edges:
        forced_edges = compute_forced_edges(adj, v_star, edge_budget, edges)
        if forced_edges:
            _print_and_log(
                f"[MIP-4L] Forced edges: {len(forced_edges)} edges",
                verbose=verbose,
                terminal_log_file=terminal_log_file,
            )

    # ========== Bound propagation ==========
    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)
    alpha3 = 1.0 + float(weights.eps3)
    alpha4 = 1.0 + float(weights.eps4)

    if use_incident_cap_tightening:
        kappa = compute_incident_caps(
            adj,
            edge_budget,
            cap_mode,
            v_star,
            forced_edges,
            node_budget=node_budget,
        )
        _print_and_log(
            f"[MIP-4L] Top-κ tightening enabled, cap_mode={cap_mode}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIP-4L] kappa range: [{kappa.min()}, {kappa.max()}]",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        kappa = None

    (
        L_pre_list, U_pre_list,
        L_u_list, U_u_list,
        L_a_list, U_a_list,
        L_h_list, U_h_list,
    ) = _compute_node_bounds_with_sequential_obbt(
        X=X,
        adj=adj,
        edges=edges,
        alpha_list=[alpha1, alpha2, alpha3, alpha4],
        W_in_list=[weights.W1_1, weights.W2_1, weights.W3_1, weights.W4_1],
        b_in_list=[weights.b1_1, weights.b2_1, weights.b3_1, weights.b4_1],
        W_out_list=[weights.W1_2, weights.W2_2, weights.W3_2, weights.W4_2],
        b_out_list=[weights.b1_2, weights.b2_2, weights.b3_2, weights.b4_2],
        use_incident_cap_tightening=use_incident_cap_tightening,
        kappa=kappa,
        use_obbt_u=use_obbt_u,
        max_obbt_u_layers=max_obbt_u_layers,
        connectivity=connectivity,
        edge_budget=edge_budget,
        min_edges=min_edges,
        wallclock_deadline_s=wallclock_deadline_s,
        verbose=verbose,
        terminal_log_file=terminal_log_file,
        log_prefix="[MIP-4L]",
    )

    L_pre1, L_pre2, L_pre3, L_pre4 = L_pre_list
    U_pre1, U_pre2, U_pre3, U_pre4 = U_pre_list
    L_u1, L_u2, L_u3, L_u4 = L_u_list
    U_u1, U_u2, U_u3, U_u4 = U_u_list
    L_a1, L_a2, L_a3, L_a4 = L_a_list
    U_a1, U_a2, U_a3, U_a4 = U_a_list
    L_h1, L_h2, L_h3, L_h4 = L_h_list
    U_h1, U_h2, U_h3, U_h4 = U_h_list

    # ========== Create Gurobi model ==========
    model = gp.Model("GINNodeGNN_4L_MILP")
    _configure_gurobi_logging(model, verbose=verbose, log_file=log_file)
    model.Params.Seed = int(seed)
    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if use_anderson_cuts:
        model.Params.PreCrush = 1

    # ========== Edge selection variables z ==========
    z: Dict[Tuple[int, int], object] = {}
    for i, j in edges:
        z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
    model.update()

    removal_z = _add_complement_edge_variables(model, z, edges) if connectivity_on_complement else None
    explanation_z = removal_z if removal_z is not None else z
    connectivity_edge_budget = (
        max(0, len(edges) - int(edge_budget))
        if connectivity_on_complement and edge_budget is not None
        else edge_budget
    )

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    # ========== Symmetry Breaking ==========
    # Create y variables early if symmetry breaking is enabled (needed for node symmetry breaking)
    # Bridge constraints reuse these variables if also enabled
    y: Optional[Dict[int, "gp.Var"]] = None
    symmetric_groups: Optional[Dict[int, List[List[int]]]] = None
    if use_symmetry_breaking:
        symmetric_groups = find_symmetric_branch_groups(adj, X, edges)
        # Edge-level symmetry breaking
        add_symmetry_breaking_constraints(
            model,
            explanation_z,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        # Create y variables for node-level symmetry breaking
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
        model.update()
        # y[i] <= sum of incident z edges
        for i in range(n):
            incident_edges = [z_ij(i, j) for j in adj.neighbors[i]]
            if incident_edges:
                model.addConstr(y[i] <= gp.quicksum(incident_edges), name=f"y_upper_{i}")
            else:
                model.addConstr(y[i] == 0, name=f"y_isolated_{i}")
        # Node-level symmetry breaking
        add_symmetry_breaking_constraints_nodes(
            model,
            y,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Forced Edges Constraints ==========
    if use_forced_edges:
        add_forced_edge_constraints(
            model,
            z,
            forced_edges,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Hierarchical Bridge Constraints ==========
    if use_hierarchical_bridge:
        adj_list: List[List[int]] = [[] for _ in range(n)]
        for (a, b) in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        add_hierarchical_bridge_constraints(
            model,
            z,
            n,
            adj_list,
            v_star,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Bridge Constraints - Node Classification ==========
    # Create y variables if not already created by symmetry breaking
    if use_bridge_constraints or use_aggregated_bridge_constraints:
        if y is None:
            y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
            model.update()
            for i in range(n):
                incident_edges = [z_ij(i, j) for j in adj.neighbors[i]]
                if incident_edges:
                    model.addConstr(y[i] <= gp.quicksum(incident_edges), name=f"y_upper_{i}")
                else:
                    model.addConstr(y[i] == 0, name=f"y_isolated_{i}")

        adj_list: List[List[int]] = [[] for _ in range(n)]
        for (a, b) in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)

        add_bridge_constraints(
            model,
            y,
            z,
            n,
            adj_list,
            v_star,
            edge_budget=edge_budget,
            node_budget=node_budget,
            use_individual=use_bridge_constraints,
            use_aggregated=use_aggregated_bridge_constraints,
            verbose=verbose,
            log_file=log_file,
            terminal_log_file=terminal_log_file,
        )

    # ========== Layer 1 ==========
    anderson_relus: List[AndersonReluSpec] = []
    p1 = {
        (i, f): model.addVar(lb=float(L_pre1[i, f]), ub=float(U_pre1[i, f]), name=f"p1_{i}_{f}")
        for i in range(n)
        for f in range(d0)
    }
    u1 = {
        (i, m): model.addVar(lb=float(L_u1[i, m]), ub=float(U_u1[i, m]), name=f"u1_{i}_{m}")
        for i in range(n)
        for m in range(m1)
    }
    a1 = {
        (i, m): model.addVar(lb=float(L_a1[i, m]), ub=float(U_a1[i, m]), name=f"a1_{i}_{m}")
        for i in range(n)
        for m in range(m1)
    }
    d1 = {}
    h1 = {
        (i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]), name=f"h1_{i}_{k}")
        for i in range(n)
        for k in range(h)
    }
    model.update()

    for i in range(n):
        for f in range(d0):
            expr = gp.LinExpr()
            expr.addConstant(float((alpha1 + adj.self_loop_counts[i]) * X[i, f]))
            for j in adj.neighbors[i]:
                expr.addTerms(float(X[j, f]), z_ij(i, j))
            model.addConstr(p1[(i, f)] == expr, name=f"p1def_{i}_{f}")

    for i in range(n):
        for m in range(m1):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_1[m]))
            for f in range(d0):
                wval = float(weights.W1_1[f, m])
                if wval != 0.0:
                    expr.addTerms(wval, p1[(i, f)])
            model.addConstr(u1[(i, m)] == expr, name=f"u1def_{i}_{m}")

    for i in range(n):
        for m in range(m1):
            lb_u, ub_u = float(L_u1[i, m]), float(U_u1[i, m])
            uvar, avar = u1[(i, m)], a1[(i, m)]
            if ub_u <= 0.0:
                model.addConstr(avar == 0.0, name=f"relu1_off_{i}_{m}")
            elif lb_u >= 0.0:
                model.addConstr(avar == uvar, name=f"relu1_on_{i}_{m}")
            else:
                dvar = model.addVar(vtype=GRB.BINARY, name=f"d1_{i}_{m}")
                d1[(i, m)] = dvar
                model.addConstr(avar >= uvar, name=f"relu1_1_{i}_{m}")
                model.addConstr(avar >= 0.0, name=f"relu1_2_{i}_{m}")
                model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu1_3_{i}_{m}")
                model.addConstr(avar <= ub_u * dvar, name=f"relu1_4_{i}_{m}")

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_2[k]))
            for m in range(m1):
                wval = float(weights.W1_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a1[(i, m)])
            model.addConstr(h1[(i, k)] == expr, name=f"h1def_{i}_{k}")

    model.update()

    if use_anderson_cuts and d1:
        for (i, m), dvar in d1.items():
            anderson_relus.append(
                AndersonReluSpec(
                    x_vars=tuple(p1[(i, f)] for f in range(d0)),
                    y_var=a1[(i, m)],
                    delta_var=dvar,
                    w=np.asarray(weights.W1_1[:, m], dtype=float).copy(),
                    L=np.asarray(L_pre1[i, :], dtype=float).copy(),
                    U=np.asarray(U_pre1[i, :], dtype=float).copy(),
                    b=float(weights.b1_1[m]),
                    tag=f"relu1_{i}_{m}",
                )
            )

    def add_selectable_layer(
        layer_idx: int,
        prev_h: Dict[Tuple[int, int], object],
        L_prev_h: np.ndarray,
        U_prev_h: np.ndarray,
        alpha: float,
        W_in: np.ndarray,
        b_in: np.ndarray,
        W_out: np.ndarray,
        b_out: np.ndarray,
        L_pre: np.ndarray,
        U_pre: np.ndarray,
        L_u: np.ndarray,
        U_u: np.ndarray,
        L_a: np.ndarray,
        U_a: np.ndarray,
        L_h_cur: np.ndarray,
        U_h_cur: np.ndarray,
        m_dim: int,
    ) -> Tuple[Dict[Tuple[int, int, int], object], Dict[Tuple[int, int], object], Dict[Tuple[int, int], object], Dict[Tuple[int, int], object], Dict[Tuple[int, int], object], Dict[Tuple[int, int], object]]:
        t = {}
        for (u, v) in edges:
            for k in range(h):
                lb_v = float(min(0.0, L_prev_h[v, k]))
                ub_v = float(max(0.0, U_prev_h[v, k]))
                t[(u, v, k)] = model.addVar(lb=lb_v, ub=ub_v, name=f"t{layer_idx}_{u}_{v}_{k}")
                lb_u = float(min(0.0, L_prev_h[u, k]))
                ub_u = float(max(0.0, U_prev_h[u, k]))
                t[(v, u, k)] = model.addVar(lb=lb_u, ub=ub_u, name=f"t{layer_idx}_{v}_{u}_{k}")

        p = {
            (i, k): model.addVar(lb=float(L_pre[i, k]), ub=float(U_pre[i, k]), name=f"p{layer_idx}_{i}_{k}")
            for i in range(n)
            for k in range(h)
        }
        u = {
            (i, m): model.addVar(lb=float(L_u[i, m]), ub=float(U_u[i, m]), name=f"u{layer_idx}_{i}_{m}")
            for i in range(n)
            for m in range(m_dim)
        }
        a = {
            (i, m): model.addVar(lb=float(L_a[i, m]), ub=float(U_a[i, m]), name=f"a{layer_idx}_{i}_{m}")
            for i in range(n)
            for m in range(m_dim)
        }
        d = {}
        h_cur = {
            (i, k): model.addVar(lb=float(L_h_cur[i, k]), ub=float(U_h_cur[i, k]), name=f"h{layer_idx}_{i}_{k}")
            for i in range(n)
            for k in range(h)
        }
        model.update()

        for (u_idx, v_idx) in edges:
            zuv = z[(u_idx, v_idx)]
            for k in range(h):
                hv = prev_h[(v_idx, k)]
                tv = t[(u_idx, v_idx, k)]
                Lb, Ub = float(L_prev_h[v_idx, k]), float(U_prev_h[v_idx, k])
                model.addConstr(tv <= hv - Lb * (1 - zuv), name=f"t{layer_idx}mc1_{u_idx}_{v_idx}_{k}")
                model.addConstr(tv >= hv - Ub * (1 - zuv), name=f"t{layer_idx}mc2_{u_idx}_{v_idx}_{k}")
                model.addConstr(tv <= Ub * zuv, name=f"t{layer_idx}mc3_{u_idx}_{v_idx}_{k}")
                model.addConstr(tv >= Lb * zuv, name=f"t{layer_idx}mc4_{u_idx}_{v_idx}_{k}")

                hu = prev_h[(u_idx, k)]
                tu = t[(v_idx, u_idx, k)]
                Lb, Ub = float(L_prev_h[u_idx, k]), float(U_prev_h[u_idx, k])
                model.addConstr(tu <= hu - Lb * (1 - zuv), name=f"t{layer_idx}mc1_{v_idx}_{u_idx}_{k}")
                model.addConstr(tu >= hu - Ub * (1 - zuv), name=f"t{layer_idx}mc2_{v_idx}_{u_idx}_{k}")
                model.addConstr(tu <= Ub * zuv, name=f"t{layer_idx}mc3_{v_idx}_{u_idx}_{k}")
                model.addConstr(tu >= Lb * zuv, name=f"t{layer_idx}mc4_{v_idx}_{u_idx}_{k}")

        for i in range(n):
            for k in range(h):
                expr = gp.LinExpr()
                expr.addTerms(float(alpha + adj.self_loop_counts[i]), prev_h[(i, k)])
                for j in adj.neighbors[i]:
                    expr.addTerms(1.0, t[(i, j, k)])
                model.addConstr(p[(i, k)] == expr, name=f"p{layer_idx}def_{i}_{k}")

        for i in range(n):
            for m in range(m_dim):
                expr = gp.LinExpr()
                expr.addConstant(float(b_in[m]))
                for k in range(h):
                    wval = float(W_in[k, m])
                    if wval != 0.0:
                        expr.addTerms(wval, p[(i, k)])
                model.addConstr(u[(i, m)] == expr, name=f"u{layer_idx}def_{i}_{m}")

        for i in range(n):
            for m in range(m_dim):
                lb_u, ub_u = float(L_u[i, m]), float(U_u[i, m])
                uvar, avar = u[(i, m)], a[(i, m)]
                if ub_u <= 0.0:
                    model.addConstr(avar == 0.0, name=f"relu{layer_idx}_off_{i}_{m}")
                elif lb_u >= 0.0:
                    model.addConstr(avar == uvar, name=f"relu{layer_idx}_on_{i}_{m}")
                else:
                    dvar = model.addVar(vtype=GRB.BINARY, name=f"d{layer_idx}_{i}_{m}")
                    d[(i, m)] = dvar
                    model.addConstr(avar >= uvar, name=f"relu{layer_idx}_1_{i}_{m}")
                    model.addConstr(avar >= 0.0, name=f"relu{layer_idx}_2_{i}_{m}")
                    model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu{layer_idx}_3_{i}_{m}")
                    model.addConstr(avar <= ub_u * dvar, name=f"relu{layer_idx}_4_{i}_{m}")
                    if use_anderson_cuts:
                        anderson_relus.append(
                            AndersonReluSpec(
                                x_vars=tuple(p[(i, k)] for k in range(h)),
                                y_var=avar,
                                delta_var=dvar,
                                w=np.asarray(W_in[:, m], dtype=float).copy(),
                                L=np.asarray(L_pre[i, :], dtype=float).copy(),
                                U=np.asarray(U_pre[i, :], dtype=float).copy(),
                                b=float(b_in[m]),
                                tag=f"relu{layer_idx}_{i}_{m}",
                            )
                        )

        for i in range(n):
            for k in range(h):
                expr = gp.LinExpr()
                expr.addConstant(float(b_out[k]))
                for m in range(m_dim):
                    wval = float(W_out[m, k])
                    if wval != 0.0:
                        expr.addTerms(wval, a[(i, m)])
                model.addConstr(h_cur[(i, k)] == expr, name=f"h{layer_idx}def_{i}_{k}")

        model.update()
        return t, p, u, a, d, h_cur

    _, _, _, _, _, h2 = add_selectable_layer(
        2, h1, L_h1, U_h1, alpha2, weights.W2_1, weights.b2_1, weights.W2_2, weights.b2_2,
        L_pre2, U_pre2, L_u2, U_u2, L_a2, U_a2, L_h2, U_h2, m2,
    )
    _, _, _, _, _, h3 = add_selectable_layer(
        3, h2, L_h2, U_h2, alpha3, weights.W3_1, weights.b3_1, weights.W3_2, weights.b3_2,
        L_pre3, U_pre3, L_u3, U_u3, L_a3, U_a3, L_h3, U_h3, m3,
    )
    _, _, _, _, _, h4 = add_selectable_layer(
        4, h3, L_h3, U_h3, alpha4, weights.W4_1, weights.b4_1, weights.W4_2, weights.b4_2,
        L_pre4, U_pre4, L_u4, U_u4, L_a4, U_a4, L_h4, U_h4, m4,
    )

    # ========== Output logits Z for v_star ==========
    Z = {c: model.addVar(lb=-GRB.INFINITY, name=f"Z_{c}") for c in range(num_classes)}
    model.update()

    for c in range(num_classes):
        expr = gp.LinExpr()
        expr.addConstant(float(weights.b_out[c]))
        if use_jk_concat:
            for k in range(h):
                wk = float(weights.W_out[c, k])
                if wk != 0.0:
                    expr.addTerms(wk, h1[(v_star, k)])
            for k in range(h):
                wk = float(weights.W_out[c, h + k])
                if wk != 0.0:
                    expr.addTerms(wk, h2[(v_star, k)])
            for k in range(h):
                wk = float(weights.W_out[c, 2 * h + k])
                if wk != 0.0:
                    expr.addTerms(wk, h3[(v_star, k)])
            for k in range(h):
                wk = float(weights.W_out[c, 3 * h + k])
                if wk != 0.0:
                    expr.addTerms(wk, h4[(v_star, k)])
        else:
            for k in range(h):
                wk = float(weights.W_out[c, k])
                if wk != 0.0:
                    expr.addTerms(wk, h4[(v_star, k)])
        model.addConstr(Z[c] == expr, name=f"Zdef_{c}")

    # ========== Margin constraints ==========
    rho_var = _add_margin_constraints(
        model=model,
        Z=Z,
        c_star=c_star,
        num_classes=num_classes,
        bar_m=bar_m,
        objective=objective,
        rho=rho,
    )

    # ========== Edge budget constraints ==========
    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    # ========== Connectivity constraints ==========
    y = None
    if connectivity != "none":
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
        model.update()
        model.addConstr(y[v_star] == 1.0, name="root_selected")
        for (i, j), zij in explanation_z.items():
            model.addConstr(zij <= y[i], name=f"edge_node1_{i}_{j}")
            model.addConstr(zij <= y[j], name=f"edge_node2_{i}_{j}")
        if use_connectivity_lower_bound:
            model.addConstr(
                gp.quicksum(explanation_z[e] for e in edges) >= gp.quicksum(y[i] for i in range(n)) - 1.0,
                name="connectivity_lower_bound",
            )

        if use_star_constraints:
            for i in range(n):
                if i == v_star:
                    continue
                incident = [explanation_z[(a, b)] for (a, b) in edges if a == i or b == i]
                if incident:
                    model.addConstr(y[i] <= gp.quicksum(incident), name=f"star_{i}")

        if connectivity == "flow":
            f = {}
            cap = _flow_cap_upper_bound(n, connectivity_edge_budget, node_budget)
            for (i, j) in edges:
                f[(i, j)] = model.addVar(lb=0.0, ub=cap, name=f"f_{i}_{j}")
                f[(j, i)] = model.addVar(lb=0.0, ub=cap, name=f"f_{j}_{i}")
            model.update()

            for i in range(n):
                out_arcs = [f[(i, j)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                in_arcs = [f[(j, i)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                if i == v_star:
                    model.addConstr(
                        gp.quicksum(out_arcs) - gp.quicksum(in_arcs)
                        == gp.quicksum(y[j] for j in range(n) if j != v_star),
                        name=f"flow_root_{i}",
                    )
                else:
                    model.addConstr(
                        gp.quicksum(in_arcs) - gp.quicksum(out_arcs) == y[i],
                        name=f"flow_balance_{i}",
                    )
            for (i, j), zij in explanation_z.items():
                model.addConstr(f[(i, j)] + f[(j, i)] <= cap * zij, name=f"flow_cap_{i}_{j}")

        elif connectivity == "cut":
            model.Params.LazyConstraints = 1

    # ========== Objective ==========
    _set_selection_objective(
        model=model,
        z=z,
        edges=edges,
        objective=objective,
        rho_var=rho_var,
        edge_budget=edge_budget,
        node_budget=node_budget,
    )

    # ========== Warm start ==========
    if initial_solution is not None:
        initial_set = set(_unique_undirected_edges(initial_solution))
        for e, zvar in z.items():
            zvar.Start = 1.0 if e in initial_set else 0.0
        _print_and_log(
            f"[MIP-4L] Warm start with {len(initial_set)} edges",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )

    return MilpBuild(
        model=model,
        z=z,
        y=y,
        edges=edges,
        n=n,
        v_star=v_star,
        tilde_a=adj,
        c_star=c_star,
        rho_var=rho_var,
        use_maxflow_separation=use_maxflow_separation,
        connectivity_z=explanation_z,
        removal_z=removal_z,
        bar_m=bar_m,
        use_nested_cuts=use_nested_cuts,
        use_back_cuts=use_back_cuts,
        max_cuts_per_round=max_cuts_per_round,
        max_root_sep_rounds=max_root_sep_rounds,
        use_anderson_cuts=use_anderson_cuts,
        anderson_max_node_count=anderson_max_node_count,
        anderson_max_cuts=anderson_max_cuts,
        anderson_relus=tuple(anderson_relus),
        log_file=log_file,
        terminal_log_file=terminal_log_file,
    )


# =============================================================================
# 2-Layer Node Classification MILP
# =============================================================================

def build_gin_nodegnn_milp_2l(
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    *,
    v_star: int,
    weights: GINNodeGnnWeights2L,
    reference_c_star: Optional[int] = None,
    reference_bar_m: Optional[float] = None,
    rho: float = 0.0,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    min_edges: Optional[int] = None,
    objective: ObjectiveMode = "min_edges",
    connectivity: ConnectivityMode = "none",
    connectivity_on_complement: bool = False,
    use_maxflow_separation: bool = False,
    use_nested_cuts: bool = False,
    use_back_cuts: bool = False,
    max_cuts_per_round: int = 25,
    max_root_sep_rounds: int = 1,
    use_anderson_cuts: bool = False,
    anderson_max_node_count: int = 0,
    anderson_max_cuts: int = 50,
    use_incident_cap_tightening: bool = False,
    cap_mode: CapMode = "global",
    use_symmetry_breaking: bool = False,
    use_star_constraints: bool = True,
    use_connectivity_lower_bound: bool = False,
    use_forced_edges: bool = False,
    use_bridge_constraints: bool = False,
    use_aggregated_bridge_constraints: bool = False,
    use_hierarchical_bridge: bool = False,
    use_obbt_u: bool = False,
    max_obbt_u_layers: Optional[int] = None,
    verbose: bool = False,
    time_limit_s: Optional[float] = None,
    wallclock_deadline_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    seed: int = 0,
    initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
) -> MilpBuild:
    """
    Build MILP for 2-layer GIN NodeGNN explanation.

    This is the 2-layer version of build_gin_nodegnn_milp.
    Uses JK-concat by default: output = [h1 || h2] @ W_out + b_out
    """
    _validate_objective_mode(objective, rho)
    if connectivity not in ("none", "flow", "cut"):
        raise ValueError("Unknown connectivity mode")

    gp, GRB = _require_gurobi()

    n, d0 = X.shape
    if not (0 <= v_star < n):
        raise ValueError("v_star out of range")

    h = weights.hidden_dim
    m1, m2 = weights.mlp_hidden1, weights.mlp_hidden2
    num_classes = weights.num_classes
    use_jk_concat = getattr(weights, 'use_jk_concat', True)

    adj = compute_adjacency(n, undirected_edges)
    edges = _unique_undirected_edges(undirected_edges)

    # Full forward pass for reference prediction
    h1_full, h2_full = forward_gin_nodegnn_full_2l(X, adj, weights)
    if use_jk_concat:
        v_embed = np.concatenate([h1_full[v_star], h2_full[v_star]], axis=0)
    else:
        v_embed = h2_full[v_star]
    bar_Z = v_embed @ weights.W_out.T + weights.b_out
    local_c_star = int(np.argmax(bar_Z))
    local_bar_m = float(bar_Z[local_c_star] - np.max(np.delete(bar_Z, local_c_star))) if num_classes >= 2 else 0.0
    if (reference_c_star is None) != (reference_bar_m is None):
        raise ValueError("reference_c_star and reference_bar_m must be provided together.")
    if reference_c_star is None:
        c_star = local_c_star
        bar_m = local_bar_m
        ref_msg = f"[MIP-2L] Reference prediction (working graph): class={c_star}, margin={bar_m:.4f}"
    else:
        c_star = int(reference_c_star)
        if not (0 <= c_star < num_classes):
            raise ValueError(f"reference_c_star={c_star} out of range for num_classes={num_classes}")
        bar_m = float(reference_bar_m)
        ref_msg = f"[MIP-2L] Reference prediction (original full graph): class={c_star}, margin={bar_m:.4f}"
        _print_and_log(
            f"[MIP-2L] Working-graph full prediction: class={local_c_star}, margin={local_bar_m:.4f}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _print_and_log(
        ref_msg,
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    _print_and_log(
        f"[MIP-2L] Graph: {n} nodes, {len(edges)} edges",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )

    # ========== Forced Edges ==========
    forced_edges: List[Tuple[int, int]] = []
    if use_forced_edges:
        forced_edges = compute_forced_edges(adj, v_star, edge_budget, edges)
        if forced_edges:
            _print_and_log(
                f"[MIP-2L] Forced edges: {len(forced_edges)} edges",
                verbose=verbose,
                terminal_log_file=terminal_log_file,
            )

    # ========== Bound propagation ==========
    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)

    # Compute incident caps for Top-κ tightening
    if use_incident_cap_tightening:
        kappa = compute_incident_caps(
            adj,
            edge_budget,
            cap_mode,
            v_star,
            forced_edges,
            node_budget=node_budget,
        )
        _print_and_log(
            f"[MIP-2L] Top-κ tightening enabled, cap_mode={cap_mode}",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIP-2L] kappa range: [{kappa.min()}, {kappa.max()}]",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        kappa = None

    (
        L_pre_list, U_pre_list,
        L_u_list, U_u_list,
        L_a_list, U_a_list,
        L_h_list, U_h_list,
    ) = _compute_node_bounds_with_sequential_obbt(
        X=X,
        adj=adj,
        edges=edges,
        alpha_list=[alpha1, alpha2],
        W_in_list=[weights.W1_1, weights.W2_1],
        b_in_list=[weights.b1_1, weights.b2_1],
        W_out_list=[weights.W1_2, weights.W2_2],
        b_out_list=[weights.b1_2, weights.b2_2],
        use_incident_cap_tightening=use_incident_cap_tightening,
        kappa=kappa,
        use_obbt_u=use_obbt_u,
        max_obbt_u_layers=max_obbt_u_layers,
        connectivity=connectivity,
        edge_budget=edge_budget,
        min_edges=min_edges,
        wallclock_deadline_s=wallclock_deadline_s,
        verbose=verbose,
        terminal_log_file=terminal_log_file,
        log_prefix="[MIP-2L]",
    )

    L_pre1, L_pre2 = L_pre_list
    U_pre1, U_pre2 = U_pre_list
    L_u1, L_u2 = L_u_list
    U_u1, U_u2 = U_u_list
    L_a1, L_a2 = L_a_list
    U_a1, U_a2 = U_a_list
    L_h1, L_h2 = L_h_list
    U_h1, U_h2 = U_h_list

    # ========== Create Gurobi model ==========
    model = gp.Model("GINNodeGNN_2L_MILP")
    _configure_gurobi_logging(model, verbose=verbose, log_file=log_file)
    model.Params.Seed = int(seed)
    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if use_anderson_cuts:
        model.Params.PreCrush = 1

    # ========== Edge selection variables z ==========
    z: Dict[Tuple[int, int], object] = {}
    for i, j in edges:
        z[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
    model.update()

    removal_z = _add_complement_edge_variables(model, z, edges) if connectivity_on_complement else None
    explanation_z = removal_z if removal_z is not None else z
    connectivity_edge_budget = (
        max(0, len(edges) - int(edge_budget))
        if connectivity_on_complement and edge_budget is not None
        else edge_budget
    )

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    # ========== Symmetry Breaking ==========
    # Create y variables early if symmetry breaking is enabled (needed for node symmetry breaking)
    # Bridge constraints reuse these variables if also enabled
    y: Optional[Dict[int, "gp.Var"]] = None
    symmetric_groups: Optional[Dict[int, List[List[int]]]] = None
    if use_symmetry_breaking:
        symmetric_groups = find_symmetric_branch_groups(adj, X, edges)
        # Edge-level symmetry breaking
        add_symmetry_breaking_constraints(
            model,
            explanation_z,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        # Create y variables for node-level symmetry breaking
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
        model.update()
        # y[i] <= sum of incident z edges
        for i in range(n):
            incident_edges = [z_ij(i, j) for j in adj.neighbors[i]]
            if incident_edges:
                model.addConstr(y[i] <= gp.quicksum(incident_edges), name=f"y_upper_{i}")
            else:
                model.addConstr(y[i] == 0, name=f"y_isolated_{i}")
        # Node-level symmetry breaking
        add_symmetry_breaking_constraints_nodes(
            model,
            y,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Forced Edges Constraints ==========
    if use_forced_edges:
        add_forced_edge_constraints(
            model,
            z,
            forced_edges,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Hierarchical Bridge Constraints ==========
    if use_hierarchical_bridge:
        adj_list: List[List[int]] = [[] for _ in range(n)]
        for (a, b) in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        add_hierarchical_bridge_constraints(
            model,
            z,
            n,
            adj_list,
            v_star,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # ========== Bridge Constraints ==========
    # Create y variables if not already created by symmetry breaking
    if use_bridge_constraints or use_aggregated_bridge_constraints:
        if y is None:
            y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
            model.update()
            for i in range(n):
                incident_edges = [z_ij(i, j) for j in adj.neighbors[i]]
                if incident_edges:
                    model.addConstr(y[i] <= gp.quicksum(incident_edges), name=f"y_upper_{i}")
                else:
                    model.addConstr(y[i] == 0, name=f"y_isolated_{i}")

        adj_list: List[List[int]] = [[] for _ in range(n)]
        for (a, b) in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)

        add_bridge_constraints(model, y, z, n, adj_list, v_star,
                               edge_budget=edge_budget,
                               node_budget=node_budget,
                               use_individual=use_bridge_constraints,
                               use_aggregated=use_aggregated_bridge_constraints,
                               verbose=verbose,
                               log_file=log_file,
                               terminal_log_file=terminal_log_file)

    # ========== Layer 1 Variables ==========
    anderson_relus: List[AndersonReluSpec] = []
    p1 = {(i, f): model.addVar(lb=float(L_pre1[i, f]), ub=float(U_pre1[i, f]),
                                name=f"p1_{i}_{f}") for i in range(n) for f in range(d0)}
    u1 = {(i, m): model.addVar(lb=float(L_u1[i, m]), ub=float(U_u1[i, m]),
                                name=f"u1_{i}_{m}") for i in range(n) for m in range(m1)}
    a1 = {(i, m): model.addVar(lb=float(L_a1[i, m]), ub=float(U_a1[i, m]),
                                name=f"a1_{i}_{m}") for i in range(n) for m in range(m1)}
    d1 = {}
    h1 = {(i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]),
                                name=f"h1_{i}_{k}") for i in range(n) for k in range(h)}
    model.update()

    # Layer 1: p1 = alpha1 * X + sum_j z_ij * X_j
    for i in range(n):
        for f in range(d0):
            expr = gp.LinExpr()
            expr.addConstant(float((alpha1 + adj.self_loop_counts[i]) * X[i, f]))
            for j in adj.neighbors[i]:
                expr.addTerms(float(X[j, f]), z_ij(i, j))
            model.addConstr(p1[(i, f)] == expr, name=f"p1def_{i}_{f}")

    # Layer 1: u1 = p1 @ W1_1 + b1_1
    for i in range(n):
        for m in range(m1):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_1[m]))
            for f in range(d0):
                wval = float(weights.W1_1[f, m])
                if wval != 0.0:
                    expr.addTerms(wval, p1[(i, f)])
            model.addConstr(u1[(i, m)] == expr, name=f"u1def_{i}_{m}")

    # Layer 1: ReLU a1 = max(0, u1)
    for i in range(n):
        for m in range(m1):
            lb_u, ub_u = float(L_u1[i, m]), float(U_u1[i, m])
            uvar, avar = u1[(i, m)], a1[(i, m)]
            if ub_u <= 0.0:
                model.addConstr(avar == 0.0, name=f"relu1_off_{i}_{m}")
            elif lb_u >= 0.0:
                model.addConstr(avar == uvar, name=f"relu1_on_{i}_{m}")
            else:
                dvar = model.addVar(vtype=GRB.BINARY, name=f"d1_{i}_{m}")
                d1[(i, m)] = dvar
                model.addConstr(avar >= uvar, name=f"relu1_1_{i}_{m}")
                model.addConstr(avar >= 0.0, name=f"relu1_2_{i}_{m}")
                model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu1_3_{i}_{m}")
                model.addConstr(avar <= ub_u * dvar, name=f"relu1_4_{i}_{m}")
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p1[(i, f)] for f in range(d0)),
                            y_var=avar,
                            delta_var=dvar,
                            w=np.asarray(weights.W1_1[:, m], dtype=float).copy(),
                            L=np.asarray(L_pre1[i, :], dtype=float).copy(),
                            U=np.asarray(U_pre1[i, :], dtype=float).copy(),
                            b=float(weights.b1_1[m]),
                            tag=f"relu1_{i}_{m}",
                        )
                    )

    # Layer 1: h1 = a1 @ W1_2 + b1_2
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_2[k]))
            for m in range(m1):
                wval = float(weights.W1_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a1[(i, m)])
            model.addConstr(h1[(i, k)] == expr, name=f"h1def_{i}_{k}")

    model.update()

    # ========== Layer 2 Variables ==========
    t2 = {}
    for (u, v) in edges:
        for k in range(h):
            lb_v = float(min(0.0, L_h1[v, k]))
            ub_v = float(max(0.0, U_h1[v, k]))
            t2[(u, v, k)] = model.addVar(lb=lb_v, ub=ub_v, name=f"t2_{u}_{v}_{k}")
            lb_u = float(min(0.0, L_h1[u, k]))
            ub_u = float(max(0.0, U_h1[u, k]))
            t2[(v, u, k)] = model.addVar(lb=lb_u, ub=ub_u, name=f"t2_{v}_{u}_{k}")

    p2 = {(i, k): model.addVar(lb=float(L_pre2[i, k]), ub=float(U_pre2[i, k]),
                                name=f"p2_{i}_{k}") for i in range(n) for k in range(h)}
    u2 = {(i, m): model.addVar(lb=float(L_u2[i, m]), ub=float(U_u2[i, m]),
                                name=f"u2_{i}_{m}") for i in range(n) for m in range(m2)}
    a2 = {(i, m): model.addVar(lb=float(L_a2[i, m]), ub=float(U_a2[i, m]),
                                name=f"a2_{i}_{m}") for i in range(n) for m in range(m2)}
    d2 = {}
    h2 = {(i, k): model.addVar(lb=float(L_h2[i, k]), ub=float(U_h2[i, k]),
                                name=f"h2_{i}_{k}") for i in range(n) for k in range(h)}
    model.update()

    # Layer 2: McCormick constraints for t2 = z * h1
    for (u, v) in edges:
        zuv = z[(u, v)]
        for k in range(h):
            hv = h1[(v, k)]
            tv = t2[(u, v, k)]
            L, U = float(L_h1[v, k]), float(U_h1[v, k])
            model.addConstr(tv <= hv - L * (1 - zuv), name=f"t2mc1_{u}_{v}_{k}")
            model.addConstr(tv >= hv - U * (1 - zuv), name=f"t2mc2_{u}_{v}_{k}")
            model.addConstr(tv <= U * zuv, name=f"t2mc3_{u}_{v}_{k}")
            model.addConstr(tv >= L * zuv, name=f"t2mc4_{u}_{v}_{k}")

            hu = h1[(u, k)]
            tu = t2[(v, u, k)]
            L, U = float(L_h1[u, k]), float(U_h1[u, k])
            model.addConstr(tu <= hu - L * (1 - zuv), name=f"t2mc1_{v}_{u}_{k}")
            model.addConstr(tu >= hu - U * (1 - zuv), name=f"t2mc2_{v}_{u}_{k}")
            model.addConstr(tu <= U * zuv, name=f"t2mc3_{v}_{u}_{k}")
            model.addConstr(tu >= L * zuv, name=f"t2mc4_{v}_{u}_{k}")

    # Layer 2: p2 = alpha2 * h1 + sum_j t2
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addTerms(float(alpha2 + adj.self_loop_counts[i]), h1[(i, k)])
            for j in adj.neighbors[i]:
                expr.addTerms(1.0, t2[(i, j, k)])
            model.addConstr(p2[(i, k)] == expr, name=f"p2def_{i}_{k}")

    # Layer 2: u2 = p2 @ W2_1 + b2_1
    for i in range(n):
        for m in range(m2):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_1[m]))
            for k in range(h):
                wval = float(weights.W2_1[k, m])
                if wval != 0.0:
                    expr.addTerms(wval, p2[(i, k)])
            model.addConstr(u2[(i, m)] == expr, name=f"u2def_{i}_{m}")

    # Layer 2: ReLU
    for i in range(n):
        for m in range(m2):
            lb_u, ub_u = float(L_u2[i, m]), float(U_u2[i, m])
            uvar, avar = u2[(i, m)], a2[(i, m)]
            if ub_u <= 0.0:
                model.addConstr(avar == 0.0, name=f"relu2_off_{i}_{m}")
            elif lb_u >= 0.0:
                model.addConstr(avar == uvar, name=f"relu2_on_{i}_{m}")
            else:
                dvar = model.addVar(vtype=GRB.BINARY, name=f"d2_{i}_{m}")
                d2[(i, m)] = dvar
                model.addConstr(avar >= uvar, name=f"relu2_1_{i}_{m}")
                model.addConstr(avar >= 0.0, name=f"relu2_2_{i}_{m}")
                model.addConstr(avar <= uvar - lb_u * (1 - dvar), name=f"relu2_3_{i}_{m}")
                model.addConstr(avar <= ub_u * dvar, name=f"relu2_4_{i}_{m}")
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p2[(i, k)] for k in range(h)),
                            y_var=avar,
                            delta_var=dvar,
                            w=np.asarray(weights.W2_1[:, m], dtype=float).copy(),
                            L=np.asarray(L_pre2[i, :], dtype=float).copy(),
                            U=np.asarray(U_pre2[i, :], dtype=float).copy(),
                            b=float(weights.b2_1[m]),
                            tag=f"relu2_{i}_{m}",
                        )
                    )

    # Layer 2: h2 = a2 @ W2_2 + b2_2
    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_2[k]))
            for m in range(m2):
                wval = float(weights.W2_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a2[(i, m)])
            model.addConstr(h2[(i, k)] == expr, name=f"h2def_{i}_{k}")

    model.update()

    # ========== Output logits Z for v_star ==========
    Z = {c: model.addVar(lb=-GRB.INFINITY, name=f"Z_{c}") for c in range(num_classes)}
    model.update()

    for c in range(num_classes):
        expr = gp.LinExpr()
        expr.addConstant(float(weights.b_out[c]))
        if use_jk_concat:
            for k in range(h):
                wk = float(weights.W_out[c, k])
                if wk != 0.0:
                    expr.addTerms(wk, h1[(v_star, k)])
            for k in range(h):
                wk = float(weights.W_out[c, h + k])
                if wk != 0.0:
                    expr.addTerms(wk, h2[(v_star, k)])
        else:
            for k in range(h):
                wk = float(weights.W_out[c, k])
                if wk != 0.0:
                    expr.addTerms(wk, h2[(v_star, k)])
        model.addConstr(Z[c] == expr, name=f"Zdef_{c}")

    # ========== Margin constraints ==========
    rho_var = _add_margin_constraints(
        model=model,
        Z=Z,
        c_star=c_star,
        num_classes=num_classes,
        bar_m=bar_m,
        objective=objective,
        rho=rho,
    )

    # ========== Edge budget constraints ==========
    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    # ========== Connectivity constraints ==========
    y = None
    if connectivity != "none":
        y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
        model.update()
        model.addConstr(y[v_star] == 1.0, name="root_selected")
        for (i, j), zij in explanation_z.items():
            model.addConstr(zij <= y[i], name=f"edge_node1_{i}_{j}")
            model.addConstr(zij <= y[j], name=f"edge_node2_{i}_{j}")
        if use_connectivity_lower_bound:
            model.addConstr(
                gp.quicksum(explanation_z[e] for e in edges) >= gp.quicksum(y[i] for i in range(n)) - 1.0,
                name="connectivity_lower_bound",
            )

        if use_star_constraints:
            for i in range(n):
                if i == v_star:
                    continue
                incident = [explanation_z[(a, b)] for (a, b) in edges if a == i or b == i]
                if incident:
                    model.addConstr(y[i] <= gp.quicksum(incident), name=f"star_{i}")

        if connectivity == "flow":
            f = {}
            cap = _flow_cap_upper_bound(n, connectivity_edge_budget, node_budget)
            for (i, j) in edges:
                f[(i, j)] = model.addVar(lb=0.0, ub=cap, name=f"f_{i}_{j}")
                f[(j, i)] = model.addVar(lb=0.0, ub=cap, name=f"f_{j}_{i}")
            model.update()

            for i in range(n):
                out_arcs = [f[(i, j)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                in_arcs = [f[(j, i)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                if i == v_star:
                    model.addConstr(
                        gp.quicksum(out_arcs) - gp.quicksum(in_arcs)
                        == gp.quicksum(y[j] for j in range(n) if j != v_star),
                        name=f"flow_root_{i}",
                    )
                else:
                    model.addConstr(
                        gp.quicksum(in_arcs) - gp.quicksum(out_arcs) == y[i],
                        name=f"flow_balance_{i}",
                    )
            for (i, j), zij in explanation_z.items():
                model.addConstr(f[(i, j)] + f[(j, i)] <= cap * zij, name=f"flow_cap_{i}_{j}")

        elif connectivity == "cut":
            model.Params.LazyConstraints = 1

    # ========== Objective ==========
    _set_selection_objective(
        model=model,
        z=z,
        edges=edges,
        objective=objective,
        rho_var=rho_var,
        edge_budget=edge_budget,
        node_budget=node_budget,
    )

    # ========== Warm start ==========
    if initial_solution is not None:
        initial_set = set(_unique_undirected_edges(initial_solution))
        for e, zvar in z.items():
            zvar.Start = 1.0 if e in initial_set else 0.0
        _print_and_log(
            f"[MIP-2L] Warm start with {len(initial_set)} edges",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )

    return MilpBuild(
        model=model,
        z=z,
        y=y,
        edges=edges,
        n=n,
        v_star=v_star,
        tilde_a=adj,
        c_star=c_star,
        rho_var=rho_var,
        use_maxflow_separation=use_maxflow_separation,
        connectivity_z=explanation_z,
        removal_z=removal_z,
        bar_m=bar_m,
        use_nested_cuts=use_nested_cuts,
        use_back_cuts=use_back_cuts,
        max_cuts_per_round=max_cuts_per_round,
        max_root_sep_rounds=max_root_sep_rounds,
        use_anderson_cuts=use_anderson_cuts,
        anderson_max_node_count=anderson_max_node_count,
        anderson_max_cuts=anderson_max_cuts,
        anderson_relus=tuple(anderson_relus),
        log_file=log_file,
        terminal_log_file=terminal_log_file,
    )


# =============================================================================
# Max-Flow / Min-Cut
# =============================================================================

def _maxflow_mincut(
    source: int,
    sink: int,
    n: int,
    neighbors: List[List[int]],
    capacity: Dict[Tuple[int, int], float],
) -> Tuple[float, set]:
    """Compute max-flow and min-cut (source side)."""
    if _HAS_SCIPY_MAXFLOW:
        return _maxflow_mincut_scipy(source, sink, n, neighbors, capacity)
    return _maxflow_mincut_python(source, sink, n, neighbors, capacity)


def _maxflow_mincut_scipy(
    source: int, sink: int, n: int,
    neighbors: List[List[int]], capacity: Dict[Tuple[int, int], float],
) -> Tuple[float, set]:
    SCALE = 1_000_000
    row, col, data = [], [], []
    for (i, j), cap in capacity.items():
        if cap > 1e-9:
            row.append(i)
            col.append(j)
            data.append(int(cap * SCALE + 0.5))
    if not data:
        return 0.0, {source}
    graph = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.int64)
    result = maximum_flow(graph, source, sink)
    max_flow_value = result.flow_value / SCALE
    flow_matrix = result.flow.tocsr()
    reachable = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in neighbors[u]:
            if v in reachable:
                continue
            cap_uv = capacity.get((u, v), 0.0) * SCALE
            flow_uv = flow_matrix[u, v] if u < n and v < n else 0
            residual = cap_uv - flow_uv
            if residual > 0.5:
                reachable.add(v)
                queue.append(v)
    return max_flow_value, reachable


def _maxflow_mincut_python(
    source: int, sink: int, n: int,
    neighbors: List[List[int]], capacity: Dict[Tuple[int, int], float],
) -> Tuple[float, set]:
    res = {k: v for k, v in capacity.items()}

    def bfs_path():
        parent = {source: None}
        visited = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            if u == sink:
                path = []
                cur = sink
                while parent[cur] is not None:
                    prev = parent[cur]
                    path.append((prev, cur))
                    cur = prev
                path.reverse()
                bottleneck = min(res.get((u2, v2), 0.0) for (u2, v2) in path)
                return path, bottleneck
            for v in neighbors[u]:
                if v not in visited and res.get((u, v), 0.0) > 1e-9:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        return None, 0.0

    max_flow = 0.0
    while True:
        path, bottleneck = bfs_path()
        if path is None or bottleneck <= 1e-9:
            break
        max_flow += float(bottleneck)
        for (u, v) in path:
            res[(u, v)] = res.get((u, v), 0.0) - bottleneck
            res[(v, u)] = res.get((v, u), 0.0) + bottleneck

    reachable = {source}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in neighbors[u]:
            if v not in reachable and res.get((u, v), 0.0) > 1e-9:
                reachable.add(v)
                stack.append(v)
    return max_flow, reachable


# =============================================================================
# Connectivity Callback with Nested/Back Cuts
# =============================================================================

def _separate_nested_cuts(
    source: int, sink: int, n: int,
    neighbors: List[List[int]],
    capacity: Dict[Tuple[int, int], float],
    y_t: float,
    max_cuts: int,
    use_back_cuts: bool = False,
) -> List[Tuple[float, set, float]]:
    """
    Perform nested cut separation with heuristic prioritization.

    Returns list of (violation, reachable_set, rhs) tuples, sorted by strength.
    - violation: y_t - flow_val (higher = more violated)
    - rhs: sum of z values on cut edges (lower = stronger cut)

    Strategy:
    1. Compute both source-cut and back-cut initially
    2. Pick the stronger one (smaller rhs) as first cut
    3. Continue with nested cuts from that direction
    4. Add cuts from other direction if budget remains
    """
    all_cuts = []  # List of (violation, reachable, rhs, cut_size)

    def compute_rhs(reachable: set) -> Tuple[float, int]:
        """Compute rhs = sum of z values on cut edges, and cut size."""
        rhs = 0.0
        cut_size = 0
        for i in reachable:
            for j in neighbors[i]:
                if j not in reachable:
                    rhs += capacity.get((i, j), 0.0)
                    cut_size += 1
        return rhs, cut_size

    def nested_from_source(cap: Dict, budget: int) -> List[Tuple[float, set, float, int]]:
        """Generate nested cuts from source side."""
        cuts = []
        current_cap = {k: v for k, v in cap.items()}
        source_set = {source}

        for _ in range(budget):
            flow_val, reachable = _maxflow_mincut(source, sink, n, neighbors, current_cap)
            if sink in reachable:
                break

            rhs, cut_size = compute_rhs(reachable)
            violation = y_t - flow_val
            if violation > 1e-6:
                cuts.append((violation, reachable, rhs, cut_size))

            # Shrink: saturate cut edges
            for i in reachable:
                for j in neighbors[i]:
                    if j not in reachable:
                        current_cap[(i, j)] = 0.0

            # Recompute reachable after shrinking
            new_reachable = {source}
            stack = [source]
            while stack:
                u = stack.pop()
                for v in neighbors[u]:
                    if v not in new_reachable and current_cap.get((u, v), 0) > 1e-9:
                        new_reachable.add(v)
                        stack.append(v)

            if sink in new_reachable or len(new_reachable) <= len(source_set):
                break
            source_set = new_reachable

        return cuts

    def nested_from_sink(cap: Dict, budget: int) -> List[Tuple[float, set, float, int]]:
        """Generate nested cuts from sink side (back cuts)."""
        cuts = []
        current_cap = {k: v for k, v in cap.items()}
        sink_set = {sink}

        for _ in range(budget):
            flow_val, reachable_from_sink = _maxflow_mincut(sink, source, n, neighbors, current_cap)
            if source in reachable_from_sink:
                break

            # Convert to source-side reachable
            source_reachable = set(range(n)) - reachable_from_sink
            source_reachable.add(source)

            rhs, cut_size = compute_rhs(source_reachable)
            violation = y_t - flow_val
            if violation > 1e-6:
                cuts.append((violation, source_reachable, rhs, cut_size))

            # Shrink sink side
            for i in reachable_from_sink:
                for j in neighbors[i]:
                    if j not in reachable_from_sink:
                        current_cap[(i, j)] = 0.0

            new_reachable = {sink}
            stack = [sink]
            while stack:
                u = stack.pop()
                for v in neighbors[u]:
                    if v not in new_reachable and current_cap.get((u, v), 0) > 1e-9:
                        new_reachable.add(v)
                        stack.append(v)

            if source in new_reachable or len(new_reachable) <= len(sink_set):
                break
            sink_set = new_reachable

        return cuts

    if use_back_cuts:
        # Strategy: compute first cut from both sides, pick stronger one first
        source_first = nested_from_source(capacity, 1)
        back_first = nested_from_sink(capacity, 1)

        # Determine which side gives stronger cut (lower rhs)
        source_stronger = True
        if source_first and back_first:
            source_stronger = source_first[0][2] <= back_first[0][2]  # Compare rhs
        elif back_first and not source_first:
            source_stronger = False

        # Allocate budget: stronger side gets more
        if source_stronger:
            source_budget = (max_cuts + 1) // 2
            back_budget = max_cuts - source_budget
            all_cuts.extend(nested_from_source(capacity, source_budget))
            all_cuts.extend(nested_from_sink(capacity, back_budget))
        else:
            back_budget = (max_cuts + 1) // 2
            source_budget = max_cuts - back_budget
            all_cuts.extend(nested_from_sink(capacity, back_budget))
            all_cuts.extend(nested_from_source(capacity, source_budget))
    else:
        # Only source-side cuts
        all_cuts.extend(nested_from_source(capacity, max_cuts))

    # Sort by strength: lower rhs first (stronger cuts)
    all_cuts.sort(key=lambda x: (x[2], -x[0]))  # rhs ascending, violation descending

    # Remove duplicates (same reachable set)
    seen = set()
    unique_cuts = []
    for viol, reach, rhs, _ in all_cuts:
        key = frozenset(reach)
        if key not in seen:
            seen.add(key)
            unique_cuts.append((viol, reach, rhs))

    return unique_cuts[:max_cuts]


def _anderson_separator_indices(
    w: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    x_hat: np.ndarray,
    delta_hat: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Proposition 13 separator for ideal ReLU cuts.

    Returns boolean mask for I* = {r : w_r (x_r - L~_r) < w_r U~_r delta}.
    """
    wv = np.asarray(w, dtype=float).reshape(-1)
    Lv = np.asarray(L, dtype=float).reshape(-1)
    Uv = np.asarray(U, dtype=float).reshape(-1)
    xv = np.asarray(x_hat, dtype=float).reshape(-1)
    if not (wv.size == Lv.size == Uv.size == xv.size):
        raise ValueError("Inconsistent dimensions in Anderson separator.")

    positive = wv >= 0.0
    breve_L = np.where(positive, Lv, Uv)
    breve_U = np.where(positive, Uv, Lv)
    lhs = wv * (xv - breve_L)
    rhs = wv * breve_U * float(delta_hat)
    return lhs < (rhs - float(tol))


def _evaluate_anderson_rhs(
    w: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    b: float,
    I_mask: np.ndarray,
    x_hat: np.ndarray,
    delta_hat: float,
) -> float:
    """Evaluate Anderson RHS at a point (x_hat, delta_hat)."""
    wv = np.asarray(w, dtype=float).reshape(-1)
    Lv = np.asarray(L, dtype=float).reshape(-1)
    Uv = np.asarray(U, dtype=float).reshape(-1)
    xv = np.asarray(x_hat, dtype=float).reshape(-1)
    Iv = np.asarray(I_mask, dtype=bool).reshape(-1)
    positive = wv >= 0.0
    breve_L = np.where(positive, Lv, Uv)
    breve_U = np.where(positive, Uv, Lv)

    rhs = 0.0
    if np.any(Iv):
        rhs += float(np.dot(wv[Iv], xv[Iv] - breve_L[Iv] * (1.0 - float(delta_hat))))
    if np.any(~Iv):
        rhs += float((float(b) + np.dot(wv[~Iv], breve_U[~Iv])) * float(delta_hat))
    else:
        rhs += float(b) * float(delta_hat)
    return rhs


def _build_anderson_rhs_expr(gp, spec: AndersonReluSpec, I_mask: np.ndarray):
    """Build linear RHS expression for one Anderson cut (fixed I mask)."""
    expr = gp.LinExpr()
    wv = np.asarray(spec.w, dtype=float).reshape(-1)
    Lv = np.asarray(spec.L, dtype=float).reshape(-1)
    Uv = np.asarray(spec.U, dtype=float).reshape(-1)
    Iv = np.asarray(I_mask, dtype=bool).reshape(-1)
    positive = wv >= 0.0
    breve_L = np.where(positive, Lv, Uv)
    breve_U = np.where(positive, Uv, Lv)

    delta_coeff = float(spec.b)
    const_term = 0.0
    for r, x_var in enumerate(spec.x_vars):
        wr = float(wv[r])
        if Iv[r]:
            expr.addTerms(wr, x_var)
            const_term -= wr * float(breve_L[r])
            delta_coeff += wr * float(breve_L[r])
        else:
            delta_coeff += wr * float(breve_U[r])

    if delta_coeff != 0.0:
        expr.addTerms(delta_coeff, spec.delta_var)
    if const_term != 0.0:
        expr.addConstant(const_term)
    return expr


def _cut_connectivity_callback(build: MilpBuild, use_maxflow: bool = False):
    """
    Create callback for:
    1) connectivity lazy/user cuts
    2) Anderson ReLU user cuts at MIPNODE
    """
    import time as _time
    gp, GRB = _require_gurobi()

    z = build.connectivity_z if build.connectivity_z is not None else build.z
    y = build.y
    v_star = build.v_star
    adj = build.tilde_a
    n = adj.n

    use_nested_cuts = getattr(build, 'use_nested_cuts', False)
    use_back_cuts = getattr(build, 'use_back_cuts', False)
    max_cuts_per_round = getattr(build, 'max_cuts_per_round', 50)
    max_root_sep_rounds = getattr(build, 'max_root_sep_rounds', 1)
    use_anderson_cuts = getattr(build, 'use_anderson_cuts', False)
    anderson_max_node_count = int(getattr(build, 'anderson_max_node_count', 0))
    anderson_max_cuts = int(getattr(build, 'anderson_max_cuts', 50))
    anderson_relus = tuple(getattr(build, 'anderson_relus', ()))
    separation_log_file = getattr(build, 'terminal_log_file', None)

    if y is None and not use_anderson_cuts:
        raise ValueError("Cut callback requires y variables")

    flow_neighbors = [list(adj.neighbors[i]) for i in range(n)] if y is not None else [[] for _ in range(n)]
    anderson_max_node_count = max(0, anderson_max_node_count)
    anderson_max_cuts = max(1, anderson_max_cuts)

    # Mutable state for post-solve metrics extraction
    root_sep_state = {
        'round_count': 0,
        'total_cuts': 0,
        'cuts_mipnode': 0,
        'cuts_mipsol_lazy': 0,
        'anderson_cuts_mipnode': 0,
        'anderson_round_count': 0,
        'anderson_done': False,
        'early_terminated': False,
        'round_stats': []  # List of (round_num, cuts_added, time_s)
    }
    setattr(build.model, "_cut_sep_state", root_sep_state)

    def _separate_connectivity(model, y_sol, z_sol, is_mipsol: bool) -> int:
        """Run connectivity separation and return number of added cuts."""
        if y is None:
            return 0

        cuts_added = 0

        selected_nodes = [i for i, val in y_sol.items() if val > 0.01 and i != v_star]
        if not selected_nodes:
            return 0

        def _build_cut_lhs_and_rhs(reachable_set: set):
            cut_edges_expr = []
            rhs_val = 0.0
            for (i, j) in build.edges:
                if (i in reachable_set) != (j in reachable_set):
                    cut_edges_expr.append(z[(i, j)])
                    rhs_val += float(z_sol.get((i, j), 0.0))
            return cut_edges_expr, rhs_val

        if use_maxflow:
            capacity = {}
            for (i, j), zval in z_sol.items():
                if zval > 1e-6:
                    capacity[(i, j)] = zval
                    capacity[(j, i)] = zval

            terminal_violations = []
            for t in selected_nodes:
                if y_sol[t] < 0.01:
                    continue
                flow_val, _ = _maxflow_mincut(v_star, t, n, flow_neighbors, capacity)
                violation = y_sol[t] - flow_val
                terminal_violations.append((violation, y_sol[t], t))

            terminal_violations.sort(key=lambda x: (-x[0], -x[1]))
            sorted_terminals = [t for (viol, _, t) in terminal_violations if viol > 1e-6]

            for t in sorted_terminals:
                if cuts_added >= max_cuts_per_round:
                    break

                if use_nested_cuts:
                    nested_cuts = _separate_nested_cuts(
                        v_star, t, n, flow_neighbors, capacity,
                        y_t=y_sol[t],
                        max_cuts=max(1, max_cuts_per_round // max(1, len(sorted_terminals))),
                        use_back_cuts=use_back_cuts
                    )
                    for viol, reachable, _ in nested_cuts:
                        if viol <= 1e-6:
                            continue
                        if (v_star not in reachable) or (t in reachable):
                            continue
                        cut_edges, rhs_now = _build_cut_lhs_and_rhs(reachable)
                        if not cut_edges:
                            continue
                        if y_sol[t] <= rhs_now + 1e-6:
                            continue
                        if is_mipsol:
                            model.cbLazy(y[t] <= gp.quicksum(cut_edges))
                        else:
                            model.cbCut(y[t] <= gp.quicksum(cut_edges))
                        cuts_added += 1
                        if cuts_added >= max_cuts_per_round:
                            break
                else:
                    flow_val, reachable = _maxflow_mincut(v_star, t, n, flow_neighbors, capacity)
                    if flow_val >= y_sol[t] - 1e-6:
                        continue

                    if use_back_cuts:
                        back_flow, back_reach_sink = _maxflow_mincut(t, v_star, n, flow_neighbors, capacity)
                        back_reachable = set(range(n)) - back_reach_sink
                        back_reachable.add(v_star)
                        fwd_rhs = sum(
                            capacity.get((i, j), 0.0)
                            for i in reachable
                            for j in flow_neighbors[i]
                            if j not in reachable
                        )
                        back_rhs = sum(
                            capacity.get((i, j), 0.0)
                            for i in back_reachable
                            for j in flow_neighbors[i]
                            if j not in back_reachable
                        )
                        if back_rhs < fwd_rhs and back_flow < y_sol[t] - 1e-6:
                            reachable = back_reachable

                    if (v_star not in reachable) or (t in reachable):
                        continue
                    cut_edges, rhs_now = _build_cut_lhs_and_rhs(reachable)
                    if not cut_edges:
                        continue
                    if y_sol[t] <= rhs_now + 1e-6:
                        continue
                    if is_mipsol:
                        model.cbLazy(y[t] <= gp.quicksum(cut_edges))
                    else:
                        model.cbCut(y[t] <= gp.quicksum(cut_edges))
                    cuts_added += 1
        else:
            adj_sol = [[] for _ in range(n)]
            for (i, j), zval in z_sol.items():
                if zval > 0.5:
                    adj_sol[i].append(j)
                    adj_sol[j].append(i)

            visited = {v_star}
            stack = [v_star]
            while stack:
                u = stack.pop()
                for vtx in adj_sol[u]:
                    if vtx not in visited:
                        visited.add(vtx)
                        stack.append(vtx)

            # Add one lazy cut per selected component disconnected from v_star.
            # The representative is deterministic: the minimum selected node ID
            # in that component. This avoids adding one near-duplicate cut per
            # terminal while still rejecting every disconnected component in the
            # incumbent. max_cuts_per_round intentionally does not apply here.
            processed = set(visited)
            for seed in sorted(selected_nodes):
                if y_sol[seed] <= 0.5 or seed in processed:
                    continue

                component = {seed}
                stack = [seed]
                processed.add(seed)
                while stack:
                    u = stack.pop()
                    for vtx in adj_sol[u]:
                        if vtx in visited or vtx in processed:
                            continue
                        processed.add(vtx)
                        component.add(vtx)
                        stack.append(vtx)

                representative = min(v for v in component if y_sol[v] > 0.5)

                cut_edges = [
                    z[(i, j)]
                    for (i, j) in build.edges
                    if (i in component) != (j in component)
                ]
                if cut_edges:
                    model.cbLazy(y[representative] <= gp.quicksum(cut_edges))
                else:
                    model.cbLazy(y[representative] <= 0.0)
                cuts_added += 1

        return cuts_added

    def _separate_anderson(model, nodecnt: float) -> int:
        """Run Anderson separation at current MIPNODE point."""
        if not use_anderson_cuts or not anderson_relus:
            return 0
        cuts_added = 0
        for spec in anderson_relus:
            if cuts_added >= anderson_max_cuts:
                break
            try:
                x_hat = np.array([float(model.cbGetNodeRel(vx)) for vx in spec.x_vars], dtype=float)
                y_hat = float(model.cbGetNodeRel(spec.y_var))
                delta_hat = float(model.cbGetNodeRel(spec.delta_var))
            except Exception:
                continue

            I_star = _anderson_separator_indices(spec.w, spec.L, spec.U, x_hat, delta_hat)
            rhs_hat = _evaluate_anderson_rhs(spec.w, spec.L, spec.U, spec.b, I_star, x_hat, delta_hat)
            violation = y_hat - rhs_hat
            if violation <= 1e-4:
                continue
            rhs_expr = _build_anderson_rhs_expr(gp, spec, I_star)
            model.cbCut(spec.y_var <= rhs_expr)
            cuts_added += 1
            anderson_msg = (
                "[AndersonCut] added "
                f"nodecnt={nodecnt:.0f} tag={spec.tag} "
                f"violation={violation:.6e} "
                f"y={y_hat:.6e} rhs={rhs_hat:.6e}"
            )
            print(anderson_msg)
            _append_log_line(separation_log_file, anderson_msg)
        return cuts_added

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if y is None:
                return
            y_sol = {i: model.cbGetSolution(y[i]) for i in range(n)}
            z_sol = {e: model.cbGetSolution(z[e]) for e in build.edges}
            cuts_added = _separate_connectivity(model, y_sol, z_sol, is_mipsol=True)
            root_sep_state['cuts_mipsol_lazy'] += cuts_added
            return

        if where != GRB.Callback.MIPNODE:
            return

        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return

        nodecnt = float(model.cbGet(GRB.Callback.MIPNODE_NODCNT))

        # Connectivity separation at root LP rounds
        if y is not None and use_maxflow and nodecnt <= 0.0:
            if not root_sep_state['early_terminated'] and root_sep_state['round_count'] < max_root_sep_rounds:
                round_start_time = _time.time()
                root_sep_state['round_count'] += 1
                current_round = root_sep_state['round_count']
                y_sol = {i: model.cbGetNodeRel(y[i]) for i in range(n)}
                z_sol = {e: model.cbGetNodeRel(z[e]) for e in build.edges}
                selected_nodes = [i for i, val in y_sol.items() if val > 0.01 and i != v_star]
                if not selected_nodes:
                    round_time = _time.time() - round_start_time
                    root_sep_state['round_stats'].append((current_round, 0, round_time))
                    root_sep_state['early_terminated'] = True
                    _log_separation_stats(separation_log_file, root_sep_state, "early_term_no_terminals")
                else:
                    cuts_added = _separate_connectivity(model, y_sol, z_sol, is_mipsol=False)
                    round_time = _time.time() - round_start_time
                    root_sep_state['round_stats'].append((current_round, cuts_added, round_time))
                    root_sep_state['total_cuts'] += cuts_added
                    root_sep_state['cuts_mipnode'] += cuts_added
                    if cuts_added == 0:
                        root_sep_state['early_terminated'] = True
                        _log_separation_stats(separation_log_file, root_sep_state, "early_term_no_cuts")
                    elif current_round >= max_root_sep_rounds:
                        _log_separation_stats(separation_log_file, root_sep_state, "max_rounds_reached")

        # Anderson user cuts: run at most one separation pass, at the first eligible MIPNODE.
        if (
            use_anderson_cuts
            and (not root_sep_state['anderson_done'])
            and nodecnt <= float(anderson_max_node_count) + 1e-9
        ):
            root_sep_state['anderson_done'] = True
            root_sep_state['anderson_round_count'] += 1
            anderson_added = _separate_anderson(model, nodecnt)
            if anderson_added > 0:
                root_sep_state['total_cuts'] += anderson_added
                root_sep_state['cuts_mipnode'] += anderson_added
                root_sep_state['anderson_cuts_mipnode'] += anderson_added
                anderson_summary = (
                    f"[AndersonCut] summary round={root_sep_state['anderson_round_count']} "
                    f"nodecnt={nodecnt:.0f} cuts_added={anderson_added}"
                )
                print(anderson_summary)
                _append_log_line(separation_log_file, anderson_summary)

    return callback


def _log_separation_stats(log_file: Optional[str], state: dict, reason: str) -> None:
    """Write separation statistics to log file."""
    if log_file is None:
        return
    try:
        with open(log_file, 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("CUT SEPARATION STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Termination reason: {reason}\n")
            f.write(f"Total rounds: {state['round_count']}\n")
            f.write(f"Total cuts added: {state['total_cuts']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Round':>6} {'Cuts':>8} {'Time (s)':>12}\n")
            f.write("-" * 40 + "\n")
            for round_num, cuts, time_s in state['round_stats']:
                f.write(f"{round_num:>6} {cuts:>8} {time_s:>12.4f}\n")
            f.write("-" * 40 + "\n")
            total_time = sum(t for _, _, t in state['round_stats'])
            f.write(f"{'Total':>6} {state['total_cuts']:>8} {total_time:>12.4f}\n")
            f.write("=" * 60 + "\n\n")
    except Exception:
        pass  # Silently ignore logging errors


# =============================================================================
# Optimize and Extract Solution
# =============================================================================

def _select_edges_from_z_values(
    z_values: Dict[Tuple[int, int], float],
    *,
    relax_edge_mask: bool = False,
    selection_edge_budget: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Extract selected edges from z-values.

    Default behavior matches the original MILP pipeline (`z > 0.5`).
    In relaxed-edge mode, select the top-K edges ranked by fractional z-value.
    """
    if not relax_edge_mask:
        return [e for e, v in z_values.items() if v > 0.5]

    if selection_edge_budget is None:
        raise ValueError("relax_edge_mask requires an explicit edge budget for top-K extraction.")

    k = max(0, min(int(selection_edge_budget), len(z_values)))
    if k == 0:
        return []

    ranked_edges = sorted(
        z_values.items(),
        key=lambda item: (-float(item[1]), item[0][0], item[0][1]),
    )
    return [edge for edge, _ in ranked_edges[:k]]


def optimize_and_extract(
    build: MilpBuild,
    *,
    relax_edge_mask: bool = False,
    selection_edge_budget: Optional[int] = None,
) -> MilpSolution:
    """Optimize MILP and extract solution."""
    import time as _time
    gp, GRB = _require_gurobi()
    model = build.model

    def _safe_float(value) -> Optional[float]:
        try:
            x = float(value)
        except Exception:
            return None
        return x if np.isfinite(x) else None

    root_stats = {
        "captured": False,
        "root_lp_value": None,
        "root_lp_time": None,
    }
    callback_stats = {
        "calls": 0,
        "time_sec": 0.0,
    }

    def _capture_root_stats(cb_model, where):
        if where != GRB.Callback.MIPNODE:
            return
        if root_stats["captured"]:
            return
        nodecnt = cb_model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecnt > 0:
            return
        status = cb_model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return
        bound = cb_model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
        if np.isfinite(bound):
            root_stats["root_lp_value"] = float(bound)
        if np.isfinite(runtime):
            root_stats["root_lp_time"] = float(runtime)
        root_stats["captured"] = True

    use_lazy = getattr(model.Params, "LazyConstraints", 0) == 1
    use_anderson = bool(getattr(build, "use_anderson_cuts", False)) and len(getattr(build, "anderson_relus", ())) > 0
    if use_lazy or use_anderson:
        cut_callback = _cut_connectivity_callback(build, use_maxflow=build.use_maxflow_separation)

        def callback(cb_model, where):
            t0 = _time.perf_counter()
            callback_stats["calls"] += 1
            _capture_root_stats(cb_model, where)
            cut_callback(cb_model, where)
            callback_stats["time_sec"] += (_time.perf_counter() - t0)

        interrupted_by_user = _optimize_gurobi_model(
            model,
            callback=callback,
            terminal_log_file=getattr(build, "terminal_log_file", None),
        )
    else:
        def callback(cb_model, where):
            t0 = _time.perf_counter()
            callback_stats["calls"] += 1
            _capture_root_stats(cb_model, where)
            callback_stats["time_sec"] += (_time.perf_counter() - t0)

        interrupted_by_user = _optimize_gurobi_model(
            model,
            callback=callback,
            terminal_log_file=getattr(build, "terminal_log_file", None),
        )

    status_code = int(model.Status)
    if interrupted_by_user or status_code == getattr(GRB, "INTERRUPTED", 11):
        status = "interrupted_by_user"
    elif status_code == GRB.OPTIMAL:
        status = "optimal"
    elif status_code == GRB.TIME_LIMIT:
        status = "time_limit"
    elif status_code == GRB.INFEASIBLE:
        status = "infeasible"
    else:
        status = f"status_{status_code}"

    sep_state = getattr(model, "_cut_sep_state", None)
    cuts_mipnode = int(sep_state.get("cuts_mipnode", 0)) if isinstance(sep_state, dict) else 0
    cuts_mipsol_lazy = int(sep_state.get("cuts_mipsol_lazy", 0)) if isinstance(sep_state, dict) else 0
    anderson_cuts_added = (
        int(sep_state.get("anderson_cuts_mipnode", 0))
        if isinstance(sep_state, dict)
        else 0
    )
    connectivity_lazy_cuts_added = cuts_mipsol_lazy
    best_bound = _safe_float(getattr(model, "ObjBound", None))
    incumbent_obj = _safe_float(getattr(model, "ObjVal", None)) if getattr(model, "SolCount", 0) > 0 else None
    final_mip_gap = _safe_float(getattr(model, "MIPGap", None))
    model_size = _safe_float(getattr(model, "NumNZs", None))

    if getattr(model, "SolCount", 0) <= 0:
        return MilpSolution(
            status=status,
            objective_value=None,
            selected_edges=[],
            z_values={},
            bar_m=build.bar_m,
            v_star=build.v_star,
            c_star=build.c_star,
            final_mip_gap=final_mip_gap,
            best_bound=best_bound,
            incumbent_obj=incumbent_obj,
            model_size=model_size,
            root_lp_value=root_stats["root_lp_value"],
            root_lp_time=root_stats["root_lp_time"],
            callback_time_sec=float(callback_stats["time_sec"]),
            callback_calls=int(callback_stats["calls"]),
            cuts_mipnode=cuts_mipnode,
            cuts_mipsol_lazy=cuts_mipsol_lazy,
            anderson_cuts_added=anderson_cuts_added,
            connectivity_lazy_cuts_added=connectivity_lazy_cuts_added,
            bnb_node_count=float(getattr(model, "NodeCount", 0.0)),
        )

    z_values = {e: float(var.X) for e, var in build.z.items()}
    selected_edges = _select_edges_from_z_values(
        z_values,
        relax_edge_mask=relax_edge_mask,
        selection_edge_budget=selection_edge_budget,
    )
    obj = _safe_float(getattr(model, "ObjVal", None))

    rho_optimal = None
    if build.rho_var is not None and getattr(model, "SolCount", 0) > 0:
        rho_optimal = float(build.rho_var.X)

    return MilpSolution(
        status=status,
        objective_value=obj,
        selected_edges=selected_edges,
        z_values=z_values,
        rho_optimal=rho_optimal,
        bar_m=build.bar_m,
        v_star=build.v_star,
        c_star=build.c_star,
        final_mip_gap=final_mip_gap,
        best_bound=best_bound,
        incumbent_obj=incumbent_obj,
        model_size=model_size,
        root_lp_value=root_stats["root_lp_value"],
        root_lp_time=root_stats["root_lp_time"],
        callback_time_sec=float(callback_stats["time_sec"]),
        callback_calls=int(callback_stats["calls"]),
        cuts_mipnode=cuts_mipnode,
        cuts_mipsol_lazy=cuts_mipsol_lazy,
        anderson_cuts_added=anderson_cuts_added,
        connectivity_lazy_cuts_added=connectivity_lazy_cuts_added,
        bnb_node_count=float(getattr(model, "NodeCount", 0.0)),
    )


# =============================================================================
# Graph Classification MILP
# =============================================================================

def build_gin_graphgnn_milp_2l(
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    weights: GINGraphGnnWeights2L,
    *,
    readout: str = "sum",
    rho: float = 0.0,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    min_edges: Optional[int] = None,
    objective: ObjectiveMode = "min_edges",
    connectivity: ConnectivityMode = "none",
    connectivity_on_complement: bool = False,
    use_incident_cap_tightening: bool = False,
    cap_mode: CapMode = "global",
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
    defer_obbt_u: bool = False,
    u_bound_overrides: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    verbose: bool = False,
    time_limit_s: Optional[float] = None,
    wallclock_deadline_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    seed: int = 0,
    initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
) -> GraphMilpBuild:
    """Build MILP for 2-layer GIN GraphGNN explanation."""
    _validate_objective_mode(objective, rho)
    if connectivity not in ("none", "flow", "cut"):
        raise ValueError(
            "Graph classification currently supports connectivity in {'none', 'flow', 'cut'} only."
        )
    if cap_mode == "distance":
        raise ValueError("cap_mode='distance' is only supported for node classification (requires v_star)")

    gp, GRB = _require_gurobi()

    X = np.asarray(X, dtype=float)
    n, d0 = X.shape
    h = weights.hidden_dim
    m1, m2 = weights.mlp_hidden1, weights.mlp_hidden2
    num_classes = weights.num_classes

    adj = compute_adjacency(n, undirected_edges)
    edges = _unique_undirected_edges(undirected_edges)

    bar_Z = forward_gin_graphgnn_full_2l(X, adj, weights, readout=readout)
    c_star = int(np.argmax(bar_Z))
    bar_m = float(bar_Z[c_star] - np.max(np.delete(bar_Z, c_star))) if num_classes >= 2 else 0.0

    _print_and_log(
        f"[MIP-Graph-2L] Full graph: class={c_star}, margin={bar_m:.4f}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    _print_and_log(
        f"[MIP-Graph-2L] Graph: {n} nodes, {len(edges)} edges, readout={readout}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )

    if use_incident_cap_tightening:
        kappa = compute_incident_caps(
            adj,
            edge_budget,
            cap_mode="global",
            v_star=None,
            node_budget=node_budget,
        )
        _print_and_log(
            "[MIP-Graph-2L] Top-k tightening enabled, cap_mode=global",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIP-Graph-2L] kappa range: [{kappa.min()}, {kappa.max()}]",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        kappa = None

    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)

    L_u1, U_u1 = _u1_bounds_direct(
        adj,
        X,
        weights.W1_1,
        weights.b1_1,
        alpha1,
        kappa if use_incident_cap_tightening else None,
    )
    L_u1, U_u1 = _intersect_u_bound_override(
        L_u1, U_u1, layer=1, u_bound_overrides=u_bound_overrides
    )
    L_a1, U_a1 = np.maximum(L_u1, 0.0), np.maximum(U_u1, 0.0)
    L_h1, U_h1 = _linear_bounds(L_a1, U_a1, weights.W1_2, weights.b1_2)

    if use_incident_cap_tightening and kappa is not None:
        L_pre2, U_pre2 = _pre_bounds_selectable_topk(adj, L_h1, U_h1, alpha2, kappa)
    else:
        L_pre2, U_pre2 = _pre_bounds_selectable(adj, L_h1, U_h1, alpha2)
    L_u2, U_u2 = _linear_bounds(L_pre2, U_pre2, weights.W2_1, weights.b2_1)
    L_u2, U_u2 = _intersect_u_bound_override(
        L_u2, U_u2, layer=2, u_bound_overrides=u_bound_overrides
    )
    L_a2, U_a2 = np.maximum(L_u2, 0.0), np.maximum(U_u2, 0.0)
    L_h2, U_h2 = _linear_bounds(L_a2, U_a2, weights.W2_2, weights.b2_2)

    model = gp.Model("GraphGNN_2L_MILP")
    _configure_gurobi_logging(model, verbose=verbose, log_file=log_file)
    model.Params.Seed = int(seed)
    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if use_anderson_cuts:
        model.Params.PreCrush = 1

    z = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}") for (i, j) in edges}
    _set_edge_branch_priority(z, edge_branch_priority)
    model.update()
    removal_z = _add_complement_edge_variables(model, z, edges) if connectivity_on_complement else None
    explanation_z = removal_z if removal_z is not None else z
    connectivity_edge_budget = (
        max(0, len(edges) - int(edge_budget))
        if connectivity_on_complement and edge_budget is not None
        else edge_budget
    )

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    symmetric_groups: Optional[Dict[int, List[List[int]]]] = None
    if use_symmetry_breaking:
        symmetric_groups = find_symmetric_branch_groups(adj, X, edges)
        add_symmetry_breaking_constraints(
            model,
            explanation_z,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
    model.update()

    incident_by_node: Dict[int, List[object]] = {i: [] for i in range(n)}
    for (i, j), zij in explanation_z.items():
        incident_by_node[i].append(zij)
        incident_by_node[j].append(zij)

    r: Optional[Dict[int, object]] = None
    eps_empty: Optional[object] = None
    if connectivity in ("flow", "cut"):
        for (i, j), zij in explanation_z.items():
            model.addConstr(zij <= y[i], name=f"edge_node1_{i}_{j}")
            model.addConstr(zij <= y[j], name=f"edge_node2_{i}_{j}")

        r = {i: model.addVar(vtype=GRB.BINARY, name=f"r_{i}") for i in range(n)}
        eps_empty = model.addVar(vtype=GRB.BINARY, name="eps_empty")
        model.update()
        model.addConstr(eps_empty + gp.quicksum(r[i] for i in range(n)) == 1.0, name="root_or_empty")
        for i in range(n):
            model.addConstr(y[i] <= 1 - eps_empty, name=f"y_empty_{i}")
            model.addConstr(r[i] <= y[i], name=f"root_implies_selected_{i}")
            if use_star_constraints:
                if incident_by_node[i]:
                    model.addConstr(
                        y[i] <= gp.quicksum(incident_by_node[i]) + r[i] + eps_empty,
                        name=f"star_or_root_{i}",
                    )
                else:
                    model.addConstr(y[i] <= r[i] + eps_empty, name=f"isolated_or_root_{i}")
        if use_connectivity_lower_bound:
            model.addConstr(
                gp.quicksum(explanation_z[e] for e in edges) >= gp.quicksum(y[i] for i in range(n)) - 1.0,
                name="connectivity_lower_bound",
            )
        if connectivity == "flow":
            cap = _flow_cap_upper_bound(n, connectivity_edge_budget, node_budget)
            f = {}
            for (i, j) in edges:
                f[(i, j)] = model.addVar(lb=0.0, ub=cap, name=f"f_{i}_{j}")
                f[(j, i)] = model.addVar(lb=0.0, ub=cap, name=f"f_{j}_{i}")
            root_supply = {i: model.addVar(lb=0.0, ub=cap, name=f"s_{i}") for i in range(n)}
            model.update()

            model.addConstr(
                gp.quicksum(root_supply[i] for i in range(n)) == gp.quicksum(y[i] for i in range(n)) - 1.0 + eps_empty,
                name="flow_total_supply",
            )
            for i in range(n):
                model.addConstr(root_supply[i] <= cap * r[i], name=f"flow_supply_root_{i}")
                out_arcs = [f[(i, j)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                in_arcs = [f[(j, i)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                model.addConstr(
                    gp.quicksum(out_arcs) - gp.quicksum(in_arcs) == root_supply[i] - (y[i] - r[i]),
                    name=f"flow_balance_{i}",
                )
            for (i, j), zij in explanation_z.items():
                model.addConstr(f[(i, j)] + f[(j, i)] <= cap * zij, name=f"flow_cap_{i}_{j}")
        else:
            model.Params.LazyConstraints = 1
    else:
        for i in range(n):
            if incident_by_node[i]:
                model.addConstr(y[i] <= gp.quicksum(incident_by_node[i]), name=f"y_upper_{i}")
            else:
                model.addConstr(y[i] == 0, name=f"y_isolated_{i}")

    if use_symmetry_breaking and symmetric_groups:
        add_symmetry_breaking_constraints_nodes(
            model,
            y,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    if use_bridge_constraints:
        _print_and_log(
            "[MIP-Graph-2L] Bridge constraints ignored for graph classification (requires v_star).",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    anderson_relus: List[AndersonReluSpec] = []

    # Layer 1. Since X is fixed, u1 can be expressed directly without p1 variables.
    XW1 = X @ weights.W1_1
    u1 = {
        (i, m): model.addVar(lb=float(L_u1[i, m]), ub=float(U_u1[i, m]), name=f"u1_{i}_{m}")
        for i in range(n)
        for m in range(m1)
    }
    a1 = {
        (i, m): model.addVar(lb=0.0, ub=float(max(U_u1[i, m], 0.0)), name=f"a1_{i}_{m}")
        for i in range(n)
        for m in range(m1)
    }
    d1 = {}
    h1 = {
        (i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]), name=f"h1_{i}_{k}")
        for i in range(n)
        for k in range(h)
    }
    model.update()

    for i in range(n):
        self_coeff = alpha1 + adj.self_loop_counts[i]
        for m in range(m1):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_1[m] + self_coeff * XW1[i, m]))
            for j in adj.neighbors[i]:
                xw = float(XW1[j, m])
                if xw != 0.0:
                    expr.addTerms(xw, z_ij(i, j))
            model.addConstr(u1[(i, m)] == expr, name=f"u1def_{i}_{m}")

    for i in range(n):
        for m in range(m1):
            L_val, U_val = float(L_u1[i, m]), float(U_u1[i, m])
            if U_val <= 0:
                model.addConstr(a1[(i, m)] == 0.0, name=f"relu1_off_{i}_{m}")
            elif L_val >= 0:
                model.addConstr(a1[(i, m)] == u1[(i, m)], name=f"relu1_on_{i}_{m}")
            else:
                d1[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d1_{i}_{m}")
                model.addConstr(a1[(i, m)] >= u1[(i, m)], name=f"relu1_1_{i}_{m}")
                model.addConstr(a1[(i, m)] >= 0.0, name=f"relu1_2_{i}_{m}")
                model.addConstr(a1[(i, m)] <= u1[(i, m)] - L_val * (1 - d1[(i, m)]), name=f"relu1_3_{i}_{m}")
                model.addConstr(a1[(i, m)] <= U_val * d1[(i, m)], name=f"relu1_4_{i}_{m}")
                if use_anderson_cuts:
                    neighbors = list(adj.neighbors[i])
                    self_coeff = alpha1 + adj.self_loop_counts[i]
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(z_ij(i, j) for j in neighbors),
                            y_var=a1[(i, m)],
                            delta_var=d1[(i, m)],
                            w=np.asarray([XW1[j, m] for j in neighbors], dtype=float),
                            L=np.zeros(len(neighbors), dtype=float),
                            U=np.ones(len(neighbors), dtype=float),
                            b=float(weights.b1_1[m] + self_coeff * XW1[i, m]),
                            tag=f"graph2l_relu1_{i}_{m}",
                        )
                    )

    model.update()

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_2[k]))
            for m in range(m1):
                wval = float(weights.W1_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a1[(i, m)])
            model.addConstr(h1[(i, k)] == expr, name=f"h1def_{i}_{k}")

    # Layer 2.
    t2 = {}
    for (u, v) in edges:
        for k in range(h):
            t2[(u, v, k)] = model.addVar(
                lb=float(min(0.0, L_h1[v, k])),
                ub=float(max(0.0, U_h1[v, k])),
                name=f"t2_{u}_{v}_{k}",
            )
            t2[(v, u, k)] = model.addVar(
                lb=float(min(0.0, L_h1[u, k])),
                ub=float(max(0.0, U_h1[u, k])),
                name=f"t2_{v}_{u}_{k}",
            )

    u2 = {
        (i, m): model.addVar(lb=float(L_u2[i, m]), ub=float(U_u2[i, m]), name=f"u2_{i}_{m}")
        for i in range(n)
        for m in range(m2)
    }
    a2 = {
        (i, m): model.addVar(lb=0.0, ub=float(max(U_u2[i, m], 0.0)), name=f"a2_{i}_{m}")
        for i in range(n)
        for m in range(m2)
    }
    d2 = {}
    h2 = {
        (i, k): model.addVar(lb=float(L_h2[i, k]), ub=float(U_h2[i, k]), name=f"h2_{i}_{k}")
        for i in range(n)
        for k in range(h)
    }
    model.update()

    for (u, v) in edges:
        zuv = z[(u, v)]
        for k in range(h):
            hv = h1[(v, k)]
            tv = t2[(u, v, k)]
            L, U = float(L_h1[v, k]), float(U_h1[v, k])
            model.addConstr(tv <= hv - L * (1 - zuv), name=f"t2mc1_{u}_{v}_{k}")
            model.addConstr(tv >= hv - U * (1 - zuv), name=f"t2mc2_{u}_{v}_{k}")
            model.addConstr(tv <= U * zuv, name=f"t2mc3_{u}_{v}_{k}")
            model.addConstr(tv >= L * zuv, name=f"t2mc4_{u}_{v}_{k}")

            hu = h1[(u, k)]
            tu = t2[(v, u, k)]
            L, U = float(L_h1[u, k]), float(U_h1[u, k])
            model.addConstr(tu <= hu - L * (1 - zuv), name=f"t2mc1_{v}_{u}_{k}")
            model.addConstr(tu >= hu - U * (1 - zuv), name=f"t2mc2_{v}_{u}_{k}")
            model.addConstr(tu <= U * zuv, name=f"t2mc3_{v}_{u}_{k}")
            model.addConstr(tu >= L * zuv, name=f"t2mc4_{v}_{u}_{k}")

    for i in range(n):
        self_coeff = alpha2 + adj.self_loop_counts[i]
        for m in range(m2):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_1[m]))
            for k in range(h):
                wval = float(weights.W2_1[k, m])
                if wval == 0.0:
                    continue
                expr.addTerms(wval * self_coeff, h1[(i, k)])
                for j in adj.neighbors[i]:
                    expr.addTerms(wval, t2[(i, j, k)])
            model.addConstr(u2[(i, m)] == expr, name=f"u2def_{i}_{m}")

    for i in range(n):
        for m in range(m2):
            L_val, U_val = float(L_u2[i, m]), float(U_u2[i, m])
            if U_val <= 0:
                model.addConstr(a2[(i, m)] == 0.0, name=f"relu2_off_{i}_{m}")
            elif L_val >= 0:
                model.addConstr(a2[(i, m)] == u2[(i, m)], name=f"relu2_on_{i}_{m}")
            else:
                d2[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d2_{i}_{m}")
                model.addConstr(a2[(i, m)] >= u2[(i, m)], name=f"relu2_1_{i}_{m}")
                model.addConstr(a2[(i, m)] >= 0.0, name=f"relu2_2_{i}_{m}")
                model.addConstr(a2[(i, m)] <= u2[(i, m)] - L_val * (1 - d2[(i, m)]), name=f"relu2_3_{i}_{m}")
                model.addConstr(a2[(i, m)] <= U_val * d2[(i, m)], name=f"relu2_4_{i}_{m}")
                if use_anderson_cuts:
                    x_vars: List[object] = []
                    coeffs: List[float] = []
                    lows: List[float] = []
                    ups: List[float] = []
                    self_coeff = alpha2 + adj.self_loop_counts[i]
                    for k in range(h):
                        wval = float(weights.W2_1[k, m])
                        if wval == 0.0:
                            continue
                        x_vars.append(h1[(i, k)])
                        coeffs.append(wval * self_coeff)
                        lows.append(float(L_h1[i, k]))
                        ups.append(float(U_h1[i, k]))
                        for j in adj.neighbors[i]:
                            x_vars.append(t2[(i, j, k)])
                            coeffs.append(wval)
                            lows.append(float(min(0.0, L_h1[j, k])))
                            ups.append(float(max(0.0, U_h1[j, k])))
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(x_vars),
                            y_var=a2[(i, m)],
                            delta_var=d2[(i, m)],
                            w=np.asarray(coeffs, dtype=float),
                            L=np.asarray(lows, dtype=float),
                            U=np.asarray(ups, dtype=float),
                            b=float(weights.b2_1[m]),
                            tag=f"graph2l_relu2_{i}_{m}",
                        )
                    )

    model.update()

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_2[k]))
            for m in range(m2):
                wval = float(weights.W2_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a2[(i, m)])
            model.addConstr(h2[(i, k)] == expr, name=f"h2def_{i}_{k}")

    pool2 = {k: model.addVar(lb=-GRB.INFINITY, name=f"pool2_{k}") for k in range(h)}
    model.update()

    scale = 1.0 if readout == "sum" else (1.0 / n)
    for k in range(h):
        expr = gp.LinExpr()
        for i in range(n):
            expr.addTerms(scale, h2[(i, k)])
        model.addConstr(pool2[k] == expr, name=f"pool2def_{k}")

    Z = {c: model.addVar(lb=-GRB.INFINITY, name=f"Z_{c}") for c in range(num_classes)}
    model.update()

    for c in range(num_classes):
        expr = gp.LinExpr()
        expr.addConstant(float(weights.b_out[c]))
        for k in range(h):
            wk = float(weights.W_out[c, k])
            if wk != 0.0:
                expr.addTerms(wk, pool2[k])
        model.addConstr(Z[c] == expr, name=f"Zdef_{c}")

    rho_var = _add_margin_constraints(
        model=model,
        Z=Z,
        c_star=c_star,
        num_classes=num_classes,
        bar_m=bar_m,
        objective=objective,
        rho=rho,
    )

    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    root_node = edges[0][0] if edges else 0

    _set_selection_objective(
        model=model,
        z=z,
        edges=edges,
        objective=objective,
        rho_var=rho_var,
        edge_budget=edge_budget,
        node_budget=node_budget,
    )

    if not defer_obbt_u:
        _apply_obbt_u_bounds(
            model,
            use_obbt_u=use_obbt_u,
            max_obbt_u_layers=max_obbt_u_layers,
            available_layers=(1, 2),
            connectivity=connectivity,
            wallclock_deadline_s=wallclock_deadline_s,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
            log_prefix="[MIP-Graph-2L]",
        )

    if initial_solution is not None:
        initial_set = set(_unique_undirected_edges(initial_solution))
        for e, zvar in z.items():
            zvar.Start = 1.0 if e in initial_set else 0.0
        _print_and_log(
            f"[MIP-Graph-2L] Warm start with {len(initial_set)} edges",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )

    return GraphMilpBuild(
        model=model,
        z=z,
        y=y if connectivity in ("flow", "cut") else None,
        r=r if connectivity in ("flow", "cut") else None,
        edges=edges,
        n=n,
        root_node=root_node,
        connectivity=cast(Literal["none", "flow", "cut"], connectivity),
        tilde_a=adj,
        c_star=c_star,
        rho_var=rho_var,
        bar_m=bar_m,
        eps_empty=eps_empty if connectivity in ("flow", "cut") else None,
        connectivity_z=explanation_z if connectivity in ("flow", "cut") else None,
        removal_z=removal_z,
        use_anderson_cuts=use_anderson_cuts,
        anderson_max_node_count=anderson_max_node_count,
        anderson_max_cuts=anderson_max_cuts,
        anderson_relus=tuple(anderson_relus),
        terminal_log_file=terminal_log_file,
    )


def _degree_table_layer1(
    X: np.ndarray,
    adj: Adjacency,
    weights: GINGraphGnnWeights,
    *,
    num_edges: int,
    edge_budget: Optional[int],
    min_edges: Optional[int],
    kappa: Optional[np.ndarray],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact layer-1 tables for identical node features.

    With x_i = x0 for every node, p1_i = (alpha1 + selfloops_i + deg_S(i)) x0, so
    h1_i is a function H1_i(d) of the selected degree d. Returns, per node, the
    feasible degrees and H1 values, the induced h1 bounds, and layer-2
    preactivation bounds p2_i = (alpha2 + selfloops_i) h1_i + sum of h1 over the
    deg_S(i) selected neighbours, each of which has degree at least 1.
    """
    X = np.asarray(X, dtype=float)
    x0 = X[0]
    if not np.array_equal(X, np.broadcast_to(x0, X.shape)):
        raise ValueError("degree_table_layer1 requires identical node features for all nodes.")
    n = X.shape[0]
    h = weights.hidden_dim
    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)
    b_max = num_edges if edge_budget is None else min(int(edge_budget), num_edges)
    b_min = 0 if min_edges is None else int(min_edges)
    tables: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        deg_g = len(adj.neighbors[i])
        d_hi = min(deg_g, b_max)
        if kappa is not None:
            d_hi = min(d_hi, int(kappa[i]))
        d_lo = max(0, deg_g - (num_edges - b_min))
        if d_lo > d_hi:
            raise ValueError(f"Node {i} has no feasible degree under the edge budget.")
        degrees = np.arange(d_lo, d_hi + 1)
        pre = (alpha1 + float(adj.self_loop_counts[i]) + degrees)[:, None] * x0[None, :]
        hidden = np.maximum(pre @ weights.W1_1 + weights.b1_1, 0.0)
        tables.append((degrees, hidden @ weights.W1_2 + weights.b1_2))
    L_h1 = np.stack([H.min(axis=0) for _, H in tables])
    U_h1 = np.stack([H.max(axis=0) for _, H in tables])

    L_pre2 = np.zeros((n, h))
    U_pre2 = np.zeros((n, h))
    for i in range(n):
        degrees_i, H_i = tables[i]
        own = (alpha2 + float(adj.self_loop_counts[i])) * H_i
        # Neighbours that can send a message have a feasible degree >= 1.
        senders = [tables[j][1][tables[j][0] >= 1] for j in adj.neighbors[i] if np.any(tables[j][0] >= 1)]
        feasible = degrees_i <= len(senders)
        if not np.any(feasible):
            raise ValueError(f"Node {i} has no feasible degree given its neighbours.")
        if senders:
            nb_max = np.stack([s.max(axis=0) for s in senders])
            nb_min = np.stack([s.min(axis=0) for s in senders])
        for k in range(h):
            top = np.zeros(1)
            bottom = np.zeros(1)
            if senders:
                top = np.concatenate([[0.0], np.cumsum(np.sort(nb_max[:, k])[::-1])])
                bottom = np.concatenate([[0.0], np.cumsum(np.sort(nb_min[:, k]))])
            idx = np.flatnonzero(feasible)
            U_pre2[i, k] = float(np.max(own[idx, k] + top[degrees_i[idx]]))
            L_pre2[i, k] = float(np.min(own[idx, k] + bottom[degrees_i[idx]]))
    return tables, L_h1, U_h1, L_pre2, U_pre2


def build_gin_graphgnn_milp(
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    weights: GINGraphGnnWeights,
    *,
    readout: str = "sum",
    rho: float = 0.0,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    min_edges: Optional[int] = None,
    objective: ObjectiveMode = "min_edges",
    connectivity: ConnectivityMode = "none",
    connectivity_on_complement: bool = False,
    use_incident_cap_tightening: bool = False,
    cap_mode: CapMode = "global",
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
    defer_obbt_u: bool = False,
    u_bound_overrides: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    verbose: bool = False,
    time_limit_s: Optional[float] = None,
    wallclock_deadline_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    seed: int = 0,
    initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
    use_degree_table_layer1: bool = False,
) -> GraphMilpBuild:
    """
    Build MILP for 3-layer GIN GraphGNN explanation.

    use_degree_table_layer1: for identical node features, represent layer 1
    exactly by one-hot degree variables and the layer-2 messages z_ij h1_j by
    exact binary products, replacing the layer-1 ReLU big-M (see
    ``_degree_table_layer1``).

    Graph classification uses global pooling:
        Z = W_out @ pool(h3) + b_out

    Args:
        connectivity: "none", "flow", or "cut" for graph classification
        use_incident_cap_tightening: Enable Top-κ incident cap tightening
        cap_mode: Only "global" supported for graph classification
        use_symmetry_breaking: Enable symmetric-branch symmetry breaking

    Note: Bridge constraints are only for node classification (requires v_star).
    """
    _validate_objective_mode(objective, rho)
    if connectivity not in ("none", "flow", "cut"):
        raise ValueError(
            "Graph classification currently supports connectivity in {'none', 'flow', 'cut'} only."
        )
    if cap_mode == "distance":
        raise ValueError("cap_mode='distance' is only supported for node classification (requires v_star)")

    gp, GRB = _require_gurobi()

    X = np.asarray(X, dtype=float)
    n, d0 = X.shape
    h = weights.hidden_dim
    m1, m2, m3 = weights.mlp_hidden1, weights.mlp_hidden2, weights.mlp_hidden3
    num_classes = weights.num_classes

    adj = compute_adjacency(n, undirected_edges)
    edges = _unique_undirected_edges(undirected_edges)

    # Full forward pass
    bar_Z = forward_gin_graphgnn_full(X, adj, weights, readout=readout)
    c_star = int(np.argmax(bar_Z))
    bar_m = float(bar_Z[c_star] - np.max(np.delete(bar_Z, c_star))) if num_classes >= 2 else 0.0

    _print_and_log(
        f"[MIP-Graph] Full graph: class={c_star}, margin={bar_m:.4f}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    _print_and_log(
        f"[MIP-Graph] Graph: {n} nodes, {len(edges)} edges, readout={readout}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )

    # Compute incident caps for Top-κ tightening
    if use_incident_cap_tightening:
        kappa = compute_incident_caps(
            adj,
            edge_budget,
            cap_mode="global",
            v_star=None,
            node_budget=node_budget,
        )
        _print_and_log(
            "[MIP-Graph] Top-κ tightening enabled, cap_mode=global",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIP-Graph] kappa range: [{kappa.min()}, {kappa.max()}]",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        kappa = None

    # Bound propagation
    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)
    alpha3 = 1.0 + float(weights.eps3)

    degree_tables: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if use_degree_table_layer1:
        if use_obbt_u or (u_bound_overrides and 1 in u_bound_overrides):
            raise ValueError("use_degree_table_layer1 cannot be combined with layer-1 OBBT bounds.")
        degree_tables, L_h1, U_h1, L_pre2, U_pre2 = _degree_table_layer1(
            X,
            adj,
            weights,
            num_edges=len(edges),
            edge_budget=edge_budget,
            min_edges=min_edges,
            kappa=kappa if use_incident_cap_tightening else None,
        )
        _print_and_log(
            "[MIP-Graph] Exact degree-table layer 1: "
            f"{sum(len(d) for d, _ in degree_tables)} degree states over {n} nodes",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        # Layer 1: Use direct u1 bounds with top-κ incident cap tightening when enabled
        L_pre1, U_pre1 = _pre_bounds_layer1(adj, X, alpha1)
        L_u1, U_u1 = _u1_bounds_direct(adj, X, weights.W1_1, weights.b1_1, alpha1,
                                        kappa if use_incident_cap_tightening else None)
        L_u1, U_u1 = _intersect_u_bound_override(
            L_u1, U_u1, layer=1, u_bound_overrides=u_bound_overrides
        )
        L_a1, U_a1 = np.maximum(L_u1, 0.0), np.maximum(U_u1, 0.0)
        L_h1, U_h1 = _linear_bounds(L_a1, U_a1, weights.W1_2, weights.b1_2)

        # Layer 2: Use per-dimension top-κ incident cap tightening when enabled
        if use_incident_cap_tightening and kappa is not None:
            L_pre2, U_pre2 = _pre_bounds_selectable_topk(adj, L_h1, U_h1, alpha2, kappa)
        else:
            L_pre2, U_pre2 = _pre_bounds_selectable(adj, L_h1, U_h1, alpha2)
    L_u2, U_u2 = _linear_bounds(L_pre2, U_pre2, weights.W2_1, weights.b2_1)
    L_u2, U_u2 = _intersect_u_bound_override(
        L_u2, U_u2, layer=2, u_bound_overrides=u_bound_overrides
    )
    L_a2, U_a2 = np.maximum(L_u2, 0.0), np.maximum(U_u2, 0.0)
    L_h2, U_h2 = _linear_bounds(L_a2, U_a2, weights.W2_2, weights.b2_2)

    # Layer 3: Use per-dimension top-κ incident cap tightening when enabled
    if use_incident_cap_tightening and kappa is not None:
        L_pre3, U_pre3 = _pre_bounds_selectable_topk(adj, L_h2, U_h2, alpha3, kappa)
    else:
        L_pre3, U_pre3 = _pre_bounds_selectable(adj, L_h2, U_h2, alpha3)
    L_u3, U_u3 = _linear_bounds(L_pre3, U_pre3, weights.W3_1, weights.b3_1)
    L_u3, U_u3 = _intersect_u_bound_override(
        L_u3, U_u3, layer=3, u_bound_overrides=u_bound_overrides
    )
    L_a3, U_a3 = np.maximum(L_u3, 0.0), np.maximum(U_u3, 0.0)
    L_h3, U_h3 = _linear_bounds(L_a3, U_a3, weights.W3_2, weights.b3_2)

    # Create model
    model = gp.Model("GraphGNN_MILP")
    _configure_gurobi_logging(model, verbose=verbose, log_file=log_file)
    model.Params.Seed = int(seed)
    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if use_anderson_cuts:
        model.Params.PreCrush = 1
    z = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}") for (i, j) in edges}
    _set_edge_branch_priority(z, edge_branch_priority)
    model.update()
    removal_z = _add_complement_edge_variables(model, z, edges) if connectivity_on_complement else None
    explanation_z = removal_z if removal_z is not None else z
    connectivity_edge_budget = (
        max(0, len(edges) - int(edge_budget))
        if connectivity_on_complement and edge_budget is not None
        else edge_budget
    )

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    # ========== Symmetry Breaking ==========
    symmetric_groups: Optional[Dict[int, List[List[int]]]] = None
    if use_symmetry_breaking:
        symmetric_groups = find_symmetric_branch_groups(adj, X, edges)
        # Edge-level symmetry breaking
        add_symmetry_breaking_constraints(
            model,
            explanation_z,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # Node selection helpers (used by symmetry breaking and optional connectivity)
    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
    model.update()

    incident_by_node: Dict[int, List[object]] = {i: [] for i in range(n)}
    for (i, j), zij in explanation_z.items():
        incident_by_node[i].append(zij)
        incident_by_node[j].append(zij)

    r: Optional[Dict[int, object]] = None
    eps_empty: Optional[object] = None
    if connectivity in ("flow", "cut"):
        # Edge-node linking for connectivity helper variables
        for (i, j), zij in explanation_z.items():
            model.addConstr(zij <= y[i], name=f"edge_node1_{i}_{j}")
            model.addConstr(zij <= y[j], name=f"edge_node2_{i}_{j}")

        # Variable root selection (graph classification)
        r = {i: model.addVar(vtype=GRB.BINARY, name=f"r_{i}") for i in range(n)}
        # Empty explanation indicator: allows selecting no edges at all
        eps_empty = model.addVar(vtype=GRB.BINARY, name="eps_empty")
        model.update()
        # Either pick exactly one root OR empty explanation
        model.addConstr(eps_empty + gp.quicksum(r[i] for i in range(n)) == 1.0, name="root_or_empty")
        for i in range(n):
            # If empty, no node can be selected
            model.addConstr(y[i] <= 1 - eps_empty, name=f"y_empty_{i}")
            model.addConstr(r[i] <= y[i], name=f"root_implies_selected_{i}")
            if use_star_constraints:
                if incident_by_node[i]:
                    model.addConstr(
                        y[i] <= gp.quicksum(incident_by_node[i]) + r[i] + eps_empty,
                        name=f"star_or_root_{i}",
                    )
                else:
                    model.addConstr(y[i] <= r[i] + eps_empty, name=f"isolated_or_root_{i}")
        if use_connectivity_lower_bound:
            model.addConstr(
                gp.quicksum(explanation_z[e] for e in edges) >= gp.quicksum(y[i] for i in range(n)) - 1.0,
                name="connectivity_lower_bound",
            )
        if connectivity == "flow":
            cap = _flow_cap_upper_bound(n, connectivity_edge_budget, node_budget)
            f = {}
            for (i, j) in edges:
                f[(i, j)] = model.addVar(lb=0.0, ub=cap, name=f"f_{i}_{j}")
                f[(j, i)] = model.addVar(lb=0.0, ub=cap, name=f"f_{j}_{i}")
            root_supply = {i: model.addVar(lb=0.0, ub=cap, name=f"s_{i}") for i in range(n)}
            model.update()

            # Empty case: sum(y)=0 => sum(s)=-1+eps_empty=0
            model.addConstr(
                gp.quicksum(root_supply[i] for i in range(n)) == gp.quicksum(y[i] for i in range(n)) - 1.0 + eps_empty,
                name="flow_total_supply",
            )
            for i in range(n):
                model.addConstr(root_supply[i] <= cap * r[i], name=f"flow_supply_root_{i}")
                out_arcs = [f[(i, j)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                in_arcs = [f[(j, i)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                model.addConstr(
                    gp.quicksum(out_arcs) - gp.quicksum(in_arcs) == root_supply[i] - (y[i] - r[i]),
                    name=f"flow_balance_{i}",
                )
            for (i, j), zij in explanation_z.items():
                model.addConstr(f[(i, j)] + f[(j, i)] <= cap * zij, name=f"flow_cap_{i}_{j}")
        else:
            model.Params.LazyConstraints = 1
    else:
        # Keep previous helper behavior when connectivity is disabled
        for i in range(n):
            if incident_by_node[i]:
                model.addConstr(y[i] <= gp.quicksum(incident_by_node[i]), name=f"y_upper_{i}")
            else:
                model.addConstr(y[i] == 0, name=f"y_isolated_{i}")

    # Node-level symmetry breaking
    if use_symmetry_breaking and symmetric_groups:
        add_symmetry_breaking_constraints_nodes(
            model,
            y,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # Note: Bridge Constraints not applicable for graph classification
    # (no target node v* to define connectivity from)

    # Layer 1
    anderson_relus: List[AndersonReluSpec] = []
    deg1: Dict[Tuple[int, int], object] = {}
    if degree_tables is not None:
        # Exact layer 1: one-hot selected degree, h1 read from the table.
        h1 = {(i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]), name=f"h1_{i}_{k}")
              for i in range(n) for k in range(h)}
        for i in range(n):
            for dval in degree_tables[i][0]:
                deg1[(i, int(dval))] = model.addVar(vtype=GRB.BINARY, name=f"deg1_{i}_{int(dval)}")
        model.update()
        for i in range(n):
            degrees_i, H_i = degree_tables[i]
            states = [deg1[(i, int(dval))] for dval in degrees_i]
            model.addConstr(gp.quicksum(states) == 1.0, name=f"deg1_onehot_{i}")
            model.addConstr(
                gp.quicksum(float(dval) * var for dval, var in zip(degrees_i, states))
                == gp.quicksum(z_ij(i, j) for j in adj.neighbors[i]),
                name=f"deg1_count_{i}",
            )
            for k in range(h):
                model.addConstr(
                    h1[(i, k)] == gp.quicksum(float(H_i[idx, k]) * var for idx, var in enumerate(states)),
                    name=f"h1def_{i}_{k}",
                )
    else:
        p1 = {
            (i, f): model.addVar(
                lb=float(L_pre1[i, f]),
                ub=float(U_pre1[i, f]),
                name=f"p1_{i}_{f}",
            )
            for i in range(n)
            for f in range(d0)
        }
        u1 = {(i, m): model.addVar(lb=float(L_u1[i, m]), ub=float(U_u1[i, m]), name=f"u1_{i}_{m}")
              for i in range(n) for m in range(m1)}
        a1 = {(i, m): model.addVar(lb=0.0, ub=float(max(U_u1[i, m], 0.0)), name=f"a1_{i}_{m}")
              for i in range(n) for m in range(m1)}
        d1 = {}
        h1 = {(i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]), name=f"h1_{i}_{k}")
              for i in range(n) for k in range(h)}
        model.update()

        for i in range(n):
            for f in range(d0):
                expr = gp.LinExpr()
                expr.addConstant(float((alpha1 + adj.self_loop_counts[i]) * X[i, f]))
                for j in adj.neighbors[i]:
                    expr.addTerms(float(X[j, f]), z_ij(i, j))
                model.addConstr(p1[(i, f)] == expr, name=f"p1def_{i}_{f}")

        for i in range(n):
            for m in range(m1):
                expr = gp.LinExpr()
                expr.addConstant(float(weights.b1_1[m]))
                for f in range(d0):
                    wval = float(weights.W1_1[f, m])
                    if wval != 0.0:
                        expr.addTerms(wval, p1[(i, f)])
                model.addConstr(u1[(i, m)] == expr, name=f"u1def_{i}_{m}")

        for i in range(n):
            for m in range(m1):
                L_val, U_val = float(L_u1[i, m]), float(U_u1[i, m])
                if U_val <= 0:
                    model.addConstr(a1[(i, m)] == 0.0)
                elif L_val >= 0:
                    model.addConstr(a1[(i, m)] == u1[(i, m)])
                else:
                    d1[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d1_{i}_{m}")
                    model.addConstr(a1[(i, m)] >= u1[(i, m)])
                    model.addConstr(a1[(i, m)] >= 0.0)
                    model.addConstr(a1[(i, m)] <= u1[(i, m)] - L_val * (1 - d1[(i, m)]))
                    model.addConstr(a1[(i, m)] <= U_val * d1[(i, m)])
                    if use_anderson_cuts:
                        anderson_relus.append(
                            AndersonReluSpec(
                                x_vars=tuple(p1[(i, f)] for f in range(d0)),
                                y_var=a1[(i, m)],
                                delta_var=d1[(i, m)],
                                w=np.asarray(weights.W1_1[:, m], dtype=float),
                                L=np.asarray(L_pre1[i, :], dtype=float),
                                U=np.asarray(U_pre1[i, :], dtype=float),
                                b=float(weights.b1_1[m]),
                                tag=f"graph3l_relu1_{i}_{m}",
                            )
                        )

        model.update()

        for i in range(n):
            for k in range(h):
                expr = gp.LinExpr()
                expr.addConstant(float(weights.b1_2[k]))
                for m in range(m1):
                    wval = float(weights.W1_2[m, k])
                    if wval != 0.0:
                        expr.addTerms(wval, a1[(i, m)])
                model.addConstr(h1[(i, k)] == expr, name=f"h1def_{i}_{k}")

    # Layer 2 (similar structure)
    t2 = {}
    for i in range(n):
        for j in adj.neighbors[i]:
            for k in range(h):
                t2[(i, j, k)] = model.addVar(lb=-GRB.INFINITY, name=f"t2_{i}_{j}_{k}")

    p2 = {
        (i, k): model.addVar(
            lb=float(L_pre2[i, k]),
            ub=float(U_pre2[i, k]),
            name=f"p2_{i}_{k}",
        )
        for i in range(n)
        for k in range(h)
    }
    u2 = {(i, m): model.addVar(lb=float(L_u2[i, m]), ub=float(U_u2[i, m]), name=f"u2_{i}_{m}")
          for i in range(n) for m in range(m2)}
    a2 = {(i, m): model.addVar(lb=0.0, ub=float(max(U_u2[i, m], 0.0)), name=f"a2_{i}_{m}")
          for i in range(n) for m in range(m2)}
    d2 = {}
    h2 = {(i, k): model.addVar(lb=float(L_h2[i, k]), ub=float(U_h2[i, k]), name=f"h2_{i}_{k}")
          for i in range(n) for k in range(h)}
    model.update()

    if degree_tables is not None:
        # Exact messages: z_ij h1_j = sum_d H1_j(d) q_ijd with q_ijd = z_ij * deg1_jd.
        for i in range(n):
            for j in adj.neighbors[i]:
                zij = z_ij(i, j)
                degrees_j, H_j = degree_tables[j]
                senders = [idx for idx, dval in enumerate(degrees_j) if dval >= 1]
                if not senders:
                    model.addConstr(zij == 0.0, name=f"msg2_none_{i}_{j}")
                    for k in range(h):
                        model.addConstr(t2[(i, j, k)] == 0.0)
                    continue
                q = {idx: model.addVar(lb=0.0, ub=1.0, name=f"msg2_{i}_{j}_{int(degrees_j[idx])}") for idx in senders}
                model.addConstr(gp.quicksum(q.values()) == zij, name=f"msg2_sum_{i}_{j}")
                for idx in senders:
                    state = deg1[(j, int(degrees_j[idx]))]
                    model.addConstr(q[idx] <= state)
                    model.addConstr(q[idx] >= state + zij - 1.0)
                for k in range(h):
                    model.addConstr(
                        t2[(i, j, k)] == gp.quicksum(float(H_j[idx, k]) * q[idx] for idx in senders)
                    )
    else:
        for i in range(n):
            for j in adj.neighbors[i]:
                a, b = (i, j) if i < j else (j, i)
                zij = z[(a, b)]
                for k in range(h):
                    L_jk = float(L_h1[j, k])
                    U_jk = float(U_h1[j, k])
                    tvar = t2[(i, j, k)]
                    model.addConstr(tvar <= h1[(j, k)] - L_jk * (1 - zij))
                    model.addConstr(tvar >= h1[(j, k)] - U_jk * (1 - zij))
                    model.addConstr(tvar <= U_jk * zij)
                    model.addConstr(tvar >= L_jk * zij)

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addTerms(float(alpha2 + adj.self_loop_counts[i]), h1[(i, k)])
            for j in adj.neighbors[i]:
                expr.add(t2[(i, j, k)])
            model.addConstr(p2[(i, k)] == expr)

    for i in range(n):
        for m in range(m2):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_1[m]))
            for k in range(h):
                wval = float(weights.W2_1[k, m])
                if wval != 0.0:
                    expr.addTerms(wval, p2[(i, k)])
            model.addConstr(u2[(i, m)] == expr)

    for i in range(n):
        for m in range(m2):
            L_val, U_val = float(L_u2[i, m]), float(U_u2[i, m])
            if U_val <= 0:
                model.addConstr(a2[(i, m)] == 0.0)
            elif L_val >= 0:
                model.addConstr(a2[(i, m)] == u2[(i, m)])
            else:
                d2[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d2_{i}_{m}")
                model.addConstr(a2[(i, m)] >= u2[(i, m)])
                model.addConstr(a2[(i, m)] >= 0.0)
                model.addConstr(a2[(i, m)] <= u2[(i, m)] - L_val * (1 - d2[(i, m)]))
                model.addConstr(a2[(i, m)] <= U_val * d2[(i, m)])
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p2[(i, k)] for k in range(h)),
                            y_var=a2[(i, m)],
                            delta_var=d2[(i, m)],
                            w=np.asarray(weights.W2_1[:, m], dtype=float),
                            L=np.asarray(L_pre2[i, :], dtype=float),
                            U=np.asarray(U_pre2[i, :], dtype=float),
                            b=float(weights.b2_1[m]),
                            tag=f"graph3l_relu2_{i}_{m}",
                        )
                    )

    model.update()

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b2_2[k]))
            for m in range(m2):
                wval = float(weights.W2_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a2[(i, m)])
            model.addConstr(h2[(i, k)] == expr)

    # Layer 3
    t3 = {}
    for i in range(n):
        for j in adj.neighbors[i]:
            for k in range(h):
                t3[(i, j, k)] = model.addVar(lb=-GRB.INFINITY, name=f"t3_{i}_{j}_{k}")

    p3 = {
        (i, k): model.addVar(
            lb=float(L_pre3[i, k]),
            ub=float(U_pre3[i, k]),
            name=f"p3_{i}_{k}",
        )
        for i in range(n)
        for k in range(h)
    }
    u3 = {(i, m): model.addVar(lb=float(L_u3[i, m]), ub=float(U_u3[i, m]), name=f"u3_{i}_{m}")
          for i in range(n) for m in range(m3)}
    a3 = {(i, m): model.addVar(lb=0.0, ub=float(max(U_u3[i, m], 0.0)), name=f"a3_{i}_{m}")
          for i in range(n) for m in range(m3)}
    d3 = {}
    h3 = {(i, k): model.addVar(lb=float(L_h3[i, k]), ub=float(U_h3[i, k]), name=f"h3_{i}_{k}")
          for i in range(n) for k in range(h)}
    model.update()

    for i in range(n):
        for j in adj.neighbors[i]:
            a, b = (i, j) if i < j else (j, i)
            zij = z[(a, b)]
            for k in range(h):
                L_jk = float(L_h2[j, k])
                U_jk = float(U_h2[j, k])
                tvar = t3[(i, j, k)]
                model.addConstr(tvar <= h2[(j, k)] - L_jk * (1 - zij))
                model.addConstr(tvar >= h2[(j, k)] - U_jk * (1 - zij))
                model.addConstr(tvar <= U_jk * zij)
                model.addConstr(tvar >= L_jk * zij)

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addTerms(float(alpha3 + adj.self_loop_counts[i]), h2[(i, k)])
            for j in adj.neighbors[i]:
                expr.add(t3[(i, j, k)])
            model.addConstr(p3[(i, k)] == expr)

    for i in range(n):
        for m in range(m3):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b3_1[m]))
            for k in range(h):
                wval = float(weights.W3_1[k, m])
                if wval != 0.0:
                    expr.addTerms(wval, p3[(i, k)])
            model.addConstr(u3[(i, m)] == expr)

    for i in range(n):
        for m in range(m3):
            L_val, U_val = float(L_u3[i, m]), float(U_u3[i, m])
            if U_val <= 0:
                model.addConstr(a3[(i, m)] == 0.0)
            elif L_val >= 0:
                model.addConstr(a3[(i, m)] == u3[(i, m)])
            else:
                d3[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d3_{i}_{m}")
                model.addConstr(a3[(i, m)] >= u3[(i, m)])
                model.addConstr(a3[(i, m)] >= 0.0)
                model.addConstr(a3[(i, m)] <= u3[(i, m)] - L_val * (1 - d3[(i, m)]))
                model.addConstr(a3[(i, m)] <= U_val * d3[(i, m)])
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p3[(i, k)] for k in range(h)),
                            y_var=a3[(i, m)],
                            delta_var=d3[(i, m)],
                            w=np.asarray(weights.W3_1[:, m], dtype=float),
                            L=np.asarray(L_pre3[i, :], dtype=float),
                            U=np.asarray(U_pre3[i, :], dtype=float),
                            b=float(weights.b3_1[m]),
                            tag=f"graph3l_relu3_{i}_{m}",
                        )
                    )

    model.update()

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b3_2[k]))
            for m in range(m3):
                wval = float(weights.W3_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a3[(i, m)])
            model.addConstr(h3[(i, k)] == expr)

    # ========== Global pooling (all nodes contribute, including self-loops) ==========
    pool3 = {k: model.addVar(lb=-GRB.INFINITY, name=f"pool3_{k}") for k in range(h)}
    model.update()

    scale = 1.0 if readout == "sum" else (1.0 / n)
    for k in range(h):
        expr = gp.LinExpr()
        for i in range(n):
            expr.addTerms(scale, h3[(i, k)])  # All nodes contribute
        model.addConstr(pool3[k] == expr, name=f"pool3def_{k}")

    # Output logits
    Z = {c: model.addVar(lb=-GRB.INFINITY, name=f"Z_{c}") for c in range(num_classes)}
    model.update()

    for c in range(num_classes):
        expr = gp.LinExpr()
        expr.addConstant(float(weights.b_out[c]))
        for k in range(h):
            wk = float(weights.W_out[c, k])
            if wk != 0.0:
                expr.addTerms(wk, pool3[k])
        model.addConstr(Z[c] == expr, name=f"Zdef_{c}")

    # Margin constraints
    rho_var = _add_margin_constraints(
        model=model,
        Z=Z,
        c_star=c_star,
        num_classes=num_classes,
        bar_m=bar_m,
        objective=objective,
        rho=rho,
    )

    # Edge constraints
    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    # Note: y is now defined above for node selection in pooling
    root_node = edges[0][0] if edges else 0

    # Objective
    _set_selection_objective(
        model=model,
        z=z,
        edges=edges,
        objective=objective,
        rho_var=rho_var,
        edge_budget=edge_budget,
        node_budget=node_budget,
    )

    if not defer_obbt_u:
        _apply_obbt_u_bounds(
            model,
            use_obbt_u=use_obbt_u,
            max_obbt_u_layers=max_obbt_u_layers,
            available_layers=(1, 2, 3),
            connectivity=connectivity,
            wallclock_deadline_s=wallclock_deadline_s,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
            log_prefix="[MIP-Graph]",
        )

    # Warm start
    if initial_solution is not None:
        initial_set = set(_unique_undirected_edges(initial_solution))
        for e, zvar in z.items():
            zvar.Start = 1.0 if e in initial_set else 0.0
        _print_and_log(
            f"[MIP-Graph] Warm start with {len(initial_set)} edges",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )

    return GraphMilpBuild(
        model=model,
        z=z,
        y=y if connectivity in ("flow", "cut") else None,
        r=r if connectivity in ("flow", "cut") else None,
        edges=edges,
        n=n,
        root_node=root_node,
        connectivity=cast(Literal["none", "flow", "cut"], connectivity),
        tilde_a=adj,
        c_star=c_star,
        rho_var=rho_var,
        bar_m=bar_m,
        eps_empty=eps_empty if connectivity in ("flow", "cut") else None,
        connectivity_z=explanation_z if connectivity in ("flow", "cut") else None,
        removal_z=removal_z,
        use_anderson_cuts=use_anderson_cuts,
        anderson_max_node_count=anderson_max_node_count,
        anderson_max_cuts=anderson_max_cuts,
        anderson_relus=tuple(anderson_relus),
        terminal_log_file=terminal_log_file,
    )


def build_gin_graphgnn_milp_4l(
    X: np.ndarray,
    undirected_edges: Sequence[Tuple[int, int]],
    weights: GINGraphGnnWeights4L,
    *,
    readout: str = "sum",
    rho: float = 0.0,
    edge_budget: Optional[int] = None,
    node_budget: Optional[int] = None,
    min_edges: Optional[int] = None,
    objective: ObjectiveMode = "min_edges",
    connectivity: ConnectivityMode = "none",
    connectivity_on_complement: bool = False,
    use_incident_cap_tightening: bool = False,
    cap_mode: CapMode = "global",
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
    defer_obbt_u: bool = False,
    u_bound_overrides: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
    verbose: bool = False,
    time_limit_s: Optional[float] = None,
    wallclock_deadline_s: Optional[float] = None,
    mip_gap: Optional[float] = None,
    seed: int = 0,
    initial_solution: Optional[Sequence[Tuple[int, int]]] = None,
    log_file: Optional[str] = None,
    terminal_log_file: Optional[str] = None,
) -> GraphMilpBuild:
    """Build MILP for 4-layer GIN GraphGNN explanation (last-layer readout)."""
    _validate_objective_mode(objective, rho)
    if connectivity not in ("none", "flow", "cut"):
        raise ValueError(
            "Graph classification currently supports connectivity in {'none', 'flow', 'cut'} only."
        )
    if cap_mode == "distance":
        raise ValueError("cap_mode='distance' is only supported for node classification (requires v_star)")

    gp, GRB = _require_gurobi()

    X = np.asarray(X, dtype=float)
    n, d0 = X.shape
    h = weights.hidden_dim
    m1, m2, m3, m4 = (
        weights.mlp_hidden1,
        weights.mlp_hidden2,
        weights.mlp_hidden3,
        weights.mlp_hidden4,
    )
    num_classes = weights.num_classes

    adj = compute_adjacency(n, undirected_edges)
    edges = _unique_undirected_edges(undirected_edges)

    # Full forward pass
    bar_Z = forward_gin_graphgnn_full_4l(X, adj, weights, readout=readout)
    c_star = int(np.argmax(bar_Z))
    bar_m = float(bar_Z[c_star] - np.max(np.delete(bar_Z, c_star))) if num_classes >= 2 else 0.0

    _print_and_log(
        f"[MIP-Graph-4L] Full graph: class={c_star}, margin={bar_m:.4f}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )
    _print_and_log(
        f"[MIP-Graph-4L] Graph: {n} nodes, {len(edges)} edges, readout={readout}",
        verbose=verbose,
        terminal_log_file=terminal_log_file,
    )

    # Compute incident caps for Top-κ tightening
    if use_incident_cap_tightening:
        kappa = compute_incident_caps(
            adj,
            edge_budget,
            cap_mode="global",
            v_star=None,
            node_budget=node_budget,
        )
        _print_and_log(
            "[MIP-Graph-4L] Top-κ tightening enabled, cap_mode=global",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
        _print_and_log(
            f"[MIP-Graph-4L] kappa range: [{kappa.min()}, {kappa.max()}]",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )
    else:
        kappa = None

    # Bound propagation
    alpha1 = 1.0 + float(weights.eps1)
    alpha2 = 1.0 + float(weights.eps2)
    alpha3 = 1.0 + float(weights.eps3)
    alpha4 = 1.0 + float(weights.eps4)

    # Layer 1: Use direct u1 bounds with top-κ incident cap tightening when enabled
    L_pre1, U_pre1 = _pre_bounds_layer1(adj, X, alpha1)
    L_u1, U_u1 = _u1_bounds_direct(adj, X, weights.W1_1, weights.b1_1, alpha1,
                                    kappa if use_incident_cap_tightening else None)
    L_u1, U_u1 = _intersect_u_bound_override(
        L_u1, U_u1, layer=1, u_bound_overrides=u_bound_overrides
    )
    L_a1, U_a1 = np.maximum(L_u1, 0.0), np.maximum(U_u1, 0.0)
    L_h1, U_h1 = _linear_bounds(L_a1, U_a1, weights.W1_2, weights.b1_2)

    # Layer 2: Use per-dimension top-κ incident cap tightening when enabled
    if use_incident_cap_tightening and kappa is not None:
        L_pre2, U_pre2 = _pre_bounds_selectable_topk(adj, L_h1, U_h1, alpha2, kappa)
    else:
        L_pre2, U_pre2 = _pre_bounds_selectable(adj, L_h1, U_h1, alpha2)
    L_u2, U_u2 = _linear_bounds(L_pre2, U_pre2, weights.W2_1, weights.b2_1)
    L_u2, U_u2 = _intersect_u_bound_override(
        L_u2, U_u2, layer=2, u_bound_overrides=u_bound_overrides
    )
    L_a2, U_a2 = np.maximum(L_u2, 0.0), np.maximum(U_u2, 0.0)
    L_h2, U_h2 = _linear_bounds(L_a2, U_a2, weights.W2_2, weights.b2_2)

    # Layer 3: Use per-dimension top-κ incident cap tightening when enabled
    if use_incident_cap_tightening and kappa is not None:
        L_pre3, U_pre3 = _pre_bounds_selectable_topk(adj, L_h2, U_h2, alpha3, kappa)
    else:
        L_pre3, U_pre3 = _pre_bounds_selectable(adj, L_h2, U_h2, alpha3)
    L_u3, U_u3 = _linear_bounds(L_pre3, U_pre3, weights.W3_1, weights.b3_1)
    L_u3, U_u3 = _intersect_u_bound_override(
        L_u3, U_u3, layer=3, u_bound_overrides=u_bound_overrides
    )
    L_a3, U_a3 = np.maximum(L_u3, 0.0), np.maximum(U_u3, 0.0)
    L_h3, U_h3 = _linear_bounds(L_a3, U_a3, weights.W3_2, weights.b3_2)

    # Layer 4: Use per-dimension top-κ incident cap tightening when enabled
    if use_incident_cap_tightening and kappa is not None:
        L_pre4, U_pre4 = _pre_bounds_selectable_topk(adj, L_h3, U_h3, alpha4, kappa)
    else:
        L_pre4, U_pre4 = _pre_bounds_selectable(adj, L_h3, U_h3, alpha4)
    L_u4, U_u4 = _linear_bounds(L_pre4, U_pre4, weights.W4_1, weights.b4_1)
    L_u4, U_u4 = _intersect_u_bound_override(
        L_u4, U_u4, layer=4, u_bound_overrides=u_bound_overrides
    )
    L_a4, U_a4 = np.maximum(L_u4, 0.0), np.maximum(U_u4, 0.0)
    L_h4, U_h4 = _linear_bounds(L_a4, U_a4, weights.W4_2, weights.b4_2)

    # Create model
    model = gp.Model("GraphGNN_4L_MILP")
    _configure_gurobi_logging(model, verbose=verbose, log_file=log_file)
    model.Params.Seed = int(seed)
    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if use_anderson_cuts:
        model.Params.PreCrush = 1
    z = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}") for (i, j) in edges}
    _set_edge_branch_priority(z, edge_branch_priority)
    model.update()
    removal_z = _add_complement_edge_variables(model, z, edges) if connectivity_on_complement else None
    explanation_z = removal_z if removal_z is not None else z
    connectivity_edge_budget = (
        max(0, len(edges) - int(edge_budget))
        if connectivity_on_complement and edge_budget is not None
        else edge_budget
    )

    def z_ij(i: int, j: int):
        a, b = (i, j) if i < j else (j, i)
        return z[(a, b)]

    # ========== Symmetry Breaking ==========
    symmetric_groups: Optional[Dict[int, List[List[int]]]] = None
    if use_symmetry_breaking:
        symmetric_groups = find_symmetric_branch_groups(adj, X, edges)
        # Edge-level symmetry breaking
        add_symmetry_breaking_constraints(
            model,
            explanation_z,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # Node selection helpers (used by symmetry breaking and optional connectivity)
    y = {i: model.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in range(n)}
    model.update()

    incident_by_node: Dict[int, List[object]] = {i: [] for i in range(n)}
    for (i, j), zij in explanation_z.items():
        incident_by_node[i].append(zij)
        incident_by_node[j].append(zij)

    r: Optional[Dict[int, object]] = None
    eps_empty: Optional[object] = None
    if connectivity in ("flow", "cut"):
        for (i, j), zij in explanation_z.items():
            model.addConstr(zij <= y[i], name=f"edge_node1_{i}_{j}")
            model.addConstr(zij <= y[j], name=f"edge_node2_{i}_{j}")

        r = {i: model.addVar(vtype=GRB.BINARY, name=f"r_{i}") for i in range(n)}
        # Empty explanation indicator: allows selecting no edges at all
        eps_empty = model.addVar(vtype=GRB.BINARY, name="eps_empty")
        model.update()
        # Either pick exactly one root OR empty explanation
        model.addConstr(eps_empty + gp.quicksum(r[i] for i in range(n)) == 1.0, name="root_or_empty")
        for i in range(n):
            # If empty, no node can be selected
            model.addConstr(y[i] <= 1 - eps_empty, name=f"y_empty_{i}")
            model.addConstr(r[i] <= y[i], name=f"root_implies_selected_{i}")
            if use_star_constraints:
                if incident_by_node[i]:
                    model.addConstr(
                        y[i] <= gp.quicksum(incident_by_node[i]) + r[i] + eps_empty,
                        name=f"star_or_root_{i}",
                    )
                else:
                    model.addConstr(y[i] <= r[i] + eps_empty, name=f"isolated_or_root_{i}")
        if use_connectivity_lower_bound:
            model.addConstr(
                gp.quicksum(explanation_z[e] for e in edges) >= gp.quicksum(y[i] for i in range(n)) - 1.0,
                name="connectivity_lower_bound",
            )
        if connectivity == "flow":
            cap = _flow_cap_upper_bound(n, connectivity_edge_budget, node_budget)
            f = {}
            for (i, j) in edges:
                f[(i, j)] = model.addVar(lb=0.0, ub=cap, name=f"f_{i}_{j}")
                f[(j, i)] = model.addVar(lb=0.0, ub=cap, name=f"f_{j}_{i}")
            root_supply = {i: model.addVar(lb=0.0, ub=cap, name=f"s_{i}") for i in range(n)}
            model.update()

            # Empty case: sum(y)=0 => sum(s)=-1+eps_empty=0
            model.addConstr(
                gp.quicksum(root_supply[i] for i in range(n)) == gp.quicksum(y[i] for i in range(n)) - 1.0 + eps_empty,
                name="flow_total_supply",
            )
            for i in range(n):
                model.addConstr(root_supply[i] <= cap * r[i], name=f"flow_supply_root_{i}")
                out_arcs = [f[(i, j)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                in_arcs = [f[(j, i)] for j in adj.neighbors[i] if (min(i, j), max(i, j)) in z]
                model.addConstr(
                    gp.quicksum(out_arcs) - gp.quicksum(in_arcs) == root_supply[i] - (y[i] - r[i]),
                    name=f"flow_balance_{i}",
                )
            for (i, j), zij in explanation_z.items():
                model.addConstr(f[(i, j)] + f[(j, i)] <= cap * zij, name=f"flow_cap_{i}_{j}")
        else:
            model.Params.LazyConstraints = 1
    else:
        for i in range(n):
            if incident_by_node[i]:
                model.addConstr(y[i] <= gp.quicksum(incident_by_node[i]), name=f"y_upper_{i}")
            else:
                model.addConstr(y[i] == 0, name=f"y_isolated_{i}")

    # Node-level symmetry breaking
    if use_symmetry_breaking and symmetric_groups:
        add_symmetry_breaking_constraints_nodes(
            model,
            y,
            symmetric_groups,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    # Layer 1
    anderson_relus: List[AndersonReluSpec] = []
    p1 = {
        (i, f): model.addVar(
            lb=float(L_pre1[i, f]),
            ub=float(U_pre1[i, f]),
            name=f"p1_{i}_{f}",
        )
        for i in range(n)
        for f in range(d0)
    }
    u1 = {(i, m): model.addVar(lb=float(L_u1[i, m]), ub=float(U_u1[i, m]), name=f"u1_{i}_{m}")
          for i in range(n) for m in range(m1)}
    a1 = {(i, m): model.addVar(lb=0.0, ub=float(max(U_u1[i, m], 0.0)), name=f"a1_{i}_{m}")
          for i in range(n) for m in range(m1)}
    d1 = {}
    h1 = {(i, k): model.addVar(lb=float(L_h1[i, k]), ub=float(U_h1[i, k]), name=f"h1_{i}_{k}")
          for i in range(n) for k in range(h)}
    model.update()

    for i in range(n):
        for f in range(d0):
            expr = gp.LinExpr()
            expr.addConstant(float((alpha1 + adj.self_loop_counts[i]) * X[i, f]))
            for j in adj.neighbors[i]:
                expr.addTerms(float(X[j, f]), z_ij(i, j))
            model.addConstr(p1[(i, f)] == expr, name=f"p1def_{i}_{f}")

    for i in range(n):
        for m in range(m1):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_1[m]))
            for f in range(d0):
                wval = float(weights.W1_1[f, m])
                if wval != 0.0:
                    expr.addTerms(wval, p1[(i, f)])
            model.addConstr(u1[(i, m)] == expr, name=f"u1def_{i}_{m}")

    for i in range(n):
        for m in range(m1):
            L_val, U_val = float(L_u1[i, m]), float(U_u1[i, m])
            if U_val <= 0:
                model.addConstr(a1[(i, m)] == 0.0)
            elif L_val >= 0:
                model.addConstr(a1[(i, m)] == u1[(i, m)])
            else:
                d1[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d1_{i}_{m}")
                model.addConstr(a1[(i, m)] >= u1[(i, m)])
                model.addConstr(a1[(i, m)] >= 0.0)
                model.addConstr(a1[(i, m)] <= u1[(i, m)] - L_val * (1 - d1[(i, m)]))
                model.addConstr(a1[(i, m)] <= U_val * d1[(i, m)])
                if use_anderson_cuts:
                    anderson_relus.append(
                        AndersonReluSpec(
                            x_vars=tuple(p1[(i, f)] for f in range(d0)),
                            y_var=a1[(i, m)],
                            delta_var=d1[(i, m)],
                            w=np.asarray(weights.W1_1[:, m], dtype=float),
                            L=np.asarray(L_pre1[i, :], dtype=float),
                            U=np.asarray(U_pre1[i, :], dtype=float),
                            b=float(weights.b1_1[m]),
                            tag=f"graph4l_relu1_{i}_{m}",
                        )
                    )

    model.update()

    for i in range(n):
        for k in range(h):
            expr = gp.LinExpr()
            expr.addConstant(float(weights.b1_2[k]))
            for m in range(m1):
                wval = float(weights.W1_2[m, k])
                if wval != 0.0:
                    expr.addTerms(wval, a1[(i, m)])
            model.addConstr(h1[(i, k)] == expr, name=f"h1def_{i}_{k}")

    def add_graph_layer(
        layer_idx: int,
        prev_h: Dict[Tuple[int, int], object],
        L_prev_h: np.ndarray,
        U_prev_h: np.ndarray,
        alpha: float,
        W_in: np.ndarray,
        b_in: np.ndarray,
        W_out: np.ndarray,
        b_out: np.ndarray,
        L_pre: np.ndarray,
        U_pre: np.ndarray,
        L_u: np.ndarray,
        U_u: np.ndarray,
        L_h_cur: np.ndarray,
        U_h_cur: np.ndarray,
        m_dim: int,
    ) -> Dict[Tuple[int, int], object]:
        t = {}
        for i in range(n):
            for j in adj.neighbors[i]:
                for k in range(h):
                    t[(i, j, k)] = model.addVar(lb=-GRB.INFINITY, name=f"t{layer_idx}_{i}_{j}_{k}")

        p = {
            (i, k): model.addVar(
                lb=float(L_pre[i, k]),
                ub=float(U_pre[i, k]),
                name=f"p{layer_idx}_{i}_{k}",
            )
            for i in range(n)
            for k in range(h)
        }
        u = {(i, m): model.addVar(lb=float(L_u[i, m]), ub=float(U_u[i, m]), name=f"u{layer_idx}_{i}_{m}")
             for i in range(n) for m in range(m_dim)}
        a = {(i, m): model.addVar(lb=0.0, ub=float(max(U_u[i, m], 0.0)), name=f"a{layer_idx}_{i}_{m}")
             for i in range(n) for m in range(m_dim)}
        d = {}
        h_cur = {(i, k): model.addVar(lb=float(L_h_cur[i, k]), ub=float(U_h_cur[i, k]), name=f"h{layer_idx}_{i}_{k}")
                 for i in range(n) for k in range(h)}
        model.update()

        for i in range(n):
            for j in adj.neighbors[i]:
                a_idx, b_idx = (i, j) if i < j else (j, i)
                zij = z[(a_idx, b_idx)]
                for k in range(h):
                    L_jk = float(L_prev_h[j, k])
                    U_jk = float(U_prev_h[j, k])
                    tvar = t[(i, j, k)]
                    model.addConstr(tvar <= prev_h[(j, k)] - L_jk * (1 - zij))
                    model.addConstr(tvar >= prev_h[(j, k)] - U_jk * (1 - zij))
                    model.addConstr(tvar <= U_jk * zij)
                    model.addConstr(tvar >= L_jk * zij)

        for i in range(n):
            for k in range(h):
                expr = gp.LinExpr()
                expr.addTerms(float(alpha + adj.self_loop_counts[i]), prev_h[(i, k)])
                for j in adj.neighbors[i]:
                    expr.add(t[(i, j, k)])
                model.addConstr(p[(i, k)] == expr)

        for i in range(n):
            for m in range(m_dim):
                expr = gp.LinExpr()
                expr.addConstant(float(b_in[m]))
                for k in range(h):
                    wval = float(W_in[k, m])
                    if wval != 0.0:
                        expr.addTerms(wval, p[(i, k)])
                model.addConstr(u[(i, m)] == expr)

        for i in range(n):
            for m in range(m_dim):
                L_val, U_val = float(L_u[i, m]), float(U_u[i, m])
                if U_val <= 0:
                    model.addConstr(a[(i, m)] == 0.0)
                elif L_val >= 0:
                    model.addConstr(a[(i, m)] == u[(i, m)])
                else:
                    d[(i, m)] = model.addVar(vtype=GRB.BINARY, name=f"d{layer_idx}_{i}_{m}")
                    model.addConstr(a[(i, m)] >= u[(i, m)])
                    model.addConstr(a[(i, m)] >= 0.0)
                    model.addConstr(a[(i, m)] <= u[(i, m)] - L_val * (1 - d[(i, m)]))
                    model.addConstr(a[(i, m)] <= U_val * d[(i, m)])
                    if use_anderson_cuts:
                        anderson_relus.append(
                            AndersonReluSpec(
                                x_vars=tuple(p[(i, k)] for k in range(h)),
                                y_var=a[(i, m)],
                                delta_var=d[(i, m)],
                                w=np.asarray(W_in[:, m], dtype=float),
                                L=np.asarray(L_pre[i, :], dtype=float),
                                U=np.asarray(U_pre[i, :], dtype=float),
                                b=float(b_in[m]),
                                tag=f"graph4l_relu{layer_idx}_{i}_{m}",
                            )
                        )

        model.update()

        for i in range(n):
            for k in range(h):
                expr = gp.LinExpr()
                expr.addConstant(float(b_out[k]))
                for m in range(m_dim):
                    wval = float(W_out[m, k])
                    if wval != 0.0:
                        expr.addTerms(wval, a[(i, m)])
                model.addConstr(h_cur[(i, k)] == expr)

        return h_cur

    h2 = add_graph_layer(
        2, h1, L_h1, U_h1, alpha2, weights.W2_1, weights.b2_1, weights.W2_2, weights.b2_2,
        L_pre2, U_pre2, L_u2, U_u2, L_h2, U_h2, m2,
    )
    h3 = add_graph_layer(
        3, h2, L_h2, U_h2, alpha3, weights.W3_1, weights.b3_1, weights.W3_2, weights.b3_2,
        L_pre3, U_pre3, L_u3, U_u3, L_h3, U_h3, m3,
    )
    h4 = add_graph_layer(
        4, h3, L_h3, U_h3, alpha4, weights.W4_1, weights.b4_1, weights.W4_2, weights.b4_2,
        L_pre4, U_pre4, L_u4, U_u4, L_h4, U_h4, m4,
    )

    # ========== Global pooling ==========
    pool4 = {k: model.addVar(lb=-GRB.INFINITY, name=f"pool4_{k}") for k in range(h)}
    model.update()

    scale = 1.0 if readout == "sum" else (1.0 / n)
    for k in range(h):
        expr = gp.LinExpr()
        for i in range(n):
            expr.addTerms(scale, h4[(i, k)])
        model.addConstr(pool4[k] == expr, name=f"pool4def_{k}")

    # Output logits
    Z = {c: model.addVar(lb=-GRB.INFINITY, name=f"Z_{c}") for c in range(num_classes)}
    model.update()
    for c in range(num_classes):
        expr = gp.LinExpr()
        expr.addConstant(float(weights.b_out[c]))
        for k in range(h):
            wk = float(weights.W_out[c, k])
            if wk != 0.0:
                expr.addTerms(wk, pool4[k])
        model.addConstr(Z[c] == expr, name=f"Zdef_{c}")

    # Margin constraints
    rho_var = _add_margin_constraints(
        model=model,
        Z=Z,
        c_star=c_star,
        num_classes=num_classes,
        bar_m=bar_m,
        objective=objective,
        rho=rho,
    )

    if edge_budget is not None:
        effective_budget = min(int(edge_budget), len(edges))
        model.addConstr(gp.quicksum(z[e] for e in edges) <= effective_budget, name="edge_budget")
    if min_edges is not None:
        model.addConstr(gp.quicksum(z[e] for e in edges) >= int(min_edges), name="min_edges")

    root_node = edges[0][0] if edges else 0

    _set_selection_objective(
        model=model,
        z=z,
        edges=edges,
        objective=objective,
        rho_var=rho_var,
        edge_budget=edge_budget,
        node_budget=node_budget,
    )

    if not defer_obbt_u:
        _apply_obbt_u_bounds(
            model,
            use_obbt_u=use_obbt_u,
            max_obbt_u_layers=max_obbt_u_layers,
            available_layers=(1, 2, 3, 4),
            connectivity=connectivity,
            wallclock_deadline_s=wallclock_deadline_s,
            verbose=verbose,
            terminal_log_file=terminal_log_file,
            log_prefix="[MIP-Graph-4L]",
        )

    if initial_solution is not None:
        initial_set = set(_unique_undirected_edges(initial_solution))
        for e, zvar in z.items():
            zvar.Start = 1.0 if e in initial_set else 0.0
        _print_and_log(
            f"[MIP-Graph-4L] Warm start with {len(initial_set)} edges",
            verbose=verbose,
            terminal_log_file=terminal_log_file,
        )

    _set_time_limit_from_deadline(
        model,
        wallclock_deadline_s=wallclock_deadline_s,
        fallback_time_limit_s=time_limit_s,
    )

    return GraphMilpBuild(
        model=model,
        z=z,
        y=y if connectivity in ("flow", "cut") else None,
        r=r if connectivity in ("flow", "cut") else None,
        edges=edges,
        n=n,
        root_node=root_node,
        connectivity=cast(Literal["none", "flow", "cut"], connectivity),
        tilde_a=adj,
        c_star=c_star,
        rho_var=rho_var,
        bar_m=bar_m,
        eps_empty=eps_empty if connectivity in ("flow", "cut") else None,
        connectivity_z=explanation_z if connectivity in ("flow", "cut") else None,
        removal_z=removal_z,
        use_anderson_cuts=use_anderson_cuts,
        anderson_max_node_count=anderson_max_node_count,
        anderson_max_cuts=anderson_max_cuts,
        anderson_relus=tuple(anderson_relus),
        terminal_log_file=terminal_log_file,
    )


def _cut_connectivity_callback_graph(build: GraphMilpBuild):
    """Create graph callback for component lazy cuts and Anderson ReLU cuts."""
    gp, GRB = _require_gurobi()

    use_cut_connectivity = build.connectivity == "cut"
    use_anderson_cuts = bool(getattr(build, "use_anderson_cuts", False))
    anderson_relus = tuple(getattr(build, "anderson_relus", ()))
    anderson_max_node_count = max(0, int(getattr(build, "anderson_max_node_count", 0)))
    anderson_max_cuts = max(1, int(getattr(build, "anderson_max_cuts", 50)))
    separation_log_file = getattr(build, "terminal_log_file", None)

    if not use_cut_connectivity and (not use_anderson_cuts or not anderson_relus):
        raise ValueError("Graph callback requires connectivity='cut' or Anderson ReLU cuts.")
    if use_cut_connectivity and (build.y is None or build.r is None):
        raise ValueError("Graph cut callback requires y and r variables.")

    y = build.y
    r = build.r
    z = build.connectivity_z if build.connectivity_z is not None else build.z
    edges = build.edges
    n = build.n
    eps_empty = build.eps_empty

    sep_state = {
        "total_cuts": 0,
        "cuts_mipnode": 0,
        "cuts_mipsol_lazy": 0,
        "anderson_cuts_mipnode": 0,
        "anderson_round_count": 0,
        "anderson_done": False,
    }
    setattr(build.model, "_cut_sep_state", sep_state)

    def _separate_anderson(model, nodecnt: float) -> int:
        """Run Anderson separation at current graph MIPNODE point."""
        if not use_anderson_cuts or not anderson_relus:
            return 0

        cuts_added = 0
        for spec in anderson_relus:
            if cuts_added >= anderson_max_cuts:
                break
            try:
                x_hat = np.array([float(model.cbGetNodeRel(vx)) for vx in spec.x_vars], dtype=float)
                y_hat = float(model.cbGetNodeRel(spec.y_var))
                delta_hat = float(model.cbGetNodeRel(spec.delta_var))
            except Exception:
                continue

            I_star = _anderson_separator_indices(spec.w, spec.L, spec.U, x_hat, delta_hat)
            rhs_hat = _evaluate_anderson_rhs(spec.w, spec.L, spec.U, spec.b, I_star, x_hat, delta_hat)
            violation = y_hat - rhs_hat
            if violation <= 1e-4:
                continue

            rhs_expr = _build_anderson_rhs_expr(gp, spec, I_star)
            model.cbCut(spec.y_var <= rhs_expr)
            cuts_added += 1
            anderson_msg = (
                "[AndersonCut-Graph] added "
                f"nodecnt={nodecnt:.0f} tag={spec.tag} "
                f"violation={violation:.6e} "
                f"y={y_hat:.6e} rhs={rhs_hat:.6e}"
            )
            print(anderson_msg)
            _append_log_line(separation_log_file, anderson_msg)
        return cuts_added

    def callback(model, where):
        if where == GRB.Callback.MIPNODE:
            if (
                not use_anderson_cuts
                or sep_state["anderson_done"]
                or not anderson_relus
            ):
                return
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status != GRB.OPTIMAL:
                return
            nodecnt = float(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
            if nodecnt > float(anderson_max_node_count) + 1e-9:
                return

            sep_state["anderson_done"] = True
            sep_state["anderson_round_count"] += 1
            anderson_added = _separate_anderson(model, nodecnt)
            if anderson_added > 0:
                sep_state["total_cuts"] += int(anderson_added)
                sep_state["cuts_mipnode"] += int(anderson_added)
                sep_state["anderson_cuts_mipnode"] += int(anderson_added)
                anderson_summary = (
                    f"[AndersonCut-Graph] summary round={sep_state['anderson_round_count']} "
                    f"nodecnt={nodecnt:.0f} cuts_added={anderson_added}"
                )
                print(anderson_summary)
                _append_log_line(separation_log_file, anderson_summary)
            return

        if where != GRB.Callback.MIPSOL or not use_cut_connectivity:
            return

        # Check if empty explanation is selected
        eps_empty_sol = 0.0
        if eps_empty is not None:
            eps_empty_sol = float(model.cbGetSolution(eps_empty))
        if eps_empty_sol > 0.5:
            # Empty explanation selected - connectivity trivially satisfied
            return

        y_sol = {i: float(model.cbGetSolution(y[i])) for i in range(n)}
        r_sol = {i: float(model.cbGetSolution(r[i])) for i in range(n)}
        z_sol = {e: float(model.cbGetSolution(z[e])) for e in edges}

        selected_nodes = {i for i, value in y_sol.items() if value > 0.5}
        if not selected_nodes:
            return
        selected_edges = [edge for edge, value in z_sol.items() if value > 0.5]
        components = _selected_connected_components(selected_nodes, selected_edges)
        if len(components) <= 1:
            return

        root = max(range(n), key=lambda i: r_sol[i])
        root_component = next(
            (component for component in components if root in component),
            None,
        )
        if root_component is None:
            return

        cuts_added = 0
        for component in components:
            if component == root_component:
                continue
            t = min(component)
            cut_edges = [
                z[(i, j)]
                for (i, j) in edges
                if (i in component) != (j in component)
            ]
            if eps_empty is not None:
                expr = (
                    gp.quicksum(cut_edges)
                    + gp.quicksum(r[i] for i in component)
                    + eps_empty
                )
            else:
                expr = gp.quicksum(cut_edges) + gp.quicksum(
                    r[i] for i in component
                )
            model.cbLazy(y[t] <= expr)
            cuts_added += 1
            _append_log_line(
                separation_log_file,
                "[ConnectivityCut-Graph] added "
                f"component_size={len(component)} representative={int(t)}",
            )

        sep_state["cuts_mipsol_lazy"] += int(cuts_added)
        sep_state["total_cuts"] += int(cuts_added)

    return callback


def optimize_and_extract_graph(
    build: GraphMilpBuild,
    *,
    relax_edge_mask: bool = False,
    selection_edge_budget: Optional[int] = None,
) -> MilpSolution:
    """Optimize graph classification MILP and extract solution.
    """
    import time as _time
    gp, GRB = _require_gurobi()
    model = build.model

    def _safe_float(value) -> Optional[float]:
        try:
            x = float(value)
        except Exception:
            return None
        return x if np.isfinite(x) else None

    root_stats = {
        "captured": False,
        "root_lp_value": None,
        "root_lp_time": None,
    }
    callback_stats = {
        "calls": 0,
        "time_sec": 0.0,
    }

    def _capture_root_stats(cb_model, where):
        if where != GRB.Callback.MIPNODE:
            return
        if root_stats["captured"]:
            return
        nodecnt = cb_model.cbGet(GRB.Callback.MIPNODE_NODCNT)
        if nodecnt > 0:
            return
        status = cb_model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return
        bound = cb_model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
        if np.isfinite(bound):
            root_stats["root_lp_value"] = float(bound)
        if np.isfinite(runtime):
            root_stats["root_lp_time"] = float(runtime)
        root_stats["captured"] = True

    use_graph_anderson = (
        bool(getattr(build, "use_anderson_cuts", False))
        and len(getattr(build, "anderson_relus", ())) > 0
    )
    cut_callback = (
        _cut_connectivity_callback_graph(build)
        if build.connectivity == "cut" or use_graph_anderson
        else None
    )

    def callback(cb_model, where):
        t0 = _time.perf_counter()
        callback_stats["calls"] += 1
        _capture_root_stats(cb_model, where)
        if cut_callback is not None:
            cut_callback(cb_model, where)
        callback_stats["time_sec"] += (_time.perf_counter() - t0)

    interrupted_by_user = _optimize_gurobi_model(
        model,
        callback=callback,
        terminal_log_file=getattr(build, "terminal_log_file", None),
    )

    status_code = int(model.Status)
    if interrupted_by_user or status_code == getattr(GRB, "INTERRUPTED", 11):
        status = "interrupted_by_user"
    elif status_code == GRB.OPTIMAL:
        status = "optimal"
    elif status_code == GRB.TIME_LIMIT:
        status = "time_limit"
    elif status_code == GRB.INFEASIBLE:
        status = "infeasible"
    else:
        status = f"status_{status_code}"

    best_bound = _safe_float(getattr(model, "ObjBound", None))
    incumbent_obj = _safe_float(getattr(model, "ObjVal", None)) if getattr(model, "SolCount", 0) > 0 else None
    final_mip_gap = _safe_float(getattr(model, "MIPGap", None))
    model_size = _safe_float(getattr(model, "NumNZs", None))

    sep_state = getattr(model, "_cut_sep_state", None)
    cuts_mipnode = int(sep_state.get("cuts_mipnode", 0)) if isinstance(sep_state, dict) else 0
    cuts_mipsol_lazy = int(sep_state.get("cuts_mipsol_lazy", 0)) if isinstance(sep_state, dict) else 0
    anderson_cuts_added = (
        int(sep_state.get("anderson_cuts_mipnode", 0))
        if isinstance(sep_state, dict)
        else 0
    )
    connectivity_lazy_cuts_added = cuts_mipsol_lazy

    if getattr(model, "SolCount", 0) <= 0:
        return MilpSolution(
            status=status,
            objective_value=None,
            selected_edges=[],
            z_values={},
            bar_m=build.bar_m,
            v_star=None,  # No target node for graph classification
            c_star=build.c_star,
            final_mip_gap=final_mip_gap,
            best_bound=best_bound,
            incumbent_obj=incumbent_obj,
            model_size=model_size,
            root_lp_value=root_stats["root_lp_value"],
            root_lp_time=root_stats["root_lp_time"],
            callback_time_sec=float(callback_stats["time_sec"]),
            callback_calls=int(callback_stats["calls"]),
            cuts_mipnode=cuts_mipnode,
            cuts_mipsol_lazy=cuts_mipsol_lazy,
            anderson_cuts_added=anderson_cuts_added,
            connectivity_lazy_cuts_added=connectivity_lazy_cuts_added,
            bnb_node_count=float(getattr(model, "NodeCount", 0.0)),
        )

    z_values = {e: float(var.X) for e, var in build.z.items()}
    selected_edges = _select_edges_from_z_values(
        z_values,
        relax_edge_mask=relax_edge_mask,
        selection_edge_budget=selection_edge_budget,
    )
    obj = _safe_float(getattr(model, "ObjVal", None))

    rho_optimal = None
    if build.rho_var is not None and getattr(model, "SolCount", 0) > 0:
        rho_optimal = float(build.rho_var.X)

    return MilpSolution(
        status=status,
        objective_value=obj,
        selected_edges=selected_edges,
        z_values=z_values,
        rho_optimal=rho_optimal,
        bar_m=build.bar_m,
        v_star=None,  # No target node for graph classification
        c_star=build.c_star,
        final_mip_gap=final_mip_gap,
        best_bound=best_bound,
        incumbent_obj=incumbent_obj,
        model_size=model_size,
        root_lp_value=root_stats["root_lp_value"],
        root_lp_time=root_stats["root_lp_time"],
        callback_time_sec=float(callback_stats["time_sec"]),
        callback_calls=int(callback_stats["calls"]),
        cuts_mipnode=cuts_mipnode,
        cuts_mipsol_lazy=cuts_mipsol_lazy,
        anderson_cuts_added=anderson_cuts_added,
        connectivity_lazy_cuts_added=connectivity_lazy_cuts_added,
        bnb_node_count=float(getattr(model, "NodeCount", 0.0)),
    )
