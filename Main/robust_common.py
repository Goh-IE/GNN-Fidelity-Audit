"""Helpers shared by the robust runner and the scenario-table MILP.

The evaluation context of one instance, batched target-class probabilities under
edge masks, and normalizers of the robust settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from Main.audit_standard import compact_node_inference_graph, edge_set

ALPHA_METRICS = {"sufficiency", "necessity", "delta"}
ALPHA_METRIC_CHOICES = ALPHA_METRICS
CONNECTIVITY_MODES = {"none", "cut", "flow"}


def normalize_alpha_metric(value: str) -> str:
    metric = str(value).strip().lower()
    if metric not in ALPHA_METRIC_CHOICES:
        raise ValueError(f"--metric must be one of {sorted(ALPHA_METRIC_CHOICES)}, got {value!r}.")
    return metric


def effective_gurobi_norel_heur_work(args: argparse.Namespace) -> float:
    if not bool(args.gurobi_norel_hybrid) or args.gurobi_norel_heur_work is None:
        return 0.0
    return max(0.0, float(args.gurobi_norel_heur_work))


def normalize_connectivity(value: str) -> str:
    connectivity = str(value).strip().lower()
    if connectivity not in CONNECTIVITY_MODES:
        raise ValueError(f"--connectivity must be one of {sorted(CONNECTIVITY_MODES)}, got {value!r}.")
    return connectivity


@dataclass
class EvalContext:
    dataset: str
    task: str
    x: np.ndarray
    edge_index: np.ndarray
    candidate_edges: List[Tuple[int, int]]
    c_star: int
    root_node: Optional[int]
    device: torch.device
    x_tensor: torch.Tensor
    edge_index_tensor: torch.Tensor
    edge_to_cols: Dict[Tuple[int, int], List[int]]
    self_loop_cols: List[int]
    batch_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = field(
        default_factory=dict
    )
    node_inference_cache: Optional[
        Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]
    ] = None

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edge_cols(self) -> int:
        return int(self.edge_index.shape[1])


def get_batch_tensors(
    ctx: EvalContext,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    batch_size = int(batch_size)
    cached = ctx.batch_cache.get(batch_size)
    if cached is not None:
        return cached

    x_base, edge_index_base, inference_root, _ = node_inference_tensors(ctx)
    n = int(x_base.shape[0])
    m = int(edge_index_base.shape[1])
    x_batch = x_base.repeat((batch_size, 1))
    offsets = torch.arange(batch_size, dtype=torch.long, device=ctx.device).repeat_interleave(m) * n
    shifted_edge_index = edge_index_base.repeat(1, batch_size) + offsets.view(1, -1)
    if ctx.task == "graph":
        batch_vec = torch.arange(batch_size, dtype=torch.long, device=ctx.device).repeat_interleave(n)
        target_idx = None
    else:
        batch_vec = None
        if inference_root is None:
            raise ValueError("Node task requires root_node.")
        target_idx = torch.arange(batch_size, dtype=torch.long, device=ctx.device) * n + int(inference_root)
    cached = (x_batch, shifted_edge_index, batch_vec, target_idx)
    ctx.batch_cache[batch_size] = cached
    return cached


def node_inference_tensors(
    ctx: EvalContext,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[torch.Tensor]]:
    """Return a compact node graph while preserving global mask/cache IDs."""
    if ctx.task != "node":
        return ctx.x_tensor, ctx.edge_index_tensor, ctx.root_node, None
    if ctx.root_node is None:
        raise ValueError("Node task requires root_node.")
    if ctx.node_inference_cache is None:
        local_x, local_edge_index, local_root, original_columns, _ = compact_node_inference_graph(
            x=ctx.x,
            edge_index=ctx.edge_index,
            candidate_edges=ctx.candidate_edges,
            root_node=int(ctx.root_node),
        )
        ctx.node_inference_cache = (
            torch.tensor(local_x, dtype=torch.float32, device=ctx.device),
            torch.tensor(local_edge_index, dtype=torch.long, device=ctx.device),
            int(local_root),
            torch.tensor(original_columns, dtype=torch.long, device=ctx.device),
        )
    return ctx.node_inference_cache


def edge_col_lists(ctx: EvalContext, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    result: List[List[int]] = []
    missing: List[Tuple[int, int]] = []
    for edge in sorted(edge_set(edges)):
        cols = ctx.edge_to_cols.get(edge)
        if not cols:
            missing.append(edge)
        else:
            result.append(cols)
    if missing:
        preview = ", ".join(str(edge) for edge in missing[:5])
        raise ValueError(f"{len(missing)} selected edges are absent from edge_index: {preview}")
    return result


def edge_weight_template(ctx: EvalContext, base_edges: Iterable[Tuple[int, int]]) -> torch.Tensor:
    template = torch.zeros(ctx.num_edge_cols, dtype=torch.float32, device=ctx.device)
    if ctx.self_loop_cols:
        template[torch.tensor(ctx.self_loop_cols, dtype=torch.long, device=ctx.device)] = 1.0
    for cols in edge_col_lists(ctx, base_edges):
        template[torch.tensor(cols, dtype=torch.long, device=ctx.device)] = 1.0
    return template


def probabilities_from_weights(
    *,
    model: torch.nn.Module,
    ctx: EvalContext,
    edge_weights: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    x_batch, shifted_edge_index, batch_vec, target_idx = get_batch_tensors(ctx, batch_size)
    _, _, _, original_edge_columns = node_inference_tensors(ctx)
    model_edge_weights = edge_weights
    if original_edge_columns is not None:
        global_expected = int(batch_size) * int(ctx.num_edge_cols)
        compact_edge_cols = int(original_edge_columns.numel())
        compact_expected = int(batch_size) * compact_edge_cols
        if int(edge_weights.numel()) == global_expected:
            model_edge_weights = (
                edge_weights.reshape(int(batch_size), int(ctx.num_edge_cols))
                .index_select(1, original_edge_columns)
                .reshape(-1)
            )
        elif int(edge_weights.numel()) == compact_expected:
            model_edge_weights = edge_weights.reshape(-1)
        else:
            raise ValueError(
                f"Expected {global_expected} global or {compact_expected} compact edge weights "
                f"for batch_size={batch_size}, "
                f"got {int(edge_weights.numel())}."
            )
    with torch.inference_mode():
        if ctx.task == "graph":
            logits = model(x_batch, shifted_edge_index, batch=batch_vec, edge_weight=model_edge_weights)
        else:
            all_logits = model(x_batch, shifted_edge_index, edge_weight=model_edge_weights)
            if target_idx is None:
                raise ValueError("Node task requires target_idx.")
            logits = all_logits.index_select(0, target_idx)
        probs = torch.softmax(logits.reshape(batch_size, -1), dim=-1)
        return probs[:, int(ctx.c_star)].detach().cpu()
