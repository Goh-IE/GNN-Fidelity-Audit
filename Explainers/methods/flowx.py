"""FlowX_suf/nec with the author's sampling/refinement and undirected projection.

The pinned upstream implementation is under methods/_vendor/flowx. We store the
first deletion group of each flow rather than its dense one-hot tensor. This
preserves the sampling distribution, node-task 100-group cap, marginal scores,
loss, normalization, stochastic gates and optimizer. Regression compares these
operations with the unmodified upstream method bodies on the same predictor.
"""
from __future__ import annotations

import ast
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parent / "_vendor/flowx"
COMMIT = "01fc6390bf8dc64201d491a1effc5b71ae37e344"


def upstream_gumbel():
    source = UPSTREAM / "utils.py"
    definition = next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "gumbel_softmax")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["gumbel_softmax"]


gumbel_softmax = upstream_gumbel()


def carrier_layout(edge_index, n):
    """One carrier per non-loop direction and one combined self carrier/node."""
    nonloop = edge_index[0] != edge_index[1]
    edges = edge_index[:, nonloop]
    counts = torch.bincount(edge_index[0, ~nonloop], minlength=n)
    mapping = torch.empty(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    mapping[nonloop] = torch.arange(edges.shape[1], device=edge_index.device)
    mapping[~nonloop] = edges.shape[1] + edge_index[0, ~nonloop]
    return edges, mapping, counts


def enumerate_flows(edge_index, n, layers, root=None):
    """Native edge-ID order, including one GIN self carrier per node.

For node tasks, backward enumeration avoids creating walks that would be
discarded by upstream's final-target filter. Lexicographic ordering recovers
the exact order of its forward DFS.
"""
    edge_index, _, _ = carrier_layout(edge_index, n)
    loops = torch.arange(n, device=edge_index.device)
    carriers = torch.cat((edge_index, torch.stack((loops, loops))), dim=1)
    endpoints = carriers.T.tolist()
    outgoing, incoming = [[] for _ in range(n)], [[] for _ in range(n)]
    for i, (a, b) in enumerate(endpoints):
        outgoing[a].append(i)
        incoming[b].append(i)
    walks = []
    if root is None:
        def extend(path):
            if len(path) == layers:
                walks.append(path)
            else:
                for j in outgoing[endpoints[path[-1]][1]]:
                    extend(path + [j])
        for i in range(len(endpoints)):
            extend([i])
    else:
        def prepend(path, node):
            if len(path) == layers:
                walks.append(path)
            else:
                for j in incoming[node]:
                    prepend([j] + path, endpoints[j][0])
        prepend([], root)
        walks.sort()
    return carriers, torch.tensor(walks, dtype=torch.long, device=edge_index.device)


def carrier_logits(model, x, edge_index, activity, root=None):
    """Apply directional/layer masks, including the (1+eps) self carrier.

activity has shape [layers,batch,directed_edges+nodes]. This is the native
FlowX intervention. The final baseline evaluator separately uses physical
edge deletion with the GIN self term fixed.
"""
    layers, batch, count = activity.shape
    n = len(x)
    edge_index, _, loop_counts = carrier_layout(edge_index, n)
    d = edge_index.shape[1]
    if count != d + n or layers != model.num_layers:
        raise ValueError("FlowX carrier dimensions disagree with the predictor")
    if model.use_batchnorm or model.use_jk_concat:
        raise ValueError("FlowX runner requires the frozen no-BN, last-layer GIN")
    h = x.unsqueeze(0).expand(batch, -1, -1)
    for layer in range(layers):
        conv = getattr(model, f"conv{layer + 1}")
        messages = h[:, edge_index[0], :] * activity[layer, :, :d, None]
        neighbors = h.new_zeros(h.shape).index_add(1, edge_index[1], messages)
        # Existing explicit loops contribute additional self messages. Combine
        # them into the single native self carrier without changing full logits.
        if bool(loop_counts.any()):
            self_coefficient = (1 + conv.eps + loop_counts.to(x.dtype))[None, :, None]
        else:
            self_coefficient = 1 + conv.eps  # preserve the original no-loop path
        pre = neighbors + self_coefficient * h * activity[layer, :, d:, None]
        h = conv.mlp(pre)
    if root is None:
        # Match the frozen graph model's index_add pooling operation.
        pooled = h.new_zeros((batch, h.shape[-1])).index_add(
            0, torch.arange(batch, device=h.device).repeat_interleave(n), h.reshape(batch * n, -1))
        return model.lin(pooled)
    if root == "all":
        return model.lin(h).reshape(batch * n, -1)
    return model.lin(h[:, root])


@torch.no_grad()
def sample_flows(model, x, edge_index, walks, root, iterations, batch_size=32,
                 progress=None):
    """Equivalent to upstream flow_shap using first-deletion group indices."""
    nonloop_edges, _, _ = carrier_layout(edge_index, len(x))
    layers, c = model.num_layers, nonloop_edges.shape[1] + len(x)
    flat = walks + torch.arange(layers, device=x.device)[None] * c
    active_carriers = torch.cat([flat[:, layer].unique() for layer in range(layers)])
    group_count = min(len(active_carriers), 100) if root is not None else len(active_carriers)
    group_width = len(active_carriers) // group_count
    full = carrier_logits(model, x, edge_index, x.new_ones((layers, 1, c)), root).softmax(-1)
    groups, weights, deltas, permutations = [], [], [], []
    counts = x.new_zeros(len(walks))
    for iteration in range(iterations):
        permutation = active_carriers[torch.randperm(len(active_carriers))]
        positions = torch.full((layers * c,), len(active_carriers), dtype=torch.long, device=x.device)
        positions[permutation] = torch.arange(len(permutation), device=x.device)
        first = (positions[flat].amin(dim=1) // group_width)
        valid = first < group_count  # Upstream's floor division leaves a remainder.
        group_sizes = torch.bincount(first[valid], minlength=group_count).to(x.dtype)
        group = first.clamp(max=group_count - 1)
        weight = torch.where(valid, 1 / (group_sizes[group] + 1e-30), 0)
        outputs = []
        for start in range(0, group_count, batch_size):
            ends = torch.arange(start + 1, min(group_count, start + batch_size) + 1,
                                device=x.device) * group_width
            masks = (positions[None] >= ends[:, None]).reshape(-1, layers, c).permute(1, 0, 2).to(x.dtype)
            outputs.append(carrier_logits(model, x, edge_index, masks, root).softmax(-1))
        outputs = torch.cat(outputs)
        marginal = torch.cat((full, outputs[:-1])) - outputs
        groups.append(group)
        weights.append(weight)
        deltas.append(marginal)
        permutations.append(permutation)
        counts += valid.to(x.dtype)
        if progress is not None:
            progress({"stage": "flow_sampling", "iteration": iteration + 1, "iterations": iterations,
                      "flow_count": len(walks), "deletion_groups": group_count})
    return {"groups": torch.stack(groups), "weights": torch.stack(weights),
            "marginals": torch.stack(deltas), "sample_counts": counts,
            "permutations": torch.stack(permutations), "group_width": group_width,
            "group_count": group_count, "carrier_count": c}


def aggregate_flows(walks, scores, carrier_count):
    """Sum contributions per layer and direction, counting repeated traversals."""
    layers = walks.shape[1]
    return torch.stack([scores.new_zeros(carrier_count).index_add(0, walks[:, layer], scores)
                        for layer in range(layers)])


def information_loss(mask, prior):
    # Preserve both epsilons in the author's second logarithm. Omitting the
    # outer epsilon makes a saturated mask=1 produce 0*log(0) in float32.
    eps = 1e-6
    return (mask * torch.log(mask / prior + eps) +
            (1 - mask) * torch.log((1 - mask) / (1 - prior + eps) + eps)).mean()


def refine_flows(model, x, edge_index, walks, root, target, sampled,
                 variant, sparsity, epochs=500, lr=0.1, progress=None):
    necessity = variant == "flowx_nec"
    if variant not in ("flowx_suf", "flowx_nec"):
        raise ValueError(variant)
    if not 0 < sparsity < 1:
        raise ValueError("FlowX information prior requires an interior sparsity")
    group_count, c = sampled["group_count"], sampled["carrier_count"]
    # Official initialization and optimizer. The manuscript sets lr=0.1 for
    # both variants. Upstream minus's 0.01 is annotated 'Graph-SST2 only'.
    coalition = nn.Parameter(0.1 * nn.init.uniform_(x.new_empty((1, group_count, 1))))
    norm = nn.InstanceNorm1d(1, affine=True, track_running_stats=False).to(x)
    optimizer = torch.optim.Adam([{"params": coalition, "lr": lr},
                                   {"params": norm.parameters(), "lr": lr}], weight_decay=0)
    groups, weights = sampled["groups"], sampled["weights"]
    marginals = sampled["marginals"][:, :, target]
    lookup = torch.arange(len(groups), device=x.device)[:, None]
    selected_marginals = marginals[lookup, groups]
    counts = sampled["sample_counts"]
    losses = []
    for epoch in range(1, epochs + 1):
        weighted = weights * coalition.sigmoid().reshape(-1)[groups]
        flows = (weighted * selected_marginals).sum(0) / (counts + 1e-18)
        layer_scores = aggregate_flows(walks, flows, c)
        directed = layer_scores.sum(0)
        gate = gumbel_softmax(norm(directed[None, None, :]).squeeze(), 1, training=True)
        current_sparsity = (gate < 0.5).to(x.dtype).mean()
        mask = 1 - gate if necessity else gate
        # Keep the exact inverse-sigmoid/forward-sigmoid roundtrip in upstream.
        logits_mask = torch.log(mask / (1 - mask + 1e-18) + 1e-18)
        activity = logits_mask.sigmoid()[None, None, :].expand(model.num_layers, 1, -1)
        logits = carrier_logits(model, x, edge_index, activity, root)
        prediction_loss = F.cross_entropy(logits, torch.tensor([target], device=x.device))
        if necessity:
            prediction_loss = -prediction_loss
        prior = sparsity if necessity else 1 - sparsity
        information = information_loss(mask, prior)
        loss = prediction_loss + information
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite FlowX refinement at epoch={epoch}: prediction={float(prediction_loss.detach())}, "
                             f"information={float(information.detach())}, mask_range="
                             f"({float(mask.detach().min())},{float(mask.detach().max())}), "
                             f"logits={logits.detach().tolist()}, prior={prior}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append({"epoch": epoch, "loss": float(prediction_loss.detach()),
                       "info": float(information.detach()), "training_sparsity": float(current_sparsity)})
        if progress is not None and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            progress({"stage": "flow_refinement", "epochs": epochs, **losses[-1]})
    # Upstream returns the flow scores from the last forward, before the last
    # optimizer step. Preserve that convention rather than recomputing here.
    return flows.detach(), layer_scores.detach(), losses, {
        "coalition_logits_after_last_step": coalition.detach().reshape(-1),
        "instance_norm_weight_after_last_step": norm.weight.detach(),
        "instance_norm_bias_after_last_step": norm.bias.detach(),
    }


def flowx_scores(model, x, edge_index, target, root, k, variant,
                 iterations=50, epochs=500, lr=0.1, progress=None):
    start = time.perf_counter()
    carriers, walks = enumerate_flows(edge_index, len(x), model.num_layers, root)
    nonloop_edges, input_mapping, loop_counts = carrier_layout(edge_index, len(x))
    c, d = carriers.shape[1], nonloop_edges.shape[1]
    # Verify the unmasked native-self-carrier implementation for every record.
    with torch.no_grad():
        native = carrier_logits(model, x, edge_index, x.new_ones((model.num_layers, 1, c)), root)
        original = model(x, edge_index)
        expected = original if root is None else original[root:root + 1]
        torch.testing.assert_close(native, expected, atol=1e-5, rtol=1e-5)
    sampled = sample_flows(model, x, edge_index, walks, root, iterations, progress=progress)
    sample_time = time.perf_counter() - start
    # Match the requested fraction of physical non-loop edges. Native training
    # still covers all directional/self carriers, as in the original algorithm.
    sparsity = 1 - k / (d / 2)
    flows, layers, losses, parameters = refine_flows(
        model, x, edge_index, walks, root, target, sampled, variant, sparsity, epochs, lr, progress)
    additive = layers.sum(0)
    presence = torch.bincount(walks.reshape(-1), minlength=c) > 0
    native_scores = additive.masked_fill(~presence, -torch.inf)
    extra = {"flow_ids": walks, "flow_scores": flows,
             "carrier_edge_index": carriers, "layer_carrier_scores": layers,
             "input_edge_to_carrier": input_mapping, "explicit_self_loop_counts": loop_counts,
             "carrier_has_flow": presence, "native_carrier_scores": native_scores,
             "additive_carrier_scores": additive, "flow_sample_counts": sampled["sample_counts"],
             "sampling_permutations": sampled["permutations"],
             "sampling_marginals": sampled["marginals"],
             "sampling_group_width": sampled["group_width"], "sampling_group_count": sampled["group_count"],
             "flowx_training_sparsity": sparsity, "flowx_sampling_time_s": sample_time,
             "flowx_max_native_logit_error": float((native - expected).abs().max()), **parameters}
    extra = {name: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
             for name, value in extra.items()}
    if not torch.isfinite(additive).all():
        raise ValueError("Nonfinite additive FlowX scores")
    # Missing flows have additive contribution zero. Preserve the author's
    # -inf ranking sentinel separately, never average it into the reverse edge.
    return additive[input_mapping].cpu().numpy(), extra, losses
