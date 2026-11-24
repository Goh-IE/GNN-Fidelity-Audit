"""GNNExplainer attribution for the frozen audit GNN."""
from __future__ import annotations

import torch
from torch_geometric.utils import add_remaining_self_loops

from Explainers.methods.dig_loader import load_dig



def gnn_scores(model, x, edge_index, target, root, epochs):
    module, _ = load_dig()
    explainer = module.GNNExplainer(model, epochs=epochs, explain_graph=root is None)
    explainer.device = x.device
    explainer.mask_features = False
    explainer.node_idx = root
    mask_edges, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
    explainer.__set_masks__(x, mask_edges)
    optimizer = torch.optim.Adam([explainer.node_feat_mask, explainer.edge_mask],
                                 lr=explainer.lr)
    losses = []
    # DIG's loss and mask initialization, with the predictor frozen. Its original
    # clipping of predictor gradients is inert for the mask optimizer and fails
    # with all predictor gradients None in recent PyTorch, so it is omitted.
    try:
        for epoch in range(epochs):
            logits = model(x, edge_index)
            loss = explainer.__loss__(logits, torch.tensor([target], device=x.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append({"epoch": epoch + 1, "loss": float(loss.detach())})
        raw = explainer.edge_mask.detach().sigmoid().cpu().numpy()
        return raw[:edge_index.shape[1]], {
            "mask_edge_index": mask_edges.cpu().numpy(), "raw_mask_scores": raw,
        }, losses
    finally:
        explainer.__clear_masks__()
