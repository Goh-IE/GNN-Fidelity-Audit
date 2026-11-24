"""PGExplainer training and attribution on the audit cohort."""
from __future__ import annotations

import time
import numpy as np
import torch

from Explainers.methods.dig_loader import load_dig
from Explainers.methods.model_adapter import clear_masks



def train_pg(model, task, examples, epochs, progress=None):
    """DIG's full-batch Adam loop over an explicit training-only subset.

    examples: (original ID, compact features, edges, local target or None).
    The dataset loader/split remains outside the explainer and is recorded.
    """
    _, module = load_dig()
    pg = module.PGExplainer(model, model.embedding_size * (2 if task == "graph" else 3),
                           device=torch.device("cpu"), explain_graph=task == "graph",
                           epochs=epochs, num_hops=model.num_layers)
    optimizer = torch.optim.Adam(pg.elayers.parameters(), lr=pg.lr)
    prepared = []
    with torch.no_grad():
        for instance_id, x, edges, root in examples:
            logits = model(x, edges)
            pred = int((logits[0] if root is None else logits[root]).argmax())
            probability = (logits[0] if root is None else logits[root]).softmax(-1)[pred]
            constant_loss = float(-torch.log(probability + module.EPS))
            prepared.append((instance_id, x, edges, root, model.get_emb(x, edges).detach(), pred,
                             constant_loss))
    losses = []
    started = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        total = 0.0
        temperature = pg.t0 * np.power(pg.t1 / pg.t0, epoch / epochs)
        for _, x, edges, root, emb, pred, constant_loss in prepared:
            if edges.shape[1] == 0:
                # An isolated training target has no mask parameters. Its
                # prediction term is constant, and both empty-mask penalties
                # are zero. Keep the target in the training count and loss,
                # avoiding DIG's undefined mean entropy on an empty tensor.
                total += constant_loss
                continue
            probs, _ = pg.explain(x, edges, emb, tmp=temperature, training=True, node_idx=root)
            loss = pg.__loss__(probs[0] if root is None else probs[root], pred)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite PGExplainer training loss")
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        losses.append({"epoch": epoch + 1, "loss": total,
                       "temperature": float(temperature), "training_count": len(prepared),
                       "empty_edge_training_ids": [row[0] for row in prepared if row[2].shape[1] == 0]})
        if progress is not None:
            progress({**losses[-1], "elapsed_s": time.perf_counter() - started})
    pg.elayers.eval()
    clear_masks(model)
    return pg, losses


def pg_scores(pg, model, x, edge_index, root):
    with torch.no_grad():
        _, scores = pg.explain(x, edge_index, model.get_emb(x, edge_index),
                               training=False, node_idx=root)
    clear_masks(model)
    return scores.cpu().numpy(), {}, []
