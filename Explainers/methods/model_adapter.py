"""Weight-preserving PyG interface and temporary edge masks."""
from __future__ import annotations

import torch

from Main.models.vanilla_pyg_compat import (
    _CompatGNNLayer, build_pyg_compatible_from_vanilla,
)



class DIGLayer(_CompatGNNLayer):
    """Consume DIG's legacy mask once, without PyG's modern masking hooks."""

    def __setattr__(self, name, value):
        if name == "__edge_mask__":
            # GNNExplainer supplies a Parameter, PGExplainer a non-leaf Tensor.
            # This is a transient reference owned by the explainer, not a GNN
            # parameter. Registering it pollutes state_dict and prevents reuse.
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def forward(self, x, edge_index, edge_weight=None):
        count = edge_index.shape[1]
        mask = getattr(self, "__edge_mask__", None) if self._explain else None
        if mask is not None:
            if mask.numel() < count:
                raise ValueError("DIG mask is shorter than directed edge_index")
            edge_weight = mask[:count].to(x)
        if edge_weight is None:
            edge_weight = x.new_ones(count)
        if edge_weight.numel() != count:
            raise ValueError("Edge-weight length mismatch")
        # Explicit self loops and the GIN self term are not explanation variables.
        edge_weight = torch.where(edge_index[0] == edge_index[1],
                                  torch.ones_like(edge_weight), edge_weight)
        messages = x[edge_index[0]] * edge_weight.reshape(-1, 1)
        neighbors = x.new_zeros(x.shape).index_add(0, edge_index[1], messages)
        return self.mlp((1 + self.eps) * x + neighbors)


def compatible_model(original, task):
    clone = build_pyg_compatible_from_vanilla(original, task)
    for idx in range(1, clone.num_layers + 1):
        old = getattr(clone, f"conv{idx}")
        layer = DIGLayer(old.mlp.fc1.in_features, old.mlp.fc2.out_features,
                         mlp_hidden=old.mlp.fc1.out_features,
                         learn_eps=isinstance(old.eps, torch.nn.Parameter),
                         use_batchnorm=old.mlp.use_batchnorm)
        layer.load_state_dict(old.state_dict(), strict=True)
        setattr(clone, f"conv{idx}", layer)
    # A plain list avoids registering each convolution twice in the state_dict.
    clone.convs = [getattr(clone, f"conv{i}") for i in range(1, clone.num_layers + 1)]
    clone.eval()
    clone.requires_grad_(False)
    return clone


def clear_masks(model):
    for layer in model.convs:
        layer._explain = False
        layer.__edge_mask__ = None
