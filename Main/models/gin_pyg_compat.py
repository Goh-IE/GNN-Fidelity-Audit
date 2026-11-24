"""PyG MessagePassing-compatible wrappers for GIN models.

These classes mirror ``Main.models.GIN`` but implement
layers with ``torch_geometric.nn.MessagePassing`` so DIG/PyG explainers can
attach edge masks.
"""

from __future__ import annotations

import torch
from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.nn import MessagePassing


def _global_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0:
        return x.new_zeros((0, x.size(-1)))
    num_graphs = int(batch.max().item()) + 1
    out = x.new_zeros((num_graphs, x.size(-1)))
    out.index_add_(0, batch, x)
    return out


class _CompatMLP2(torch.nn.Module):
    """Linear -> ReLU -> Linear, optionally followed by BatchNorm1d."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_batchnorm: bool = False):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.act = ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(x)))
        if self.use_batchnorm:
            out = self.bn(out)
        return out


class _CompatGNNLayer(MessagePassing):
    """MessagePassing implementation matching GIN.GNNLayer semantics."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        mlp_hidden: int | None = None,
        learn_eps: bool = True,
        use_batchnorm: bool = False,
    ):
        super().__init__(aggr="add", flow="source_to_target")
        if mlp_hidden is None:
            mlp_hidden = out_dim
        self.mlp = _CompatMLP2(in_dim, int(mlp_hidden), out_dim, use_batchnorm=use_batchnorm)
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("eps", torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_edges = int(edge_index.size(1))
        dig_mask = None
        if bool(getattr(self, "__explain_flow__", False)):
            dig_mask = getattr(self, "layer_edge_mask", None)
        elif bool(getattr(self, "_explain", False)):
            dig_mask = getattr(self, "__edge_mask__", None)

        if dig_mask is not None:
            edge_weight = dig_mask.to(dtype=x.dtype, device=x.device).reshape(-1)
            if edge_weight.numel() < num_edges:
                raise ValueError(
                    f"DIG edge mask has fewer entries than edge_index: "
                    f"{edge_weight.numel()} vs {num_edges}"
                )
            edge_weight = edge_weight[:num_edges]
        elif edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        else:
            edge_weight = edge_weight.to(dtype=x.dtype, device=x.device)
            if edge_weight.numel() != num_edges:
                raise ValueError(
                    f"edge_weight must have shape (num_edges,), got {tuple(edge_weight.shape)} vs {num_edges}"
                )

        neigh = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)
        pre = (1.0 + self.eps) * x + neigh
        return self.mlp(pre)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.view(-1, 1)


class CompatNodeGNN(torch.nn.Module):
    """MessagePassing-compatible counterpart of ``GIN.NodeGNN``."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        hidden_dim: int = 20,
        mlp_hidden: int | None = None,
        learn_eps: bool = True,
        num_layers: int = 3,
        use_batchnorm: bool = False,
        use_jk_concat: bool = True,
    ):
        super().__init__()
        if num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")
        self.num_layers = num_layers
        self.hidden_dim = int(hidden_dim)
        self.use_jk_concat = use_jk_concat
        self.use_batchnorm = use_batchnorm

        if use_jk_concat:
            self.embedding_size = self.hidden_dim * num_layers
        else:
            self.embedding_size = self.hidden_dim

        self.conv1 = _CompatGNNLayer(num_features, self.hidden_dim, mlp_hidden=mlp_hidden,
                                     learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        self.conv2 = _CompatGNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden,
                                     learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 3:
            self.conv3 = _CompatGNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden,
                                         learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 4:
            self.conv4 = _CompatGNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden,
                                         learn_eps=learn_eps, use_batchnorm=use_batchnorm)

        self.lin = Linear(self.embedding_size, int(num_classes))

    def embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h1 = self.conv1(x, edge_index, edge_weight)
        h2 = self.conv2(h1, edge_index, edge_weight)
        if self.num_layers == 2:
            return torch.cat([h1, h2], dim=1) if self.use_jk_concat else h2

        h3 = self.conv3(h2, edge_index, edge_weight)
        if self.num_layers == 3:
            return torch.cat([h1, h2, h3], dim=1) if self.use_jk_concat else h3

        h4 = self.conv4(h3, edge_index, edge_weight)
        return torch.cat([h1, h2, h3, h4], dim=1) if self.use_jk_concat else h4

    def get_emb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embedding(x, edge_index, edge_weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ew = edge_weights if edge_weights is not None else edge_weight
        return self.lin(self.embedding(x, edge_index, ew))


class CompatGraphGNN(torch.nn.Module):
    """MessagePassing-compatible counterpart of ``GIN.GraphGNN``."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        hidden_dim: int = 20,
        mlp_hidden: int | None = None,
        learn_eps: bool = True,
        num_layers: int = 3,
        use_batchnorm: bool = False,
        use_jk_concat: bool = True,
    ):
        super().__init__()
        if num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")
        self.num_layers = num_layers
        self.hidden_dim = int(hidden_dim)
        self.use_jk_concat = use_jk_concat
        self.use_batchnorm = use_batchnorm

        if use_jk_concat:
            self.embedding_size = self.hidden_dim * num_layers
        else:
            self.embedding_size = self.hidden_dim

        self.conv1 = _CompatGNNLayer(num_features, self.hidden_dim, mlp_hidden=mlp_hidden,
                                     learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        self.conv2 = _CompatGNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden,
                                     learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 3:
            self.conv3 = _CompatGNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden,
                                         learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 4:
            self.conv4 = _CompatGNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden,
                                         learn_eps=learn_eps, use_batchnorm=use_batchnorm)

        self.lin = Linear(self.embedding_size, int(num_classes))

    def _get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        h1 = self.conv1(x, edge_index, edge_weight)
        h2 = self.conv2(h1, edge_index, edge_weight)
        if self.num_layers == 2:
            return (h1, h2)

        h3 = self.conv3(h2, edge_index, edge_weight)
        if self.num_layers == 3:
            return (h1, h2, h3)

        h4 = self.conv4(h3, edge_index, edge_weight)
        return (h1, h2, h3, h4)

    def embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embs = self._get_layer_embeddings(x, edge_index, edge_weight)
        return torch.cat(embs, dim=1) if self.use_jk_concat else embs[-1]

    def get_emb(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embedding(x, edge_index, edge_weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ew = edge_weights if edge_weights is not None else edge_weight
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if self.use_jk_concat:
            embs = self._get_layer_embeddings(x, edge_index, ew)
            pooled = torch.cat([_global_add_pool(h, batch) for h in embs], dim=1)
        else:
            h = self.embedding(x, edge_index, ew)
            pooled = _global_add_pool(h, batch)
        return self.lin(pooled)


def build_pyg_compatible_from_gin(model: torch.nn.Module, task: str) -> torch.nn.Module:
    """Create a MessagePassing-compatible clone from a pretrained GIN model."""
    num_features = int(model.conv1.mlp.fc1.in_features)
    num_classes = int(model.lin.out_features)
    kwargs = dict(
        hidden_dim=int(model.hidden_dim),
        num_layers=int(model.num_layers),
        use_batchnorm=bool(getattr(model, "use_batchnorm", False)),
        use_jk_concat=bool(getattr(model, "use_jk_concat", True)),
    )

    if task == "node":
        compat_model = CompatNodeGNN(num_features, num_classes, **kwargs)
    elif task == "graph":
        compat_model = CompatGraphGNN(num_features, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task}")

    compat_model.load_state_dict(model.state_dict(), strict=True)
    compat_model.eval()
    return compat_model
