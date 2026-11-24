"""GCN classifiers with the same depth and readout interface as Vanilla GIN.

The convolution uses PyG's canonical symmetric degree normalization and added
self-loops.  ``cached=False`` is intentional because explanation evaluation
changes ``edge_index`` from one masked graph to another.
"""

import torch
from torch.nn import BatchNorm1d, Linear
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class _GCNBackbone(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        hidden_dim: int = 20,
        num_layers: int = 3,
        use_batchnorm: bool = False,
        use_jk_concat: bool = False,
    ):
        super().__init__()
        if num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")

        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.use_batchnorm = bool(use_batchnorm)
        self.use_jk_concat = bool(use_jk_concat)
        self.embedding_size = self.hidden_dim * self.num_layers if self.use_jk_concat else self.hidden_dim

        in_channels = int(num_features)
        for layer_idx in range(1, self.num_layers + 1):
            setattr(
                self,
                f"conv{layer_idx}",
                GCNConv(
                    in_channels,
                    self.hidden_dim,
                    improved=False,
                    cached=False,
                    add_self_loops=True,
                    normalize=True,
                    bias=True,
                ),
            )
            if self.use_batchnorm:
                setattr(self, f"bn{layer_idx}", BatchNorm1d(self.hidden_dim))
            in_channels = self.hidden_dim

    def _get_layer_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        embeddings = []
        h = x
        for layer_idx in range(1, self.num_layers + 1):
            conv = getattr(self, f"conv{layer_idx}")
            h = conv(h, edge_index, edge_weight=edge_weight)
            if self.use_batchnorm:
                h = getattr(self, f"bn{layer_idx}")(h)
            h = F.relu(h)
            embeddings.append(h)
        return tuple(embeddings)

    def embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self._get_layer_embeddings(x, edge_index, edge_weight)
        if self.use_jk_concat:
            return torch.cat(embeddings, dim=1)
        return embeddings[-1]


class NodeGCN(_GCNBackbone):
    """Node-classification GCN."""

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(num_features, **kwargs)
        self.lin = Linear(self.embedding_size, int(num_classes))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ew = edge_weights if edge_weights is not None else edge_weight
        return self.lin(self.embedding(x, edge_index, ew))


class GraphGCN(_GCNBackbone):
    """Graph-classification GCN with the same sum readout as Vanilla GIN."""

    def __init__(self, num_features: int, num_classes: int, **kwargs):
        super().__init__(num_features, **kwargs)
        self.lin = Linear(self.embedding_size, int(num_classes))

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

        embeddings = self._get_layer_embeddings(x, edge_index, ew)
        if self.use_jk_concat:
            pooled = torch.cat([global_add_pool(h, batch) for h in embeddings], dim=1)
        else:
            pooled = global_add_pool(embeddings[-1], batch)
        return self.lin(pooled)
