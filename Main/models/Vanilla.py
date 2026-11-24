import torch
from torch.nn import BatchNorm1d, Linear, ReLU


class NeighborSum(torch.nn.Module):
    """
    Computes neighbor sum: (A_w * x)_i = sum_{j -> i} w_{j,i} * x_j
    No self-loop here; self-term is handled explicitly by the GNN layer.
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (num_nodes, num_features), got {tuple(x.shape)}")
        if edge_index.numel() == 0:
            return x.new_zeros((x.size(0), x.size(1)))
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index shape (2, num_edges), got {tuple(edge_index.shape)}")

        src, dst = edge_index[0], edge_index[1]
        num_edges = int(edge_index.size(1))

        if edge_weight is None:
            edge_weight = x.new_ones((num_edges,))
        else:
            edge_weight = edge_weight.to(dtype=x.dtype, device=x.device)
            if edge_weight.numel() != num_edges:
                raise ValueError(
                    f"edge_weight must have shape (num_edges,), got {tuple(edge_weight.shape)} vs num_edges={num_edges}"
                )

        msg = x.index_select(0, src) * edge_weight.view(-1, 1)
        agg = x.new_zeros((x.size(0), x.size(1)))
        agg.index_add_(0, dst, msg)
        return agg


class MLP2(torch.nn.Module):
    """MILP-friendly MLP: Linear -> ReLU -> Linear (no BN/Dropout)."""

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


class GNNLayer(torch.nn.Module):
    """
    GNN update:
      h_next = MLP( (1 + eps) * h + sum_{j->i} w_{j,i} * h_j )

    eps can be learnable (recommended) or fixed.
    use_batchnorm: If True, apply BatchNorm1d after the MLP output.
    """

    def __init__(self, in_dim: int, out_dim: int, *, mlp_hidden: int | None = None, learn_eps: bool = True, use_batchnorm: bool = False):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = out_dim
        self.agg = NeighborSum()
        self.mlp = MLP2(in_dim, int(mlp_hidden), out_dim, use_batchnorm=use_batchnorm)
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.zeros(()))  # scalar
        else:
            self.register_buffer("eps", torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        neigh = self.agg(x, edge_index, edge_weight)
        pre = (1.0 + self.eps) * x + neigh
        return self.mlp(pre)


def _global_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0:
        return x.new_zeros((0, x.size(-1)))
    num_graphs = int(batch.max().item()) + 1
    out = x.new_zeros((num_graphs, x.size(-1)))
    out.index_add_(0, batch, x)
    return out


class NodeGNN(torch.nn.Module):
    """
    GNN for node classification with configurable number of layers (2, 3, or 4).
    
    use_jk_concat: If True, uses JK-concat (concatenate embeddings from all layers).
                   This is required for MIP-based explanations.
                   If False, uses only the last layer embedding.
    use_batchnorm: If True, apply BatchNorm1d after each GNN layer's MLP.
    """

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
        use_jk_concat: bool = True,  # Default True for MIP compatibility
    ):
        super().__init__()
        if num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")
        self.num_layers = num_layers
        self.hidden_dim = int(hidden_dim)
        self.use_jk_concat = use_jk_concat
        self.use_batchnorm = use_batchnorm
        
        # Embedding size depends on JK-concat setting
        if use_jk_concat:
            self.embedding_size = self.hidden_dim * num_layers  # JK-concat
        else:
            self.embedding_size = self.hidden_dim  # Last layer only

        self.conv1 = GNNLayer(num_features, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        self.conv2 = GNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 3:
            self.conv3 = GNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 4:
            self.conv4 = GNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)

        self.lin = Linear(self.embedding_size, int(num_classes))

    def embedding(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h1 = self.conv1(x, edge_index, edge_weight)
        h2 = self.conv2(h1, edge_index, edge_weight)
        if self.num_layers == 2:
            if self.use_jk_concat:
                return torch.cat([h1, h2], dim=1)
            else:
                return h2
        h3 = self.conv3(h2, edge_index, edge_weight)
        if self.num_layers == 3:
            if self.use_jk_concat:
                return torch.cat([h1, h2, h3], dim=1)
            else:
                return h3
        h4 = self.conv4(h3, edge_index, edge_weight)
        if self.use_jk_concat:
            return torch.cat([h1, h2, h3, h4], dim=1)
        else:
            return h4

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None, edge_weights: torch.Tensor | None = None) -> torch.Tensor:
        # Support both edge_weight and edge_weights (PGExplainer uses edge_weights)
        ew = edge_weights if edge_weights is not None else edge_weight
        return self.lin(self.embedding(x, edge_index, ew))


class GraphGNN(torch.nn.Module):
    """
    GNN for graph classification with configurable number of layers (2, 3, or 4).
    
    use_jk_concat: If True, uses JK-concat (concatenate pooled embeddings from all layers).
                   This is required for MIP-based explanations.
                   If False, uses only the last layer embedding (default for training).
    use_batchnorm: If True, apply BatchNorm1d after each GNN layer's MLP.
    """

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
        use_jk_concat: bool = True,  # Default True for MIP compatibility
    ):
        super().__init__()
        if num_layers not in (2, 3, 4):
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")
        self.num_layers = num_layers
        self.hidden_dim = int(hidden_dim)
        self.use_jk_concat = use_jk_concat
        self.use_batchnorm = use_batchnorm
        
        # Embedding size depends on JK-concat setting
        if use_jk_concat:
            self.embedding_size = self.hidden_dim * num_layers  # JK-concat
        else:
            self.embedding_size = self.hidden_dim  # Last layer only

        self.conv1 = GNNLayer(num_features, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        self.conv2 = GNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 3:
            self.conv3 = GNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)
        if num_layers >= 4:
            self.conv4 = GNNLayer(self.hidden_dim, self.hidden_dim, mlp_hidden=mlp_hidden, learn_eps=learn_eps, use_batchnorm=use_batchnorm)

        self.lin = Linear(self.embedding_size, int(num_classes))

    def _get_layer_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> tuple:
        """Returns tuple of layer embeddings."""
        h1 = self.conv1(x, edge_index, edge_weight)
        h2 = self.conv2(h1, edge_index, edge_weight)
        if self.num_layers == 2:
            return (h1, h2)
        h3 = self.conv3(h2, edge_index, edge_weight)
        if self.num_layers == 3:
            return (h1, h2, h3)
        h4 = self.conv4(h3, edge_index, edge_weight)
        return (h1, h2, h3, h4)

    def embedding(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """Returns embedding based on use_jk_concat setting."""
        embeddings = self._get_layer_embeddings(x, edge_index, edge_weight)
        if self.use_jk_concat:
            # Concatenate all layer embeddings
            return torch.cat(embeddings, dim=1)
        else:
            # Return only last layer embedding
            return embeddings[-1]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Support both edge_weight and edge_weights (PGExplainer uses edge_weights)
        ew = edge_weights if edge_weights is not None else edge_weight
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.use_jk_concat:
            # JK-concat: pool each layer separately then concatenate
            embeddings = self._get_layer_embeddings(x, edge_index, ew)
            pooled_layers = [_global_add_pool(h, batch) for h in embeddings]
            pooled = torch.cat(pooled_layers, dim=1)
        else:
            # Last layer only
            h = self.embedding(x, edge_index, ew)
            pooled = _global_add_pool(h, batch)
        
        return self.lin(pooled)

