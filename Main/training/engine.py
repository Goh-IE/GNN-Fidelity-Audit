import os
import random
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from typing import Optional

from Main.data.dataset_loaders import load_dataset
from Main.models.model_selector import model_selector, normalize_dataset_name

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_ROOT = os.path.join(MAIN_DIR, "checkpoints")


def _build_model_dir_name(
    canonical_name: str,
    num_layers: int,
    hidden_dim: int,
    use_batchnorm: bool = False,
    use_jk_concat: bool = True,
    normalize_features: bool = True,
    nn_seed: Optional[int] = None,
) -> str:
    """Build a model directory name with optional seed suffix."""
    bn_suffix = "_BN" if use_batchnorm else ""
    jk_suffix = "_JK" if use_jk_concat else ""
    feat_suffix = "" if normalize_features else "_rawfeat"
    seed_suffix = f"_seed{int(nn_seed)}" if nn_seed is not None else ""
    return f"{canonical_name}_{num_layers}L_{hidden_dim}H{bn_suffix}{jk_suffix}{feat_suffix}{seed_suffix}"


def create_data_list(graphs, features, labels, mask):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in indices:
        x = torch.tensor(features[i])
        edge_index = torch.tensor(graphs[i])
        # Handle both one-hot encoded labels and scalar labels
        label = labels[i]
        if hasattr(label, '__len__') and len(label) > 1:
            # One-hot encoded: use argmax
            y = torch.tensor(label.argmax())
        else:
            # Scalar label (possibly shape (1,) or scalar)
            if hasattr(label, 'item'):
                y = torch.tensor(int(label.item()))
            elif hasattr(label, '__len__'):
                y = torch.tensor(int(label[0]))
            else:
                y = torch.tensor(int(label))
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the model
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc


def store_checkpoint(
    paper,
    dataset,
    model,
    train_acc,
    val_acc,
    test_acc,
    epoch=-1,
    num_layers=3,
    hidden_dim=20,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features=True,
    config=None,
    nn_seed: Optional[int] = None,
):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper 
    :param dataset: str, the dataset (supports both canonical and legacy names)
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :param num_layers: number of GNN layers (2 or 3). Only used for GIN models.
    :param hidden_dim: hidden dimension for GIN models.
    :param use_batchnorm: whether batch normalization is used. Only for GIN models.
    :param use_jk_concat: whether JK-concat is used. Only for GIN models.
    :param config: training configuration object
    :retunrs: None
    """
    # Normalize to canonical name for directory
    canonical_name = normalize_dataset_name(dataset)
    
    if nn_seed is None and config is not None:
        nn_seed = getattr(config, "seed", None)

    # Use canonical name for directory path
    save_dir = os.path.join(
        CHECKPOINT_ROOT,
        paper,
        _build_model_dir_name(
            canonical_name,
            num_layers,
            hidden_dim,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
            normalize_features=normalize_features,
            nn_seed=nn_seed,
        ),
    )
    
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc,
                  'model_config': {
                      'paper': paper,
                      'dataset': canonical_name,
                      'num_layers': num_layers,
                      'hidden_dim': hidden_dim,
                      'use_batchnorm': use_batchnorm,
                      'use_jk_concat': use_jk_concat,
                      'normalize_features': normalize_features,
                      'nn_seed': nn_seed,
                  }}
    
    # Add training hyperparameters if config is provided
    if config is not None:
        checkpoint['training_config'] = {
            'lr': getattr(config, 'lr', None),
            'epochs': getattr(config, 'epochs', None),
            'clip_max': getattr(config, 'clip_max', None),
            'batch_size': getattr(config, 'batch_size', None),
            'early_stopping': getattr(config, 'early_stopping', None),
            'normalize_features': getattr(config, 'normalize_features', normalize_features),
            'seed': getattr(config, 'seed', None),
            'eval_enabled': getattr(config, 'eval_enabled', None)
        }
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def load_best_model(
    best_epoch,
    paper,
    dataset,
    model,
    eval_enabled,
    num_layers=3,
    hidden_dim=20,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features=True,
    nn_seed: Optional[int] = None,
):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper 
    :param dataset: str, the dataset (supports both canonical and legacy names)
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :param num_layers: number of GNN layers (2 or 3). Only used for GIN models.
    :param hidden_dim: hidden dimension for GIN models.
    :param use_batchnorm: whether batch normalization is used. Only for GIN models.
    :param use_jk_concat: whether JK-concat is used. Only for GIN models.
    :return: model with pramaters taken from the checkpoint
    """
    print(best_epoch)
    # Normalize to canonical name
    canonical_name = normalize_dataset_name(dataset)
    
    seeded_dir = os.path.join(
        CHECKPOINT_ROOT,
        paper,
        _build_model_dir_name(
            canonical_name,
            num_layers,
            hidden_dim,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
            normalize_features=normalize_features,
            nn_seed=nn_seed,
        ),
    )
    legacy_dir = os.path.join(
        CHECKPOINT_ROOT,
        paper,
        _build_model_dir_name(
            canonical_name,
            num_layers,
            hidden_dim,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
            normalize_features=normalize_features,
            nn_seed=None,
        ),
    )
    save_dir = seeded_dir
    if nn_seed is not None and not os.path.isdir(seeded_dir) and os.path.isdir(legacy_dir):
        save_dir = legacy_dir

    if best_epoch == -1:
        checkpoint = torch.load(f"{save_dir}/best_model")
    else:
        checkpoint = torch.load(f"{save_dir}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model


def train_node(_dataset, _paper, config, num_layers=3, hidden_dim=20, use_batchnorm=False, use_jk_concat=True):
    """
    Train a GIN for node classification.
    :param _dataset: the dataset we wish to use for training
    :param _paper: checkpoint namespace of the model, "GIN"
    :param config: training configuration object
    :param num_layers: number of GNN layers (2 or 3). Only used for GIN models.
    :param hidden_dim: hidden dimension for GIN models.
    :param use_batchnorm: whether to use batch normalization. Only for GIN models.
    :param use_jk_concat: whether to use JK-concat. Only for GIN models.
    """
    normalize_features = getattr(config, "normalize_features", True)
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset(
        _dataset,
        normalize_features=normalize_features,
    )
    nn_seed = getattr(config, "seed", None)
    
    # Set random seeds for reproducibility (affects model initialization)
    if nn_seed is not None:
        random.seed(nn_seed)
        np.random.seed(nn_seed)
        torch.manual_seed(nn_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(nn_seed)
            torch.cuda.manual_seed_all(nn_seed)
    
    model = model_selector(_paper, _dataset, False, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)

    x = torch.tensor(features)
    edge_index = torch.tensor(graph)
    labels = torch.tensor(labels)

    # Define graph
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, config.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max)
        optimizer.step()

        if config.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        checkpoint_stored = False
        if val_acc > best_val_acc: # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(
                _paper,
                _dataset,
                model,
                train_acc,
                val_acc,
                test_acc,
                best_epoch,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                use_batchnorm=use_batchnorm,
                use_jk_concat=use_jk_concat,
                normalize_features=normalize_features,
                config=config,
                nn_seed=nn_seed,
            )
            checkpoint_stored = True

        if val_acc >= 1.0:
            best_epoch = epoch
            if not checkpoint_stored:
                store_checkpoint(
                    _paper,
                    _dataset,
                    model,
                    train_acc,
                    val_acc,
                    test_acc,
                    best_epoch,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    use_batchnorm=use_batchnorm,
                    use_jk_concat=use_jk_concat,
                    normalize_features=normalize_features,
                    config=config,
                    nn_seed=nn_seed,
                )
            print(f"Perfect validation accuracy reached at epoch {epoch}; stopping training early.")
            break
    model = load_best_model(
        best_epoch,
        _paper,
        _dataset,
        model,
        config.eval_enabled,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        use_batchnorm=use_batchnorm,
        use_jk_concat=use_jk_concat,
        normalize_features=normalize_features,
        nn_seed=nn_seed,
    )
    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(
        _paper,
        _dataset,
        model,
        train_acc,
        val_acc,
        test_acc,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        use_batchnorm=use_batchnorm,
        use_jk_concat=use_jk_concat,
        normalize_features=normalize_features,
        config=config,
        nn_seed=nn_seed,
    )
    # Return performance metrics
    return {
        'train_acc': train_acc,
        'val_acc': val_acc, 
        'test_acc': test_acc,
        'best_epoch': best_epoch
    }

def train_graph(_dataset, _paper, config, num_layers=3, hidden_dim=20, use_batchnorm=False, use_jk_concat=True):
    """
    Train a GIN for graph classification.
    :param _dataset: the dataset we wish to use for training
    :param _paper: checkpoint namespace of the model, "GIN"
    :param config: training configuration object
    :param num_layers: number of GNN layers (2 or 3). Only used for GIN models.
    :param hidden_dim: hidden dimension for GIN models.
    :param use_batchnorm: whether to use batch normalization. Only for GIN models.
    :param use_jk_concat: whether to use JK-concat. Only for GIN models.
    """
    normalize_features = getattr(config, "normalize_features", True)
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(
        _dataset,
        normalize_features=normalize_features,
    )
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    nn_seed = getattr(config, "seed", None)
    
    # Set random seeds for reproducibility (affects model initialization)
    if nn_seed is not None:
        random.seed(nn_seed)
        np.random.seed(nn_seed)
        torch.manual_seed(nn_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(nn_seed)
            torch.cuda.manual_seed_all(nn_seed)
    
    model = model_selector(_paper, _dataset, False, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, config.epochs):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max)
            optimizer.step()

        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                loss += criterion(out, data.y)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))
            train_loss = float(loss) / int(len(train_loader))

            eval_data = next(iter(test_loader)) # Loads all test samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader)) # Loads all eval samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            val_acc = evaluate(out, eval_data.y)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        checkpoint_stored = False
        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(
                _paper,
                _dataset,
                model,
                train_acc,
                val_acc,
                test_acc,
                best_epoch,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                use_batchnorm=use_batchnorm,
                use_jk_concat=use_jk_concat,
                normalize_features=normalize_features,
                config=config,
                nn_seed=nn_seed,
            )
            checkpoint_stored = True

        if val_acc >= 1.0:
            best_epoch = epoch
            if not checkpoint_stored:
                store_checkpoint(
                    _paper,
                    _dataset,
                    model,
                    train_acc,
                    val_acc,
                    test_acc,
                    best_epoch,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    use_batchnorm=use_batchnorm,
                    use_jk_concat=use_jk_concat,
                    normalize_features=normalize_features,
                    config=config,
                    nn_seed=nn_seed,
                )
            print(f"Perfect validation accuracy reached at epoch {epoch}; stopping training early.")
            break

    model = load_best_model(
        best_epoch,
        _paper,
        _dataset,
        model,
        config.eval_enabled,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        use_batchnorm=use_batchnorm,
        use_jk_concat=use_jk_concat,
        normalize_features=normalize_features,
        nn_seed=nn_seed,
    )

    with torch.no_grad():
        train_sum = 0
        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))

        eval_data = next(iter(test_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader))
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
        val_acc = evaluate(out, eval_data.y)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(
        _paper,
        _dataset,
        model,
        train_acc,
        val_acc,
        test_acc,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        use_batchnorm=use_batchnorm,
        use_jk_concat=use_jk_concat,
        normalize_features=normalize_features,
        config=config,
        nn_seed=nn_seed,
    )
    
    # Return performance metrics
    return {
        'train_acc': train_acc,
        'val_acc': val_acc, 
        'test_acc': test_acc,
        'best_epoch': best_epoch
    }
