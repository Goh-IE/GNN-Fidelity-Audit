#!/usr/bin/env python3
"""Systematic GIN training with seeded checkpoints."""

import os
import sys
import torch
import argparse
import json
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MAIN_DIR)
CHECKPOINT_ROOT = os.path.join(MAIN_DIR, "checkpoints")
PRETRAINED_ROOT = os.path.join(MAIN_DIR, "models", "pretrained")
TRAINING_CONFIG_DIR = os.path.join(MAIN_DIR, "training", "configs")
TRAINING_LOG_DIR = os.path.join(MAIN_DIR, "training_logs")

# Add the self-contained Main workspace to the import path.
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from Main.training.engine import train_node, train_graph
from Main.training.config import Selector
from Main.models.model_selector import (
    normalize_dataset_name, get_path_name, get_dataset_config,
    NODE_DATASETS, GRAPH_DATASETS, ALL_DATASETS
)


def save_hyperparameters(
    paper,
    dataset,
    num_layers,
    hidden_dim,
    config,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features=True,
    performance_metrics=None,
    nn_seed=None,
):
    """Save hyperparameters and performance metrics to both checkpoint and pretrained directories"""
    # Normalize to canonical name for directory
    canonical_name = normalize_dataset_name(dataset)
    
    # Prepare hyperparameter dict
    hyperparams = {
        'model_config': {
            'paper': paper,
            'dataset': canonical_name,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'use_batchnorm': use_batchnorm,
            'use_jk_concat': use_jk_concat,
            'normalize_features': normalize_features,
        },
        'training_config': {
            'lr': config.lr,
            'epochs': config.epochs,
            'clip_max': config.clip_max,
            'batch_size': config.batch_size,
            'early_stopping': config.early_stopping,
            'normalize_features': getattr(config, 'normalize_features', normalize_features),
            'seed': config.seed,
            'eval_enabled': config.eval_enabled
        },
        'saved_at': datetime.now().isoformat()
    }
    
    # Add performance metrics if provided
    if performance_metrics:
        hyperparams['performance'] = performance_metrics
    
    if nn_seed is None:
        nn_seed = getattr(config, "seed", None)

    # Build model name suffix
    bn_suffix = "_BN" if use_batchnorm else ""
    jk_suffix = "_JK" if use_jk_concat else ""
    feat_suffix = "" if normalize_features else "_rawfeat"
    seed_suffix = f"_seed{int(nn_seed)}" if nn_seed is not None else ""
    
    # Define directories with canonical names
    model_dir_name = f"{canonical_name}_{num_layers}L_{hidden_dim}H{bn_suffix}{jk_suffix}{feat_suffix}{seed_suffix}"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, paper, model_dir_name)
    pretrained_dir = os.path.join(PRETRAINED_ROOT, paper, model_dir_name)
    
    # Save to both directories
    for directory in [checkpoint_dir, pretrained_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        hyperparams_file = os.path.join(directory, "hyperparameters.json")
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Saved hyperparameters to {hyperparams_file}")


def copy_best_model_to_pretrained(
    paper,
    dataset,
    num_layers,
    hidden_dim,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features=True,
    nn_seed=None,
):
    """Copy the best model and hyperparameters from checkpoints to pretrained directory"""
    import shutil
    
    # Normalize to canonical name
    canonical_name = normalize_dataset_name(dataset)
    
    # Build model name suffix
    bn_suffix = "_BN" if use_batchnorm else ""
    jk_suffix = "_JK" if use_jk_concat else ""
    feat_suffix = "" if normalize_features else "_rawfeat"
    seed_suffix = f"_seed{int(nn_seed)}" if nn_seed is not None else ""
    
    # Use canonical names for directories
    model_dir_name = f"{canonical_name}_{num_layers}L_{hidden_dim}H{bn_suffix}{jk_suffix}{feat_suffix}{seed_suffix}"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, paper, model_dir_name)
    pretrained_dir = os.path.join(PRETRAINED_ROOT, paper, model_dir_name)
    
    # Create pretrained directory if it doesn't exist
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    
    # Files to copy
    files_to_copy = ["best_model", "hyperparameters.json"]
    
    for filename in files_to_copy:
        src_path = os.path.join(checkpoint_dir, filename)
        dst_path = os.path.join(pretrained_dir, filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")
        else:
            print(f"Warning: {src_path} not found!")


def load_dataset_config(dataset, config_dir):
    """Load dataset-specific configuration"""
    canonical_name = normalize_dataset_name(dataset)
    path_name = get_path_name(dataset)

    candidate_files = [
        os.path.join(config_dir, f"model_gin_{canonical_name}.json"),
        os.path.join(config_dir, f"model_gin_{path_name}.json"),
    ]

    from Main.data.pyg_infection_seed_registry import seeded_infection_names
    seeded_names = {name for name, _, _, _ in seeded_infection_names()}
    if canonical_name in seeded_names:
        family = "ER" if "-ER-" in canonical_name else "BA"
        candidate_files.append(
            os.path.join(config_dir, f"model_gin_PyG-Infection-{family}-L3-Split43.json")
        )

    # Preserve order while removing duplicates.
    candidate_files = list(dict.fromkeys(candidate_files))
    existing_config = next((path for path in candidate_files if os.path.exists(path)), None)

    if existing_config is not None:
        config_selector = Selector(existing_config)
        return config_selector.args.model
    else:
        # Fallback to default config - determine task type
        try:
            _, _, task_type = get_dataset_config(canonical_name)
            is_node = task_type == 'node'
        except ValueError:
            is_node = canonical_name in NODE_DATASETS
        
        class DefaultConfig:
            def __init__(self):
                self.lr = 0.003
                self.epochs = 1000
                self.clip_max = 2.0
                self.batch_size = 32 if is_node else 64
                self.early_stopping = None
                self.eval_enabled = True
        return DefaultConfig()


def train_single_model(
    dataset,
    paper,
    num_layers,
    hidden_dim,
    config_dir,
    use_batchnorm=False,
    use_jk_concat=True,
    normalize_features=True,
    nn_seed=42,
):
    """Train a single model configuration"""
    # Normalize to canonical name
    canonical_name = normalize_dataset_name(dataset)
    path_name = get_path_name(dataset)
    
    # Load dataset-specific config (canonical filename preferred, legacy filename fallback)
    dataset_config = load_dataset_config(canonical_name, config_dir)
    dataset_config.seed = int(nn_seed)
    dataset_config.normalize_features = bool(normalize_features)

    print(f"\n{'='*80}")
    print(f"Training {paper} model for {canonical_name}")
    print(f"Layers: {num_layers}, Hidden dim: {hidden_dim}, BatchNorm: {use_batchnorm}, JK-concat: {use_jk_concat}")
    print(f"Config: lr={dataset_config.lr}, epochs={dataset_config.epochs}, batch_size={dataset_config.batch_size}")
    print(f"Feature normalization: {dataset_config.normalize_features}")
    print(f"NN seed: {dataset_config.seed}")
    print(f"{'='*80}")
    
    try:
        # Determine task type from config
        _, _, task_type = get_dataset_config(canonical_name)
        
        if task_type == 'node':
            # Node classification
            metrics = train_node(canonical_name, paper, dataset_config, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)
        else:
            # Graph classification
            metrics = train_graph(canonical_name, paper, dataset_config, num_layers=num_layers, hidden_dim=hidden_dim, use_batchnorm=use_batchnorm, use_jk_concat=use_jk_concat)
        
        jk_suffix = "_JK" if use_jk_concat else ""
        feat_suffix = "" if dataset_config.normalize_features else "_rawfeat"
        print(f"✓ Training completed for {canonical_name}_{num_layers}L_{hidden_dim}H{jk_suffix}{feat_suffix}_seed{dataset_config.seed}")
        print(f"  Best epoch: {metrics['best_epoch']}")
        print(f"  Final accuracies - Train: {metrics['train_acc']:.4f}, Val: {metrics['val_acc']:.4f}, Test: {metrics['test_acc']:.4f}")
        
        # Save hyperparameters with performance metrics (use canonical_name for file paths)
        save_hyperparameters(
            paper,
            canonical_name,
            num_layers,
            hidden_dim,
            dataset_config,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
            normalize_features=dataset_config.normalize_features,
            performance_metrics=metrics,
            nn_seed=dataset_config.seed,
        )
        
        # Copy to pretrained directory (use canonical_name)
        copy_best_model_to_pretrained(
            paper,
            canonical_name,
            num_layers,
            hidden_dim,
            use_batchnorm=use_batchnorm,
            use_jk_concat=use_jk_concat,
            normalize_features=dataset_config.normalize_features,
            nn_seed=dataset_config.seed,
        )
        
        return True, dataset_config, metrics
        
    except Exception as e:
        jk_suffix = "_JK" if use_jk_concat else ""
        feat_suffix = "" if dataset_config.normalize_features else "_rawfeat"
        print(f"✗ Training failed for {canonical_name}_{num_layers}L_{hidden_dim}H{jk_suffix}{feat_suffix}_seed{dataset_config.seed}: {e}")
        import traceback
        traceback.print_exc()
        return False, dataset_config, None


def parse_datasets_arg(dataset_arg: str):
    """Parse dataset CLI argument into canonical dataset list."""
    if dataset_arg.strip().lower() == "all":
        return list(ALL_DATASETS)

    raw_items = [item.strip() for item in dataset_arg.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("--dataset is empty. Provide a dataset name, comma-separated list, or 'all'.")

    parsed = []
    for name in raw_items:
        canonical = normalize_dataset_name(name)
        if canonical not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {ALL_DATASETS}")
        parsed.append(canonical)

    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for name in parsed:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def parse_nn_seeds_arg(nn_seed_arg):
    """Parse nn_seed CLI argument into a list of integer seeds."""
    if isinstance(nn_seed_arg, int):
        return [nn_seed_arg]

    text = str(nn_seed_arg).strip()
    if not text:
        raise ValueError("--nn_seed is empty. Provide an integer or comma-separated integers.")

    raw_items = [item.strip() for item in text.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("--nn_seed is empty. Provide an integer or comma-separated integers.")

    parsed = []
    for item in raw_items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid nn_seed '{item}'. Use integer values like '0' or '0,1,2'.") from exc

    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for seed in parsed:
        if seed not in seen:
            unique.append(seed)
            seen.add(seed)
    return unique


def parse_hidden_dims_arg(hidden_dim_arg: str):
    """Parse hidden_dim CLI argument into a list of positive integers."""
    text = str(hidden_dim_arg).strip()
    if not text:
        raise ValueError("--hidden_dim is empty. Provide an integer or comma-separated integers.")

    raw_items = [item.strip() for item in text.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("--hidden_dim is empty. Provide an integer or comma-separated integers.")

    parsed = []
    for item in raw_items:
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid hidden_dim '{item}'. Use integer values like '8' or '8,16'.") from exc
        if value <= 0:
            raise ValueError(f"Invalid hidden_dim '{item}'. Values must be positive integers.")
        parsed.append(value)

    seen = set()
    unique = []
    for hidden_dim in parsed:
        if hidden_dim not in seen:
            unique.append(hidden_dim)
            seen.add(hidden_dim)
    return unique


def exact_model_directories(
    paper,
    dataset,
    num_layers,
    hidden_dim,
    use_batchnorm,
    use_jk_concat,
    normalize_features,
    nn_seed,
):
    """Return the exact checkpoint and pretrained directories for one run."""
    canonical_name = normalize_dataset_name(dataset)
    bn_suffix = "_BN" if use_batchnorm else ""
    jk_suffix = "_JK" if use_jk_concat else ""
    feat_suffix = "" if normalize_features else "_rawfeat"
    seed_suffix = f"_seed{int(nn_seed)}"
    model_dir_name = (
        f"{canonical_name}_{num_layers}L_{hidden_dim}H"
        f"{bn_suffix}{jk_suffix}{feat_suffix}{seed_suffix}"
    )
    return (
        os.path.join(CHECKPOINT_ROOT, paper, model_dir_name),
        os.path.join(PRETRAINED_ROOT, paper, model_dir_name),
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train GIN models for selected datasets")
    parser.add_argument("--config_dir", type=str, default=TRAINING_CONFIG_DIR, help="Directory containing config files")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Infection3-one-hot-binary-Bernoulli4",
        help="Dataset name(s): e.g. 'Infection4' or 'BA-Shapes,Infection3' or 'all'",
    )
    parser.add_argument("--use_batchnorm", action="store_true", help="Use batch normalization in the model")
    parser.add_argument("--use_jk_concat", action="store_true", help="Use JK-concat (concatenate all layer outputs). Default: False")
    parser.add_argument(
        "--disable_feature_normalization",
        action="store_true",
        help="Disable row-wise feature normalization for synthetic node datasets; saves checkpoints with _rawfeat suffix",
    )
    parser.add_argument(
        "--hidden_dim",
        type=str,
        default="5",
        help="Hidden dimension(s): single int or comma-separated list, e.g. '8' or '8,16'",
    )
    parser.add_argument(
        "--nn_seed",
        type=str,
        default="1,2,3,4,0",
        help="NN training seed(s): single int or comma-separated list, e.g. '3' or '0,1,2,3,4'",
    )
    parser.add_argument(
        "--fail_if_exists",
        action="store_true",
        help="Fail before training if any requested checkpoint or pretrained directory already exists.",
    )
    args = parser.parse_args()
    
    # Handle JK-concat flag
    use_jk_concat = args.use_jk_concat
    
    # Configuration
    config_dir = args.config_dir if os.path.isabs(args.config_dir) else os.path.join(MAIN_DIR, args.config_dir)

    datasets = parse_datasets_arg(args.dataset)
    hidden_dims = parse_hidden_dims_arg(args.hidden_dim)
    nn_seeds = parse_nn_seeds_arg(args.nn_seed)
    args.architecture = "gin"  # Only GIN checkpoints are trained.
    paper = 'GIN'
    num_layers = 2 if datasets == ["PubMed"] else 3
    use_batchnorm =args.use_batchnorm
    normalize_features = not args.disable_feature_normalization

    if args.fail_if_exists:
        existing_dirs = []
        for dataset in datasets:
            for hidden_dim in hidden_dims:
                for nn_seed in nn_seeds:
                    for directory in exact_model_directories(
                        paper,
                        dataset,
                        num_layers,
                        hidden_dim,
                        use_batchnorm,
                        use_jk_concat,
                        normalize_features,
                        nn_seed,
                    ):
                        if os.path.exists(directory):
                            existing_dirs.append(directory)
        if existing_dirs:
            formatted = "\n".join(f"  {path}" for path in existing_dirs)
            raise FileExistsError(
                "Refusing to overwrite requested model artifacts:\n"
                f"{formatted}\n"
                "Choose only missing configurations or remove partial artifacts after explicit review."
            )
    
    print(f"Starting systematic training at {datetime.now()}")
    print(f"Datasets: {datasets}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Layers: {num_layers}")
    print(f"Paper: {paper}")
    print(f"Architecture: {args.architecture}")
    print(f"BatchNorm: {use_batchnorm}")
    print(f"JK-concat: {use_jk_concat}")
    print(f"Feature normalization: {normalize_features}")
    print(f"NN seeds: {nn_seeds}")
    print(f"Config directory: {config_dir}")
    
    # Track results
    results = []
    configs_used = {}
    
    # Train all combinations
    for dataset in datasets:
        for hidden_dim in hidden_dims:
            for nn_seed in nn_seeds:
                success, dataset_config, metrics = train_single_model(
                    dataset,
                    paper,
                    num_layers,
                    hidden_dim,
                    config_dir,
                    use_batchnorm=use_batchnorm,
                    use_jk_concat=use_jk_concat,
                    normalize_features=normalize_features,
                    nn_seed=nn_seed,
                )
                result = {
                    'architecture': args.architecture,
                    'paper': paper,
                    'dataset': dataset,
                    'num_layers': num_layers,
                    'hidden_dim': hidden_dim,
                    'use_jk_concat': use_jk_concat,
                    'normalize_features': normalize_features,
                    'nn_seed': nn_seed,
                    'success': success
                }
                if metrics:
                    result['metrics'] = metrics
                results.append(result)
                configs_used[(dataset, nn_seed)] = dataset_config
    
    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    successful = 0
    for result in results:
        status = "✓" if result['success'] else "✗"
        jk_suffix = "_JK" if result.get('use_jk_concat', True) else ""
        model_name = f"{result['dataset']}_{result['num_layers']}L_{result['hidden_dim']}H{jk_suffix}_seed{result['nn_seed']}"
        if result['success'] and 'metrics' in result:
            metrics = result['metrics']
            print(f"{status} {model_name} - Train: {metrics['train_acc']:.4f}, Val: {metrics['val_acc']:.4f}, Test: {metrics['test_acc']:.4f} (epoch {metrics['best_epoch']})")
        else:
            print(f"{status} {model_name}")
        if result['success']:
            successful += 1
    
    print(f"\nSuccessfully trained {successful}/{len(results)} models")
    print(f"Completed at {datetime.now()}")
    
    # Create log file
    log_dir = TRAINING_LOG_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.architecture}_{'-'.join(str(value) for value in hidden_dims)}H"
    log_file = f"{log_dir}/training_summary_{run_tag}_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Training Summary - {datetime.now()}\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Datasets: {datasets}\n")
        f.write(f"  Hidden dims: {hidden_dims}\n")
        f.write(f"  Layers: {num_layers}\n")
        f.write(f"  Paper: {paper}\n")
        f.write(f"  Architecture: {args.architecture}\n")
        f.write(f"  BatchNorm: {use_batchnorm}\n")
        f.write(f"  JK-concat: {use_jk_concat}\n")
        f.write(f"  NN seeds: {nn_seeds}\n\n")
        
        f.write("Results:\n")
        for result in results:
            status = "SUCCESS" if result['success'] else "FAILED"
            jk_suffix = "_JK" if result.get('use_jk_concat', True) else ""
            model_name = f"{result['dataset']}_{result['num_layers']}L_{result['hidden_dim']}H{jk_suffix}_seed{result['nn_seed']}"
            dataset = result['dataset']
            config = configs_used[(dataset, result['nn_seed'])]
            
            if result['success'] and 'metrics' in result:
                metrics = result['metrics']
                f.write(f"  {model_name}: {status} (lr={config.lr}, epochs={config.epochs})\n")
                f.write(f"    Performance - Train: {metrics['train_acc']:.4f}, Val: {metrics['val_acc']:.4f}, Test: {metrics['test_acc']:.4f} (best epoch: {metrics['best_epoch']})\n")
            else:
                f.write(f"  {model_name}: {status} (lr={config.lr}, epochs={config.epochs})\n")
        
        f.write(f"\nDataset-specific configurations:\n")
        for (dataset, nn_seed), config in configs_used.items():
            f.write(f"  {dataset} (seed={nn_seed}): lr={config.lr}, epochs={config.epochs}, batch_size={config.batch_size}, early_stopping={config.early_stopping}\n")
        
        f.write(f"\nTotal: {successful}/{len(results)} successful\n")
    
    print(f"\nLog saved to: {log_file}")
    
    # Create performance summary CSV file
    csv_file = f"{log_dir}/performance_summary_{run_tag}_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Architecture,Paper,Dataset,Layers,Hidden_Dim,BatchNorm,JK_Concat,NN_Seed,Status,Train_Acc,Val_Acc,Test_Acc,Best_Epoch,LR,Epochs,Batch_Size\n")
        for result in results:
            dataset = result['dataset']
            config = configs_used[(dataset, result['nn_seed'])]
            status = "SUCCESS" if result['success'] else "FAILED"
            bn_str = "True" if use_batchnorm else "False"
            jk_str = "True" if result.get('use_jk_concat', True) else "False"
            
            if result['success'] and 'metrics' in result:
                metrics = result['metrics']
                f.write(
                    f"{result['architecture']},{result['paper']},{dataset},{result['num_layers']},{result['hidden_dim']},{bn_str},{jk_str},{result['nn_seed']},{status},"
                    f"{metrics['train_acc']:.4f},{metrics['val_acc']:.4f},{metrics['test_acc']:.4f},"
                    f"{metrics['best_epoch']},{config.lr},{config.epochs},{config.batch_size}\n"
                )
            else:
                f.write(
                    f"{result['architecture']},{result['paper']},{dataset},{result['num_layers']},{result['hidden_dim']},{bn_str},{jk_str},{result['nn_seed']},{status},"
                    f"N/A,N/A,N/A,N/A,{config.lr},{config.epochs},{config.batch_size}\n"
                )
    
    print(f"Performance summary CSV saved to: {csv_file}")

    if successful != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
