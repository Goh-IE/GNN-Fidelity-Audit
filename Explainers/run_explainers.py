#!/usr/bin/env python3
"""Run explanation methods on an explicit existing audit cohort.

One frozen candidate is evaluated on four probability objectives. There is no
optimization bound, no GT exclusion and no use of held-out scores in selection.
Existing run directories with result/config files are never overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/26gnn-baseline-matplotlib")
import numpy as np
import torch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MAIN = REPO / "Main"
sys.path.insert(0, str(REPO))

from Explainers.methods.model_adapter import compatible_model, clear_masks
from Explainers.methods.gnnexplainer import gnn_scores
from Explainers.methods.pgexplainer import train_pg, pg_scores
from Main.data.dataset_loaders import load_dataset
from Main.models.model_selector import get_pretrained_path
from Main.audit_robust import load_dataset_data, load_model, build_eval_context, resolve_repo_path
from Main.audit_standard import compact_node_inference_graph, candidate_edges_for_node

DEFAULT_METHODS = ("random", "gnnexplainer", "pgexplainer")
METHODS = DEFAULT_METHODS + ("flowx_suf", "flowx_nec")
OBJECTIVES = ("standard_sufficiency", "standard_necessity", "robust_sufficiency", "robust_necessity")


def json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_write(path, rows, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def edges_read(path):
    with Path(path).open() as handle:
        return sorted({tuple(sorted((int(r["src"]), int(r["dst"])))) for r in csv.DictReader(handle)})


def edges_write(path, edges):
    csv_write(path, [{"src": int(a), "dst": int(b)} for a, b in edges], ["src", "dst"])


def environment():
    versions = {}
    for package in ("torch", "torch-geometric", "captum", "numpy", "scipy", "matplotlib", "gurobipy", "networkx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"interpreter": sys.executable, "python": sys.version, "versions": versions,
            "platform": platform.platform(), "processor": platform.processor(),
            "device": "cpu", "torch_threads": torch.get_num_threads()}


def source_rows(run, seed, limit):
    path = run / f"seed{seed}" / "instance_fid.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    eligible = [r for r in rows if r.get("stage") == "gt_size"
                and str(r.get("is_correct", "")).lower() in ("1", "true")
                and int(float(r.get("gt_size") or 0)) > 0
                and int(float(r.get("candidate_size") or 0)) > int(float(r["gt_size"]))]
    eligible.sort(key=lambda r: int(r["row_id"]))
    seen = set()
    for r in eligible:
        key = r["instance_id"]
        if key in seen:
            raise ValueError("Source cohort contains repeated instance IDs")
        seen.add(key)
    return eligible if limit is None else eligible[:limit]


def record_input(dataset, data, source, source_config):
    ctx = build_eval_context(dataset=dataset, data=data, row=source,
                             config=source_config, device=torch.device("cpu"))
    if ctx.task == "node":
        x, edge_index, root, _, mapping = compact_node_inference_graph(
            x=ctx.x, edge_index=ctx.edge_index, candidate_edges=ctx.candidate_edges,
            root_node=ctx.root_node)
        node_ids = np.asarray(sorted(mapping, key=mapping.get), dtype=np.int64)
    else:
        x, edge_index, root = ctx.x, ctx.edge_index, None
        node_ids = np.arange(x.shape[0], dtype=np.int64)
    edges = np.asarray(sorted(ctx.candidate_edges), dtype=np.int64)
    indices = {tuple(edge): idx for idx, edge in enumerate(edges)}
    col_to_edge = np.asarray([-1 if a == b else indices[tuple(sorted((int(node_ids[a]), int(node_ids[b]))))]
                              for a, b in edge_index.T], dtype=np.int64)
    gt = edges_read(resolve_repo_path(source["gt_edge_file"]))
    if not set(gt).issubset(indices):
        raise ValueError("Reference is not representable in working graph")
    reference = np.asarray([tuple(e) in set(gt) for e in edges], dtype=bool)
    if int(reference.sum()) != int(source["budget_k"]) or len(edges) != int(source["candidate_size"]):
        raise ValueError("Source budget or carrier mismatch")
    return {"x": np.asarray(x), "edge_index": np.asarray(edge_index), "node_ids": node_ids,
            "edges": edges, "col_to_edge": col_to_edge, "root": root,
            "root_global": ctx.root_node, "task": ctx.task, "target": ctx.c_star,
            "reference": reference, "gt": gt, "k": len(gt), "ctx": ctx}


def logits(model, record, weights=None):
    x = torch.as_tensor(record["x"], dtype=next(model.parameters()).dtype)
    e = torch.as_tensor(record["edge_index"], dtype=torch.long)
    y = model(x, e, edge_weight=weights)
    return y[0] if record["root"] is None else y[record["root"]]


def weights_from_mask(record, masks):
    masks = np.atleast_2d(masks)
    columns = record["col_to_edge"]
    weights = np.ones((len(masks), len(columns)), dtype=np.float32)
    weights[:, columns >= 0] = masks[:, columns[columns >= 0]]
    return weights


def probabilities(model, record, masks, batch_size=128):
    weights = weights_from_mask(record, masks)
    x = torch.tensor(record["x"], dtype=torch.float32)
    edges = torch.tensor(record["edge_index"], dtype=torch.long)
    n, m = len(x), edges.shape[1]
    results = []
    with torch.no_grad():
        for start in range(0, len(weights), batch_size):
            w = torch.tensor(weights[start:start + batch_size]).reshape(-1)
            b = min(batch_size, len(weights) - start)
            e = edges.repeat(1, b) + (torch.arange(b).repeat_interleave(m) * n)[None]
            features = x.repeat(b, 1)
            if record["root"] is None:
                y = model(features, e, edge_weight=w, batch=torch.arange(b).repeat_interleave(n))
            else:
                y = model(features, e, edge_weight=w)[torch.arange(b) * n + record["root"]]
            results.extend(y.softmax(-1)[:, record["target"]].double().numpy())
    return np.asarray(results)


def verify_model(original, clone, record, source):
    checks = {}
    m = len(record["edges"])
    probe = np.zeros(m, dtype=bool)
    probe[np.random.default_rng(471).choice(m, record["k"], replace=False)] = True
    masks = np.stack([np.ones(m, bool), record["reference"], probe, np.zeros(m, bool)])
    max_error = 0.0
    for w in weights_from_mask(record, masks):
        weights = torch.tensor(w)
        with torch.no_grad():
            expected = logits(original, record, weights)
            actual = logits(clone, record, weights)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        max_error = max(max_error, float((actual - expected).abs().max()))
    checks["clone_max_logit_error"] = max_error
    with torch.no_grad():
        full = logits(original, record)
    if int(full.argmax()) != record["target"] or record["target"] != int(source["label"]):
        raise ValueError("Source prediction/label does not match checkpoint")
    if record["task"] == "node":
        ctx = record["ctx"]
        with torch.no_grad():
            entire = original(ctx.x_tensor, ctx.edge_index_tensor)[ctx.root_node]
        torch.testing.assert_close(full, entire, rtol=1e-5, atol=1e-5)
        checks["full_vs_working_max_logit_error"] = float((full - entire).abs().max())
    p = probabilities(original, record, masks[:2])
    gt_prob = float(source["gt_sufficiency_probability"])
    np.testing.assert_allclose(p[1], gt_prob, atol=2e-5, rtol=1e-5)
    checks["reference_probability_source_error"] = float(abs(p[1] - gt_prob))
    # Compare the legacy DIG mask path, explicit edge weights and gradients.
    w = torch.tensor(weights_from_mask(record, np.full(m, 0.73))[0], requires_grad=True)
    expected_score = logits(original, record, w)[record["target"]]
    expected_grad, = torch.autograd.grad(expected_score, w)
    for layer in clone.convs:
        layer._explain = True
        layer.__edge_mask__ = w
    score = logits(clone, record)[record["target"]]
    grad, = torch.autograd.grad(score, w)
    clear_masks(clone)
    torch.testing.assert_close(score, expected_score, atol=1e-5, rtol=1e-5)
    nonloops = record["col_to_edge"] >= 0
    torch.testing.assert_close(grad[nonloops], expected_grad[nonloops], atol=1e-5, rtol=1e-5)
    checks["mask_gradient_max_error"] = float((grad[nonloops] - expected_grad[nonloops]).abs().max())
    checks["max_absolute_mask_gradient"] = float(grad[nonloops].abs().max())
    # A finite difference must stay inside the same ReLU activation region.
    # Float64 prevents cancellation in this diagnostic only. The actual model,
    # explanations and probability evaluations remain float32.
    diagnostic = copy.deepcopy(original).double()
    wd = w.detach().double().requires_grad_(True)
    patterns = []
    hooks = [module.register_forward_pre_hook(
        lambda module, args: patterns.append((args[0].detach() > 0).clone()))
        for module in diagnostic.modules() if isinstance(module, torch.nn.ReLU)]
    try:
        center = logits(diagnostic, record, wd)[record["target"]]
        derivative, = torch.autograd.grad(center, wd)
        center_patterns = list(patterns)
        idxs = np.flatnonzero(nonloops)
        index = int(idxs[int(derivative[nonloops].abs().argmax())])
        checks["finite_difference_status"] = "activation_boundary_at_probe"
        for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
            sides = []
            stable = True
            for direction in (-1, 1):
                perturbed = wd.detach().clone()
                perturbed[index] += direction * eps
                patterns.clear()
                with torch.no_grad():
                    sides.append(float(logits(diagnostic, record, perturbed)[record["target"]]))
                stable &= len(patterns) == len(center_patterns) and all(
                    torch.equal(a, b) for a, b in zip(patterns, center_patterns))
            if stable:
                finite = (sides[1] - sides[0]) / (2 * eps)
                np.testing.assert_allclose(finite, float(derivative[index]), rtol=1e-4, atol=1e-6)
                checks.update(finite_difference_status="passed", finite_difference_step=eps,
                              finite_difference_gradient_error=abs(finite - float(derivative[index])))
                break
    finally:
        for hook in hooks:
            hook.remove()
    # Hard edge deletion is an independent check of weighted batching semantics.
    w_hard = weights_from_mask(record, probe)[0]
    x = torch.tensor(record["x"])
    e = torch.tensor(record["edge_index"][:, w_hard.astype(bool)])
    with torch.no_grad():
        y = original(x, e)
        direct = float((y[0] if record["root"] is None else y[record["root"]]).softmax(-1)[record["target"]])
    np.testing.assert_allclose(probabilities(original, record, probe)[0], direct, atol=1e-6, rtol=1e-5)
    checks["passed"] = True
    return checks


def training_examples(dataset, task, config, limit):
    graphs, features, labels, train, _, test = load_dataset(dataset, normalize_features=config.get("normalize_features", True))
    train_ids = np.flatnonzero(np.asarray(train)).tolist()
    if limit is not None:
        train_ids = train_ids[:limit]
    if not train_ids:
        raise ValueError("No PGExplainer training examples")
    if set(train_ids) & set(np.flatnonzero(np.asarray(test)).tolist()):
        raise ValueError("PG training split overlaps test targets")
    examples = []
    for idx in train_ids:
        if task == "graph":
            x, e, root = features[idx], graphs[idx], None
        else:
            all_edges = np.asarray(graphs)
            candidates = candidate_edges_for_node(node_idx=idx, k_hop=int(config["num_layers"]),
                                                   edge_index=all_edges, num_nodes=len(features))
            x, e, root, _, _ = compact_node_inference_graph(x=features, edge_index=all_edges,
                                                           candidate_edges=candidates, root_node=idx)
        examples.append((idx, torch.tensor(x, dtype=torch.float32), torch.tensor(e, dtype=torch.long), root))
    return examples


def score_edges(record, directed):
    values = []
    for idx in range(len(record["edges"])):
        v = np.asarray(directed)[record["col_to_edge"] == idx]
        if not len(v) or not np.isfinite(v).all():
            raise ValueError("Missing or nonfinite edge scores")
        values.append(float(v.mean()))
    return np.asarray(values)


def evaluate(original, record, selected, sample_count, scenario_seed, tau):
    ref = record["reference"]
    m = len(ref)
    standard = probabilities(original, record, np.stack([np.ones(m, bool), selected, ~selected, ref, ~ref]))
    full, keep, remove, ref_keep, ref_remove = map(float, standard)
    arrays = {"standard_probabilities": standard}
    output = []
    for objective in OBJECTIVES:
        necessity = objective.endswith("necessity")
        robust = objective.startswith("robust")
        alpha = (0.1 if necessity else 0.9) if robust else (1.0 if necessity else 0.0)
        if robust:
            seed = derived_seed(scenario_seed, objective)
            draws = np.random.default_rng(seed).random((sample_count, m)) < alpha
            active = (~(selected & draws)) if necessity else (selected | draws)
            ref_active = (~(ref & draws)) if necessity else (ref | draws)
            cand_p = probabilities(original, record, active)
            ref_p = probabilities(original, record, ref_active)
            arrays[objective + "_draws"] = draws
            arrays[objective + "_candidate_probabilities"] = cand_p
            arrays[objective + "_reference_probabilities"] = ref_p
            differences = ref_p - cand_p if necessity else cand_p - ref_p
            advantage = float(differences.mean())
            sd = float(differences.std(ddof=1))
            if sd > 0:
                se = sd / np.sqrt(sample_count)
                lcb = advantage - float(student_t.ppf(0.95, sample_count - 1)) * se
                pvalue = float(student_t.sf((advantage - tau) / se, sample_count - 1))
                test_status = "valid"
            else:
                # A constant observed sample does not prove population constancy.
                lcb, pvalue = None, 1.0
                test_status = "identical_mask" if np.array_equal(selected, ref) else "zero_sample_variance"
            relation = "candidate_improvement_supported" if pvalue < 0.05 else "not_supported"
            stats = {"paired_advantage_std": sd, "paired_scenario_count": sample_count,
                     "lcb_95": lcb, "pvalue": pvalue, "test_status": test_status,
                     "evaluation_seed": seed}
            pc, pr = float(cand_p.mean()), float(ref_p.mean())
        else:
            pc, pr = (remove, ref_remove) if necessity else (keep, ref_keep)
            advantage = pr - pc if necessity else pc - pr
            relation = "candidate_better" if advantage > tau else ("reference_better" if advantage < -tau else "within_tolerance")
            stats = {}
        output.append({"objective": objective, "alpha": alpha, "p_full": full,
                       "candidate_perturbed_probability": pc, "reference_perturbed_probability": pr,
                       "candidate_raw_fidelity": full - pc, "reference_raw_fidelity": full - pr,
                       "candidate_score": full - pc if necessity else pc,
                       "reference_score": full - pr if necessity else pr,
                       "advantage": advantage, "comparison_status": relation, "tau": tau, **stats})
    return output, arrays


def holm(rows):
    for objective in OBJECTIVES[2:]:
        group = [r for r in rows if r["objective"] == objective]
        group.sort(key=lambda r: float(r.get("pvalue", 1)))
        previous = 0.0
        for i, row in enumerate(group):
            value = min(1.0, (len(group) - i) * float(row.get("pvalue", 1)))
            previous = max(previous, value)
            row["holm_pvalue"] = previous
            row["holm_supported"] = previous < 0.05
            row["holm_family"] = "dataset_method_objective_all_requested_model_seeds"


def summarize(rows, by_seed):
    groups = {}
    for r in rows:
        key = (r["nn_seed"] if by_seed else "all", r["objective"])
        groups.setdefault(key, []).append(r)
    output = []
    for (seed, objective), group in groups.items():
        ok = [r for r in group if r["status"] == "ok"]
        output.append({"nn_seed": seed, "objective": objective, "n_records": len(group),
                       "n_success": len(ok), "n_failed": len(group) - len(ok),
                       "mean_overlap": float(np.mean([r["overlap"] for r in ok])) if ok else None,
                       "mean_advantage": float(np.mean([r["advantage"] for r in ok])) if ok else None,
                       "n_candidate_better": sum(r["comparison_status"] == "candidate_better" for r in ok),
                       "n_within_tolerance": sum(r["comparison_status"] == "within_tolerance" for r in ok),
                       "n_reference_better": sum(r["comparison_status"] == "reference_better" for r in ok),
                       "n_holm_supported": sum(bool(r.get("holm_supported")) for r in ok)})
    return output


def prepare_run(args, method, dataset, source, source_config, env):
    base = Path(args.output_root).resolve() if args.output_root else ROOT / "results"
    parent = base / "runs" / method if args.phase != "formal" else base / method
    run = parent / dataset / args.run_name
    if (run / "config.json").exists() or (run / "summary.csv").exists():
        raise FileExistsError(f"Refusing to overwrite {run}. Use a new run_name.")
    run.mkdir(parents=True, exist_ok=True)
    command = "conda run --no-capture-output -n 26GNN python " + shlex.join(sys.argv)
    if not (run / "experiment.md").exists():
        (run / "experiment.md").write_text(
            f"# {method} {args.phase}\n\n## Plan\n\nEvaluate fixed exact-K masks on the explicit audit cohort. "
            "Model seeds and source row order are fixed before scoring. No outcome-based selection. "
            "No model retraining or solver. GT masks may be returned. "
            "Only sufficiency and necessity are evaluated. Success requires forward/gradient equivalence, "
            "budget feasibility and artifact replay. Failure rows remain in the denominator. "
            "For smoke/pilot, results are not paper conclusions.\n\n"
            "Sources: Ying et al. (2019), https://arxiv.org/abs/1903.03894 ; "
            "Luo et al. (2020), https://arxiv.org/abs/2011.04573 ; "
            "Faber et al. (2021), related-work/faber.txt. "
            "DIG supplies the attribution algorithms, while the edge-budget comparison is this study's protocol.\n")
    code = [Path(__file__),
            ROOT / "methods/dig_loader.py", ROOT / "methods/model_adapter.py",
            ROOT / "methods/gnnexplainer.py", ROOT / "methods/pgexplainer.py",
            MAIN / "models/vanilla_pyg_compat.py",
            MAIN / "audit_standard.py", MAIN / "audit_robust.py"]
    from Explainers.methods.dig_loader import DIG_ROOT
    if DIG_ROOT is not None:
        code += list((DIG_ROOT / "dig/xgraph/method").glob("*.py"))
        code += list((DIG_ROOT / "dig/xgraph/models").glob("*.py"))
    if method.startswith("flowx_"):
        code += [ROOT / "methods/flowx.py"]
        code += list((ROOT / "methods/_vendor/flowx").glob("*.*"))
    config = {**vars(args), "method": method, "dataset": dataset,
              "source_run": str(source), "source_config": source_config,
              "environment": env, "command": command, "cwd": str(REPO),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "dirty_before_run": subprocess.check_output(["git", "status", "--short", "--", "Main", "Explainers"], cwd=REPO, text=True),
              "code_hashes": {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): digest(p) for p in code},
              "connectivity": "none", "exclude_gt": False, "budget": "exact_reference_edge_count",
              "edge_score_reduction": "mean_directed_scores", "tie_break": "canonical_global_edge_order",
              "optimization_scenarios": 0, "evaluation_selection_independent": True,
              "random_repeats": 1, "evaluation_scenarios_shared_between_methods": True,
              "evaluation_scenarios_shared_with_previous_audit": False,
              "gnn_defaults": {"lr": 0.01, "edge_size": 0.001, "edge_entropy": 0.001, "mask_features": False},
              "pg_defaults": {"lr": 0.005, "size": 0.01, "entropy": 0.0005, "t0": 5.0, "t1": 1.0},
              "inference": "one-sided paired t test against tau, Holm per dataset/method/objective",
              "zero_variance_test": "p=1, no population certificate from zero observed variance"}
    if method.startswith("flowx_"):
        from Explainers.methods.flowx import COMMIT
        config["flowx"] = {
            "repository": "https://github.com/CM-BF/FlowX", "commit": COMMIT,
            "paper": "https://doi.org/10.1109/TPAMI.2023.3347470",
            "implementation": "compact first-deletion-group representation, checked against upstream method bodies",
            "training_sparsity": "1-K/number_of_selectable_undirected_edges",
            "learning_rate": args.flowx_lr, "sampling_iterations": args.flowx_samples,
            "refinement_epochs": args.flowx_epochs,
            "node_deletion_groups": "upstream min(active_layer_carriers,100), floor grouping retained",
            "self_carriers": "native sampling/training masks (1+eps) self term; fixed in final edge evaluation",
            "explicit_input_loops": "combined native self coefficient (1+eps+loop_count), original evaluation input retained",
            "projection": "sum flows per directed edge across layers, then mean reverse directions, then exact-K",
            "missing_flow_direction": "empty additive sum=0; upstream -inf ranking sentinel stored separately",
            "target": "unperturbed predicted class only",
            "native_hard_mask_used_for_selection": False,
            "learning_rate_note": "paper 0.1 for both; upstream minus's 0.01 is annotated Graph-SST2 only",
        }
        with (run / "experiment.md").open("a") as handle:
            handle.write("\n## FlowX protocol fixed before evaluation\n\n"
                         "Gui et al. (2024), FlowX: Towards Explainable Graph Neural Networks via Message Flows, "
                         "https://doi.org/10.1109/TPAMI.2023.3347470 and "
                         "https://arxiv.org/html/2206.12987v3#A2.SS4. Pinned official source and license: "
                         "Explainers/methods/_vendor/flowx. Native directional/self-carrier sampling and refinement "
                         "are retained. Flow contributions are summed over layers, reverse directions averaged, "
                         "then exactly K physical edges selected. Self carriers are excluded only from final "
                         "edge selection/evaluation. An absent flow direction contributes the empty sum zero, "
                         "while the native -infinity ranking sentinel is also saved. "
                         "The training sparsity is 1-K/m. Defaults are 50 MC iterations, 500 refinement epochs "
                         "and lr=0.1 from the paper, including sufficiency (upstream 0.01 is marked Graph-SST2 only). "
                         "No held-out objective is used for tuning or candidate selection. "
                         "This is FlowX under a declared undirected projection, not a reproduction of its "
                         "directional-budget evaluation. Whether it yields reference-improving masks remains "
                         "an empirical question, and all contrary results/failures remain in the denominator.\n")
    json_write(run / "config.json", config)
    with (run / "experiment.md").open("a") as handle:
        handle.write(f"\n## Execution configuration\n\n```bash\n{command}\n```\n\nEnvironment and full configuration: [config.json](config.json). "
                     "One CPU torch thread. Fresh held-out scenarios are paired across baselines but differ from historical audit draws. "
                     "No SAA objective is optimized. Zero empirical variance is not treated as a population certificate.\n")
    return run


def reuse_successful_record(previous, directory, stem, info, previous_manifest):
    """Copy a completed formal record with its original producer provenance.

Only a new run can receive these copies. Failed records are computed afresh,
and Holm is recomputed over the full cohort after all records are present.
"""
    metrics_path = previous / f"seed{info['nn_seed']}" / "evaluation" / (stem + "_metrics.json")
    if not metrics_path.exists():
        return None
    rows = json.loads(metrics_path.read_text())
    if len(rows) != 4 or any(row.get("status") != "ok" for row in rows):
        return None
    old_info = previous_manifest[(info['nn_seed'], info['row_id'])]
    for key in ("checkpoint_sha256", "source_csv_sha256", "input_hashes", "explainer_seed", "budget_k", "c_star"):
        if old_info[key] != info[key]:
            raise ValueError(f"Unsafe formal reuse: changed {key}")
    old_directory = metrics_path.parent.parent
    # This implementation repair changes explicit-loop carrier support only.
    with np.load(old_directory / "inputs" / (stem + ".npz")) as arrays:
        if np.any(arrays["edge_index"][0] == arrays["edge_index"][1]):
            raise ValueError("Do not reuse explicit-loop records across the carrier repair")
    copied = {}
    for folder in ("inputs", "scores", "edges", "evaluation", "logs"):
        for source in (old_directory / folder).glob("*" + stem + "*"):
            if source.is_file():
                target = directory / folder / source.name
                shutil.copyfile(source, target)
                copied[str(target.relative_to(directory))] = digest(source)
                if digest(target) != copied[str(target.relative_to(directory))]:
                    raise ValueError("Reused artifact hash mismatch")
    provenance = {"producer_run": str(previous), "producer_config_sha256": digest(previous / "config.json"),
                  "copied_artifact_hashes": copied, "reason": "successful formal record, no explicit loops"}
    json_write(directory / "logs" / (stem + "_reuse.json"), provenance)
    info["reused_provenance"] = provenance
    for row in rows:
        row["explanation_reused"] = True
        row["artifact_producer_run"] = str(previous)
    return rows


def run_dataset(args, source):
    source = resolve_repo_path(source).resolve()
    source_config = json.loads((source / "config.json").read_text())
    dataset = source.parent.name
    if source_config.get("connectivity", "none") != "none":
        raise ValueError("This runner currently supports unrestricted source cohorts only")
    if source_config.get("architecture", "gin") != "gin":
        raise ValueError("DIG adapter requires the audited Vanilla GIN")
    runs = {method: prepare_run(args, method, dataset, source, source_config, environment()) for method in args.methods}
    manifests = {method: [] for method in args.methods}
    all_rows = {method: [] for method in args.methods}
    failure = False
    started = time.perf_counter()
    planned = {seed: source_rows(source, seed, args.max_instances) for seed in args.nn_seeds}
    reusable = {}
    if args.reuse_successful_run_name:
        for method, run in runs.items():
            previous = run.parent / args.reuse_successful_run_name
            prior = json.loads((previous / "config.json").read_text())
            current = json.loads((run / "config.json").read_text())
            for key in ("interpreter", "python", "versions", "device", "torch_threads"):
                if prior["environment"][key] != current["environment"][key]:
                    raise ValueError(f"Unsafe formal reuse: changed environment {key}")
            for key in ("phase", "method", "dataset", "source_run", "source_config", "nn_seeds", "max_instances",
                        "flowx_samples", "flowx_epochs", "flowx_lr", "explainer_seed", "evaluation_seed",
                        "eval_samples", "tau", "connectivity", "exclude_gt", "edge_score_reduction"):
                if prior[key] != current[key]:
                    raise ValueError(f"Unsafe formal reuse: changed {key}")
            reusable[method] = (previous, {(r["nn_seed"], r["row_id"]): r for r in
                                           json.loads((previous / "cohort_manifest.json").read_text())})
    for method, run in runs.items():
        json_write(run / "planned_cohort.json", [{"nn_seed": seed, "source_row": row}
                   for seed, cohort in planned.items() for row in cohort])
    try:
        data = load_dataset_data(dataset, bool(source_config.get("normalize_features", True)))
        for seed in args.nn_seeds:
            cohort = planned[seed]
            if not cohort:
                raise ValueError(f"No source records for {dataset} seed {seed}")
            original = load_model(dataset, seed, source_config, torch.device("cpu"))
            original.eval().requires_grad_(False)
            initial = {k: v.clone() for k, v in original.state_dict().items()}
            clone = compatible_model(original, data.task)
            checkpoint = Path(get_pretrained_path("Vanilla", dataset, num_layers=original.num_layers,
                              hidden_dim=original.hidden_dim, use_batchnorm=original.use_batchnorm,
                              use_jk_concat=original.use_jk_concat,
                              normalize_features=bool(source_config.get("normalize_features", True)), nn_seed=seed))
            # Verify the recorded path is the actual loaded predictor, including fallback detection.
            loaded_state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
            for key, value in initial.items():
                torch.testing.assert_close(value, loaded_state[key], atol=0, rtol=0)
            model_meta = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                          "source_csv": str(source / f"seed{seed}" / "instance_fid.csv"),
                          "source_csv_sha256": digest(source / f"seed{seed}" / "instance_fid.csv")}
            pg = None
            pg_error = None
            pg_time = 0.0
            if "pgexplainer" in runs:
                directory = runs["pgexplainer"] / f"seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    set_seed(derived_seed(args.explainer_seed, dataset, seed, "pg_training"))
                    examples = training_examples(dataset, data.task, source_config, args.pg_train_limit)
                    json_write(directory / "training_manifest.json", {"training_ids": [r[0] for r in examples],
                                "training_seed": derived_seed(args.explainer_seed, dataset, seed, "pg_training"), **model_meta})
                    tic = time.perf_counter()
                    def training_progress(epoch_row):
                        with (directory / "training_progress.jsonl").open("a") as handle:
                            handle.write(json.dumps(epoch_row) + "\n")
                        json_write(runs["pgexplainer"] / "progress.json",
                                   {"state": "training", "nn_seed": seed, **epoch_row})
                        print(f"{dataset} seed={seed} PG training epoch={epoch_row['epoch']}/{args.pg_epochs}", flush=True)
                    pg, losses = train_pg(clone, data.task, examples, args.pg_epochs,
                                          progress=training_progress)
                    pg_time = time.perf_counter() - tic
                    torch.save(pg.elayers.state_dict(), directory / "explainer.pt")
                    csv_write(directory / "logs/training_loss.csv", losses)
                    json_write(directory / "training_result.json", {"status": "ok", "training_time_s": pg_time,
                               "epochs": args.pg_epochs, "training_count": len(examples),
                               "explainer_sha256": digest(directory / "explainer.pt")})
                except Exception:
                    pg_error = traceback.format_exc()
                    json_write(directory / "training_result.json", {"status": "failed", "error": pg_error})
                    clear_masks(clone)
            for source_row in cohort:
                record = record_input(dataset, data, source_row, source_config)
                checks = verify_model(original, clone, record, source_row)
                row_id, instance_id = int(source_row["row_id"]), int(source_row["instance_id"])
                arrays = {k: record[k] for k in ("x", "edge_index", "node_ids", "edges", "col_to_edge", "reference")}
                hashes = {k: array_digest(v) for k, v in arrays.items()}
                stem = f"seed{seed}_row{row_id:06d}"
                scenario_seed = derived_seed(args.evaluation_seed, dataset, seed, instance_id, "heldout")
                for method, run in runs.items():
                    directory = run / f"seed{seed}"
                    for name in ("edges", "scores", "evaluation", "logs", "inputs"):
                        (directory / name).mkdir(parents=True, exist_ok=True)
                    explain_seed = derived_seed(args.explainer_seed, dataset, seed, instance_id, method)
                    base = {"dataset": dataset, "method": method, "phase": args.phase, "task": data.task,
                            "nn_seed": seed, "row_id": row_id, "instance_id": instance_id,
                            "explainer_seed": explain_seed, "repeat_id": 0, "budget_k": record["k"],
                            "candidate_size": len(record["edges"]), "c_star": record["target"],
                            "label": int(source_row["label"]), "is_correct": True}
                    manifests[method].append({**base, **model_meta, "input_hashes": hashes,
                                              "root_local": record["root"], "source_row": source_row})
                    json_write(run / "cohort_manifest.json", manifests[method])
                    json_write(directory / "logs" / (stem + "_validation.json"), checks)
                    np.savez_compressed(directory / "inputs" / (stem + ".npz"), **arrays)
                    edges_write(directory / "edges" / ("gt_edges_" + stem + ".csv"), record["gt"])
                    try:
                        if method in reusable:
                            previous, previous_manifest = reusable[method]
                            reused = reuse_successful_record(previous, directory, stem,
                                                             manifests[method][-1], previous_manifest)
                            if reused is not None:
                                all_rows[method].extend(reused)
                                json_write(run / "cohort_manifest.json", manifests[method])
                                json_write(run / "progress.json", {"state": "reusing_verified_inputs", "nn_seed": seed,
                                           "row_id": row_id, "completed_records": len(all_rows[method]) // 4,
                                           "planned_records": sum(map(len, planned.values()))})
                                continue
                        set_seed(explain_seed)
                        x = torch.tensor(record["x"], dtype=torch.float32)
                        e = torch.tensor(record["edge_index"], dtype=torch.long)
                        tic = time.perf_counter()
                        if method == "random":
                            selected = np.zeros(len(record["edges"]), dtype=bool)
                            selected[np.random.default_rng(explain_seed).choice(len(selected), record["k"], replace=False)] = True
                            directed, extra, losses, scores = None, {}, [], None
                        else:
                            if method == "gnnexplainer":
                                directed, extra, losses = gnn_scores(clone, x, e, record["target"], record["root"], args.gnn_epochs)
                            elif method.startswith("flowx_"):
                                from Explainers.methods.flowx import flowx_scores
                                def flow_progress(detail):
                                    json_write(run / "progress.json", {"state": "explaining", "nn_seed": seed,
                                               "row_id": row_id, "completed_records": len(all_rows[method]) // 4,
                                               "planned_records": sum(map(len, planned.values())), **detail})
                                directed, extra, losses = flowx_scores(
                                    clone, x, e, record["target"], record["root"], record["k"], method,
                                    iterations=args.flowx_samples, epochs=args.flowx_epochs,
                                    lr=args.flowx_lr, progress=flow_progress)
                            else:
                                if pg_error:
                                    raise RuntimeError(pg_error)
                                directed, extra, losses = pg_scores(pg, clone, x, e, record["root"])
                            scores = score_edges(record, directed)
                            order = np.argsort(-scores, kind="stable")
                            selected = np.zeros(len(scores), dtype=bool)
                            selected[order[:record["k"]]] = True
                        explain_time = time.perf_counter() - tic
                        clear_masks(clone)
                        if selected.sum() != record["k"]:
                            raise ValueError("Baseline violated exact-K")
                        for key, value in initial.items():
                            torch.testing.assert_close(original.state_dict()[key], value, atol=0, rtol=0)
                            torch.testing.assert_close(clone.state_dict()[key], value, atol=0, rtol=0)
                        # Saving raw scores and masks precedes held-out evaluation.
                        np.savez_compressed(directory / "scores" / (stem + ".npz"),
                                            **arrays, selected=selected,
                                            **({"edge_scores": scores, "directed_scores": directed} if scores is not None else {}), **extra)
                        edges_write(directory / "edges" / ("explanation_edges_" + stem + ".csv"), record["edges"][selected])
                        if losses:
                            csv_write(directory / "logs" / (stem + "_loss.csv"), losses)
                        tic = time.perf_counter()
                        metrics, evaluation = evaluate(original, record, selected, args.eval_samples, scenario_seed, args.tau)
                        eval_time = time.perf_counter() - tic
                        np.savez_compressed(directory / "evaluation" / (stem + ".npz"), **evaluation)
                        overlap = float((selected & record["reference"]).sum() / record["k"])
                        all_rows[method].extend({**base, "status": "ok", "selected_size": int(selected.sum()),
                                                "overlap": overlap, "is_reference": bool(np.array_equal(selected, record["reference"])),
                                                "explanation_time_s": explain_time, "evaluation_time_s": eval_time,
                                                "pg_training_time_s": pg_time if method == "pgexplainer" else 0.0,
                                                **metric} for metric in metrics)
                        json_write(directory / "evaluation" / (stem + "_metrics.json"), all_rows[method][-4:])
                        json_write(run / "progress.json", {"state": "evaluating", "nn_seed": seed,
                                   "row_id": row_id, "completed_records": len(all_rows[method]) // 4,
                                   "planned_records": sum(map(len, planned.values()))})
                        print(f"{dataset} seed={seed} row={row_id} {method}: K={record['k']} overlap={overlap:.3f}", flush=True)
                    except Exception:
                        failure = True
                        clear_masks(clone)
                        error = traceback.format_exc()
                        (directory / "logs" / (stem + "_error.txt")).write_text(error)
                        all_rows[method].extend({**base, "objective": objective, "status": "failed", "error": error,
                                                "pvalue": 1.0} for objective in OBJECTIVES)
                        json_write(directory / "evaluation" / (stem + "_metrics.json"), all_rows[method][-4:])
                        print(f"FAILED {dataset} {method} row={row_id}: {error.splitlines()[-1]}", flush=True)
    except Exception:
        failure = True
        error = traceback.format_exc()
        for run in runs.values():
            (run / "failure.txt").write_text(error)
        print(error, flush=True)
    finally:
        for method, run in runs.items():
            rows = all_rows[method]
            # Preserve the complete intended denominator even if a dataset/model
            # validation failure prevents later records from being attempted.
            for seed, cohort in planned.items():
                seen = {int(r["row_id"]) for r in rows if r["nn_seed"] == seed}
                for source_row in cohort:
                    row_id = int(source_row["row_id"])
                    if row_id not in seen:
                        rows.extend({"dataset": dataset, "method": method, "phase": args.phase,
                                     "nn_seed": seed, "row_id": row_id,
                                     "instance_id": int(source_row["instance_id"]),
                                     "objective": objective, "status": "not_attempted_after_error", "pvalue": 1.0,
                                     "error": "See failure.txt or earlier record errors"} for objective in OBJECTIVES)
            holm(rows)
            for seed in args.nn_seeds:
                part = [r for r in rows if r["nn_seed"] == seed]
                csv_write(run / f"seed{seed}/instance_fid.csv", part)
                csv_write(run / f"seed{seed}/skipped_targets.csv", [r for r in part if r["status"] != "ok"],
                          fields=None if any(r["status"] != "ok" for r in part) else ["row_id", "instance_id", "status", "error"])
            csv_write(run / "summary_by_seed.csv", summarize(rows, True))
            csv_write(run / "summary.csv", summarize(rows, False))
            status = "failed" if (run / "failure.txt").exists() else ("partial" if any(r["status"] != "ok" for r in rows) else "complete")
            json_write(run / "completion.json", {"status": status, "metric_rows": len(rows),
                        "failed_metric_rows": sum(r["status"] != "ok" for r in rows),
                        "dataset_total_wall_time_s": time.perf_counter() - started})
            json_write(run / "progress.json", {"state": status, "completed_records": len(rows) // 4,
                       "failed_records": sum(r["status"] != "ok" for r in rows) // 4})
            with (run / "experiment.md").open("a") as handle:
                handle.write(f"\n## Outcome\n\nStatus: {status}. Metric rows: {len(rows)}. "
                             "See summary.csv, completion.json and seed*/logs/*_validation.json. "
                             "Four objective rows represent one record, not four independent observations. "
                             "These are candidate comparisons, not global optimality certificates. "
                             "Any smoke result remains a development diagnostic. No performance-based tuning. "
                             "A failed/partial run is retained and must not be overwritten. Rerun with a new run_name after fixing the recorded failure.\n")
    return failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", help="Optional directory for new baseline outputs.")
    parser.add_argument("--source-runs", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--nn-seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--phase", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--reuse-successful-run-name", default=None,
                        help="New formal FlowX completion run: preserve successful no-loop records and their provenance")
    parser.add_argument("--gnn-epochs", type=int, default=100)
    parser.add_argument("--pg-epochs", type=int, default=20)
    parser.add_argument("--pg-train-limit", type=int, default=None)
    parser.add_argument("--flowx-samples", type=int, default=50)
    parser.add_argument("--flowx-epochs", type=int, default=500)
    parser.add_argument("--flowx-lr", type=float, default=0.1)
    parser.add_argument("--explainer-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=1729)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=1e-4)
    args = parser.parse_args()
    if args.eval_samples < 2 or min(args.gnn_epochs, args.pg_epochs) < 1:
        parser.error("At least two evaluation samples and positive epochs required")
    if min(args.flowx_samples, args.flowx_epochs) < 1 or args.flowx_lr <= 0:
        parser.error("FlowX sampling iterations, epochs and learning rate must be positive")
    args.run_name = args.run_name or (datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S") + f"_{args.phase}_explainers")
    if Path(args.run_name).name != args.run_name:
        parser.error("run_name must be a directory name")
    if args.phase == "formal" and args.pg_train_limit is not None:
        parser.error("Formal PGExplainer must use the complete training split")
    if args.reuse_successful_run_name and (args.phase != "formal" or any(not m.startswith("flowx_") for m in args.methods)):
        parser.error("Formal reuse is restricted to the verified FlowX explicit-loop repair")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    failures = [run_dataset(args, source) for source in args.source_runs]
    return int(any(failures))


if __name__ == "__main__":
    raise SystemExit(main())
