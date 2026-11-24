# GNN Fidelity Audit

Code and saved results for **Are Ground-Truth Explanations Optimal under Fidelity
Objectives? A Systematic Audit of Synthetic Benchmarks for Graph Neural
Networks**, by Woo Seok Goh and Kyungsik Lee.

| Directory | Contents |
| --- | --- |
| [Main/](Main/) | Standard and robust fidelity audit, model training, data, checkpoints and audit results |
| [Explainers/](Explainers/) | Random, GNNExplainer, PGExplainer and FlowX implementations or adapters and their saved results |

The repository holds the runs that the manuscript reports, and only those.
The audit covers seven benchmarks: BA-House, BA2Grid, BA-HouseANDGrid,
BA-HouseORGrid, Tree-Cycles, BA-2Motifs and Infection. Each benchmark has five
3-layer GINs with hidden dimension 5 (the main models) and five each with
hidden dimensions 20 and 80 (the capacity study). The main audit has 937
instance/model pairs. Paper sources, rendering and reporting tools, tests and
diagnostic runs are omitted.

## Install

Use Python 3.11 and run commands from the repository root.

```bash
python -m pip install -r requirements.txt
# Also needed for GNNExplainer and PGExplainer:
python -m pip install -r Explainers/requirements.txt
```

The verification environment used Python 3.11.14, CPU PyTorch 2.9.1, PyG 2.7.0
and Gurobi 13.0.0. MILP execution requires a Gurobi license. Enumeration imports
the shared solver module but does not optimize a Gurobi model. A C++ compiler
is optional for the accelerated enumeration helpers in `Main/enumeration/native/`.

## Run the audit

`Main/audit_standard.py` runs deterministic fidelity audits and
`Main/audit_robust.py` runs robust SAA audits. `Main/train_gnn.py` trains the
predictors. Shared dataset loaders and frozen inputs live in `Main/data/`,
models and checkpoints in `Main/models/`, the MILP implementation in
`Main/solvers/`, and training and configuration helpers in `Main/training/`.

```bash
python Main/audit_standard.py --help
python Main/audit_robust.py --help
python Main/train_gnn.py --help
```

Standard fidelity on the four MILP benchmarks (BA-House, BA2Grid,
BA-HouseANDGrid, BA-HouseORGrid), as run for the manuscript. Use
`--objective max_necessity` for necessity.

```bash
python Main/audit_standard.py \
  --datasets BA-House --nn_seeds 1,2,3,4,5 --instance_filter correct \
  --solver milp --stages gt_size --objective max_sufficiency \
  --node_edge_budget_mode eq --architecture gin --num_layers 3 --hidden_dim 5 \
  --connectivity none --exclude_gt --no-incumbent --max_instances 30 \
  --time_limit 1200 --gurobi_method 1 --gurobi_threads 1 \
  --use_o1_tightening --use_o3_symmetry_breaking \
  --use_anderson_cuts --anderson_max_node_count 0 --anderson_max_cuts 300 \
  --no-use_o4_star_constraints --no-use_o9_connectivity_lb \
  --no-gurobi_norel_hybrid --no-use_obbt_u --no-degree_table_layer1 \
  --output_dir /tmp/gnn-audit-standard --run_name example_standard
```

Tree-Cycles, BA-2Motifs and Infection use `--solver enumeration` with the same
model, budget and GT options and without `--max_instances 30`.

Robust fidelity uses the scenario-table MILP (`--saa_milp_objective
prob_table`, the only option). It precomputes the model's class probabilities
for every scenario and does not encode the network. All robust results,
including the connected-mask ablation, come from this formulation.

```bash
python Main/audit_robust.py \
  --datasets BA-House --nn_seeds 1,2,3,4,5 --architecture gin \
  --num_layers 3 --hidden_dim 5 --solver milp \
  --metric sufficiency --alphas 0.9 --max_instances 30 \
  --opt_samples 50 --eval_samples 1000 --sampling_seed 0 \
  --counterexample_ci_level 0.95 \
  --saa_milp_objective prob_table --saa_table_mip_gap 0 \
  --connectivity none --exclude_gt --no-incumbent --time_limit 1200 \
  --gurobi_method 1 --gurobi_threads 1 --gurobi_numeric_focus 3 \
  --gurobi_symmetry 2 --no-gurobi_norel_hybrid --torch_threads 1 --device cpu \
  --output_root /tmp/gnn-audit-robust --run_name example_robust
```

Use `--metric necessity --alphas 0.1` for robust necessity. Several later runs
also pass `--direct_row_numbering pool` (run names ending in `poolrow`), which
numbers rows by their position in the instance pool. The flag changes row
labels only. Larger models use `--hidden_dim 20` or `--hidden_dim 80`. The
connectivity ablation uses `--connectivity cut`.

Default result roots are separated by metric family and solver:

```text
Main/results/
  standard_fidelity_mip/
  standard_fidelity_enumerations/
  robust_fidelity_mip/
```

The `_mip` directories contain MILP runs. The standard runner takes
`--solver milp` or `--solver enumeration`, and the robust runner takes
`--solver milp` only. Each family keeps the original `<dataset>/<run-name>/seedN/`
layout. Dataset folders `BA2Grid-Longa` and `PyG-Infection-ER-L3-Split43` hold
BA2Grid and Infection.

## Which runs back which result

Runs are identified by the timestamp at the start of their names.

| Result | Standard fidelity | Robust fidelity |
| --- | --- | --- |
| Main audit, MILP benchmarks | `20261109_191411` | `20261109_150927` (BA-House: `20261109_160036`) |
| Main audit, Tree-Cycles and BA-2Motifs | `20260811_051355`, `20260811_052610` | `20261115_021703` (Tree-Cycles), `20261115_003921` (BA-2Motifs) |
| Main audit, Infection | `20260825_213900`, `20260825_213901` | `20261115_003921` |
| Hidden dimension 20 | `20261114_230238`, Infection `20260903_224506` | `20261114_230238` (MILP benchmarks), `20261115_021703` (Tree-Cycles, BA-2Motifs), `20261115_003921` (Infection) |
| Hidden dimension 80 | `20261115_125458` | `20261115_125458` |
| Connected masks, MILP benchmarks | `20261112_011751` | `20261111_131758` |
| Connected masks, enumeration benchmarks | `20261026_191500`, Infection `20260907_021500`, `20260907_021501` | `20261123_154018` |

Standard runs at hidden dimension 20 and 80 cover the three enumeration
benchmarks only. BA-HouseANDGrid has no connectivity ablation because its
references are disconnected. The BA-2Motifs model with hidden dimension 80 and
seed 4 failed to learn the task (test accuracy 44.0%). Its rows are kept in the
run files, and the manuscript excludes them.

## How the manuscript reads these files

The manuscript applies its own decision rule to the saved rows. The rule
differs from the `counterexample_status` column that the runners write.

- **Cohort.** For each benchmark, the main cohort is the first 30 instances of
  the model-independent standard pool, in pool order (all instances when fewer
  exist). The same instances are used for every model, width and objective.
  Standard enumeration runs and some robust runs contain the full pool, so the
  cohort is applied when reading. Pairs whose model misclassifies the instance are
  dropped.
- **Tolerance.** The score tolerance is `tau = 1e-4`.
- **Standard fidelity.** Each pair falls into one of four classes: reference
  inferiority (a candidate beats the reference by more than `tau`), tie within
  tolerance, reference superiority, or unresolved. For MILP runs the classes
  come from the incumbent and from the Gurobi dual bound in `best_bound`,
  mapped to probability space with `bar_m`. A time-limited run can still give
  a certified class when the bound settles it.
- **Robust fidelity.** For each pair, a one-sided paired t test on the 1,000
  held-out scenarios checks whether the expected improvement over the
  reference exceeds `tau`. It uses `paired_advantage_mean`,
  `paired_advantage_std` and `paired_scenario_count`. A zero standard deviation
  gives `p = 1`. Holm correction at level 0.05 is applied within each
  benchmark, hidden dimension, mask family and objective.

## Run the explainers

`Explainers/run_explainers.py` is the shared entry point. Implementation and
results are grouped as follows:

```text
Explainers/
  run_explainers.py
  requirements.txt
  methods/
    gnnexplainer.py
    pgexplainer.py
    dig_loader.py    # Shared loader for the pinned DIG implementation
    model_adapter.py # Shared frozen-model and edge-mask interface
    flowx.py         # FlowX sufficiency and necessity adapters
    _vendor/flowx/   # Pinned upstream sources, provenance and license
  results/<method>/
  results/robust_reevaluations/
```

All methods use the same explicit audit cohort and exact reference-size edge
budget. Available method names are `random`, `gnnexplainer`, `pgexplainer`,
`flowx_suf` and `flowx_nec`.

```bash
python Explainers/run_explainers.py --help
python Explainers/run_explainers.py \
  --source-runs Main/results/standard_fidelity_enumerations/BA-2Motifs/*Fid^-* \
  --methods random --nn-seeds 1 --phase smoke --max-instances 1 \
  --output-root /tmp/gnn-explainer-example --run-name example_random
```

Formal outputs use `Explainers/results/<method>/<dataset>/<run-name>/`.
Existing runs are never overwritten. FlowX uses the pinned upstream sampling
and refinement with an undirected edge projection and fixed GIN self terms
during final evaluation. On BA-2Motifs the FlowX results are the
explicit-loop completion runs (`20260917_152207`).

The robust explainer scores in the manuscript come from
`Explainers/results/robust_reevaluations/`. These runs evaluate the saved
explainer masks on the same 1,000 held-out scenarios as the robust audit
(the sampling seeds and reference scores match the audit runs above). Each
`evaluation/*.npz` file stores the masks, the scenario draws and the resulting
probabilities. The producing script needs per-instance attribution scores that
are not part of this repository, so these runs are records rather than a
command you can rerun here. Their files still list a Grad-CAM method, which
the manuscript does not report.

## Saved results and interpretation

Each run keeps its original `config.json`, per-seed `instance_fid.csv`,
selected and GT edge masks, solver records and available machine-readable
provenance. Rendered figures and per-run Markdown notes are omitted. Result
bytes and run names are unchanged. Paths written inside saved artifacts refer
to the original workspace. The execution readers map those paths to this
layout. Other external tools must account for them.

Raw sufficiency fidelity is smaller-is-better and necessity is larger-is-better.
The robust protocol uses 50 optimization scenarios and 1,000 independent,
paired evaluation scenarios. A time-limited incumbent is not a certified
optimum, and a fixed-SAA optimum is not a population optimum.

This is a current-code snapshot with historical result artifacts. Producer
code hashes and environment records are kept where available. The full formal
training and optimization were not rerun for this snapshot. Small checks
reproduced saved rows for standard MILP, standard enumeration, the robust
scenario-table MILP at hidden dimensions 5 and 80 with unrestricted and
connected masks, and all five explainers.

## Attribution and licenses

The original source workspace did not specify a repository-wide license.
Existing source attribution is kept. The pinned FlowX sources under
[Explainers/methods/_vendor/flowx/](Explainers/methods/_vendor/flowx/) keep their
[MIT license](Explainers/methods/_vendor/flowx/LICENSE) and `provenance.json` for
CM-BF/FlowX commit `01fc6390bf8dc64201d491a1effc5b71ae37e344`.
DIG is installed separately through the explainer requirements. Third-party
packages keep their own licenses. Dataset definitions and provenance are in
`Main/data/pyg_datasets.py`. See `CITATION.cff` for the accompanying research
citation.
