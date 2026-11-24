# GNN Fidelity Audit

Code and saved results for the paper **Are Ground-Truth Explanations Optimal
under Fidelity Objectives? An Optimization-Based Audit of Synthetic Benchmarks
for Graph Neural Networks** by Woo Seok Goh and Kyungsik Lee.

| Directory | Contents |
| --- | --- |
| [Main/](Main/) | Standard and robust fidelity audit, model training, data, checkpoints and audit results |
| [Explainers/](Explainers/) | Random, GNNExplainer, PGExplainer and FlowX, with their saved results |

The audit covers seven benchmarks: BA-House, BA2Grid, BA-HouseANDGrid,
BA-HouseORGrid, Tree-Cycles, BA-2Motifs and Infection. For each benchmark we
trained five 3-layer GINs with hidden dimension 5, which are the main models,
and five more at each of hidden dimensions 20 and 80 for the capacity study.
The main audit has 937 instance/model pairs.

This repository contains only the runs that the paper reports. It does not
include the paper sources, the tools that build the paper's tables and
figures, or tests.

## Install

Use Python 3.11 and run all commands from the repository root.

```bash
python -m pip install -r requirements.txt
# Also needed for GNNExplainer and PGExplainer:
python -m pip install -r Explainers/requirements.txt
```

We tested with Python 3.11.14, CPU PyTorch 2.9.1, PyG 2.7.0 and Gurobi 13.0.0.
The MILP searches need a Gurobi license. Enumeration does not, but `gurobipy`
must still be installed because the solver module is imported. A C++ compiler
is optional. If one is available, the enumeration helpers in
`Main/enumeration/native/` run faster.

## Code layout

- `Main/audit_standard.py` runs the standard fidelity audit.
- `Main/audit_robust.py` runs the robust fidelity audit.
- `Main/train_gnn.py` trains the GINs.
- `Main/data/` holds the dataset loaders and the frozen inputs.
- `Main/models/GIN.py` defines the GIN, and its checkpoints are in
  `Main/models/pretrained/GIN/`.
- `Main/training/` holds the training helpers and configurations.
- `Main/solvers/` holds the MILP searches:

```text
Main/solvers/
  standard_milp.py   # standard fidelity search over exact-K edge masks
  gin_milp.py        # MILP encoding of the trained GIN, used by standard_milp.py
  robust_milp.py     # scenario-table SAA-MILP for robust fidelity
```

`Main/robust_common.py` holds helpers shared by the robust runner and
`robust_milp.py`.

## Run the audit

```bash
python Main/audit_standard.py --help
python Main/audit_robust.py --help
python Main/train_gnn.py --help
```

### Standard fidelity

The command below runs standard sufficiency on BA-House with the settings used
in the paper. The other three MILP benchmarks (BA2Grid, BA-HouseANDGrid and
BA-HouseORGrid) use the same command. For necessity, use
`--objective max_necessity`.

```bash
python Main/audit_standard.py \
  --datasets BA-House --nn_seeds 1,2,3,4,5 --instance_filter correct \
  --solver milp --stages gt_size --objective max_sufficiency \
  --node_edge_budget_mode eq --num_layers 3 --hidden_dim 5 \
  --connectivity none --exclude_gt --no-incumbent --max_instances 30 \
  --time_limit 1200 --gurobi_method 1 --gurobi_threads 1 \
  --use_incident_cap_tightening --use_gurobi_symmetry_detection \
  --use_anderson_cuts --anderson_max_node_count 0 --anderson_max_cuts 300 \
  --no-use_star_constraints --no-use_connectivity_lower_bound \
  --no-gurobi_norel_hybrid --no-use_obbt_u --no-degree_table_layer1 \
  --output_dir /tmp/gnn-audit-standard --run_name example_standard
```

Tree-Cycles, BA-2Motifs and Infection are searched by enumeration instead. Use
`--solver enumeration` and drop `--max_instances 30`.

### Robust fidelity

Robust fidelity is searched with a scenario-table MILP. Before solving, the
code runs the model to get the target-class probability for every relevant
edge subset in each of the 50 fixed optimization scenarios. The MILP then picks
the mask with the best average over these scenarios. Because the network is
not encoded in the MILP, a search that ends optimal gives the exact optimum of
this fixed sample average. All robust results in the paper, including the
connectivity ablation, use this method.

```bash
python Main/audit_robust.py \
  --datasets BA-House --nn_seeds 1,2,3,4,5 --num_layers 3 --hidden_dim 5 \
  --metric sufficiency --alphas 0.9 --max_instances 30 \
  --opt_samples 50 --eval_samples 1000 --sampling_seed 0 \
  --counterexample_ci_level 0.95 --saa_table_mip_gap 0 \
  --connectivity none --exclude_gt --no-incumbent --time_limit 1200 \
  --gurobi_method 1 --gurobi_threads 1 --gurobi_numeric_focus 3 \
  --gurobi_symmetry 2 --no-gurobi_norel_hybrid --torch_threads 1 --device cpu \
  --output_root /tmp/gnn-audit-robust --run_name example_robust
```

For robust necessity, use `--metric necessity --alphas 0.1`. Some runs also
pass `--direct_row_numbering pool`, which numbers rows by their position in the
instance pool. Each run's `config.json` shows which numbering it used. Row IDs
determine the scenario seeds, so compare two runs only if they use the same
numbering.

Other settings used in the paper:

- Larger models: `--hidden_dim 20` or `--hidden_dim 80`.
- Connectivity ablation: `--connectivity cut --use_star_constraints
  --use_connectivity_lower_bound`.

## Saved results

Results are stored by fidelity type and search method:

```text
Main/results/
  standard_fidelity_mip/
  standard_fidelity_enumerations/
  robust_fidelity_mip/
```

The `_mip` folders hold MILP runs. Dataset folders use the dataset names that
the command line expects, so BA2Grid is in `BA2Grid-Longa` and Infection is in
`PyG-Infection-ER-L3-Split43`. Each run folder is named after the model width
and the objective, with `_connected` added for the connectivity ablation:

```text
Main/results/<family>/<dataset>/
  hidden5_sufficiency/             # main audit
  hidden5_necessity/
  hidden5_sufficiency_connected/   # connectivity ablation
  hidden5_necessity_connected/
  hidden20_sufficiency/            # larger hidden dimensions
  hidden80_necessity/
  ...
    seed1/ ... seed5/
```

Each run's `config.json` records its settings, such as hidden dimension,
connectivity, time limit, alpha and row numbering, and keeps the original run
name in `run_name`. The trained models are in
`Main/models/pretrained/GIN/<dataset>_3L_<width>H_seed<seed>/`.

A few cases to note:

- Standard runs at hidden dimensions 20 and 80 exist only for the three
  enumeration benchmarks.
- BA-HouseANDGrid has no connectivity ablation, because its reference
  explanations are not connected.
- The BA-2Motifs model with hidden dimension 80 and seed 4 did not learn the
  task (test accuracy 44.0%). Its rows are in the run files, but the paper
  leaves them out.

## How the paper counts the results

The paper applies its own decision rule to the saved rows. This rule is not
the same as the `counterexample_status` column written by the runners.

- **Instances.** For each benchmark, the paper uses the first 30 instances of
  the standard pool in pool order, or all of them if there are fewer. The same
  instances are used for every model, width and objective. Some runs contain
  more instances, and the paper keeps only these 30 when reading them. Pairs
  whose model misclassifies the instance are dropped.
- **Tolerance.** Scores within `tau = 1e-4` count as tied.
- **Standard fidelity.** Each pair is sorted into one of four outcomes: the
  reference is beaten by more than `tau`, a tie within `tau`, the reference
  wins, or unresolved. For MILP runs, the outcome comes from the best mask
  found and from the Gurobi bound in `best_bound`, converted to a probability
  with `bar_m`. A run that hit the time limit can still be settled by its bound.
- **Robust fidelity.** For each pair, a one-sided paired t test on the 1,000
  held-out scenarios asks whether the best mask beats the reference by more
  than `tau`. It uses `paired_advantage_mean`, `paired_advantage_std` and
  `paired_scenario_count`. A zero standard deviation gives `p = 1`. Holm
  correction at level 0.05 is applied within each benchmark, hidden dimension,
  mask family and objective.

## Explainers

`Explainers/run_explainers.py` runs all explainers. The folder is organized as
follows:

```text
Explainers/
  run_explainers.py
  requirements.txt
  methods/
    gnnexplainer.py
    pgexplainer.py
    dig_loader.py    # loads GNNExplainer and PGExplainer from DIG
    model_adapter.py # wraps the trained GIN and its edge masks
    flowx.py         # FlowX for sufficiency and necessity
    _vendor/flowx/   # FlowX source code with its license
  results/<method>/
  results/robust_reevaluations/
```

Every explainer is run on the same instances as the audit and selects exactly
as many edges as the reference explanation has. The available methods are
`random`, `gnnexplainer`, `pgexplainer`, `flowx_suf` and `flowx_nec`.

```bash
python Explainers/run_explainers.py --help
python Explainers/run_explainers.py \
  --source-runs Main/results/standard_fidelity_enumerations/BA-2Motifs/hidden5_sufficiency \
  --methods random --nn-seeds 1 --phase smoke --max-instances 1 \
  --output-root /tmp/gnn-explainer-example --run-name example_random
```

The saved runs are in `Explainers/results/<method>/<dataset>/hidden5/`. New
formal runs are written to `Explainers/results/<method>/<dataset>/<run-name>/`,
and existing runs are never overwritten.

FlowX runs its original sampling and refinement code. We make its edge scores
undirected and keep the GIN self term fixed in the final evaluation. On
BA-2Motifs, the FlowX results come from a second run. It copied the finished
records of an earlier run that had stopped, and recomputed the records whose
input graphs contain self-loops after we fixed how those loops are handled.

The robust explainer scores in the paper come from
`Explainers/results/robust_reevaluations/`. These runs score the saved
explainer masks on the same 1,000 held-out scenarios as the robust audit, with
the same sampling seeds and the same reference scores. Each `evaluation/*.npz`
file stores the masks, the scenarios and the resulting probabilities. The
script that made these runs needs per-instance attribution scores that are
not in this repository, so the runs cannot be repeated from here. Their files
also list Grad-CAM, which the paper does not report.

## Notes on the saved files

Paths inside the saved files follow the original workspace layout (for example
`Version5/...`). The code here maps them to this repository. After the runs
were produced, some names in the saved files were changed to match the current
code. No values changed. SHA-256 hashes recorded inside the files refer to the
original versions.

## License and citation

This repository is released under the [MIT License](LICENSE).

The FlowX code in `Explainers/methods/_vendor/flowx/` is copied from the
[FlowX repository](https://github.com/CM-BF/FlowX) (commit `01fc639`) and keeps
its own [MIT license](Explainers/methods/_vendor/flowx/LICENSE). DIG and the
other packages are installed through the requirements files and keep their own
licenses.

Dataset definitions are in `Main/data/pyg_datasets.py`. To cite this work, see
`CITATION.cff`.
