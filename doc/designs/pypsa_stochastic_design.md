# PyPSA Stochastic Programming via mpi-sppy (LP-file interface) — Design

Status: draft for review. Spans two repos: **PyPSA** (work on the `DLWoodruff/PyPSA`
fork) and **mpi-sppy**. The required mpi-sppy support (`.lp` files + per-nonant
rho from `{s}_rho.csv`) is **merged upstream** — Pyomo/mpi-sppy `main` — so
all remaining work is on the PyPSA side. No use of the mpi-sppy *agnostic/guest*
framework — the coupling is at the **file boundary**. Core feasibility has been
validated end to end (Phase 0); the **PyPSA exporter + read-back (Phase 1)** and
the **inline `solve_stochastic_mpisppy` driver (Phase 3)** are implemented and validated
against the mpi-sppy reader, the EF oracle, and a real PH run that converges to the
EF first stage (§12). The capacity-fixed **dispatch re-solve** (`dispatch="resolve"`,
per-scenario operations + scenario-conditional duals; `n_mod` → `n_mod_opt`) is
also implemented (§13.7), as are the **user docs** (the PyPSA user-guide page
`docs/user-guide/optimization/stochastic-decomposition.md`). Remaining (Phase 4):
`dispatch="read"` (mpi-sppy tree solution, §13.7 option 2) as a follow-up PR.

## 1. Goal

Let a PyPSA user solve a stochastic energy-system model by **decomposition**
(Progressive Hedging + bounding cylinders) using mpi-sppy, as an alternative to
PyPSA's existing *monolithic* Extensive Form (EF) solve.

**The point of this work is large-scale, *parallel* decomposition** — running PH
with bounding cylinders across many MPI ranks, typically as separate scheduler
jobs on an HPC cluster. **§13 is the heart of this design.** The integrated,
single-process (serial) path is provided **only for testing and small problems**,
not as the intended production mode.

The user-facing entry point is a new PyPSA method:

```python
n.set_scenarios({"low": 0.3, "med": 0.4, "high": 0.3})   # existing PyPSA UX
# ... set per-scenario data ...
n.optimize.solve_stochastic_mpisppy(method="ph", solver_name="gurobi", ...)   # NEW
```

## 2. Background and motivation

### 2.1 PyPSA already builds and solves the EF

PyPSA has native multi-scenario support. `n.set_scenarios({...})`
(`pypsa/network/index.py`, ~L785) adds a `scenario` index level to component
data; `n.optimize()` then builds a single `linopy` model in which:

- **capacity** variables (`*_nom`: `Generator-p_nom`, `Line-s_nom`,
  `Link-p_nom`, `Store-e_nom`, `StorageUnit-p_nom`) have **no** `scenario`
  dimension — they are shared **here-and-now / first-stage** decisions;
- **operational** variables (`Generator-p`, dispatch, status, …) carry a
  `(scenario, name, snapshot)` dimension — they are **second-stage / recourse**;
- the objective is probability-weighted expected cost (with optional CVaR).

This monolithic EF *is* the "simple solver." Because PyPSA can already form and
solve the EF directly, **solving the EF through mpi-sppy adds no value** — it is
useful only as a correctness oracle (see §11). The real target is **PH/cylinder
decomposition**, which PyPSA does not have.

### 2.2 Why the file interface

mpi-sppy already ships a complete file-based scenario loader
(`mpisppy/problem_io/mps_module.py`). PyPSA/`linopy` can write each scenario's
subproblem to a standard LP file (MPS is also supported). This keeps PyPSA free
of any Pyomo dependency and keeps mpi-sppy free of any PyPSA dependency: the two
programs meet only through files. mpi-sppy needs essentially **no new code** —
its reader already parses LP, and the only additions (merged in #770) were two
small file-path hooks: resolving the `.lp` / `.mps` filename and reading
`{s}_rho.csv` (§10).

## 3. Architecture

Three layers, all on the PyPSA side except the unchanged mpi-sppy solve:

```
n.set_scenarios({...})                       # EXISTING PyPSA UX
        │
        ▼
n.optimize.solve_stochastic_mpisppy(method="ph", solver_name="gurobi", ...)   # NEW
        │
        ├─ (1) EXPORT  — for each scenario s (file stem  scenario{i}, §4.3):
        │        slice scenario s  (n.get_scenario(s)) → standalone single-scenario Network
        │        create_model()  → linopy.Model
        │        write  {stem}.lp           (model; implicit  x{label}  names; MPS ok too)
        │        write  {stem}_nonants.json (probability + first-stage var names)
        │        write  {stem}_rho.csv      (per-nonant cost-proportional rho)
        │
        ├─ (2) SOLVE  — mpi-sppy driver as a subprocess (MPI for cylinders):
        │        mpiexec -np K python -m mpi4py generic_cylinders.py
        │            --mps-files-directory=DIR [--config-file FILE]
        │            --xhatshuffle --lagrangian --default-rho ...
        │        → converged first-stage capacities + bounds
        │
        └─ (3) WRITE-BACK
                 set  *_nom_opt  on n
                 (optional) re-solve each scenario with capacities fixed
                 → full per-scenario dispatch
```

## 4. The file-interface contract

For a directory `DIR` (`cfg.mps_files_directory`), each scenario `s` contributes:

| File | Producer | Consumer | Content |
|------|----------|----------|---------|
| `{s}.lp` (or `{s}.mps`) | PyPSA (linopy) | mpi-sppy `mps_reader` | the scenario subproblem |
| `{s}_nonants.json` | PyPSA | mpi-sppy `mps_module` | probability + scenario tree + nonant names |
| `{s}_rho.csv` | PyPSA | mpi-sppy | per-nonant rho (cost-proportional default) |

### 4.1 `{s}_nonants.json`

Schema consumed by `mpisppy/problem_io/mps_module.py:scenario_creator` (~L38–87):

```json
{
  "scenarioData": { "scenProb": 0.3 },
  "treeData": {
    "globalNodeCount": 1,
    "nodes": {
      "ROOT": { "serialNumber": 0, "condProb": 1.0,
                "nonAnts": ["x0", "x3", "x7"] }
    }
  }
}
```

The `nonAnts` entries are variable names **exactly as they appear in the LP (or
MPS) file**. The reader normalizes `(` and `)` to `_` when resolving them against the
Pyomo model (`mps_reader.py` ~L114; `mps_module.py` ~L67) — but it does **not**
normalize `#`. See the naming decision in §6.

The exact format is mirrored by mpi-sppy's own writer
`mpisppy/extensions/scenario_lp_mps_files.py` (~L24–87) and by the reference
artifacts in `mpi-sppy-1/_delme_test_write_mp_mps_dir/`
(`Scenario*.{mps,lp,json}`, `Scenario*_rho.csv`). PyPSA reproduces this format
directly rather than going through that extension.

### 4.2 `{s}_rho.csv`

Per-nonant PH rho, in mpi-sppy's convention (header `varname,rho`, then one row
per nonant; `scenario_lp_mps_files.py` ~L84–87):

```csv
varname,rho
x0,1000
x1,1500
```

PyPSA emits this by default with **cost-proportional rho** (§7.1). Consumed by
mpi-sppy's `mps_module._rho_setter` as of #770 (§10). **Per-node rho consistency
is the writer's job:** mpi-sppy applies rhos per scenario and does *not* reconcile
them across scenarios, so a given nonant must get the **same** rho in every
scenario's file. For two-stage problems that means every scenario's `_rho.csv` is
identical — PyPSA computes rho once and replicates it (§7.1).

### 4.3 Scenario file naming (integer stems)

mpi-sppy derives each scenario's *name* from its model file **stem** and
**requires the stem to end in an integer**: `mps_module.scenario_names_creator`
globs `*.lp` / `*.mps`, sorts the stems lexicographically, and does
`re.search(r"\d+$", ...)` (preferring zero-based, warning otherwise). PyPSA
scenario names (`"low"`, `"med"`, `"high"`) therefore **cannot** be the file
names. The exporter writes **zero-padded integer stems** — `scenario0.lp`,
`scenario1.lp`, … (`scenario00…` once there are ≥ 10 scenarios, so the
lexicographic sort matches the numeric order) — and preserves the human-readable
PyPSA name in `{stem}_nonants.json` `scenarioData.name` and in the manifest's
`scenario_files` map (`{index, name, stem, probability}`). This renaming is
invisible to the rest of the machinery: nonant names are `x{label}` (§6) and
write-back keys off the manifest's nonant→component map (§13.5), both independent
of the scenario file name.

## 5. Nonant (first-stage) identification

Default first stage = **all extendable capacity variables**, derived in code from
`pypsa.descriptors.nominal_attrs` (`Generator-p_nom`, `Line-s_nom`,
`Transformer-s_nom`, `Link-p_nom`, `Process-p_nom`, `Store-e_nom`,
`StorageUnit-p_nom`) plus modular counts `{component}-n_mod`, filtered to those
variables actually present in the built model. The user may override with an
explicit list (to restrict the first stage, or to add e.g. committable
first-stage decisions later).

For each first-stage `linopy` variable, PyPSA recovers its on-file names from the
integer `labels` (see §6) and writes them into `nonAnts`.

## 6. Variable naming — the enabling fact

### 6.1 Determinism (verified)

`linopy` assigns variable *labels* strictly by **creation order**, independent of
any numeric data (objective coefficients, bounds, RHS). Empirically, two models
built with the same sequence of `add_variables` calls produce **identical**
on-file names even when the data differ. So scenario networks built with the same
sequence of variable creations emit LP (or MPS) files whose variables line up by
name — which is what mpi-sppy needs to enforce nonanticipativity across scenarios.

This determinism is the load-bearing fact. mpi-sppy matches nonants
**positionally**, so its actual requirement is only that the **first-stage
nonants share names and order across scenarios** (§9.1) — weaker than full
structural identity, which is merely a *sufficient* (and, for PyPSA, simple) way
to guarantee it.

### 6.2 Use implicit `x{label}` names

`linopy.io` can write two name styles
(`linopy/io.py`, `get_printers_scalar`, ~L99–135):

- **implicit** (default): `x{label}` (e.g. `x0`, `x42`) — opaque, no special
  characters;
- **explicit** (`explicit_coordinate_names=True`):
  `Generator_p_nom(wind)#0` — readable, but contains `(`, `)`, and `#`.

**Decision: use implicit `x{label}`.** The mpi-sppy reader normalizes `()` but
not `#`, and `#` in a Pyomo component name is fragile through
`find_component`. `x{label}` round-trips cleanly through the file writer →
coin-or `mip` → Pyomo (validated for LP and MPS, §6.3). PyPSA maps a first-stage
`linopy` variable to its `x{label}` names via:

```python
labels = n.model["Generator-p_nom"].labels.data.ravel()       # ints
names  = ["x" + str(int(L)) for L in labels if L >= 0]         # mask -1 (inactive)
```

(equivalently `linopy.io.get_printers_scalar(n.model, explicit_coordinate_names=False)`).

### 6.3 File format: LP primary (naming validated in Phase 0)

Both formats were validated end to end in Phase 0 (PyPSA → file → mpi-sppy
reader): implicit names are `x{label}`, deterministic across structurally-
identical scenario networks built with different data, resolved by
`find_component("x{label}")`, with a matching numeric round-trip (see §12).

**Decision: prefer LP.** `linopy` writes **LP** with its own writer
(`to_lp_file`, `io.py` ~L588), giving full, stable control of the on-file names
(no dependence on a third party's write behavior) and producing human-readable
files — valuable for debugging the scenario subproblems and the nonant matching.

**MPS remains supported** as an alternative. `linopy` writes MPS by delegating to
`highspy.Highs.writeModel` (`io.py` ~L690–701); Phase 0 confirmed HiGHS preserves
the implicit `x{label}` names, so MPS round-trips cleanly too.

## 7. Objective convention (no double counting)

`create_model()` for a single network builds
`objective = Σ capital_cost · (*_nom) + Σ marginal_cost · dispatch`
(`pypsa/optimization/optimize.py:define_objective`, ~L139; capex on `*_nom`
~L312–333). Each scenario's file therefore carries the **full** capex (on the
nonants) **plus that scenario's** opex. This is correct for PH/EF because

```
Σ_s p_s · (capex + opex_s) = capex + Σ_s p_s · opex_s = capex + E[opex]
```

since the scenario probabilities sum to 1. mpi-sppy applies the `scenProb`
weighting; capex is not double-counted.

### 7.1 Cost-proportional rho (default)

A good PH rho for a nonant scales with its objective coefficient, so PyPSA sets,
for each first-stage variable `i` with objective coefficient `c_i` (the
`capital_cost` on the `*_nom` variable):

```
rho_i = max(rho_floor, alpha * |c_i|)
```

with `rho_alpha` (default 1.0) and a small `rho_floor` (default 1e-3) so a
zero-cost nonant still gets a positive rho. In code `c_i` is read from the built
model's `model.objective.flat` (its `vars`/`coeffs` table), which equals
`capital_cost` for the `*_nom` nonants — robust to any user-chosen first stage.
PyPSA writes the values to `{stem}_rho.csv` (§4.2). The `rho` argument (§8)
selects this default, a flat scalar, or an explicit mapping keyed by the on-file
nonant name (`{x{label}: rho}`, missing entries → `rho_floor`).

Because rho must be **identical per nonant across scenarios** (§4.2 — mpi-sppy
does not reconcile), PyPSA computes the rho vector **once** (capital costs are
first-stage, hence scenario-invariant) and writes the same values to every
scenario's `_rho.csv`. If a use case ever varied a nonant's cost by scenario,
PyPSA would still emit a single agreed rho (e.g. the mean) to keep PH well-defined.

## 8. Public API (PyPSA)

The API exposes the **three phases** (§13.6) as methods on the existing
`OptimizationAccessor`. The write/read phases are **dependency-free** (no
mpi-sppy needed); only the inline `solve_stochastic_mpisppy` runs the driver.

```python
# Phase 1 — write the problem for mpi-sppy (1 rank, PyPSA env)
manifest = n.optimize.write_stochastic_problem_mpisppy(
    directory,
    clean=True,                  # clear stale scenario files in `directory` first (§13.6)
    first_stage=None,            # None = all extendable capacities (+ *-n_mod); else explicit list
    rho="cost-proportional",     # "cost-proportional" | float | {on-file name: rho}  (§7.1)
    rho_alpha=1.0, rho_floor=1e-3,
    file_format="lp",            # "lp" | "mps"
    cylinders=("lagrangian", "xhatshuffle"),
    solver_name="gurobi",        # recorded in the manifest command
    default_rho=1.0,             # mpi-sppy --default-rho fallback in the manifest command
    config_file=None,            # mpi-sppy --config-file to reference (§8.2)
    mpisppy_args=None,           # extra mpi-sppy CLI args (§8.2)
    model_kwargs=None,           # forwarded to create_model() per scenario
)
# manifest (also written to DIR/pypsa_stochastic_manifest.json) carries:
#   scenarios, scenario_files (name↔stem↔prob), first_stage, nonants, nonant_map,
#   rho, solution_file, solve_command (the exact phase-2 mpiexec line), mpi_ranks,
#   sbatch_template.

# Phase 3 — read the incumbent back onto the network (1 rank, PyPSA env)
n.optimize.read_stochastic_solution_mpisppy(
    directory,
    solution_file=None,          # defaults to the manifest's solution_file (DIR/xhat.csv)
)   # sets *_nom_opt from mpi-sppy's --write-xhat-file CSV (§13.5)

# Inline convenience (laptop / single node) = phase 1 + subprocess solve + phase 3
n.optimize.solve_stochastic_mpisppy(
    working_dir=None,            # None = temp dir, removed afterwards unless keep_files
    method="ph",                 # "ph" (target) | "ef" (correctness oracle only)
    solver_name="gurobi",        # QP/MIQP for the PH proximal term; or LP/MILP if proximal terms linearized (§9.3)
    first_stage=None, rho="cost-proportional", rho_alpha=1.0, rho_floor=1e-3,
    file_format="lp",
    cylinders=("lagrangian", "xhatshuffle"),
    default_rho=1.0, max_iterations=50,
    config_file=None, mpisppy_options=None, mpisppy_args=None,   # (§8.2)
    nprocs=None,                 # ranks for the inline mpiexec run (PH only)
    keep_files=False,            # keep a temp working_dir after solving
    tee=True,                    # stream the driver output live
    model_kwargs=None,           # forwarded to create_model() per scenario
)   # sets *_nom_opt on n; returns {on-file nonant name: value}
```

Implemented in a new `pypsa/optimization/stochastic_mpisppy.py`:

- `write_stochastic_problem_mpisppy(...)` and `read_stochastic_solution_mpisppy(...)` — the two
  **dependency-free** phase functions (also bound as accessor methods); they
  touch files only and do **not** import mpi-sppy;
- `solve_stochastic_mpisppy(...)` = phase 1 + subprocess `mpiexec … generic_cylinders.py`
  + phase 3; this is the **only** entry that needs the mpi-sppy driver.

The optional extra `pypsa[mpisppy]` (= `{mpi-sppy, mpi4py, mip}`) is therefore
needed only where the inline solve runs — not for the decoupled write/read jobs
(§13.6).

### 8.1 Optional-feature integration (PyPSA conventions)

The stochastic backend follows PyPSA's established optional-feature pattern; the
closest precedent is `tsam` time-series aggregation
(`pypsa/clustering/temporal.py`). The contract: importing PyPSA never requires
the extra deps, and invoking the feature without them fails loudly with an
install hint.

**Extra.** Declare in `pyproject.toml` `[project.optional-dependencies]`, named
after the backend to match the `tsam` / `cartopy` / `gurobipy` convention.
The extra pulls only the PyPI-installable helpers (`mpi4py`, `mip`); `mpi-sppy`
itself is installed separately from GitHub until a PyPI release ships the MPS-file
loader support, after which it returns here as a plain `mpi-sppy>=<version>` pin:

```toml
mpisppy = ["mpi4py", "mip"]
```

(`mip` is coin-or python-mip, the LP/MPS parser behind mpi-sppy's `mps_module`.)

**Gate + lazy import.** Reuse PyPSA's centralized helper
`check_optional_dependency(module_name, install_message)` (`pypsa/common.py`,
~L725) at the entry of `solve_stochastic_mpisppy`, then import inside the function.
**Import names differ from PyPI names**: `mpi-sppy` → `mpisppy`,
`python-mip` → `mip`.

```python
from pypsa.common import check_optional_dependency

def solve_stochastic_mpisppy(self, ...):
    check_optional_dependency(
        "mpisppy",
        "Missing optional dependencies for stochastic decomposition. "
        "Install via `pip install pypsa[mpisppy]`.",
    )
    import mpisppy            # noqa: PLC0415  (lazy)
    ...
```

This shared helper is preferred over `tsam`'s hand-rolled `find_spec` check for
consistency across the codebase. Only `solve_stochastic_mpisppy` gates on mpi-sppy;
`write_stochastic_problem_mpisppy` / `read_stochastic_solution_mpisppy` are file-only and import
cleanly without it (§13.6).

**Accessor placement.** `solve_stochastic_mpisppy` is a method on the **existing**
`OptimizationAccessor` (`pypsa/optimization/optimize.py`), so the public call is
`n.optimize.solve_stochastic_mpisppy(...)` — alongside `create_model` and
`optimize_transmission_expansion_iteratively`. No new top-level accessor.

**Config.** Defaults such as `solver_name` come from the existing options system
(`pypsa.options.params.optimize.*`, `pypsa/_options.py`); feature-specific
defaults (rho, cylinders) could later be registered under
`params.optimize.stochastic.*`.

### 8.2 Passing mpi-sppy options

mpi-sppy already has a full configuration system (`mpisppy/utils/config.py`):
`parse_command_line` reads `--config-file FILE` (declared ~L363, applied
~L1582–1595) to load *all* options from a file, plus `--solver-options-file`
(~L204) for layered solver options. A Python dict must therefore **not** be the
only input mechanism.

PyPSA invokes the mpi-sppy **driver as a subprocess** (under `mpiexec` for the
cylinders), so options are layered, lowest → highest precedence:

1. **PyPSA-managed essentials** it always sets: `--mps-files-directory`, the
   chosen cylinders, `--solver-name` (default from
   `pypsa.options.params.optimize`), and the `method`/`file_format` plumbing;
2. **`config_file=`** — forwarded verbatim as mpi-sppy `--config-file`; this is
   the **primary way** to pass the full option surface from a file;
3. **`mpisppy_options=`** — optional dict, translated to CLI flags (programmatic
   convenience for a few common knobs);
4. **`mpisppy_args=`** — explicit extra CLI args, last word.

PyPSA never reimplements mpi-sppy's option set; it forwards to mpi-sppy's own
config (file + CLI) — consistent with the file-boundary philosophy. For example,
mpi-sppy's runtime **rho setters** — which adapt the PH penalty during the solve
(`--coeff-rho`, `--sensi-rho`, `--grad-rho`, …) — are reached this way (e.g.
`mpisppy_args=["--coeff-rho"]`), complementing the *initial* per-nonant rho PyPSA
writes to `{s}_rho.csv` (§7.1).

## 9. Assumptions and constraints

1. **Consistent nonants (the necessary condition).** mpi-sppy matches nonants
   positionally, so the only hard requirement is that the **first-stage nonants
   appear with the same names in the same order in every scenario**. Building all
   scenarios with identical structure (same components and snapshots, data-only
   differences) is a *sufficient* — and, for PyPSA, the simplest — way to
   guarantee this via linopy's deterministic labels (§6.1); it is **not
   necessary** (scenarios may differ in second-stage structure as long as the
   nonant name-list matches). The exporter must **assert** the nonant lists agree
   (names + order) across scenarios and fail loudly on mismatch.
2. **Two-stage first.** capacity = stage 1, dispatch = stage 2. The JSON tree
   format and mpi-sppy support multi-stage, but v1 targets two-stage.
3. **Solver.** The PH proximal term is quadratic, so subproblems are QPs (MIQPs
   with integer first stage, §9.4); a QP/MIQP solver such as Gurobi (the default,
   present in the dev env) handles them natively. A QP-capable solver is **not
   required**, though: mpi-sppy can linearize the proximal term
   (`--linearize-proximal-terms`, a refined piecewise-linear approximation;
   binaries exactly via `--linearize-binary-proximal-terms`), so an LP/MILP solver
   (HiGHS, CBC, …) can be used instead.
4. **Integer / committable first-stage is supported.** First-stage variables may
   be integer (e.g. modular `*-n_mod`, committable on/off). PH consensus on
   integers is heuristic, but mpi-sppy's bounding cylinders (Lagrangian lower
   bound + xhat upper bound) bracket the optimum and report a valid optimality
   gap. With the PH proximal term, integer first-stage makes the subproblems
   MIQPs (Gurobi handles these directly, or linearize the proximal — quadratic —
   term, §9.3, to keep them MILPs).

## 10. Required mpi-sppy support — MERGED (#770)

The file-path support PyPSA relies on — `.lp` scenario files and per-nonant rho
from `{s}_rho.csv` — is merged on Pyomo/mpi-sppy `main` (#770); PyPSA just needs
an mpi-sppy at or past it (no driver/reader/PH changes were required). For details
see mpi-sppy's user docs (`doc/src/agnostic.rst`, `doc/src/extensions.rst`) and
the companion `doc/designs/mps_module_lp_rho_design.md`. The one constraint it
places on PyPSA — identical per-nonant rho across scenarios — is in §4.2 / §7.1.

## 11. Correctness and testing

- **EF oracle:** solve via mpi-sppy EF over the written files and compare the
  objective to PyPSA's native EF (`n.optimize()` on the `set_scenarios`
  network). They must match to tolerance. (This is the only role of the EF path.)
- **PH convergence:** PH first-stage (capacities) must converge to the EF
  first-stage solution. For **integer** first-stage, check instead that the
  cylinders' bound gap brackets the EF objective (PH consensus is heuristic; the
  bounds are valid — §9.4).
- **Fixture:** a tiny 2-bus, ~3-scenario, few-snapshot network — small enough to
  solve the EF directly and to inspect the LP/JSON by eye.

## 12. Phased plan

- **Phase 0 — de-risk — DONE.** Installed coin-or `mip` (1.17.6, bundles CBC via
  `cbcbox`) in `pypsa-sppy`; built two structurally-identical PyPSA networks
  (1 bus, 2 extendable generators, 3 snapshots) with different data; wrote each
  to MPS and LP with implicit names; round-tripped through
  `mps_reader.read_mps_and_create_pyomo_model`. **Results:** nonants are
  `x0`, `x1`; names identical across the two networks (determinism holds through
  the full PyPSA → HiGHS → `mip` path); `find_component` resolves every nonant
  for both formats; and the model round-trips numerically (PyPSA-native objective
  == LP/MPS-via-Pyomo objective = 123100). **Conclusion:** the file-based
  approach is sound; LP is the primary format (full naming control,
  human-readable), MPS also works.
- **Phase 1 — exporter — DONE (validated 2026-06-26).**
  `pypsa/optimization/stochastic_mpisppy.py` with `write_stochastic_problem_mpisppy` +
  `read_stochastic_solution_mpisppy` (both dependency-free), bound as `n.optimize.*`
  methods; 14 dependency-free tests in `test/test_stochastic_export.py` (no solver,
  no mpi-sppy). Scenario slicing reuses the existing `n.get_scenario(s)` (§14
  resolved); writes integer-stem (§4.3) LP/MPS + nonant JSON + cost-proportional
  `_rho.csv` + `pypsa_stochastic_manifest.json`. **Validated** against
  `mpisppy.problem_io` in-process (`scenario_names_creator` + `scenario_creator`
  resolve every nonant, consuming LP + JSON + rho) and via the **EF oracle: native
  PyPSA EF == mpi-sppy EF over the written files, exact match (97210.0)**, with the
  incumbent `xhat.csv` read back to identical `*_nom_opt`. (`n_mod` write-back +
  capacity-fixed dispatch re-solve deferred to Phase 4.)
- **Phase 2 — mpi-sppy file-path support — DONE (#770, upstream main):** `.lp`
  filename lookup + `_rho.csv` consumption + tests (§10).
- **Phase 3 — inline driver — DONE (validated 2026-06-26).**
  `solve_stochastic_mpisppy(...)` added to `pypsa/optimization/stochastic_mpisppy.py` and bound as
  `n.optimize.solve_stochastic_mpisppy` (§8): write (Phase 1) + blocking mpi-sppy
  subprocess + read-back (§13.5). Gates on `check_optional_dependency("mpisppy",
  …)`; PyPSA never joins MPI — mpi-sppy runs wholly in the subprocess. `method="ph"`
  runs the cylinders under `mpiexec`; `method="ef"` solves the extensive form in a
  single process (`--EF --EF-solver-name`, the oracle). `mpisppy_options`
  dict→CLI, `config_file`/`mpisppy_args` passthrough, `nprocs`/`max_iterations`
  knobs, temp-dir `working_dir` with `keep_files`, `tee` streaming. The
  `pypsa[mpisppy]` extra (`mpi-sppy`, `mpi4py`, `mip`) landed here too. **Validated**
  in `pypsa-sppy`: 25 dependency-free + 1 guarded e2e test (mpi-sppy EF first stage
  == native PyPSA EF first stage); 38 pre-existing stochastic tests still pass; and
  a real **PH run under `mpiexec -np 3` converges exactly to the EF first stage
  (max abs diff 0.0; EF obj 97210.0)**.
- **Phase 4 — write-back completion + docs.** `n_mod` → `n_mod_opt` **done**; the
  capacity-fixed dispatch re-solve `dispatch="resolve"` (per-scenario operations +
  scenario-conditional duals, with a `scenarios=` subset, §13.7) **done**; and the
  **user docs done** — `docs/user-guide/optimization/stochastic-decomposition.md`
  covers the inline vs decoupled/SLURM workflow (§13.6) and the file-interface
  performance caveat (§14). Remaining: `dispatch="read"` (mpi-sppy tree solution,
  §13.7 option 2) as a **follow-up PR**. The public methods, the `pypsa[mpisppy]`
  extra and the §11 correctness tests already landed in Phases 1 and 3.

## 13. Parallel execution and scaling

PyPSA is **not** part of the MPI job. It writes the scenario files (§4) and later
reads the incumbent solution back — both serial, 1-rank steps. The mpi-sppy solve
runs separately across many ranks, either as a blocking subprocess (inline) or as
its own scheduler job (§13.6). The whole coupling is files at both ends; PyPSA
never joins the MPI communicator.

### 13.1 Rank ↔ cylinder mapping

With `C` cylinders (e.g. 3 = PH hub + Lagrangian spoke + xhatshuffle spoke),
mpi-sppy requires `N % C == 0` and gives each cylinder `N/C` ranks. **Every
cylinder holds the full scenario set**, distributed across its `N/C` ranks; a
given `{s}.lp` is read **once per cylinder** (C× total), by the rank that owns it.
Example, S = 30 scenarios, C = 3:

| `np` (N) | ranks/cylinder | scenarios/rank |
|---|---|---|
| 3  | 1  | 30 |
| 30 | 10 | 3  |
| 90 | 30 | 1  |

A rank reads only *its* scenarios, so per-rank startup I/O is `S / (N/C)` files —
exactly one `.lp` at full scale.

### 13.2 File I/O is startup-only

Files are touched only when each rank builds its local scenarios in
`scenario_creator` (`.lp` → Pyomo via coin-or `mip`, plus `_nonants.json` and
`_rho.csv` via `_rho_setter`). Thereafter PH iterations are in-memory subproblem
solves plus MPI reductions — **no per-iteration I/O**. The file boundary is a
one-time startup cost, not a steady-state bottleneck.

### 13.3 Correctness linchpin under MPI

mpi-sppy enforces nonanticipativity by reducing nonant vectors **positionally**
across scenarios — within a cylinder (the `xbar` reduction) and across cylinders
(bound/incumbent exchange). Therefore the `nonAnts` list must be **identical in
both names and order across every scenario's `_nonants.json`**. The determinism
result (§6.1) and matching nonant lists (§9.1) guarantee this; the exporter must
assert it (same ordered name-list for all scenarios). This is *why* deterministic
naming is load-bearing rather than cosmetic.

### 13.4 Scaling considerations

- **Shared filesystem (multi-node).** `DIR` and the solution file must be visible
  to all nodes (shared FS), or staged per node. Single-node is a non-issue.
- **Startup I/O.** `C × S` reads total, distributed across ranks, one-time.
- **Parallelism ceiling.** ranks/cylinder ≤ S, so **max useful N ≈ C × S**.
  Scenario count — not rank count — sets the ceiling; to use many ranks you need
  many scenarios (or bundling — see below).
- **Bundling (large S).** Proper bundles compose with the file path (#771):
  `--scenarios-per-bundle` works directly — mpi-sppy infers the scenario count
  from the directory (no `--num-scens`) — and per-nonant rho from `{s}_rho.csv` is
  assembled bundle-aware. This relies on the scenario-constant rho invariant
  (§7.1): the bundle takes each nonant's shared rho and errors if its scenarios
  disagree.
- **Solver licensing.** Each rank runs its own solver engine concurrently (QP, or
  MIQP with integer first stage, §9.4 — or LP/MILP if the proximal term is
  linearized, §9.3); at large N this needs a Gurobi
  floating/cluster license permitting that many simultaneous engines.
- **Memory.** ~one Pyomo scenario model per rank at full scale — scales well.

### 13.5 Result write-back

mpi-sppy writes the incumbent's first-stage (nonant) values with
**`--write-xhat-file PATH`** — a by-name CSV with `#` comment lines then
`node_name, variable_name, value` rows (`sputils.write_nonant_tree_csv`); it works
identically for EF and cylinders runs. `read_stochastic_solution_mpisppy` parses the
`ROOT` rows, maps each `x{label}` back to its component via the manifest's
`nonant_map`, and sets `*_nom_opt` (shared across all scenarios — first-stage) —
the return half of the file boundary, consistent with §3. (`x{label}` names
contain no `(` / `)`, so mpi-sppy's name normalisation is a no-op here.) The
inline `solve_stochastic_mpisppy` (§8) drives this whole loop — write, subprocess solve,
read-back — in one call. Modular module counts (`*-n_mod` → `n_mod_opt`) are also
written back (the integer-derived `*_nom_opt = n_mod_opt × *_nom_mod`). Recovering
the **second stage** (per-scenario dispatch and marginal prices), which the xhat
file does not carry, is Phase 4 — see §13.7.

### 13.6 Execution paths: inline vs decoupled (SLURM)

**Inline** (laptop / single interactive node): `solve_stochastic_mpisppy` writes the
files, runs `mpiexec -np N … generic_cylinders.py …` as a blocking subprocess,
and reads the solution back — one Python call.

**Decoupled** (HPC scheduler): the three phases (§8) are submitted as **separate
jobs**, because you cannot hold a process/allocation open while a large MPI job
queues and runs, the phases have very different resource profiles (1 core / many
cores / 1 core), and even different environments — **PyPSA** for write + read,
**mpi-sppy** for solve (§8). `write_stochastic_problem_mpisppy` emits the exact phase-2
command and an `sbatch` template; the user submits a dependency chain:

```bash
DIR=/shared/run42
rm -f "$DIR"/*.lp "$DIR"/*.mps "$DIR"/*_nonants.json "$DIR"/*_rho.csv  # hygiene, see below
j1=$(sbatch --parsable write.sbatch)                           # -n 1,  PyPSA env
j2=$(sbatch --parsable --dependency=afterok:$j1 solve.sbatch)  # -N …,  mpi-sppy env,
                                                               #   srun … generic_cylinders.py
sbatch            --dependency=afterok:$j2 read.sbatch         # -n 1,  PyPSA env
```

**Directory hygiene (important).** mpi-sppy discovers scenarios by scanning the
transfer directory for model files — every `{s}.lp` / `{s}.mps` it finds becomes a
scenario (`mps_module.scenario_names_creator`).
A previous run that wrote **more** scenarios leaves stale `{s}.lp` (+
`_nonants.json` / `_rho.csv`) that a smaller new run would silently pick up as
**phantom scenarios** — wrong scenario set and probabilities. So **clear the
transfer directory before phase 1**: `write_stochastic_problem_mpisppy(clean=True)` (the
default) does this in Python, and the docs / `write.sbatch` should *also* `rm` the
stale files as belt-and-suspenders (covering a prior partial/aborted write).

Sizing: `--ntasks` divisible by `C`; ranks/cylinder ≤ S (§13.1). Job 2 uses
`srun` or `mpiexec`/`mpirun` per the site's MPI build. The shared `DIR` (and the
solution file, §13.5) must be visible to all three jobs (§13.4).

### 13.7 Recovering the second stage (dispatch and prices) — Phase 4

`--write-xhat-file` returns only the **first stage** (`*_nom`, plus `*-n_mod` →
`n_mod_opt`). The per-scenario **dispatch** (operational time series) and
**marginal prices** are not in that file. Two ways to recover them (a third,
emitting duals from mpi-sppy, is future work):

1. **Re-solve dispatch in PyPSA — IMPLEMENTED (`dispatch="resolve"`).** For each
   scenario, slice it
   (`n.get_scenario(s)`), fix capacities to the incumbent (`*_nom = *_nom_opt`,
   `*_nom_extendable = False` — which also drops the modular `n_mod`↔`*_nom`
   constraint), call `n.optimize()`, and merge the per-scenario results onto the
   stochastic network's `(scenario, name)` columns (the inverse of
   `get_scenario`'s `xs`, via `pd.concat(keys=…)`; copy `status == "Output"`
   series). Each subproblem is an independent LP (capacities are now parameters),
   so the work is **parallelizable** — but PyPSA's orchestration is **serial** (one
   model, one blocking solver call, no built-in scenario parallelism), so the loop
   runs serially unless a process pool is added or the re-solves are pushed out as
   separate scheduler jobs. This is the **only** path that yields marginal prices /
   duals and PyPSA's native statistics; for committable (unit-commitment) models it
   is also the only correct way to get prices (fix the binaries, re-solve the LP —
   which PyPSA already does natively). Cost: extra solves; a small chance of a
   different-but-equal-cost dispatch under degeneracy.

   **Scenario-conditional duals (DLW).** The duals from a per-scenario re-solve
   are *conditional on the realized scenario* — the marginal values in that
   scenario's world — which are typically what a decision maker wants, and are a
   distinct object from the extensive form's **probability-weighted** duals (the
   standalone dual equals the EF dual divided by the scenario probability, modulo
   dual degeneracy; verified on the `elec` bus of the test fixture, where
   EF/standalone = `prob_s` exactly). This scenario-conditional information is a
   substantive reason to re-solve rather than read the (dual-free) tree solution.
   Because the re-solve is inherently per-scenario, it naturally takes a **scenario
   subset** (`scenarios=`): re-solve — and obtain scenario-conditional duals — for
   only the scenarios of interest (e.g. the stressed ones), avoiding the serial
   cost of re-solving all of them. A user can equally hand-roll this in their own
   script (`n.get_scenario(s)` → `fix_optimal_capacities()` → `n.optimize()`).

2. **Read mpi-sppy's full tree solution (primal only) — NEXT PR.** The
   xhatter already solves every scenario subproblem to optimality — it fixes the
   nonants and solves (`Xhat_Eval.evaluate` → `_fix_nonants` → `solve_loop`) — so
   the complete primal solution exists in memory and mpi-sppy can write it:
   `generic_cylinders --solution-base-name BASE` writes `BASE_soldir/{stem}.csv`,
   one file per scenario, each row `var_name,value` for every variable
   (`sputils.scenario_tree_solution_writer`, selectable via the
   `module.tree_solution_writer` hook — no core fork). **Verified (2026-06-29,**
   real `mpiexec -np 3` + Gurobi PH on the test fixture**):** the CSV uses our
   `x{label}` names verbatim — all 20 variables (both stages) present, zero
   extraneous rows. This avoids the extra solves and gives exactly the incumbent
   dispatch, but carries **no duals/prices**.

   **Implementation note (why it is its own PR).** Mapping the tree solution back
   is more than a name lookup: the CSV holds raw *optimization variable* values,
   whereas PyPSA's dynamic outputs are *derived* by `assign_solution` with
   component-specific logic (`Line s` → `p0 = s, p1 = -s`; `Link p` →
   `p0 = p, p1 = -p·η` with delay shifts; `marginal_price` from **constraint**
   duals, not variables — confirming the tree read is dual-free). So the read path
   must, per scenario: build the model (`create_model`, no solve) → inject the tree
   values as linopy's `model.solution` → run `assign_solution()` → merge (reuse the
   option-1 merge). It saves the *solve* but not the model *build*. It also needs
   the manifest to store `model_kwargs` and the tree-solution directory, and
   `solve_stochastic_mpisppy` to add `--solution-base-name {dir}/sol` (supported by both the
   PH driver, `generic/decomp.py`, and the EF driver, `generic/ef.py`). A
   `_variable_records(model)` generalising `_nonant_records` supplies the
   `x{label} → (component, attr, asset, snapshot)` map (`'snapshot' in dims`
   distinguishes operational from first-stage variables).

3. **Have mpi-sppy emit duals (future work).** Possible in principle but reduces to
   a conditionally-valid price-recovery re-solve inside the xhatter, and duals are
   not defined for general (MIP/nonconvex) subproblems — see §14.

**Decision (mode API).** Expose **both** via
`dispatch="none" | "read" | "resolve"` on `read_stochastic_solution_mpisppy` /
`solve_stochastic_mpisppy` (default `"none"` = first stage only):

- `"resolve"` re-solves per scenario (1) — **implemented** — yielding primal
  dispatch **and scenario-conditional duals**, optionally restricted to a scenario
  subset (`scenarios=`). Use it when you want prices/duals (conditional on the
  scenario) or PyPSA-native statistics, or when only the xhat file is available.
- `"read"` parses mpi-sppy's tree solution (2) — the cheap, exact-incumbent,
  primal-only path; recommended when you just want dispatch. **Next PR** (see the
  implementation note in (2)); inline `solve_stochastic_mpisppy` will auto-add
  `--solution-base-name`. Until then it raises `NotImplementedError`.

Emitting duals from mpi-sppy itself remains future work (§14), not a near-term
dependency.

## 14. Open questions / future work

Resolved in Phase 1: **scenario slicing** — reuse the existing `n.get_scenario(s)`
(§12); **solution write-back flag** — mpi-sppy `--write-xhat-file` (§13.5);
**scenario file naming** — integer stems (§4.3). Resolved in Phase 3: the **inline
driver** `solve_stochastic_mpisppy` (§8, §12).

- **Dispatch write-back (Phase 4).** `read_stochastic_solution_mpisppy` sets the `*_nom`
  capacities and writes modular module counts back (`*-n_mod` → `n_mod_opt`).
  Second-stage recovery is exposed via the `dispatch=` mode (§13.7).
  `"resolve"` (per-scenario re-solve, primal **plus scenario-conditional duals**,
  with a `scenarios=` subset) is **implemented**. Decision makers are often
  specifically interested in duals *conditional on a scenario* (distinct from the
  EF's probability-weighted duals) and may want them for only a subset of
  scenarios — both reasons to provide the re-solve path, not just the tree read.
  `"read"` (mpi-sppy tree solution, primal-only) is the **next PR** — it needs
  `assign_solution` to translate raw variables into PyPSA outputs (§13.7 option 2),
  so it is a self-contained follow-up.
- **Duals / marginal prices from mpi-sppy (special opt-in — future work to
  consider).** mpi-sppy could be extended to return constraint duals from the
  xhatter's fixed-nonant subproblem solves, letting PyPSA reconstruct
  `buses_t.marginal_price` etc. without a re-solve. But mpi-sppy is
  **general-purpose** — scenario subproblems may be MIP, nonconvex, or solved by a
  backend with no dual interface — so **duals are not defined in general**; this
  could never be default behaviour and would need an **explicit opt-in option**
  valid only on the LP-subproblem subclass. It is an issue of the **xhatter's
  solve**, not the LP reader: no `dual` Suffix is declared on the scenario models,
  and the incumbent cache is **primal-only** (`update_best_solution_if_improving`
  stores a `ComponentMap` of `(var → value)`), so the winning solve's duals are
  discarded before write — you would re-solve at write time anyway, and a
  committable second stage would need a fix-binaries-then-LP re-solve. The
  near-term path (§13.7) keeps price recovery on the PyPSA side, where the model
  class is known and the machinery exists; revisit a special mpi-sppy dual option
  if a concrete need arises.
- **Multi-stage** investment horizons (PyPSA `investment_periods`).
- **CVaR / risk** interaction with decomposition (PyPSA already has CVaR in the
  monolithic EF; cf. mpi-sppy `doc/designs/cvar_design.md`).
- **File-interface performance (noted in the user docs).** The file interface is
  not fast (per-run file I/O + LP/MPS write & parse). For hard-to-solve problems
  this is negligible — wall time is dominated by the subproblem solves — and the
  **user docs state this caveat** (the Performance note in
  `stochastic-decomposition.md`). If a faster, in-memory interface is ever needed,
  implement PyPSA as an mpi-sppy *agnostic/guest* instead of the file boundary.

## 15. Key code references

**PyPSA** (`DLWoodruff/PyPSA` fork):
- `pypsa/optimization/optimize.py` — `create_model` (~L600), `define_objective`
  (~L139; capex on `*_nom` ~L312–333), `assign_solution` (~L901),
  `write_stochastic_problem_mpisppy` / `read_stochastic_solution_mpisppy` accessor methods (end of
  `OptimizationAccessor`).
- `pypsa/optimization/stochastic_mpisppy.py` — the exporter + read-back (implemented).
- `pypsa/network/index.py` — `set_scenarios` (~L719), `get_scenario` (~L933, the
  scenario slicer the exporter reuses), `scenarios` / `scenario_weightings`.
- `pypsa/descriptors.py` — `nominal_attrs` (default first-stage source, §5).
- `pypsa/optimization/variables.py` — variable definitions (scenario dim on
  operational vars; none on nominal).
- `test/test_stochastic_export.py` — dependency-free exporter/read-back tests.

**linopy** (0.8.0):
- `linopy/io.py` — `to_file` (~L655), `to_lp_file` (~L588), MPS via HiGHS
  (~L690–701), `get_printers_scalar` (~L99–135).

**mpi-sppy** (`Pyomo/mpi-sppy` main, post-#770):
- `mpisppy/problem_io/mps_module.py` — `_scenario_model_path`, `scenario_creator`
  (stashes `model._rho_csv_path`), `scenario_names_creator`, `_rho_setter`.
- `mpisppy/problem_io/mps_reader.py` — `read_mps_and_create_pyomo_model` (coin-or
  `mip`; reads `.lp` and `.mps`; returns the Pyomo model directly).
- `mpisppy/utils/sputils.py` — `write_nonant_tree_csv` (the `--write-xhat-file`
  format, §13.5); `scenario_tree_solution_writer` / `write_tree_solution` (the full
  per-scenario primal dump behind `--solution-base-name`, §13.7);
  `mpisppy/utils/config.py` — `write_xhat_file` arg (~L1275),
  `solution_base_name` arg.
- `mpisppy/utils/xhat_eval.py` — `Xhat_Eval.evaluate` → `_fix_nonants` →
  `solve_loop` (fixes nonants and fully solves each scenario subproblem; the primal
  source for §13.7 option 2 and the dual discussion in §14).
- `mpisppy/cylinders/spoke.py` / `mpisppy/spbase.py` —
  `update_best_solution_if_improving` (incumbent cache is primal-only, §14).
- `mpisppy/utils/rho_utils.py` — `rho_list_from_csv` (accepts `varname`/`fullname`).
- `mpisppy/generic/decomp.py` — `_get_rho_setter` (auto-discovers
  `module._rho_setter`); `_write_solutions` (`--solution-base-name` →
  `write_tree_solution`, `--write-xhat-file` → `write_tree_nonants`, §13.7).
- `mpisppy/generic/parsing.py` — maps `--mps-files-directory` → `mps_module`.
- `mpisppy/extensions/scenario_lp_mps_files.py` — reference writer (lp/mps/json/rho).
- `mpisppy/tests/examples/mps_module_data/` — `.lp` + json + rho test fixture (#770).
- `doc/src/agnostic.rst`, `doc/src/extensions.rst` — user docs for the file format.
