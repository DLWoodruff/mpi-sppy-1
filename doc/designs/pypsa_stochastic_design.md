# PyPSA Stochastic Programming via mpi-sppy (LP-file interface) — Design

Status: draft for review. Spans two repos: **PyPSA** (work on the `DLWoodruff/PyPSA`
fork) and **mpi-sppy**. The required mpi-sppy support (`.lp` files + per-nonant
rho from `{s}_rho.csv`) is **merged upstream** — Pyomo/mpi-sppy `main`, #770 — so
all remaining work is on the PyPSA side. No use of the mpi-sppy *agnostic/guest*
framework — the coupling is at the **file boundary**. Core feasibility has been
validated end to end (Phase 0, §12).

## 1. Goal

Let a PyPSA user solve a stochastic energy-system model by **decomposition**
(Progressive Hedging + bounding cylinders) using mpi-sppy, as an alternative to
PyPSA's existing *monolithic* Extensive Form (EF) solve.

The user-facing entry point is a new PyPSA method:

```python
n.set_scenarios({"low": 0.3, "med": 0.4, "high": 0.3})   # existing PyPSA UX
# ... set per-scenario data ...
n.optimize.solve_stochastic(method="ph", solver_name="gurobi", ...)   # NEW
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
its reader already parses LP (Phase 0, §10); only the file-name lookup needs a
small tweak.

## 3. Architecture

Three layers, all on the PyPSA side except the unchanged mpi-sppy solve:

```
n.set_scenarios({...})                       # EXISTING PyPSA UX
        │
        ▼
n.optimize.solve_stochastic(method="ph", solver_name="gurobi", ...)   # NEW
        │
        ├─ (1) EXPORT  — for each scenario s:
        │        slice scenario s → standalone single-scenario Network
        │        create_model()  → linopy.Model
        │        write  {s}.lp             (model; implicit  x{label}  names; MPS ok too)
        │        write  {s}_nonants.json   (probability + first-stage var names)
        │        write  {s}_rho.csv        (per-nonant cost-proportional rho)
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

## 5. Nonant (first-stage) identification

Default first stage = **all extendable capacity variables**: `Generator-p_nom`,
`Line-s_nom`, `Link-p_nom`, `Store-e_nom`, `StorageUnit-p_nom`, plus modular
counts `*-n_mod` where present. The user may override with an explicit list (to
restrict the first stage, or to add e.g. committable first-stage decisions
later).

For each first-stage `linopy` variable, PyPSA recovers its on-file names from the
integer `labels` (see §6) and writes them into `nonAnts`.

## 6. Variable naming — the enabling fact

### 6.1 Determinism (verified)

`linopy` assigns variable *labels* strictly by **creation order**, independent of
any numeric data (objective coefficients, bounds, RHS). Empirically, two models
built with the same sequence of `add_variables` calls produce **identical**
on-file names even when the data differ. Therefore two structurally-identical
scenario networks emit LP (or MPS) files whose variables line up by name — which
is exactly what mpi-sppy needs to enforce nonanticipativity across scenarios.

This is the load-bearing assumption of the whole approach. It holds **iff** all
scenarios are structurally identical (§9).

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

with `alpha` (default 1.0) and a small `rho_floor` so a zero-cost nonant still
gets a positive rho. PyPSA already holds every `capital_cost`, so it computes rho
directly and writes `{s}_rho.csv` (§4.2). The `rho` argument (§8) selects this
default, a flat scalar, or an explicit per-variable mapping.

Because rho must be **identical per nonant across scenarios** (§4.2 — mpi-sppy
does not reconcile), PyPSA computes the rho vector **once** (capital costs are
first-stage, hence scenario-invariant) and writes the same values to every
scenario's `_rho.csv`. If a use case ever varied a nonant's cost by scenario,
PyPSA would still emit a single agreed rho (e.g. the mean) to keep PH well-defined.

## 8. Public API (PyPSA)

The API exposes the **three phases** (§13.6) as methods on the existing
`OptimizationAccessor`. The write/read phases are **dependency-free** (no
mpi-sppy needed); only the inline `solve_stochastic` runs the driver.

```python
# Phase 1 — write the problem for mpi-sppy (1 rank, PyPSA env)
manifest = n.optimize.write_stochastic_problem(
    directory,
    clean=True,                  # clear stale scenario files in `directory` first (§13.6)
    first_stage=None,            # None = all extendable capacities; else explicit list
    rho="cost-proportional",     # "cost-proportional" | float | {varname: rho}  (§7.1)
    file_format="lp",            # "lp" | "mps"
    cylinders=("lagrangian", "xhatshuffle"),
    solver_name="gurobi",
    config_file=None,            # mpi-sppy --config-file to reference (§8.2)
    mpisppy_args=None,           # extra mpi-sppy CLI args (§8.2)
)
# manifest carries the exact phase-2 mpi-sppy command + an sbatch template.

# Phase 3 — read the incumbent back onto the network (1 rank, PyPSA env)
n.optimize.read_stochastic_solution(directory)   # sets *_nom_opt; optional dispatch re-solve

# Inline convenience (laptop / single node) = phase 1 + subprocess solve + phase 3
n.optimize.solve_stochastic(
    method="ph",                 # "ph" (target) | "ef" (correctness oracle only)
    solver_name="gurobi",        # QP-capable (PH proximal ⇒ QP; MIQP if integer first stage)
    first_stage=None, rho="cost-proportional",
    cylinders=("lagrangian", "xhatshuffle"),
    config_file=None, mpisppy_args=None, mpisppy_options=None,   # (§8.2)
    nprocs=None,                 # ranks for the inline mpiexec run
    working_dir=None, file_format="lp", keep_files=False,
)
```

Implemented in a new `pypsa/optimization/stochastic.py`:

- `write_stochastic_problem(...)` and `read_stochastic_solution(...)` — the two
  **dependency-free** phase functions (also bound as accessor methods); they
  touch files only and do **not** import mpi-sppy;
- `solve_stochastic(...)` = phase 1 + subprocess `mpiexec … generic_cylinders.py`
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
after the backend to match the `tsam` / `cartopy` / `gurobipy` convention (PyPI
names):

```toml
mpisppy = ["mpi-sppy", "mpi4py", "mip"]
```

**Gate + lazy import.** Reuse PyPSA's centralized helper
`check_optional_dependency(module_name, install_message)` (`pypsa/common.py`,
~L725) at the entry of `solve_stochastic`, then import inside the function.
**Import names differ from PyPI names**: `mpi-sppy` → `mpisppy`,
`python-mip` → `mip`.

```python
from pypsa.common import check_optional_dependency

def solve_stochastic(self, ...):
    check_optional_dependency(
        "mpisppy",
        "Missing optional dependencies for stochastic decomposition. "
        "Install via `pip install pypsa[mpisppy]`.",
    )
    import mpisppy            # noqa: PLC0415  (lazy)
    ...
```

This shared helper is preferred over `tsam`'s hand-rolled `find_spec` check for
consistency across the codebase. Only `solve_stochastic` gates on mpi-sppy;
`write_stochastic_problem` / `read_stochastic_solution` are file-only and import
cleanly without it (§13.6).

**Accessor placement.** `solve_stochastic` is a method on the **existing**
`OptimizationAccessor` (`pypsa/optimization/optimize.py`), so the public call is
`n.optimize.solve_stochastic(...)` — alongside `create_model` and
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
config (file + CLI) — consistent with the file-boundary philosophy.

## 9. Assumptions and constraints

1. **Structural identity.** All scenarios share the same components and the same
   snapshots; only data differ. This guarantees identical `linopy` labels and
   thus consistent nonant names (§6.1). The exporter must **assert** this — e.g.
   compare the first-stage label→name maps across scenarios and fail loudly on
   mismatch.
2. **Two-stage first.** capacity = stage 1, dispatch = stage 2. The JSON tree
   format and mpi-sppy support multi-stage, but v1 targets two-stage.
3. **QP-capable solver** for PH (the proximal term makes subproblems QPs); Gurobi
   is the default and is present in the dev env.
4. **Integer / committable first-stage is supported.** First-stage variables may
   be integer (e.g. modular `*-n_mod`, committable on/off). PH consensus on
   integers is heuristic, but mpi-sppy's bounding cylinders (Lagrangian lower
   bound + xhat upper bound) bracket the optimum and report a valid optimality
   gap. With the PH proximal term, integer first-stage makes the subproblems
   MIQPs — Gurobi handles these.

## 10. Required mpi-sppy support — MERGED (#770)

Both changes are implemented on **Pyomo/mpi-sppy `main`** (PR #770, *"mps_module:
accept .lp files and per-nonant rho from {s}_rho.csv"*); PyPSA needs an mpi-sppy
at or past that commit. No driver/reader/PH changes were required. See the
companion design `doc/designs/mps_module_lp_rho_design.md`.

1. **`.lp` files.** `mps_module._scenario_model_path` resolves `{s}.lp` / `{s}.mps`
   (`.lp` preferred when both exist); `scenario_creator` and
   `scenario_names_creator` use it (the latter globs both extensions and derives
   base names with `os.path.splitext`). The reader was already LP-capable
   (coin-or `mip` auto-detects).
2. **Per-nonant rho.** `mps_module._rho_setter` (auto-discovered by
   `generic/decomp.py:_get_rho_setter`) reads `{s}_rho.csv` whose path
   `scenario_creator` stashes as `model._rho_csv_path`; it returns `[]` when the
   file is absent (so `--default-rho` still applies). `rho_utils.rho_list_from_csv`
   now accepts a `varname` *or* `fullname` column and normalizes `()`→`_`.

**Constraint PyPSA must honor:** mpi-sppy applies rho **per scenario without
cross-scenario reconciliation**, so the writer must give a nonant the same rho in
every scenario (§4.2, §7.1). User-facing format docs live in mpi-sppy's
`doc/src/agnostic.rst` (scenario files + `_rho.csv`) and `doc/src/extensions.rst`.

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
- **Phase 1 — exporter.** scenario slicing + `write_stochastic_problem` (clears
  the directory, then writes LP + nonant JSON + cost-proportional `_rho.csv` + the
  phase-2 command/sbatch), validated against mpi-sppy's
  `mpisppy/tests/examples/mps_module_data/` fixture (#770: `Scenario*.lp`,
  `_nonants.json`, `_rho.csv`).
- **Phase 2 — mpi-sppy file-path support — DONE (#770, upstream main):** `.lp`
  filename lookup + `_rho.csv` consumption + tests (§10).
- **Phase 3 — driver + write-back.** subprocess/scheduler PH run;
  `read_stochastic_solution`.
- **Phase 4 — public methods, `pypsa[mpisppy]` extra, tests (§11), docs
  (incl. the SLURM workflow, §13.6).**

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
result (§6.1) plus structural identity (§9.1) guarantee this; the exporter must
assert it (same ordered name-list for all scenarios). This is *why* deterministic
naming is load-bearing rather than cosmetic.

### 13.4 Scaling considerations

- **Shared filesystem (multi-node).** `DIR` and the solution file must be visible
  to all nodes (shared FS), or staged per node. Single-node is a non-issue.
- **Startup I/O.** `C × S` reads total, distributed across ranks, one-time.
- **Parallelism ceiling.** ranks/cylinder ≤ S, so **max useful N ≈ C × S**.
  Scenario count — not rank count — sets the ceiling; to use many ranks you need
  many scenarios (or bundling, §14).
- **Solver licensing.** Each rank runs its own solver engine concurrently (QP, or
  MIQP with integer first stage, §9.4); at large N this needs a Gurobi
  floating/cluster license permitting that many simultaneous engines.
- **Memory.** ~one Pyomo scenario model per rank at full scale — scales well.

### 13.5 Result write-back

mpi-sppy writes the best incumbent's first-stage (nonant) values to a file (e.g.
`--solution-base-name`; exact flag to confirm), which PyPSA reads to set
`*_nom_opt` — the return half of the file boundary, consistent with §3. PyPSA may
optionally re-solve each scenario with capacities fixed for full dispatch.

### 13.6 Execution paths: inline vs decoupled (SLURM)

**Inline** (laptop / single interactive node): `solve_stochastic` writes the
files, runs `mpiexec -np N … generic_cylinders.py …` as a blocking subprocess,
and reads the solution back — one Python call.

**Decoupled** (HPC scheduler): the three phases (§8) are submitted as **separate
jobs**, because you cannot hold a process/allocation open while a large MPI job
queues and runs, the phases have very different resource profiles (1 core / many
cores / 1 core), and even different environments — **PyPSA** for write + read,
**mpi-sppy** for solve (§8). `write_stochastic_problem` emits the exact phase-2
command and an `sbatch` template; the user submits a dependency chain:

```bash
DIR=/shared/run42
rm -f "$DIR"/*.lp "$DIR"/*.mps "$DIR"/*_nonants.json "$DIR"/*_rho.csv  # hygiene, see below
j1=$(sbatch --parsable write.sbatch)                           # -n 1,  PyPSA env
j2=$(sbatch --parsable --dependency=afterok:$j1 solve.sbatch)  # -N …,  mpi-sppy env,
                                                               #   srun … generic_cylinders.py
sbatch            --dependency=afterok:$j2 read.sbatch         # -n 1,  PyPSA env
```

**Directory hygiene (important).** mpi-sppy discovers scenarios by *globbing* the
transfer directory (`mps_module.scenario_names_creator` globs `*.lp` / `*.mps`).
A previous run that wrote **more** scenarios leaves stale `{s}.lp` (+
`_nonants.json` / `_rho.csv`) that a smaller new run would silently pick up as
**phantom scenarios** — wrong scenario set and probabilities. So **clear the
transfer directory before phase 1**: `write_stochastic_problem(clean=True)` (the
default) does this in Python, and the docs / `write.sbatch` should *also* `rm` the
stale files as belt-and-suspenders (covering a prior partial/aborted write).

Sizing: `--ntasks` divisible by `C`; ranks/cylinder ≤ S (§13.1). Job 2 uses
`srun` or `mpiexec`/`mpirun` per the site's MPI build. The shared `DIR` (and the
solution file, §13.5) must be visible to all three jobs (§13.4).

## 14. Open questions / future work

- **Scenario slicing.** Carving a clean single-scenario `Network` out of a
  `set_scenarios` network (static `(scenario, name)` MultiIndex; dynamic
  `(scenario, name)` columns). If fiddly, the fallback is to accept a dict of
  pre-built per-scenario networks; the public method can support both inputs.
- **Multi-stage** investment horizons (PyPSA `investment_periods`).
- **CVaR / risk** interaction with decomposition (PyPSA already has CVaR in the
  monolithic EF; cf. mpi-sppy `doc/designs/cvar_design.md`).
- **Bundling for large S.** Whether the LP-file path (`mps_module`) composes with
  proper bundles (`--scenarios-per-bundle`) to right-size subproblems and cut
  comm when S is large; may be a separate `mps_module` item (§13.4).
- **Solution write-back flag.** Pin down the exact `generic_cylinders` mechanism
  that emits the incumbent first stage to a file for PyPSA to read (§13.5).

## 15. Key code references

**PyPSA** (`DLWoodruff/PyPSA` fork):
- `pypsa/optimization/optimize.py` — `create_model` (~L600), `define_objective`
  (~L139; capex on `*_nom` ~L312–333), `assign_solution` (~L901).
- `pypsa/network/index.py` — `set_scenarios` (~L785), `scenarios` /
  `scenario_weightings` (~L800–890).
- `pypsa/optimization/variables.py` — variable definitions (scenario dim on
  operational vars; none on nominal).
- (new) `pypsa/optimization/stochastic.py` — exporter + write-back.

**linopy** (0.8.0):
- `linopy/io.py` — `to_file` (~L655), `to_lp_file` (~L588), MPS via HiGHS
  (~L690–701), `get_printers_scalar` (~L99–135).

**mpi-sppy** (`Pyomo/mpi-sppy` main, post-#770):
- `mpisppy/problem_io/mps_module.py` — `_scenario_model_path`, `scenario_creator`
  (stashes `model._rho_csv_path`), `scenario_names_creator`, `_rho_setter`.
- `mpisppy/problem_io/mps_reader.py` — `read_mps_and_create_pyomo_model` (coin-or
  `mip`; reads `.lp` and `.mps`).
- `mpisppy/utils/rho_utils.py` — `rho_list_from_csv` (accepts `varname`/`fullname`).
- `mpisppy/generic/decomp.py` — `_get_rho_setter` (auto-discovers `module._rho_setter`).
- `mpisppy/generic/parsing.py` — maps `--mps-files-directory` → `mps_module`.
- `mpisppy/extensions/scenario_lp_mps_files.py` — reference writer (lp/mps/json/rho).
- `mpisppy/tests/examples/mps_module_data/` — `.lp` + json + rho test fixture (#770).
- `doc/src/agnostic.rst`, `doc/src/extensions.rst` — user docs for the file format.
