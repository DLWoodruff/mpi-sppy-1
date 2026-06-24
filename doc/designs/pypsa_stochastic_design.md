# PyPSA Stochastic Programming via mpi-sppy (LP-file interface) — Design

Status: draft for review. Spans two repos: **PyPSA** (work on the `DLWoodruff/PyPSA`
fork) and **mpi-sppy** (`main`). The mpi-sppy change is small (accept `.lp`
filenames in mps_module; its reader already parses LP); all new modeling logic
lives in PyPSA. No use of the mpi-sppy
*agnostic/guest* framework — the coupling is at the **file boundary**. Core
feasibility has been validated end to end (Phase 0, §12).

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

PyPSA emits this by default with **cost-proportional rho** (§7.1). For two-stage
problems every scenario's rho file is identical. (The file path does not yet
*consume* `_rho.csv` — a small mpi-sppy wiring item, §10.)

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

## 8. Public API (PyPSA)

```python
n.optimize.solve_stochastic(
    method="ph",                 # "ph" (target) | "ef" (correctness oracle only)
    solver_name="gurobi",        # QP-capable (PH proximal ⇒ QP; MIQP if integer first stage)
    first_stage=None,            # None = all extendable capacities; else explicit list
    rho="cost-proportional",     # "cost-proportional" | float | {varname: rho}  (§7.1)
    cylinders=("lagrangian", "xhatshuffle"),
    config_file=None,            # path to an mpi-sppy --config-file (primary options input, §8.2)
    mpisppy_args=None,           # extra mpi-sppy CLI args, list[str] (§8.2)
    mpisppy_options=None,        # optional dict convenience → CLI flags (§8.2)
    working_dir=None,            # temp dir if None
    file_format="lp",            # "lp" | "mps"
    keep_files=False,
)
```

Thin sugar over two **dependency-free** functions in a new
`pypsa/optimization/stochastic.py`:

- `write_mpisppy_files(scenarios, directory, first_stage=..., file_format=...) -> manifest`
- `assign_stochastic_solution(n, solution) -> None`

`mpi-sppy`, `mpi4py`, and `mip` are imported **lazily** inside `solve_stochastic`
so that importing PyPSA never requires them. A new optional extra
`pypsa[mpisppy]` declares `{mpi-sppy, mpi4py, mip}`.

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
consistency across the codebase.

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

## 10. Changes required in mpi-sppy (small)

LP is the primary format (§6.3), so mpi-sppy must load `{s}.lp`. Phase 0 showed
the **reader already handles LP**: `read_mps_and_create_pyomo_model`
(`mps_reader.py` ~L158–168) calls coin-or `mip.Model().read()`, which
auto-detects `.lp` by extension — a `.lp` path round-trips with **no reader
change** (the Phase 0 LP test went through the unchanged reader).

Two small additions:

1. **`.lp` filename lookup** in
   `mpisppy/problem_io/mps_module.py:scenario_creator` (~L47–49), which currently
   hardcodes `sharedPath + ".mps"`. It should accept `.lp` (detect whichever of
   `{s}.lp` / `{s}.mps` exists, or take the extension from config). MPS stays
   supported.
2. **Consume `{s}_rho.csv`** on the file path. The extension *writes* it
   (`scenario_lp_mps_files.py` ~L84–87) but `mps_module`/the driver do not yet
   *read* it; the file path should apply the per-nonant rho (directly, or via an
   existing rho_setter / config hook). Until wired, rho falls back to
   `--default-rho`.

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
- **Phase 1 — exporter.** scenario slicing + `write_mpisppy_files` (LP +
  nonant JSON + cost-proportional `_rho.csv`), validated against the
  `_delme_test_write_mp_mps_dir/` reference.
- **Phase 2 — mpi-sppy file-path tweaks** (§10): `.lp` filename lookup + consume
  `_rho.csv`.
- **Phase 3 — driver + write-back.** programmatic PH run; `assign_stochastic_solution`.
- **Phase 4 — public method, `pypsa[mpisppy]` extra, tests (§11), docs.**

## 13. Open questions / future work

- **Scenario slicing.** Carving a clean single-scenario `Network` out of a
  `set_scenarios` network (static `(scenario, name)` MultiIndex; dynamic
  `(scenario, name)` columns). If fiddly, the fallback is to accept a dict of
  pre-built per-scenario networks; the public method can support both inputs.
- **Multi-stage** investment horizons (PyPSA `investment_periods`).
- **CVaR / risk** interaction with decomposition (PyPSA already has CVaR in the
  monolithic EF; cf. mpi-sppy `doc/designs/cvar_design.md`).

## 14. Key code references

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

**mpi-sppy** (`main`):
- `mpisppy/problem_io/mps_module.py` — `scenario_creator` (~L38–87).
- `mpisppy/problem_io/mps_reader.py` — `read_mps_to_mip_model` (~L72–80),
  `mip_model_to_pyomo` (~L83–155), `read_mps_and_create_pyomo_model` (~L158–168).
- `mpisppy/extensions/scenario_lp_mps_files.py` — reference writer (~L24–87).
- `mpisppy/generic_cylinders.py` — `--mps-files-directory` driver.
- `mpi-sppy-1/_delme_test_write_mp_mps_dir/` — reference artifacts.
