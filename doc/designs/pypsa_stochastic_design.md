# PyPSA Stochastic Programming via mpi-sppy (MPS-file interface) — Design

Status: draft for review. Spans two repos: **PyPSA** (work on the `DLWoodruff/PyPSA`
fork) and **mpi-sppy** (`main`). The mpi-sppy change is small and purely additive
(LP-file reading); all new modeling logic lives in PyPSA. No use of the mpi-sppy
*agnostic/guest* framework — the coupling is at the **file boundary**.

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
subproblem to a standard MPS file. This keeps PyPSA free of any Pyomo dependency
and keeps mpi-sppy free of any PyPSA dependency: the two programs meet only
through files. mpi-sppy needs essentially **no new code** beyond adding LP-file
reading alongside the existing MPS reader.

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
        │        write  {s}.mps            (model; implicit  x{label}  names)
        │        write  {s}_nonants.json   (probability + first-stage var names)
        │        write  {s}_rho.csv        (optional: per-nonant rho)
        │
        ├─ (2) SOLVE  — mpi-sppy, UNCHANGED except LP support:
        │        generic_cylinders.py --mps-files-directory=DIR
        │            --xhatshuffle --lagrangian --default-rho ...
        │        → converged first-stage capacities
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
| `{s}.mps` (or `{s}.lp`) | PyPSA (linopy) | mpi-sppy `mps_reader` | the scenario subproblem |
| `{s}_nonants.json` | PyPSA | mpi-sppy `mps_module` | probability + scenario tree + nonant names |
| `{s}_rho.csv` | PyPSA (optional) | mpi-sppy | per-nonant rho hints |

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

The `nonAnts` entries are variable names **exactly as they appear in the MPS
file**. The reader normalizes `(` and `)` to `_` when resolving them against the
Pyomo model (`mps_reader.py` ~L114; `mps_module.py` ~L67) — but it does **not**
normalize `#`. See the naming decision in §6.

The exact format is mirrored by mpi-sppy's own writer
`mpisppy/extensions/scenario_lp_mps_files.py` (~L24–87) and by the reference
artifacts in `mpi-sppy-1/_delme_test_write_mp_mps_dir/`
(`Scenario*.{mps,lp,json}`, `Scenario*_rho.csv`). PyPSA reproduces this format
directly rather than going through that extension.

## 5. Nonant (first-stage) identification

Default first stage = **all extendable capacity variables**: `Generator-p_nom`,
`Line-s_nom`, `Link-p_nom`, `Store-e_nom`, `StorageUnit-p_nom`, plus modular
counts `*-n_mod` where present. The user may override with an explicit list (to
restrict the first stage, or to add e.g. committable first-stage decisions
later).

For each first-stage `linopy` variable, PyPSA recovers its on-file names from the
integer `labels` (see §6) and writes them into `nonAnts`.

## 6. Variable naming — the enabling fact and the one risk

### 6.1 Determinism (verified)

`linopy` assigns variable *labels* strictly by **creation order**, independent of
any numeric data (objective coefficients, bounds, RHS). Empirically, two models
built with the same sequence of `add_variables` calls produce **identical**
on-file names even when the data differ. Therefore two structurally-identical
scenario networks emit MPS files whose variables line up by name — which is
exactly what mpi-sppy needs to enforce nonanticipativity across scenarios.

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
`find_component`. `x{label}` round-trips cleanly HiGHS → coin-or `mip` →
Pyomo. PyPSA maps a first-stage `linopy` variable to its `x{label}` names via:

```python
labels = n.model["Generator-p_nom"].labels.data.ravel()       # ints
names  = ["x" + str(int(L)) for L in labels if L >= 0]         # mask -1 (inactive)
```

(equivalently `linopy.io.get_printers_scalar(n.model, explicit_coordinate_names=False)`).

### 6.3 The risk: MPS naming is delegated to HiGHS

`linopy` writes **MPS** by delegating to `highspy.Highs.writeModel`
(`linopy/io.py` ~L690–701), so the on-file names are only as controllable as
HiGHS makes them. Exploration showed HiGHS *preserves* the names linopy supplies,
but this must be confirmed for implicit names before relying on it (Phase 0,
§12).

**Fallback / why LP support matters:** `linopy`'s **LP** writer (`to_lp_file`,
`io.py` ~L588) is linopy's own code and gives **full** control of the on-file
names. Adding LP reading to mpi-sppy therefore doubles as our naming-control
safety net, not merely a convenience.

## 7. Objective convention (no double counting)

`create_model()` for a single network builds
`objective = Σ capital_cost · (*_nom) + Σ marginal_cost · dispatch`
(`pypsa/optimization/optimize.py:define_objective`, ~L139; capex on `*_nom`
~L312–333). Each scenario's MPS therefore carries the **full** capex (on the
nonants) **plus that scenario's** opex. This is correct for PH/EF because

```
Σ_s p_s · (capex + opex_s) = capex + Σ_s p_s · opex_s = capex + E[opex]
```

since the scenario probabilities sum to 1. mpi-sppy applies the `scenProb`
weighting; capex is not double-counted.

## 8. Public API (PyPSA)

```python
n.optimize.solve_stochastic(
    method="ph",                 # "ph" (target) | "ef" (correctness oracle only)
    solver_name="gurobi",        # QP-capable: PH proximal term ⇒ QP subproblems
    first_stage=None,            # None = all extendable capacities; else explicit list
    cylinders=("lagrangian", "xhatshuffle"),
    mpisppy_options={"default_rho": 1.0, "max_iterations": 50, "convthresh": 1e-3},
    working_dir=None,            # temp dir if None
    file_format="mps",           # "mps" | "lp"
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

## 10. Changes required in mpi-sppy (additive only)

Add LP reading next to the MPS reader:

- `mpisppy/problem_io/mps_reader.py`: a `read_lp_to_mip_model(path)` paralleling
  `read_mps_to_mip_model` (~L72–80) — coin-or `mip.Model().read()` auto-detects
  `.lp`; and let `read_mps_and_create_pyomo_model` (~L158–168) dispatch on
  extension.
- `mpisppy/problem_io/mps_module.py:scenario_creator` (~L47–49): try `{s}.lp`
  when `{s}.mps` is absent.

The MPS path itself is unchanged and already works.

## 11. Correctness and testing

- **EF oracle:** solve via mpi-sppy EF over the written files and compare the
  objective to PyPSA's native EF (`n.optimize()` on the `set_scenarios`
  network). They must match to tolerance. (This is the only role of the EF path.)
- **PH convergence:** PH first-stage (capacities) must converge to the EF
  first-stage solution.
- **Fixture:** a tiny 2-bus, ~3-scenario, few-snapshot network — small enough to
  solve the EF directly and to inspect the MPS/JSON by eye.

## 12. Phased plan

- **Phase 0 — de-risk (do first).** `pip install mip` in the `pypsa-sppy` env;
  write a PyPSA single-network MPS (implicit names); round-trip it through
  `mpisppy.problem_io.mps_reader.read_mps_and_create_pyomo_model`; confirm the
  capacity nonants resolve via `find_component("x{label}")`. Repeat for LP. This
  validates the entire approach for minimal cost and decides MPS-vs-LP as the
  primary format.
- **Phase 1 — exporter.** scenario slicing + `write_mpisppy_files` (MPS +
  nonant JSON), validated against the `_delme_test_write_mp_mps_dir/` reference.
- **Phase 2 — mpi-sppy LP support** (§10).
- **Phase 3 — driver + write-back.** programmatic PH run; `assign_stochastic_solution`.
- **Phase 4 — public method, `pypsa[mpisppy]` extra, tests (§11), docs.**

## 13. Open questions / future work

- **Scenario slicing.** Carving a clean single-scenario `Network` out of a
  `set_scenarios` network (static `(scenario, name)` MultiIndex; dynamic
  `(scenario, name)` columns). If fiddly, the fallback is to accept a dict of
  pre-built per-scenario networks; the public method can support both inputs.
- **Rho setting.** Emit `{s}_rho.csv` from capital-cost magnitudes (cost-
  proportional rho) as a better default than a flat `default_rho`.
- **Multi-stage** investment horizons (PyPSA `investment_periods`).
- **Committable / integer first-stage** decisions (PH becomes heuristic).
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
