# Design Plan: ADMM in `generic_cylinders`

## Goal

Add `--admm` and `--stoch-admm` CLI flags to `generic_cylinders.py` so that
ADMM-based decomposition can be used with any compatible model module, without
requiring a bespoke `*_admm_cylinders.py` driver per problem.

## Prerequisite

The `mpisppy/generic/` refactor (see `refactor_generic_cylinders_plan.md`) should
be completed first. That refactor splits `generic_cylinders.py` into focused
modules and creates the `mpisppy/generic/admm.py` slot for this work.

## CLI Interface

```bash
# Deterministic ADMM (replaces distr_admm_cylinders.py pattern)
mpiexec -np 6 python -m mpi4py mpisppy/generic_cylinders.py \
    --module-name distr --admm --num-scens 3 \
    --default-rho 1.0 --max-iterations 100 --solver-name cplex \
    --lagrangian --xhatxbar

# Stochastic ADMM (replaces stoch_distr_admm_cylinders.py pattern)
mpiexec -np 6 python -m mpi4py mpisppy/generic_cylinders.py \
    --module-name stoch_distr --stoch-admm --num-admm-subproblems 3 \
    --num-stoch-scens 3 --default-rho 1.0 --max-iterations 100 \
    --solver-name cplex --lagrangian --xhatxbar
```

## Model Module Interface

### Standard (non-ADMM) — already required by `generic_cylinders`

- `scenario_creator(scenario_name, **kwargs)`
- `scenario_names_creator(num_scens, ...)`
- `scenario_denouement(rank, scenario_name, scenario)`
- `kw_creator(cfg)`
- `inparser_adder(cfg)`

### Additional for `--admm` (deterministic)

- `consensus_vars_creator(num_scens, ...)` → returns `consensus_vars` dict
  - Keys: subproblem names (= scenario names)
  - Values: list of consensus variable name strings

### Additional for `--stoch-admm`

- `consensus_vars_creator(admm_subproblem_names, stoch_scenario_name, **kwargs)` → `consensus_vars` dict
  - Keys: ADMM subproblem names
  - Values: list of `(var_name, stage)` tuples
- `admm_subproblem_names_creator(num_admm_subproblems)` → list of subproblem name strings
- `stoch_scenario_names_creator(num_stoch_scens)` → list of stochastic scenario name strings
- `admm_stoch_subproblem_scenario_names_creator(admm_subproblem_names, stoch_scenario_names)` → list of composite names
- `split_admm_stoch_subproblem_scenario_name(name)` → `(admm_subproblem_name, stoch_scenario_name)`

## New Config Args

Added when `--admm` or `--stoch-admm` is present:

| Arg | Domain | Description |
|---|---|---|
| `--admm` | bool | Enable deterministic ADMM decomposition |
| `--stoch-admm` | bool | Enable stochastic ADMM decomposition |
| `--num-admm-subproblems` | int | Number of ADMM subproblems (stoch-admm only) |
| `--num-stoch-scens` | int | Number of stochastic scenarios (stoch-admm only) |

## Implementation: `mpisppy/generic/admm.py`

```python
def admm_args(cfg):
    """Register ADMM-specific config args."""
    cfg.add_to_config("admm", description="Use ADMM decomposition",
                      domain=bool, default=False)
    cfg.add_to_config("stoch_admm", description="Use stochastic ADMM decomposition",
                      domain=bool, default=False)

def setup_admm(module, cfg, n_cylinders):
    """Create AdmmWrapper for deterministic ADMM.

    Modifies cfg by attaching variable_probability.
    Returns modified scenario_creator, scenario_creator_kwargs,
    all_scenario_names, all_nodenames.
    """
    all_scenario_names = module.scenario_names_creator(cfg.num_scens)
    scenario_creator_kwargs = module.kw_creator(cfg)
    consensus_vars = module.consensus_vars_creator(
        cfg.num_scens, **scenario_creator_kwargs  # or subset of kwargs
    )

    admm = AdmmWrapper(
        options={},
        all_scenario_names=all_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    cfg.quick_assign("variable_probability", object, admm.var_prob_list)

    return (admm.admmWrapper_scenario_creator, None,
            all_scenario_names, None)


def setup_stoch_admm(module, cfg, n_cylinders):
    """Create Stoch_AdmmWrapper for stochastic ADMM.

    Modifies cfg by attaching variable_probability.
    Returns modified scenario_creator, scenario_creator_kwargs,
    all_scenario_names, all_nodenames.
    """
    admm_subproblem_names = module.admm_subproblem_names_creator(cfg.num_admm_subproblems)
    stoch_scenario_names = module.stoch_scenario_names_creator(cfg.num_stoch_scens)
    all_names = module.admm_stoch_subproblem_scenario_names_creator(
        admm_subproblem_names, stoch_scenario_names)

    scenario_creator_kwargs = module.kw_creator(cfg)
    stoch_scenario_name = stoch_scenario_names[0]
    consensus_vars = module.consensus_vars_creator(
        admm_subproblem_names, stoch_scenario_name, **scenario_creator_kwargs)

    admm = Stoch_AdmmWrapper(
        options={},
        all_admm_stoch_subproblem_scenario_names=all_names,
        split_admm_stoch_subproblem_scenario_name=module.split_admm_stoch_subproblem_scenario_name,
        admm_subproblem_names=admm_subproblem_names,
        stoch_scenario_names=stoch_scenario_names,
        scenario_creator=module.scenario_creator,
        consensus_vars=consensus_vars,
        n_cylinders=n_cylinders,
        mpicomm=MPI.COMM_WORLD,
        scenario_creator_kwargs=scenario_creator_kwargs,
        BFs=None,  # could be extracted from cfg if multi-stage
    )

    cfg.quick_assign("variable_probability", object, admm.var_prob_list)

    return (admm.admmWrapper_scenario_creator, None,
            all_names, admm.all_nodenames)
```

## Integration with `do_decomp`

After the refactor, `do_decomp` in `mpisppy/generic/decomp.py` will read
`variable_probability` from cfg:

```python
variable_probability = cfg.get("variable_probability")  # None unless ADMM
```

And pass it through to `build_hub_dict()` and `build_spoke_list()`, which forward
it to `vanilla.ph_hub(...)`, `vanilla.xhatxbar_spoke(...)`, etc.

## Integration with `generic_cylinders.py` main block

In the slim `__main__` section:

```python
from mpisppy.generic.admm import admm_args, setup_admm, setup_stoch_admm
from mpisppy.utils.sputils import count_cylinders

# ... after parsing args ...

if cfg.admm or cfg.stoch_admm:
    n_cylinders = count_cylinders(cfg)
    if cfg.admm:
        scenario_creator, scenario_creator_kwargs, all_scenario_names, all_nodenames = \
            setup_admm(module, cfg, n_cylinders)
    else:
        scenario_creator, scenario_creator_kwargs, all_scenario_names, all_nodenames = \
            setup_stoch_admm(module, cfg, n_cylinders)
    # Skip normal scenario_creator setup; go straight to do_decomp
```

## Constraints and Limitations

- **FWPH spoke does not work with `variable_probability`** — should raise an error
  if both `--admm`/`--stoch-admm` and `--fwph` are enabled.
- **EF mode with ADMM** — the existing `distr_ef.py` pattern (EF of the wrapped
  scenarios) should work, but needs testing. Could be supported via `--EF --admm`.
- **Bundle support** — ADMM + proper bundles interaction is untested and likely
  unsupported initially. Should raise an error if both are specified.
- **`n_cylinders` must be computed before wrapper creation** — uses
  `sputils.count_cylinders(cfg)`.

## Testing Plan

1. **Unit test:** Verify `setup_admm` and `setup_stoch_admm` produce correct
   wrapper objects with known inputs (distr, stoch_distr models).
2. **Integration test:** Run existing distr/stoch_distr examples through
   `generic_cylinders.py --admm` and compare results to existing bespoke drivers.
3. **Error cases:** Test that `--admm --fwph`, `--admm --stoch-admm`,
   `--admm --scenarios-per-bundle` all raise clear errors.

## Migration Path for Existing Examples

Once working, the bespoke drivers can be simplified to thin wrappers or deprecated:

```python
# examples/distr/distr_admm_cylinders.py (simplified)
# Now just: mpiexec -np 6 python -m mpi4py mpisppy/generic_cylinders.py \
#   --module-name distr --admm --num-scens 3 ...
```
