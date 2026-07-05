###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# General-purpose bootstrap code for data-based, two-stage stochastic programs.
# These are the empirical methods (classical, extended, subsampling, bagging);
# the smoothed methods arrive in a follow-on merge.

import os
from statistics import NormalDist
import numpy as np
from numpy.random import default_rng
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
from mpisppy.confidence_intervals.bootsp.batch_executor import BatchExecutor

# The communicators live in boot_utils so there is a single source of truth.
# These module globals are the K = 1 view (each rank its own batch worker) and
# are still used by the smoothed methods and the simulation prep, which are
# K = 1 only. The empirical estimators below instead route their parallelism
# through a BatchExecutor so they generalize to K > 1 (a wheel per batch); at
# K = 1 the executor reproduces exactly what these globals describe.
comm = boot_utils.comm
n_proc = boot_utils.n_proc
my_rank = boot_utils.my_rank
rankcomm = boot_utils.rankcomm


def _name_of_position_fn(cfg):
    """Return a function mapping a dataset position to a scenario name.

    By default a position ``p`` maps to the historical name ``Scenario{p}``,
    so ``user_boot`` / ``simulate_boot`` are unchanged. generic_cylinders's
    ``do_boot`` installs a resolver (``cfg.boot_name_of_position``) that maps a
    position to the canonical scenario name for that record, so the integrated
    path addresses records by list position rather than a scraped integer.
    """
    resolver = cfg.get("boot_name_of_position", None)
    if resolver is not None:
        return resolver
    return lambda p: "Scenario" + str(p)


def _best_bound(ef, results):
    """The solver's outer bound on the EF objective.

    Lower bound for a minimization, upper bound for a maximization; falls back
    to the incumbent objective value when the solver reports no bound.
    """
    bound = None
    try:
        prob = results.problem[0]
        bound = prob.lower_bound if ef.EF_Obj.sense == pyo.minimize \
            else prob.upper_bound
    except (AttributeError, IndexError, KeyError, TypeError):
        bound = None
    if bound is None:
        return pyo.value(ef.EF_Obj)
    return float(bound)


def _ef_optimal_value(ef):
    """The bootstrap "optimal" of a solved batch EF: the solver's outer bound
    (stashed on the ef by solve_routine), not the incumbent objective.

    A mixed-integer EF is solved only to a MIP gap, so its incumbent
    (``pyo.value(EF_Obj)``) is an inner bound on the batch optimal; using it
    would make the optimality gap (value at xhat minus the optimal) read
    optimistically. The solver's best bound is the correct, conservative choice
    and unifies with the cylinders batch executor, whose decomposition bound
    plays the same role.
    """
    val = getattr(ef, "_mpisppy_boot_optimal", None)
    return pyo.value(ef.EF_Obj) if val is None else val


def _sample_names_and_mapping(cfg, scenarios, duplication):
    """Build the (distinct) sample scenario names and their position mapping.

    A resample can select the same record more than once, but an mpi-sppy
    extensive form needs *distinct* scenario names, so with ``duplication`` we
    mint fresh ``SampleScenario{i}`` names and map each back to its record's
    canonical name (via the positional resolver). Without duplication the
    scenarios are distinct positions and can be named directly. Shared by the
    optimal solve and the xhat evaluation so the two always agree.
    """
    name_of_pos = _name_of_position_fn(cfg)
    if duplication:
        names = ['SampleScenario' + str(i) for i in range(len(scenarios))]
        mapping = {names[i]: name_of_pos(scenarios[i]) for i in range(len(scenarios))}
    else:
        names = [name_of_pos(s) for s in scenarios]
        mapping = None
    return names, mapping


def _batch_optimal_value(cfg, module, scenarios, executor, duplication):
    """The batch optimal (outer bound) ``L_b`` for one resampled batch.

    For K = 1 this is a direct extensive form solved to its best (outer) bound
    (``_ef_optimal_value``). For K > 1 the batch is solved by a wheel on the
    group's communicator via ``executor.batch_optimal_solver``, which returns
    the wheel's decomposition (outer) bound. Both play the same role in the
    optimality-gap estimators (design section 9.4.1), so the estimator code is
    identical across K.
    """
    if executor.uses_cylinders:
        names, mapping = _sample_names_and_mapping(cfg, scenarios, duplication)
        return executor.batch_optimal_solver(names, mapping, executor.groupcomm)
    ef = solve_routine(cfg, module, scenarios, num_threads=2, duplication=duplication)
    return _ef_optimal_value(ef)


def _scenario_creator_w_mapping(scenario_name, module=None, mapping=None, **kwargs):
    """ A wrapper to allow for bootstrap samples to map to actual samples
    Args:
        scenario_name (str): the scenario number will be peeled off the end
        module (Python module): contains the scenario creator function and helpers
        mapping (dict): maps the scenario_name argument to a scenario sent to
                        the module scenario creator
        kwargs (dict): arguments for the module scenario creator
    Returns:
        model (Pyomo ConcreteModel): the instantiated scenarios

    Note: w is *not* the PH W, it is for resampling
    """
    if mapping is not None:
        return module.scenario_creator(mapping[scenario_name], **kwargs)
    else:
        return module.scenario_creator(scenario_name, **kwargs)


def slice_lens(nB):
    """ compute the share of nB for every MPI rank
    Args:
        nB (int): number of batches
    Returns:
        slice_lens (list): an allocation of nB to n_proc (a global) slices
    """

    avg = nB / n_proc
    slice_lens = [int((i + 1) * avg) - int(i * avg) for i in range(n_proc)]
    # we don't really need this assert, but it is harmless
    assert sum(slice_lens) == nB

    return slice_lens


def process_optimal(cfg, module):
    """ For simulations we need a known or assumed z*
        Args:
            cfg (Config): parameters
            module (Python module): contains the scenario creator function and helpers
        Returns:
            opt_obj (float): z*
            opt_gap (float): gap if provided by solver
    """

    if cfg.optimal_fname is not None and cfg.optimal_fname != "None":
        if not os.path.exists(cfg.optimal_fname):
            raise ValueError(f"File {cfg.optimal_fname} does not exist.\n"
                             "Maybe you need to run bootsp.boot_general_prep")
        print(f"Reading pre-computed optimal value from {cfg.optimal_fname}", flush=True)
        tmp = np.load(cfg["optimal_fname"], 'r')
        opt_obj = tmp[0]
        opt_gap = tmp[1]
        print(f"   ...optimal value: {opt_obj}")
        print(f"   ...optimality gap: {opt_gap}")
    else:
        print('No calculated optimal found, starting computing the "actual" optimal')
        print("Computing optimal function value on Rank 0 only")
        opt_ef = solve_routine(cfg, module, range(cfg.max_count), num_threads=2)
        opt_obj = pyo.value(opt_ef.EF_Obj)
        print(f"optimal EF objective: {opt_obj}; using zero gap (this should be verified visually)")
        opt_gap = 0
    return opt_obj, opt_gap


def solve_routine(cfg, module, scenarios, num_threads=None, duplication=False):
    """
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenarios (iterable; e.g., list): scenario numbers
        num_threads (int): number of solver threads
        duplication (bool): sample with duplication
    Returns:
        ef (EF): the full extensive form from mpi-sppy
    """

    tee_rank0_solves = False
    scenario_creator = _scenario_creator_w_mapping
    scenario_creator_kwargs = module.kw_creator(cfg)  # we get a new one every time...
    scenario_creator_kwargs['module'] = module  # we are going to call a wrapper

    scenario_names, scenario_creator_kwargs['mapping'] = \
        _sample_names_and_mapping(cfg, scenarios, duplication)

    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(cfg.solver_name)
    solver.options["threads"] = num_threads
    # optional batch-solver options (e.g. from generic_cylinders --boot-solver-
    # options); absent for the standalone drivers, so this is a no-op there
    solver_options = cfg.get("solver_options", None)
    if solver_options:
        for k, v in solver_options.items():
            solver.options[k] = v
    teeme = tee_rank0_solves if my_rank == 0 else False
    if 'persistent' in cfg.solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        results = solver.solve(tee=teeme)
    else:
        results = solver.solve(ef, tee=teeme, symbolic_solver_labels=True)

    # Stash the solver's best (outer) bound; this is the principled "optimal"
    # for the bootstrap gap (see _ef_optimal_value).
    ef._mpisppy_boot_optimal = _best_bound(ef, results)

    return ef


def evaluate_routine(cfg, module, xhat, scenario_names, sample_mapping, mpicomm=None):
    """ evaluate a given xhat over given scenario names

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        scenario_names (list of str): the scenario number will be peeled off the ends
        sample_mapping (dict): If not None, maps the scenario_name argument to a scenario sent to
            the module scenario creator
        mpicomm (MPI comm): the communicator to spread the evaluation scenarios
            over; defaults to the private single-rank rankcomm (the K = 1 view).
            The empirical estimators pass the batch's group communicator so an
            xhat evaluation is spread across the group's ranks (design 9.4).

    Returns:
        zhat (float): the computed expected value
    """
    if mpicomm is None:
        mpicomm = rankcomm
    # optional batch-solver options (see solve_routine); None for the standalone
    # drivers, so the xhat-evaluation solves are unchanged there
    solver_options = cfg.get("solver_options", None)
    xhat_eval_options = {"iter0_solver_options": solver_options,
                         "iterk_solver_options": solver_options,
                         "display_timing": False,
                         "solver_name": cfg.solver_name,
                         "verbose": False,
                         "toc": False
                         }

    scenario_creator = _scenario_creator_w_mapping
    scenario_creator_kwargs = module.kw_creator(cfg)  # we get a new one every time...
    scenario_creator_kwargs['module'] = module  # we are going to call a wrapper
    scenario_creator_kwargs["mapping"] = sample_mapping

    ev = xhat_eval.Xhat_Eval(xhat_eval_options,
                scenario_names,
                scenario_creator,
                mpicomm=mpicomm,
                scenario_creator_kwargs=scenario_creator_kwargs
                )

    zhat = ev.evaluate(xhat)

    return zhat


def evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True, mpicomm=None):
    """ evaluate xhat using a list of (sampled) scenario numbers

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenarios (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        duplication (bool): indicates scenarios may be duplicated in the scenarios list

    Returns:
        zhat (float): the computed expectation

    """
    # Take a list of indices of the original scenarios with possible replications
    # If need mapping, create a set of scenario names and a mapping function that maps the scenario names to the original ones
    # Return the function value evaluated for a given xhat

    scenario_names, sample_mapping = _sample_names_and_mapping(cfg, scenarios, duplication)

    return evaluate_routine(cfg, module, xhat, scenario_names, sample_mapping, mpicomm=mpicomm)


def _bootstrap_resample(cfg, module, scenario_pool, xhat, executor):
    """ Get gaps and optimal values for classic bootstrap.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenario_pool (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        executor (BatchExecutor): the rank grouping (design 9.4)
    Returns:
        numpy arrays (vector) with this group's gaps, optimal values, and uppers

    """
    # loop over this group's share of the batches; all ranks of a group share
    # the seed (so they resample the same batch) and cooperate on the solve.

    rng = default_rng(executor.group_seed(cfg.seed_offset))
    local_nB = executor.batch_share(cfg.nB)
    local_boot_gaps = np.empty(local_nB, dtype=np.float64)
    local_boot_optimals = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers = np.empty(local_nB, dtype=np.float64)
    for iter in range(local_nB):
        scenarios = rng.choice(scenario_pool, size=cfg.sample_size, replace=True)
        boot_ev = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True,
                                     mpicomm=executor.groupcomm)
        local_boot_optimals[iter] = _batch_optimal_value(cfg, module, scenarios, executor, duplication=True)
        local_boot_uppers[iter] = boot_ev
        local_boot_gaps[iter] = local_boot_uppers[iter] - local_boot_optimals[iter]

    return local_boot_gaps, local_boot_optimals, local_boot_uppers


def classical_bootstrap(cfg, module, xhat, quantile=True, executor=None):
    """ perform a classic bootstrap estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        quantile (bool): use the quantile method (else the gaussian method)
        executor (BatchExecutor or None): the rank grouping; None => K = 1

    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    if executor is None:
        executor = BatchExecutor(1)
    rng = default_rng(cfg.seed_offset)

    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False)
    dag_upper = evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False,
                                   mpicomm=executor.groupcomm)
    dag_optimal = _batch_optimal_value(cfg, module, scenario_pool, executor, duplication=False)
    dag_gap = dag_upper - dag_optimal  # this is gamma(D) in the note

    # tron is a "secret" way to turn on internal trace information
    if cfg.get("tron", False):
        print(f"rank {my_rank} at dag barrier", flush=True)
    executor.comm.Barrier()

    # bootstrap from pool
    local_boot_gaps, local_boot_optimals, local_boot_uppers = _bootstrap_resample(cfg, module, scenario_pool, xhat, executor)

    executor.comm.Barrier()

    # gather every group's batches to global rank 0 for analysis
    boot_gaps = executor.gather(local_boot_gaps, cfg.nB)
    boot_optimals = executor.gather(local_boot_optimals, cfg.nB)
    boot_uppers = executor.gather(local_boot_uppers, cfg.nB)
    if cfg.get("tron", False) and executor.is_root:
        print("*** rank 0 ends gather", flush=True)

    if executor.is_root:
        if quantile:
            alpha = cfg.alpha / 2
            ci_optimal = np.quantile(2 * dag_optimal - boot_optimals, [alpha, 1 - alpha])
            ci_upper = np.quantile(2 * dag_upper - boot_uppers, [alpha, 1 - alpha])
            ci_gap = np.quantile(2 * dag_gap - boot_gaps, [alpha, 1 - alpha])
        else:
            dd = NormalDist().inv_cdf(1 - cfg.alpha / 2)
            std_optimal = np.std(boot_optimals, ddof=1)
            std_gap = np.std(boot_gaps, ddof=1)
            std_upper = np.std(boot_uppers, ddof=1)

            ci_optimal = [dag_optimal - dd * std_optimal, dag_optimal + dd * std_optimal]
            ci_upper = [dag_upper - dd * std_upper, dag_upper + dd * std_upper]
            ci_gap = [dag_gap - dd * std_gap, dag_gap + dd * std_gap]

        return ci_optimal, ci_upper, ci_gap, dag_optimal, dag_upper, dag_gap
    else:
        return None, None, None, None, None, None


def _sub_resample(cfg, module, scenario_pool, xhat, executor):
    """ Get gaps and optimal values for subsampling method.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenario_pool (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        executor (BatchExecutor): the rank grouping (design 9.4)
    Returns:
        numpy arrays (vector) with this group's gaps, optimal values, and uppers

    """
    # loop over this group's share of the batches
    # only difference from _bootstrap_resample is the batch size: subsample_size here, sample_size there

    rng = default_rng(executor.group_seed(cfg.seed_offset))
    local_nB = executor.batch_share(cfg.nB)
    local_boot_gaps = np.empty(local_nB, dtype=np.float64)
    local_boot_optimals = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers = np.empty(local_nB, dtype=np.float64)
    for iter in range(local_nB):
        scenarios = rng.choice(scenario_pool, size=cfg.subsample_size, replace=False)
        boot_ev = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True,
                                     mpicomm=executor.groupcomm)
        local_boot_optimals[iter] = _batch_optimal_value(cfg, module, scenarios, executor, duplication=True)
        local_boot_uppers[iter] = boot_ev
        local_boot_gaps[iter] = local_boot_uppers[iter] - local_boot_optimals[iter]

    return local_boot_gaps, local_boot_optimals, local_boot_uppers


def subsampling(cfg, module, xhat, executor=None):
    """ perform a subsampling estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        executor (BatchExecutor or None): the rank grouping; None => K = 1

    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    if executor is None:
        executor = BatchExecutor(1)
    rng = default_rng(cfg.seed_offset)

    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False)
    dag_upper = evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=False,
                                   mpicomm=executor.groupcomm)
    dag_optimal = _batch_optimal_value(cfg, module, scenario_pool, executor, duplication=False)
    dag_gap = dag_upper - dag_optimal  # this is gamma(D) in the note

    executor.comm.Barrier()

    # subsampling from pool
    local_boot_gaps, local_boot_optimals, local_boot_uppers = _sub_resample(cfg, module, scenario_pool, xhat, executor)
    executor.comm.Barrier()

    # gather every group's batches to global rank 0 for analysis
    boot_gaps = executor.gather(local_boot_gaps, cfg.nB)
    boot_optimals = executor.gather(local_boot_optimals, cfg.nB)
    boot_uppers = executor.gather(local_boot_uppers, cfg.nB)

    if executor.is_root:
        alpha = cfg.alpha / 2
        err_optimal = np.sqrt(cfg.subsample_size / cfg.sample_size) * np.quantile(boot_optimals - dag_optimal, [1 - alpha, alpha])
        ci_optimal = dag_optimal - err_optimal

        err_upper = np.sqrt(cfg.subsample_size / cfg.sample_size) * np.quantile(boot_uppers - dag_upper, [1 - alpha, alpha])
        ci_upper = dag_upper - err_upper

        err_gap = np.sqrt(cfg.subsample_size / cfg.sample_size) * np.quantile(boot_gaps - dag_gap, [1 - alpha, alpha])
        ci_gap = dag_gap - err_gap

        return ci_optimal, ci_upper, ci_gap, dag_optimal, dag_upper, dag_gap
    else:
        return None, None, None, None, None, None


def _extended_resample(cfg, module, xhat, executor):
    """ Get gaps and optimal values differences for extended bootstrap.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        executor (BatchExecutor): the rank grouping (design 9.4)
    Returns:
        numpy arrays (vector) with this group's gap / optimal / upper differences

    """
    # loop over this group's share of the batches

    rng = default_rng(executor.group_seed(cfg.seed_offset) + 1)
    local_nB = executor.batch_share(cfg.nB)

    local_boot_optimals_diff = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers_diff = np.empty(local_nB, dtype=np.float64)
    local_boot_gaps_diff = np.empty(local_nB, dtype=np.float64)

    for iter in range(local_nB):
        scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=True)
        dag_optimal = _batch_optimal_value(cfg, module, scenario_pool, executor, duplication=True)
        dag_upper = evaluate_scenarios(cfg, module, scenario_pool, xhat, duplication=True,
                                       mpicomm=executor.groupcomm)

        scenarios = rng.choice(scenario_pool, size=cfg.sample_size, replace=True)
        boot_optimal = _batch_optimal_value(cfg, module, scenarios, executor, duplication=True)
        boot_upper = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True,
                                        mpicomm=executor.groupcomm)

        local_boot_optimals_diff[iter] = boot_optimal - dag_optimal
        local_boot_uppers_diff[iter] = boot_upper - dag_upper

        local_boot_gaps_diff[iter] = local_boot_uppers_diff[iter] - local_boot_optimals_diff[iter]

    return local_boot_gaps_diff, local_boot_optimals_diff, local_boot_uppers_diff


def extended_bootstrap(cfg, module, xhat, executor=None):
    """ perform an extended bootstrap estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        executor (BatchExecutor or None): the rank grouping; None => K = 1

    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    if executor is None:
        executor = BatchExecutor(1)
    rng = default_rng(cfg.seed_offset)

    # extended bootstrap
    local_boot_gaps_diff, local_boot_optimals_diff, local_boot_uppers_diff = _extended_resample(cfg, module, xhat, executor)
    executor.comm.Barrier()

    # gather every group's batch differences to global rank 0 for analysis
    boot_gaps_diff = executor.gather(local_boot_gaps_diff, cfg.nB)
    boot_optimals_diff = executor.gather(local_boot_optimals_diff, cfg.nB)
    boot_uppers_diff = executor.gather(local_boot_uppers_diff, cfg.nB)

    # The center is a fresh set of solves; unlike the per-batch differences it
    # is computed once, on group 0 (all of whose ranks must cooperate on the
    # solves when K > 1). At K = 1 group 0 is exactly global rank 0, so this is
    # the original "on rank 0" behavior. The final CI needs the gathered
    # differences, which land only on global rank 0 (is_root, itself in group 0).
    if executor.group_index == 0:

        # get center
        scenarios = rng.choice(cfg.max_count, size=cfg.sample_size, replace=True)
        dag_optimal = _batch_optimal_value(cfg, module, scenarios, executor, duplication=True)
        dag_upper = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=True,
                                       mpicomm=executor.groupcomm)

        scenarios_ = rng.choice(cfg.max_count, size=cfg.sample_size, replace=True)
        scenarios_combined = np.concatenate([scenarios, scenarios_])

        dag_optimal_combined = _batch_optimal_value(cfg, module, scenarios_combined, executor, duplication=True)
        dag_upper_combined = evaluate_scenarios(cfg, module, scenarios_combined, xhat, duplication=True,
                                                mpicomm=executor.groupcomm)

        center_optimal = 2 * dag_optimal_combined - dag_optimal
        center_upper = 2 * dag_upper_combined - dag_upper
        center_gap = center_upper - center_optimal

        if executor.is_root:
            alpha = cfg.alpha / 2
            ci_optimal = center_optimal - np.quantile(boot_optimals_diff, [1 - alpha, alpha])
            ci_upper = center_upper - np.quantile(boot_uppers_diff, [1 - alpha, alpha])
            ci_gap = center_gap - np.quantile(boot_gaps_diff, [1 - alpha, alpha])

            return ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap

    return None, None, None, None, None, None


def _bagging_resample(cfg, module, scenario_pool, xhat, executor, replacement=True):
    """ Get gaps and optimal values differences for bagging.
    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        scenario_pool (iterable; e.g., list): scenario numbers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        executor (BatchExecutor): the rank grouping (design 9.4)
        replacement (bool): sample the subsample with replacement
    Returns:
        numpy arrays (vector) with this group's gaps, optimal values, uppers, and boot counts

    """
    # loop over this group's share of the batches

    rng = default_rng(executor.group_seed(cfg.seed_offset))
    local_nB = executor.batch_share(cfg.nB)
    local_boot_gaps = np.empty(local_nB, dtype=np.float64)
    local_boot_optimals = np.empty(local_nB, dtype=np.float64)
    local_boot_uppers = np.empty(local_nB, dtype=np.float64)
    local_boot_counts = np.zeros((local_nB, cfg.sample_size))
    for iter in range(local_nB):
        scenarios_index = rng.choice(len(scenario_pool), size=cfg.subsample_size, replace=replacement)
        scenarios = [scenario_pool[index] for index in scenarios_index]
        boot_ev = evaluate_scenarios(cfg, module, scenarios, xhat, duplication=replacement,
                                     mpicomm=executor.groupcomm)
        local_boot_optimals[iter] = _batch_optimal_value(cfg, module, scenarios, executor, duplication=replacement)
        local_boot_uppers[iter] = boot_ev
        local_boot_gaps[iter] = local_boot_uppers[iter] - local_boot_optimals[iter]

        for index in scenarios_index:
            local_boot_counts[iter, index] += 1

    local_boot_counts = np.reshape(local_boot_counts, local_nB * cfg.sample_size)

    return local_boot_gaps, local_boot_optimals, local_boot_uppers, local_boot_counts


def bagging_bootstrap(cfg, module, xhat, replacement=True, executor=None):
    """ perform a bagging-based estimation of confidence intervals

    Args:
        cfg (Config): parameters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
            (i.e. the specification of a candidate solution)
        replacement (bool): sample the subsample with replacement
        executor (BatchExecutor or None): the rank grouping; None => K = 1

    Returns:
        tuple with confidence interval if on MPI rank 0
    """
    if executor is None:
        executor = BatchExecutor(1)
    rng = default_rng(cfg.seed_offset)
    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False)

    # bootstrap from pool
    local_boot_gaps, local_boot_optimals, local_boot_uppers, local_boot_counts = _bagging_resample(cfg, module, scenario_pool, xhat, executor, replacement=replacement)
    executor.comm.Barrier()

    # gather every group's batches (counts are sample_size floats per batch) to
    # global rank 0 for analysis
    boot_gaps = executor.gather(local_boot_gaps, cfg.nB)
    boot_optimals = executor.gather(local_boot_optimals, cfg.nB)
    boot_uppers = executor.gather(local_boot_uppers, cfg.nB)
    boot_counts = executor.gather(local_boot_counts, cfg.nB, item_len=cfg.sample_size)

    if executor.is_root:
        center_gap = np.mean(boot_gaps)
        center_optimal = np.mean(boot_optimals)
        center_upper = np.mean(boot_uppers)
        boot_counts = np.reshape(boot_counts, (cfg.nB, cfg.sample_size))

        cov_gap = np.matmul(boot_gaps - center_gap, boot_counts - cfg.subsample_size / cfg.sample_size) / cfg.nB
        cov_gap = np.linalg.norm(cov_gap)

        cov_optimal = np.matmul(boot_optimals - center_optimal, boot_counts - cfg.subsample_size / cfg.sample_size) / cfg.nB
        cov_optimal = np.linalg.norm(cov_optimal)

        cov_upper = np.matmul(boot_uppers - center_upper, boot_counts - cfg.subsample_size / cfg.sample_size) / cfg.nB
        cov_upper = np.linalg.norm(cov_upper)

        if not replacement:
            cov_gap *= cfg.sample_size / (cfg.sample_size - cfg.subsample_size)
            cov_optimal *= cfg.sample_size / (cfg.sample_size - cfg.subsample_size)
            cov_upper *= cfg.sample_size / (cfg.sample_size - cfg.subsample_size)

        if cfg.get("tron", False) and my_rank == 0:
            print(f"cov_gap:{cov_gap}")
            print(f"cov_optimal: {cov_optimal}")
            print(f"cov_upper: {cov_upper}")

        dd = NormalDist().inv_cdf(1 - cfg.alpha / 2)
        ci_optimal = [center_optimal - dd * cov_optimal, center_optimal + dd * cov_optimal]
        ci_upper = [center_upper - dd * cov_upper, center_upper + dd * cov_upper]
        ci_gap = [center_gap - dd * cov_gap, center_gap + dd * cov_gap]

        return ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap
    else:
        return None, None, None, None, None, None


def compute_ci(cfg, module, xhat, executor=None):
    """ Dispatch to the requested bootstrap method and return its result.

    Args:
        cfg (Config): parameters (cfg.boot_method selects the method)
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): a candidate solution in mpi-sppy nonant format
        executor (BatchExecutor or None): the rank grouping for the batch solves
            (design 9.4). None selects the default K = 1 grouping (each rank its
            own batch worker, a direct EF per batch) used by the standalone
            user_boot / simulate_boot drivers; generic_cylinders' do_boot passes
            an executor with K = --boot-ranks-per-batch and a wheel-based batch
            solver.

    Returns:
        (ci_optimal, ci_upper, ci_gap, center_optimal, center_upper, center_gap);
        the ci_* entries are None on MPI ranks other than 0.

    Note:
        This is the empirical dispatch point shared by user_boot and
        simulate_boot. The smoothed methods have a different (gap-only) return
        signature and are dispatched by smoothed_boot_sp.compute_smoothed_ci;
        a smoothed method reaching here is an error.
    """
    method = cfg.boot_method
    boot_utils.BootMethods.check_for_it(method)
    if boot_utils.is_smoothed(method):
        raise ValueError(
            f"boot_method={method} is a smoothed method; it is dispatched by "
            "smoothed_boot_sp.compute_smoothed_ci, not boot_sp.compute_ci "
            "(the drivers route smoothed methods automatically).")
    if executor is None:
        executor = BatchExecutor(1)
    if method == "Extended":
        return extended_bootstrap(cfg, module, xhat, executor=executor)
    elif method == "Bagging_with_replacement":
        return bagging_bootstrap(cfg, module, xhat, replacement=True, executor=executor)
    elif method == "Bagging_without_replacement":
        return bagging_bootstrap(cfg, module, xhat, replacement=False, executor=executor)
    elif method == "Classical_quantile":
        return classical_bootstrap(cfg, module, xhat, quantile=True, executor=executor)
    elif method == "Classical_gaussian":
        return classical_bootstrap(cfg, module, xhat, quantile=False, executor=executor)
    elif method == "Subsampling":
        return subsampling(cfg, module, xhat, executor=executor)
    else:
        raise ValueError(f"boot_method={method} is not supported.")


if __name__ == "__main__":
    print("boot_sp contains only functions and is not directly runnable.")
    print("Try, e.g., user_boot.py")
