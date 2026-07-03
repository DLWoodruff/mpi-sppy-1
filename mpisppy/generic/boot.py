###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Data-based bootstrap/bagging confidence interval for generic_cylinders.

This is the data-based analog of the MMW confidence interval (see mmw.py).
Given a dataset (interpreted by the model), it reports a bootstrap/bagging
confidence interval on the optimality gap of a candidate solution xhat. xhat
comes from the configured solve (the wheel) or from a file; the interval is
estimated by resampling a pool of dataset records held strictly disjoint from
the records that produced xhat.

The estimator itself lives in mpisppy.confidence_intervals.bootsp; this module
is the positional layer (design section 9.2) that reconciles mpi-sppy's
name-based scenario_creator with the position-based resampling: the dataset is
the ordered list from scenario_names_creator, the candidate (M) and pool (N)
blocks are disjoint slices of positions, and a resolver maps each pool position
to its canonical scenario name.
"""

import os
import tempfile

import mpisppy.utils.sputils as sputils
from mpisppy import global_toc, MPI
from mpisppy.utils.solver_spec import solver_specification


def boot_requested(cfg):
    """Return True iff a bootstrap CI was requested (--boot-method is set).

    Mirrors mmw_requested. Validates the candidate-size / xhat-file exclusivity
    and rejects the smoothed methods, which are not yet available in
    generic_cylinders (use the standalone user_boot for those).

    Raises ValueError on an unknown method, a smoothed method, or a request
    that gives both a positive candidate sample size and an xhat file.
    """
    method = cfg.get("boot_method")
    if method is None:
        return False

    import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
    boot_utils.BootMethods.check_for_it(method)
    if boot_utils.is_smoothed(method):
        raise ValueError(
            f"boot_method={method} is a smoothed method, which is not yet "
            "available in generic_cylinders. Use one of the empirical methods "
            f"({boot_utils.empirical_members()}) here, or run the standalone "
            "mpisppy.confidence_intervals.bootsp.user_boot for smoothed methods.")

    M = cfg.get("boot_candidate_sample_size")
    xhat_fname = cfg.get("boot_xhat_input_file_name")
    if xhat_fname is not None and M is not None and M > 0:
        raise ValueError(
            "boot_candidate_sample_size (M) and boot_xhat_input_file_name are "
            "mutually exclusive: give a positive M to find xhat from the "
            "candidate records, or an xhat file to read it (with M omitted or 0).")
    return True


def _check_compatibility(cfg):
    """Refuse combinations that do not apply to a data-based bootstrap (9.3)."""
    from mpisppy.generic.mmw import mmw_requested
    if mmw_requested(cfg):
        raise ValueError(
            "MMW and the bootstrap CI are mutually exclusive: MMW draws fresh "
            "samples from a distribution, while the bootstrap resamples a fixed "
            "dataset. Sequential sampling is excluded for the same reason. "
            "Run one confidence-interval method or the other.")
    if cfg.get("branching_factors") is not None:
        raise ValueError(
            "The bootstrap CI is two-stage only; it does not support the "
            "multistage (branching_factors) case.")
    for opt in ("admm", "stoch_admm", "cvar"):
        if cfg.get(opt, ifmissing=False):
            raise ValueError(
                f"The bootstrap CI cannot be combined with --{opt.replace('_', '-')}.")

    # K > 1 (a wheel per group of ranks on a sub-communicator) is a scheduled
    # follow-on; the first integration ships K = 1 (a per-rank extensive form).
    K = cfg.get("boot_ranks_per_batch", 1)
    if K is not None and K != 1:
        raise ValueError(
            f"boot_ranks_per_batch={K}: only K=1 (a per-rank extensive form) "
            "is supported so far.")


def _estimator_cfg(module_basename, module, cfg, N, M, pool_names):
    """Build the Config the bootsp estimator expects from the boot_* options.

    The estimator (boot_sp) reads its own historical option names (max_count,
    sample_size, ...). Rather than alias them onto the generic cfg, we build a
    dedicated estimator cfg so the generic cfg is left untouched. The estimator
    addresses records by position 0..N-1; we set max_count = N so its resampling
    pool *is* the disjoint N-record block, and install a resolver mapping each
    position to its canonical scenario name.
    """
    import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils

    boot_cfg = boot_utils.cfg_for_boot()
    boot_cfg.module_name = module_basename

    # carry the model's own data options (e.g. data_file) over from the generic
    # run, but not num_scens: that is the main run's candidate count, whereas the
    # estimator resamples variable-size batches and must use uniform (None)
    # scenario probabilities.
    before = set(boot_cfg.keys())
    if hasattr(module, "inparser_adder"):
        module.inparser_adder(boot_cfg)
    for k in (set(boot_cfg.keys()) - before) - {"num_scens"}:
        v = cfg.get(k)
        if v is not None:
            boot_cfg[k] = v

    # translate the boot_* flags to the estimator's option names
    boot_cfg.boot_method = cfg.boot_method
    boot_cfg.max_count = N               # the estimator's address space = the pool
    boot_cfg.candidate_sample_size = M   # informational (xhat is already in hand)
    boot_cfg.sample_size = N
    boot_cfg.subsample_size = cfg.get("boot_subsample_size")
    boot_cfg.nB = cfg.get("boot_nB")
    boot_cfg.alpha = cfg.get("boot_alpha")
    boot_cfg.seed_offset = cfg.get("boot_seed_offset")

    # the batch ("boot") solver role, falling back to the generic solver_name
    # (and its options), so the batch EF solves are independent of any xhat-EF
    # solver
    _, boot_solver_name, boot_solver_options = solver_specification(cfg, ["boot", ""])
    boot_cfg.solver_name = boot_solver_name
    boot_cfg.add_to_config(
        "solver_options",
        description="options dict for the bootstrap batch solver",
        domain=None, default=None, argparse=False)
    boot_cfg.solver_options = boot_solver_options

    # the positional resolver: estimator position p -> canonical pool name
    boot_cfg.add_to_config(
        "boot_name_of_position",
        description="maps a resampling-pool position to its scenario name",
        domain=None, default=None, argparse=False)
    boot_cfg.boot_name_of_position = lambda p: pool_names[p]

    return boot_cfg


def do_boot(module_fname, cfg, wheel=None):
    """Run a data-based bootstrap/bagging CI after the main algorithm.

    Mirrors do_mmw: generic_cylinders calls this after the configured solve.

    Args:
        module_fname (str): module name or path (e.g. 'farmer' or '/path/farmer')
        cfg (Config): config with boot_* options
        wheel (WheelSpinner or None): the configured solve; if None,
            boot_xhat_input_file_name must be set.

    Returns:
        The estimator 6-tuple (ci_optimal, ci_upper, ci_gap, center_optimal,
        center_upper, center_gap) on rank 0; the ci_* entries are None on other
        ranks. Returns None if the main algorithm produced no feasible xhat.
    """
    import mpisppy.confidence_intervals.ciutils as ciutils
    import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
    import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp

    _check_compatibility(cfg)

    # load_module (parsing.py) added the module's directory to sys.path, so the
    # basename is importable; we want the raw model module (not a wheel wrapper).
    module_basename = os.path.basename(module_fname)
    module = boot_utils.module_name_to_module(module_basename)

    if not hasattr(module, "scenario_names_creator"):
        raise ValueError(
            f"module {module_basename} must provide scenario_names_creator for "
            "the bootstrap CI")

    # The dataset is the ordered list of every scenario name it implies (9.1).
    all_names = list(module.scenario_names_creator(None))
    dataset_size = len(all_names)

    N = cfg.get("boot_sample_size")
    if N is None or N <= 0:
        raise ValueError("boot_sample_size (N) must be a positive integer")
    M = cfg.get("boot_candidate_sample_size") or 0

    # Strictly-disjoint candidate | pool position blocks (9.1).
    if M + N > dataset_size:
        raise ValueError(
            f"boot: candidate M={M} + pool N={N} exceeds the dataset size "
            f"({dataset_size}); the candidate and resampling pool must be disjoint.")
    pool_names = all_names[M:M + N]

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()

    # xhat: from a precomputed file, or written out from the wheel's best solution.
    xhat_fname = cfg.get("boot_xhat_input_file_name")
    if xhat_fname is not None:
        xhat = ciutils.read_xhat(xhat_fname)
    else:
        if wheel is None:
            raise RuntimeError(
                "do_boot: boot_xhat_input_file_name must be set when no wheel "
                "is provided")
        if M == 0:
            raise ValueError(
                "boot_candidate_sample_size (M) must be positive when xhat comes "
                "from the wheel; it names the candidate block kept disjoint from "
                "the resampling pool.")
        num_scens = cfg.get("num_scens")
        if num_scens is not None and num_scens != M:
            raise ValueError(
                f"boot_candidate_sample_size (M={M}) must match the number of "
                f"scenarios used to find xhat (num_scens={num_scens}).")
        if global_rank == 0:
            tmp_path = tempfile.mktemp(suffix=".npy")
        else:
            tmp_path = None
        tmp_path = global_comm.bcast(tmp_path, root=0)
        wheel.write_first_stage_solution(
            tmp_path,
            first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer,
        )
        global_comm.Barrier()
        if not os.path.exists(tmp_path):
            global_toc("boot CI skipped: no feasible solution found by the main algorithm.")
            return None
        xhat = ciutils.read_xhat(tmp_path)
        if global_rank == 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    boot_cfg = _estimator_cfg(module_basename, module, cfg, N, M, pool_names)

    result = boot_sp.compute_ci(boot_cfg, module, xhat)

    if global_rank == 0:
        ci_optimal, ci_upper, ci_gap, c_optimal, c_upper, c_gap = result
        ci_gap[0] = max(0, ci_gap[0])
        conf = 1.0 - cfg.boot_alpha
        global_toc(f"boot ({cfg.boot_method}) point estimate of the optimality gap: {c_gap}")
        global_toc(f"boot {conf:g} CI for the optimality gap: {ci_gap}")
        global_toc(f"boot {conf:g} CI for the optimal function value: {ci_optimal}")
        global_toc(f"boot {conf:g} CI for the function value at xhat: {ci_upper}")
    return result
