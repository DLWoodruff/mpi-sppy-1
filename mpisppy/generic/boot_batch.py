###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Solve one bootstrap batch with cylinders (design section 9.4, K > 1).

For ``--boot-ranks-per-batch`` (K) greater than 1 a resampled batch is not a
direct extensive form but a full hub/spoke wheel run on the batch's group of K
ranks. How that wheel is configured -- solver, rho, which spokes, convergence,
the relative gap -- is a *different* problem from the xhat solve (a batch has N
scenarios, usually far more than the M candidate records), so it is supplied
separately as ``--boot-batch-config-file``: literally a file of
``generic_cylinders`` flags. This module parses that file into a Config and
turns it into a callable that the estimator invokes per batch, returning the
wheel's outer (decomposition) bound as the batch optimal ``L_b`` (design 9.4.1).

The batch config must produce an *outer* bound (a Lagrangian/subgradient spoke,
or the subgradient hub); a bare PH hub with no bounding spoke leaves the outer
bound undefined and is rejected at solve time with a clear message.
"""

import shlex

from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.generic.parsing import register_generic_args
from mpisppy.generic.hub import build_hub_dict
from mpisppy.generic.spokes import build_spoke_list
from mpisppy.generic.extensions import configure_extensions
from mpisppy.generic import decomp
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.utils.config as config
from mpisppy.confidence_intervals.bootsp.boot_sp import _scenario_creator_w_mapping


def _batch_scenario_denouement(rank, name, scenario):
    """No-op denouement for the batch wheels (nothing to report per scenario)."""
    pass


def parse_batch_config_file(path, module):
    """Parse a batch config file (generic_cylinders flags) into a Config.

    The file is read as if its contents were the generic_cylinders command line
    (``--solver-name gurobi --lagrangian --default-rho 1.0`` ...); ``#`` starts a
    comment. It is parsed by the same Config machinery as the main run, so it is
    exactly a batch generic_cylinders configuration. The framework -- not the
    file -- supplies the batch's scenario set (its count and the positional
    sample->record mapping), so the file must not set the scenario-formation
    options; those are the ``--boot-*`` flags on the main command line.

    Args:
        path (str): path to the batch config file.
        module: the model module (its inparser_adder registers model options).

    Returns:
        Config: the parsed batch configuration.
    """
    cfg = config.Config()
    register_generic_args(cfg, module)
    parser = cfg.create_parser("boot batch config")
    with open(path, "r") as f:
        tokens = shlex.split(f.read(), comments=True)
    args = parser.parse_args(tokens)
    cfg.import_argparse(args)
    cfg.checker()  # same inconsistency checks as the main run
    return cfg


def _outer_bound_over_group(wheel, groupcomm):
    """The wheel's outer bound, made available on every rank of the group.

    ``WheelSpinner`` populates ``BestOuterBound`` only on the hub ranks
    (strata_rank 0); the rest carry None. The group leader (group_rank 0), which
    reports the batch result into the cross-group gather, is not necessarily a
    hub rank, so share the bound across the whole group and let every rank return
    the same value.
    """
    candidates = [b for b in groupcomm.allgather(wheel.BestOuterBound) if b is not None]
    if not candidates:
        raise RuntimeError(
            "boot batch solve produced no outer bound. The --boot-batch-config-file "
            "must configure a wheel that yields an outer (decomposition) bound on "
            "the batch optimal -- e.g. add a Lagrangian or subgradient spoke, or "
            "use the subgradient hub. A bare PH hub does not provide one.")
    bound = float(candidates[0])
    if bound in (float("inf"), float("-inf")):
        raise RuntimeError(
            "boot batch solve reported an infinite outer bound (no bounding "
            "progress). Check the --boot-batch-config-file solver/spoke settings.")
    return bound


def make_batch_optimal_solver(batch_cfg, module):
    """Return a callable that solves one batch with cylinders for its outer bound.

    The returned callable has the signature expected by the estimator's
    ``BatchExecutor.batch_optimal_solver``:

        solver(scenario_names, sample_mapping, groupcomm) -> outer_bound (float)

    It builds a fresh wheel over the batch's sample scenarios (mapped back to
    their resampled records) using ``batch_cfg`` and runs it on ``groupcomm``
    (the K ranks assigned to this batch), returning the wheel's outer bound.
    """
    rho_setter = decomp._get_rho_setter(module, batch_cfg)
    ph_converger = decomp._get_converger(batch_cfg)
    average_scenario_creator = getattr(module, "average_scenario_creator", None)
    feasible_xhat_creator = vanilla._find_feasible_xhat_creator(module, batch_cfg)

    def solver(scenario_names, sample_mapping, groupcomm):
        # a fresh kwargs dict each call: the mapping (and thus the batch) changes
        scenario_creator_kwargs = module.kw_creator(batch_cfg)
        scenario_creator_kwargs["module"] = module
        scenario_creator_kwargs["mapping"] = sample_mapping

        # a batch has this many scenarios; set num_scens so a model that reads it
        # for uniform probabilities matches the direct-EF (K=1) semantics.
        batch_cfg.num_scens = len(scenario_names)

        beans = (batch_cfg, _scenario_creator_w_mapping,
                 _batch_scenario_denouement, scenario_names)

        hub_dict = build_hub_dict(batch_cfg, beans, scenario_creator_kwargs,
                                  rho_setter, None, ph_converger)
        configure_extensions(hub_dict, module, batch_cfg)
        if batch_cfg.reduced_costs:
            vanilla.add_reduced_costs_fixer(hub_dict, batch_cfg)

        list_of_spoke_dict = build_spoke_list(
            batch_cfg, beans, scenario_creator_kwargs, rho_setter, None,
            average_scenario_creator=average_scenario_creator,
            feasible_xhat_creator=feasible_xhat_creator)

        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.run(comm_world=groupcomm)
        return _outer_bound_over_group(wheel, groupcomm)

    return solver
