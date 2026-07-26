###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Create npy files for xhat and optimal that are used in simulations.
#
#   python -m mpisppy.confidence_intervals.bootsp.boot_general_prep <json>
#
# Compute the optimal function value with max_count scenarios (or read it from
# a file), then find a candidate solution using the reserved
# candidate_sample_size scenarios and compute the corresponding optimality gap.

import sys
import json
import warnings
import numpy as np
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
import mpisppy.confidence_intervals.bootsp.boot_sp as boot_sp


def find_optimal(cfg, module):
    """The reference optimal z* that a coverage study counts against.

    This is the *incumbent*, deliberately not the outer bound the estimators
    use for a batch optimal (boot_sp._ef_optimal_value). The two play opposite
    roles: a batch optimal feeds a reported gap that must not read
    optimistically, so it takes the conservative side, while the reference z*
    stands in for the true optimum, so it wants the best available estimate of
    that optimum. Biasing the reference either way biases the measured
    coverage, and biasing it toward the bound is the worse of the two here.
    The estimators already use an outer bound for every batch optimal, so the
    gaps they report are shifted upward; a reference gap built from an outer
    bound would be shifted upward too, and the two would move together. The
    intervals would then cover the reference more often than they cover the
    true gap, so the study would report a coverage rate higher than the one
    the method actually achieves -- concealing the bound-slack effect that a
    coverage study is run to expose.

    Neither side is right when the reference solve has not converged: the truth
    lies between the incumbent and the bound, so a study built on an unconverged
    reference is not measuring coverage of anything in particular. Say so.
    """
    opt_ef = boot_sp.solve_routine(cfg, module, range(cfg.max_count), num_threads=16)
    opt_obj = pyo.value(opt_ef.EF_Obj)
    bound = boot_sp._ef_optimal_value(opt_ef)
    slack = abs(opt_obj - bound)
    if slack > 1e-6 * max(1.0, abs(opt_obj)) and boot_utils.my_rank == 0:
        warnings.warn(
            "the reference optimal solve did not converge: incumbent "
            f"{opt_obj}, best bound {bound} (gap {slack}). z* lies between "
            "them, so a coverage study against this reference is not measuring "
            "coverage of the true optimum. Tighten the solver's gap.")
    return opt_obj


def find_candidate(cfg, module):
    # the same scenario block that boot_utils.compute_xhat reserves (and that
    # boot_sp.eligible_scenarios excludes from confidence-interval sampling)
    scenarios = range(cfg.sample_size,
                      cfg.sample_size + cfg.candidate_sample_size)
    if len(scenarios) == 1:
        print(f"only one scenario, {scenarios},  for candidate solution")
    candidate_ef = boot_sp.solve_routine(cfg, module, scenarios, num_threads=2, duplication=False)

    xhat = sputils.nonant_cache_from_ef(candidate_ef)
    return xhat


def find_gap(cfg, module, xhat, opt_obj):
    obj_hat = boot_sp.evaluate_scenarios(cfg, module, range(cfg.max_count), xhat, duplication=False)
    opt_gap = obj_hat - opt_obj
    return opt_gap


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("need json file")
        print("usage (e.g.): python -m mpisppy.confidence_intervals.bootsp.boot_general_prep little_schultz.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)

    boot_utils.check_BFs(cfg)

    cfg.add_to_config(name="use_fitted",
                    description="a boolean to control use of fitted distribution",
                    domain=bool,
                    default=None,
                    argparse=False)
    cfg.use_fitted = False

    if "deterministic_data_json" in cfg:
        json_fname = cfg.deterministic_data_json
        try:
            with open(json_fname, "r") as read_file:
                detdata = json.load(read_file)
        except Exception:
            print(f"Could not read the json file: {json_fname}")
            raise
        cfg.add_to_config("detdata",
                        description="determinstic data from json file",
                        domain=dict,
                        default=detdata)

    module = boot_utils.module_name_to_module(cfg.module_name)

    xhat_fname = cfg["xhat_fname"]

    opt_obj = find_optimal(cfg, module)
    xhat = find_candidate(cfg, module)
    opt_gap = find_gap(cfg, module, xhat, opt_obj)

    np.save(cfg.optimal_fname, [opt_obj, opt_gap])
    ciutils.write_xhat(xhat, path=xhat_fname)

    print(f"opt_obj: {opt_obj}")
    print(f"opt_gap: {opt_gap}")
