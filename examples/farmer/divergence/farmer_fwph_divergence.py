###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Divergence study, FWPH: does large rho break Frank-Wolfe PH the same way?

Companion to farmer_ph_divergence.py, same instance (three crops, three
scenarios, continuous) and same rho sweep, so the two are directly comparable.

FWPH is measured two ways, because it reports two things PH does not have
together:

  conv   fwph_convergence_diff(), the check of Algorithm 3 in Boland et al.:
         sum_s p_s ||x^QP_s - xbar||_2^2.  Note this is a probability-weighted
         SQUARED two-norm and is not scaled by the variable count, so it is
         not on the same scale as PH's ||x_s - xbar||_1 / n -- compare the
         shape of the trajectory, not the magnitude.

  bound  the FWPH outer bound.  This is the quantity that runs away in aph-fw
         when rho is mis-scaled, so it is the one that makes "diverges" mean
         the same thing here as it does there.

usage: python farmer_fwph_divergence.py [--itermax N] [--solver NAME]
"""

import argparse
import csv
import os
import sys

import pyomo.environ as pyo

from mpisppy.opt.fwph import FWPH
from mpisppy.extensions.extension import Extension

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import farmer  # noqa: E402

from farmer_ph_divergence import (  # noqa: E402
    CROPS_MULTIPLIER,
    DEFAULT_RHOS,
    NUM_SCEN,
    ef_reference,
)


class FWConvTracer(Extension):
    """Record FWPH's convergence metric and outer bound at every major iteration.

    miditer() runs immediately after fwph's iterk_loop assigns self.conv, and
    after _swap_nonant_vars_back(), so the scenario nonants are the MIP ones.
    """

    def __init__(self, fw):
        super().__init__(fw)
        self.trace = []
        self.nonant_names = None

    def miditer(self):
        fw = self.opt
        xbar = None
        for s in fw.local_scenarios.values():
            if xbar is None:
                xbar = [
                    s._mpisppy_model.xbars[ndn_i]._value
                    for ndn_i in s._mpisppy_data.nonant_indices
                ]
                self.nonant_names = [
                    v.name for v in s._mpisppy_data.nonant_indices.values()
                ]
        wmax = 0.0
        for s in fw.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                wmax = max(wmax, abs(pyo.value(s._mpisppy_model.W[ndn_i])))
        self.trace.append(
            {
                "iter": fw._PHIter,
                "conv": float(fw.conv) if fw.conv is not None else float("nan"),
                "bound": float(getattr(fw, "_fwph_best_bound", float("nan"))),
                "wmax": wmax,
                "xbar": xbar,
            }
        )


def run_one(rho, itermax, solver_name, fw_iter_limit, use_integer=False):
    options = {
        "solver_name": solver_name,
        "PHIterLimit": itermax,
        "defaultPHrho": rho,
        "convthresh": -1.0,  # never stop early; we want the whole trajectory
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "display_convergence_detail": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "tee-rank0-solves": False,
        # Frank-Wolfe inner loop (Boland Algorithm 2)
        "FW_iter_limit": fw_iter_limit,
        "FW_weight": 0.0,
        "FW_conv_thresh": 1e-5,
        "FW_LP_start_iterations": 0,
        "FW_verbose": False,
        "mip_solver_options": {},
        "qp_solver_options": {},
    }
    fw = FWPH(
        options,
        ["scen{}".format(sn) for sn in range(NUM_SCEN)],
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs={
            "use_integer": use_integer,
            "crops_multiplier": CROPS_MULTIPLIER,
            "num_scens": NUM_SCEN,
        },
        extensions=FWConvTracer,
    )
    fw.fwph_main()
    return fw.extobject.trace, fw.extobject.nonant_names


# Below this the Boland metric is machine zero for a squared quantity, and its
# ups and downs are solver noise rather than algorithmic behaviour.  Counting
# them as "rises" would report a diverging metric for a run that has in fact
# frozen, which is exactly the mistake this study is about.
CONV_FLOOR_CONTINUOUS = 1e-12
CONV_FLOOR_INTEGER = 1e-5


def classify(trace, names, xstar, conv_floor=CONV_FLOOR_CONTINUOUS):
    convs = [t["conv"] for t in trace]
    bounds = [t["bound"] for t in trace]
    tail = convs[1:] if len(convs) > 1 else convs
    rises = sum(
        1
        for a, b in zip(tail, tail[1:])
        if b > a * (1.0 + 1e-9) and b > conv_floor
    )
    xbar = trace[-1]["xbar"]
    xerr = max(abs(xb - xstar[n]) for xb, n in zip(xbar, names))
    finite = [b for b in bounds if b == b]
    maxjump = max(
        (b / a for a, b in zip(tail, tail[1:]) if a > 0 and b > a and b > conv_floor),
        default=1.0,
    )
    return {
        "conv_last": convs[-1],
        "rises": rises,
        "maxjump": maxjump,
        "bound_first": finite[0] if finite else float("nan"),
        "bound_last": finite[-1] if finite else float("nan"),
        "xerr": xerr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itermax", type=int, default=100)
    parser.add_argument("--solver", type=str, default="gurobi_direct")
    parser.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS)
    parser.add_argument("--fw-iter-limit", type=int, default=10)
    parser.add_argument("--integer", action="store_true",
                        help="make the first-stage acreage variables integer")
    parser.add_argument("--conv-floor", type=float, default=None,
                        help="ignore conv changes below this when counting "
                             "rises (default: %g continuous, %g integer)"
                             % (CONV_FLOOR_CONTINUOUS, CONV_FLOOR_INTEGER))
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.conv_floor is None:
        args.conv_floor = (CONV_FLOOR_INTEGER if args.integer
                           else CONV_FLOOR_CONTINUOUS)
    if args.csv is None:
        args.csv = ("farmer_fwph_divergence%s.csv"
                    % ("_int" if args.integer else ""))

    xstar, efobj = ef_reference(args.solver, args.integer)

    rows, summary = [], []
    for rho in args.rhos:
        trace, names = run_one(rho, args.itermax, args.solver,
                               args.fw_iter_limit, args.integer)
        summary.append((rho, classify(trace, names, xstar, args.conv_floor)))
        for t in trace:
            xerr = max(abs(xb - xstar[n]) for xb, n in zip(t["xbar"], names))
            rows.append(
                {
                    "rho": rho,
                    "iter": t["iter"],
                    "conv": t["conv"],
                    "bound": t["bound"],
                    "wmax": t["wmax"],
                    "xerr": xerr,
                    "xbar": " ".join("%.12g" % v for v in t["xbar"]),
                }
            )

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["rho", "iter", "conv", "bound", "wmax", "xerr", "xbar"]
        )
        w.writeheader()
        w.writerows(rows)

    print("\n=== FWPH: farmer, %d crops%s, %d scenarios, %d major iterations, "
          "FW_iter_limit %d ==="
          % (3 * CROPS_MULTIPLIER, ", INTEGER" if args.integer else "",
             NUM_SCEN, args.itermax, args.fw_iter_limit))
    print("EF optimum %.2f" % efobj)
    print("%10s %13s %6s %10s %13s %13s %11s"
          % ("rho", "conv_last", "rises", "max jump", "bound first",
             "bound last", "|xbar-x*|"))
    for rho, d in summary:
        print("%10g %13.4g %6d %10.4g %13.6g %13.6g %11.4g"
              % (rho, d["conv_last"], d["rises"], d["maxjump"],
                 d["bound_first"], d["bound_last"], d["xerr"]))
    print("\n  rises/max jump ignore changes below conv = %g, where the "
          "squared\n  metric is machine zero and its motion is solver noise"
          % args.conv_floor)
    print("\nfull per-iteration trace written to %s" % args.csv)


if __name__ == "__main__":
    main()
