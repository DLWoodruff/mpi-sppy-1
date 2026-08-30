###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Divergence study: does the PH convergence metric go up for large rho?

Serial PH hub only (no spokes) on the three-crop, three-scenario farmer.
For each rho in a sweep we run a fixed number of PH iterations with the
convergence threshold turned off, and record

    conv  = ||x_s - xbar||_1 / (num nonants * num scenarios)   (phbase.convergence_diff)
    ||W|| = max_s ||W_s||_inf
    xbar  = the root-node xbar vector

at every iteration.  A run "diverges" here in the empirical sense: the
convergence metric is larger at the last iteration than it was at the
first, and/or it never settles.

usage: python farmer_ph_divergence.py [--itermax N] [--solver NAME]
"""

import argparse
import csv
import os
import sys

import numpy as np
import pyomo.environ as pyo

from mpisppy.opt.ph import PH
from mpisppy.opt.ef import ExtensiveForm
from mpisppy.extensions.extension import Extension

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import farmer  # noqa: E402


# rho values to sweep; the last few are the "very large" end of the study
DEFAULT_RHOS = [0.1, 1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e8]

NUM_SCEN = 3
CROPS_MULTIPLIER = 1  # three crops


class ConvTracer(Extension):
    """Record the PH convergence metric (and friends) at every iteration.

    miditer() runs immediately after phbase sets self.conv, so the trace is
    exactly the sequence phbase would have compared against convthresh.
    """

    def __init__(self, ph):
        super().__init__(ph)
        self.trace = []  # list of dicts, one per PH iteration
        self.nonant_names = None

    def miditer(self):
        ph = self.opt
        wmax = 0.0
        xbar = None
        for s in ph.local_scenarios.values():
            for ndn_i in s._mpisppy_data.nonant_indices:
                wmax = max(wmax, abs(pyo.value(s._mpisppy_model.W[ndn_i])))
            if xbar is None:
                xbar = [
                    s._mpisppy_model.xbars[ndn_i]._value
                    for ndn_i in s._mpisppy_data.nonant_indices
                ]
                if self.nonant_names is None:
                    self.nonant_names = [
                        v.name for v in s._mpisppy_data.nonant_indices.values()
                    ]
        self.trace.append(
            {"iter": ph._PHIter, "conv": ph.conv, "wmax": wmax, "xbar": xbar}
        )


def run_one(rho, itermax, solver_name):
    """Run PH (hub only) at a fixed rho; return the ConvTracer trace."""
    options = {
        "solver_name": solver_name,
        "PHIterLimit": itermax,
        "defaultPHrho": rho,
        # negative threshold: never stop early, so we see the whole trajectory
        "convthresh": -1.0,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "display_convergence_detail": False,
        "iter0_solver_options": dict(),
        "iterk_solver_options": dict(),
        "tee-rank0-solves": False,
    }
    all_scenario_names = ["scen{}".format(sn) for sn in range(NUM_SCEN)]
    ph = PH(
        options,
        all_scenario_names,
        farmer.scenario_creator,
        farmer.scenario_denouement,
        scenario_creator_kwargs={
            "use_integer": False,
            "crops_multiplier": CROPS_MULTIPLIER,
            "num_scens": NUM_SCEN,
        },
        extensions=ConvTracer,
    )
    ph.ph_main()
    return ph.extobject.trace, ph.extobject.nonant_names


def ef_reference(solver_name):
    """Root-node solution and objective of the extensive form."""
    ef = ExtensiveForm(
        {"solver": solver_name},
        ["scen{}".format(sn) for sn in range(NUM_SCEN)],
        farmer.scenario_creator,
        scenario_creator_kwargs={
            "use_integer": False,
            "crops_multiplier": CROPS_MULTIPLIER,
            "num_scens": NUM_SCEN,
        },
    )
    ef.solve_extensive_form()
    soln = ef.get_root_solution()
    return soln, ef.get_objective_value()


def classify(trace, names, xstar):
    """Summarize one rho's trajectory.

    Divergence of the *metric* is judged two ways:
      rises   -- how many iterations had conv strictly larger than the previous
      maxjump -- the largest such increase, as a ratio
    Whether PH actually solved the problem is a separate question, answered by
    xerr = ||xbar_final - x*||_inf against the extensive-form root solution.
    """
    convs = [t["conv"] for t in trace]
    # skip iteration 1: W is still 0 there, so conv is just the wait-and-see spread
    tail = convs[1:] if len(convs) > 1 else convs
    rises = sum(1 for a, b in zip(tail, tail[1:]) if b > a * (1.0 + 1e-9))
    maxjump = max(
        (b / a for a, b in zip(tail, tail[1:]) if a > 0 and b > a), default=1.0
    )
    xbar = trace[-1]["xbar"]
    xerr = max(abs(xb - xstar[n]) for xb, n in zip(xbar, names))
    # drift: how fast xbar is still moving at the end of the run
    if len(trace) > 1:
        drift = max(abs(a - b) for a, b in zip(trace[-1]["xbar"], trace[-2]["xbar"]))
    else:
        drift = float("nan")
    return {
        "conv_last": convs[-1],
        "rises": rises,
        "maxjump": maxjump,
        "xerr": xerr,
        "drift": drift,
    }


def make_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    by_rho = {}
    for r in rows:
        by_rho.setdefault(r["rho"], []).append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for rho in sorted(by_rho):
        rs = by_rho[rho]
        it = [r["iter"] for r in rs]
        ax1.semilogy(it, [max(r["conv"], 1e-14) for r in rs], label="rho=%g" % rho)
        ax2.plot(it, [r["xerr"] for r in rs], label="rho=%g" % rho)
    ax1.set_xlabel("PH iteration")
    ax1.set_ylabel(r"conv $= \|x_s-\bar{x}\|_1$ / n")
    ax1.set_title("PH convergence metric")
    ax2.set_xlabel("PH iteration")
    ax2.set_ylabel(r"$\|\bar{x}-x^*\|_\infty$")
    ax2.set_title("actual distance to the EF optimum")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("plot written to %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itermax", type=int, default=100)
    parser.add_argument("--solver", type=str, default="gurobi_direct")
    parser.add_argument("--rhos", type=float, nargs="+", default=DEFAULT_RHOS)
    parser.add_argument("--csv", type=str, default="farmer_divergence.csv")
    parser.add_argument("--plot", type=str, default="farmer_divergence.png")
    args = parser.parse_args()

    xstar, efobj = ef_reference(args.solver)

    rows = []
    summary = []
    for rho in args.rhos:
        trace, names = run_one(rho, args.itermax, args.solver)
        summary.append((rho, classify(trace, names, xstar)))
        for t in trace:
            xerr = max(abs(xb - xstar[n]) for xb, n in zip(t["xbar"], names))
            rows.append(
                {
                    "rho": rho,
                    "iter": t["iter"],
                    "conv": t["conv"],
                    "wmax": t["wmax"],
                    "xerr": xerr,
                    "xbar": " ".join("%.12g" % v for v in t["xbar"]),
                }
            )

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["rho", "iter", "conv", "wmax", "xerr", "xbar"]
        )
        w.writeheader()
        w.writerows(rows)

    print("\n=== farmer, %d crops, %d scenarios, %d PH iterations ==="
          % (3 * CROPS_MULTIPLIER, NUM_SCEN, args.itermax))
    print("EF optimum %.2f at %s"
          % (efobj, ", ".join("%s=%g" % (k.split("[")[1].rstrip("]"), v)
                              for k, v in sorted(xstar.items()))))
    print("%10s %12s %7s %11s %12s %12s"
          % ("rho", "conv_last", "rises", "max jump", "|xbar-x*|", "drift/iter"))
    for rho, d in summary:
        print("%10g %12.4g %7d %11.4g %12.4g %12.4g"
              % (rho, d["conv_last"], d["rises"], d["maxjump"], d["xerr"],
                 d["drift"]))
    print("\n  rises     = iterations (after the first) where conv went UP")
    print("  max jump  = largest one-iteration increase in conv, as a ratio")
    print("  |xbar-x*| = how far the final xbar really is from the EF answer")
    print("  drift     = how far xbar still moved on the last iteration")
    print("\nfull per-iteration trace written to %s" % args.csv)
    if args.plot:
        make_plot(rows, args.plot)


if __name__ == "__main__":
    main()
