###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


import pyomo.environ as pyo
from math import log10, floor

from mpisppy.utils import sputils


def limit_solver_threads(solver, solver_name, threads=1):
    """Cap thread count on a directly-constructed Pyomo solver so test
    solves do not fan out across every core. Reuses the canonical->native
    option translator so we do not hardcode per-solver key names. Safe to
    call before or after set_instance for persistent solvers (thread
    options are applied at solve time)."""
    solver.options.update(
        sputils.translate_solver_options({"threads": threads}, solver_name))


def get_solver(persistent_OK=True):
    solvers = ["cplex","gurobi","xpress"]
    if persistent_OK:
        solvers = [n+e for e in ('_persistent', '') for n in solvers]
    
    for solver_name in solvers:
        try:
            solver_available = pyo.SolverFactory(solver_name).available()
        except Exception:
            solver_available = False
        if solver_available:
            break
    
    if '_persistent' in solver_name:
        persistent_solver_name = solver_name
    else:
        persistent_solver_name = solver_name+"_persistent"
    try:
        persistent_available = pyo.SolverFactory(persistent_solver_name).available()
    except Exception:
        persistent_available = False
    
    return solver_available, solver_name, persistent_available, persistent_solver_name

def round_pos_sig(x, sig=1):
    return round(x, sig-int(floor(log10(abs(x))))-1)


def solver_takes_model_size(solver_name, num_vars, num_cons):
    """Can `solver_name` solve a model with this many variables and constraints?

    The pip-installed community editions of cplex, gurobi and xpress -- which
    is what CI has -- cap model size, and they report the cap as a license
    error partway through a solve rather than up front, so the only way to
    ask is to hand one over. The probe is a trivial bounded LP of the
    requested size; it says nothing about difficulty, only about size.

    A solver that is not installed at all also answers False, so check
    availability first where the two answers differ.
    """
    model = pyo.ConcreteModel()
    model.varset = pyo.RangeSet(num_vars)
    model.x = pyo.Var(model.varset, bounds=(0, 1))
    model.conset = pyo.RangeSet(num_cons)
    model.c = pyo.Constraint(
        model.conset, rule=lambda m, j: m.x[(j - 1) % num_vars + 1] <= 1)
    model.obj = pyo.Objective(expr=sum(model.x[i] for i in model.varset))
    try:
        solver = pyo.SolverFactory(solver_name)
        if sputils.is_persistent(solver):
            # The legacy persistent interface refuses solve(model).
            solver.set_instance(model)
            solver.solve()
        else:
            solver.solve(model)
    except Exception:
        return False
    return True
