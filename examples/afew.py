###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Run a few examples; dlw June 2020
# See also runall.py
# Assumes you run from the examples directory.
# Optional command line arguments: solver_name mpiexec_arg
# E.g. python run_all.py
#      python run_all.py cplex
#      python run_all.py gurobi_persistent --oversubscribe

import os
import sys

solver_name = "gurobi_persistent"
if len(sys.argv) > 1:
    solver_name = sys.argv[1]

# Use oversubscribe if your computer does not have enough cores.
# Don't use this unless you have to.
# (This may not be allowed on versions of mpiexec)
mpiexec_arg = ""  # "--oversubscribe"
if len(sys.argv) > 2:
    mpiexec_arg = sys.argv[2]

badguys = dict()

def do_one(dirname, progname, np, argstring):
    os.chdir(dirname)
    runstring = "mpiexec {} -np {} python -m mpi4py {} {}".\
                format(mpiexec_arg, np, progname, argstring)
    print(runstring)
    code = os.system(runstring)
    if code != 0:
        if dirname not in badguys:
            badguys[dirname] = [runstring]
        else:
            badguys[dirname].append(runstring)
    if '/' not in dirname:
        os.chdir("..")
    else:
        os.chdir("../..")   # hack for one level of subdirectories


# for farmer, the first arg is num_scens and is required
do_one("farmer/archive", "farmer_cylinders.py", 3,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=50 "
       "--default-rho=1 --sep-rho --display-convergence-detail "
       "--solver-name={} --xhatshuffle --lagrangian --use-norm-rho-updater".format(solver_name))
do_one("farmer", "farmer_lshapedhub.py", 2,
       "--num-scens=3 --bundles-per-rank=0 --max-iterations=50 "
       "--solver-name={} --rel-gap=0.0 "
       " --xhatlshaped --max-solver-threads=1".format(solver_name))
do_one("hydro", "hydro_cylinders_pysp.py", 3,
       "--bundles-per-rank=0 --max-iterations=10000 "
       "--default-rho=1 --xhatshuffle --lagrangian "
       "--abs-gap=0 --rel-gap=0 --time-limit=2 "
       "--solver-name={}".format(solver_name))

if len(badguys) > 0:
    print("\nBad Guys:")
    for i,v in badguys.items():
        print("Directory={}".format(i))
        for c in v:
            print("    {}".format(c))
        sys.exit(1)
else:
    print("\nAll OK.")
