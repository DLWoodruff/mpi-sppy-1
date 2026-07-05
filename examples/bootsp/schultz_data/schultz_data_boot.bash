#!/bin/bash
# Data-based bootstrap confidence interval directly from generic_cylinders
# (the end-goal workflow: solve for xhat with the cylinder system, then report a
# bootstrap/bagging CI on its optimality gap from a disjoint part of the data).
#
# The dataset is schultz_data.csv (200 rows). The main run finds xhat from the
# first M = 5 records (--num-scens 5); the bootstrap then resamples the next
# N = 100 records (kept strictly disjoint from the candidate records) for the CI.
#
# Pass a solver name as the first argument (default: gurobi_direct).
#
# How each resampled batch is solved is configured by a separate file of
# generic_cylinders flags, --boot-batch-config-file, because a batch (N
# scenarios) is a different problem from the xhat solve (M candidate records).
# Here K = 1 (--boot-ranks-per-batch defaults to 1), so each batch is a direct
# extensive form and the batch config need only name the solver.

SOLVER=${1:-gurobi_direct}

BATCH_CONFIG=$(mktemp)
echo "--solver-name ${SOLVER}" > "${BATCH_CONFIG}"

BOOT="--boot-method Classical_quantile \
      --boot-candidate-sample-size 5 \
      --boot-sample-size 100 \
      --boot-subsample-size 20 \
      --boot-nB 20 \
      --boot-alpha 0.1 \
      --boot-seed-offset 100 \
      --boot-batch-config-file ${BATCH_CONFIG}"

echo "Find xhat with a PH hub + xhatshuffle/lagrangian spokes, then a bootstrap"
echo "CI on its optimality gap from a disjoint part of the dataset."
echo
mpiexec -np 3 python -m mpisppy.generic_cylinders \
    --module-name schultz_data \
    --num-scens 5 \
    --max-iterations 20 \
    --default-rho 1.0 \
    --solver-name ${SOLVER} \
    --xhatshuffle --lagrangian \
    --max-solver-threads 2 \
    ${BOOT}

rm -f "${BATCH_CONFIG}"
