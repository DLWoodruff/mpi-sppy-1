#!/bin/bash
# Data-based bootstrap CI where each resampled batch is solved WITH CYLINDERS
# (--boot-ranks-per-batch K > 1). This is the K > 1 generalization of
# schultz_data_boot.bash (which solves each batch as a direct extensive form).
#
# The 6 ranks first find xhat together (PH hub + xhatshuffle + lagrangian over
# the M = 5 candidate records), then re-form into G = 6 // K groups of K ranks;
# each group solves its share of the bootstrap batches with a wheel (configured
# by schultz_wheel_batch.txt) and the per-group results are gathered to rank 0.
#
# Here K = 3, so G = 2 groups of 3 ranks run concurrently -- one batch each at a
# time -- until all the batches are done.
#
# Pass a solver name as the first argument (default: gurobi_direct).

SOLVER=${1:-gurobi_direct}
HERE=$(dirname "$0")

# The batch config file names gurobi_direct; substitute the chosen solver into a
# temporary copy so the demo runs with any solver.
BATCH_CONFIG=$(mktemp)
sed "s/gurobi_direct/${SOLVER}/" "${HERE}/schultz_wheel_batch.txt" > "${BATCH_CONFIG}"

BOOT="--boot-method Classical_quantile \
      --boot-candidate-sample-size 5 \
      --boot-sample-size 20 \
      --boot-subsample-size 8 \
      --boot-nB 12 \
      --boot-alpha 0.1 \
      --boot-seed-offset 100 \
      --boot-ranks-per-batch 3 \
      --boot-batch-config-file ${BATCH_CONFIG}"

echo "Find xhat with 6 ranks, then solve each bootstrap batch with a 3-rank"
echo "cylinder wheel (2 concurrent groups) for its optimality-gap CI."
echo
mpiexec -np 6 python -m mpisppy.generic_cylinders \
    --module-name schultz_data \
    --num-scens 5 \
    --max-iterations 20 \
    --default-rho 1.0 \
    --solver-name ${SOLVER} \
    --xhatshuffle --lagrangian \
    --max-solver-threads 2 \
    ${BOOT}

rm -f "${BATCH_CONFIG}"
