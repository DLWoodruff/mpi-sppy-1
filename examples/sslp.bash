SOLVER="xpress_persistent"
SPB=10
INSTANCE_NAME="sslp_10_50_200"
# INSTANCE_NAME="sslp_15_45_15"

echo "^^^ sslp fwph ^^^"
cd sslp
mpiexec -np 10 python -u -m mpi4py ../../mpisppy/generic_cylinders.py --module-name sslp --sslp-data-path ./data --instance-name ${INSTANCE_NAME} --solver-name ${SOLVER} --max-iterations 10 --max-solver-threads 1 --default-rho 1.0 --fwph-hub --xhatshuffle --rel-gap 1e-6 --abs-gap 1e-6 --scenarios-per-bundle=${SPB} --iter0-mipgap=0.0 --iterk-mipgap=0.0 --solver-options="MAXNODE=10000" --surrogate-nonant
# --solver-log-dir=./sslp_logs
cd ..
