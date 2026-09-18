# Diagnostic for issue #869; not for merging.
#
# Same pattern as mpi_one_sided_test.py, but it reports instead of asserting:
# rank 0 fills its window with 3.0 and then goes quiet for QUIET seconds,
# either sleeping (no MPI calls) or polling Iprobe (MPI calls, no RMA).
# Rank 1 waits 1 s, does Lock/Get/Unlock on rank 0's window, and prints how
# long each call took and what it read.  After the quiet period rank 0
# overwrites its window with arange, so the value read says when the Get
# was served:
#   3.0 -> served while rank 0 was quiet (passive target progressed)
#   9.0 -> served only after rank 0 woke up and made MPI calls
#   1.0 -> rank 1's buffer untouched (the Get delivered nothing)
import sys
import time

import mpi4py.MPI as mpi
import numpy as np

QUIET = 3.0
N = 10


def main():
    mode = sys.argv[1]            # sleep | poll
    lock_name = sys.argv[2]       # exclusive | shared
    lock_type = {"exclusive": mpi.LOCK_EXCLUSIVE,
                 "shared": mpi.LOCK_SHARED}[lock_name]
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()

    win = mpi.Win.Allocate(mpi.DOUBLE.size * N, mpi.DOUBLE.size, comm=comm)
    wbuf = np.ndarray(buffer=win.tomemory(), dtype="d", shape=(N,))
    if rank == 0:
        wbuf[:] = 3.0
    comm.Barrier()
    t0 = mpi.Wtime()

    if rank == 0:
        if mode == "sleep":
            time.sleep(QUIET)
        else:
            while mpi.Wtime() - t0 < QUIET:
                comm.Iprobe()
                time.sleep(0.001)
        t_woke = mpi.Wtime() - t0
        wbuf[:] = np.arange(N)
        comm.send(t_woke, dest=1)
    elif rank == 1:
        buff = np.ones(N, dtype="d")
        time.sleep(1.0)
        t1 = mpi.Wtime()
        win.Lock(0, lock_type)
        t2 = mpi.Wtime()
        win.Get((buff, N, mpi.DOUBLE), target_rank=0)
        t3 = mpi.Wtime()
        win.Unlock(0)
        t4 = mpi.Wtime()
        t_woke = comm.recv(source=0)
        last = buff[-1]
        meaning = {3.0: "served while target quiet",
                   9.0: "served only after target woke",
                   1.0: "Get delivered nothing"}.get(last, "unexpected value")
        print(f"PROBE mode={mode} lock={lock_name} "
              f"start={t1 - t0:.3f} lock={t2 - t1:.3f} get={t3 - t2:.3f} "
              f"unlock={t4 - t3:.3f} done_at={t4 - t0:.3f} "
              f"target_woke_at={t_woke:.3f} read={last} -> {meaning}",
              flush=True)

    comm.Barrier()
    del wbuf
    win.Free()


if __name__ == "__main__":
    main()
