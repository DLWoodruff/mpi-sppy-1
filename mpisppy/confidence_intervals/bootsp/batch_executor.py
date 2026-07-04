###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Rank grouping for the bootstrap batch solves (design section 9.4).

A bootstrap run has two things that want the MPI ranks and pull in opposite
directions: a *solve* wants ranks arranged as a hub plus spokes (a wheel),
while the bootstrap is embarrassingly parallel across its ``nB`` batches and
wants ranks as independent batch workers. The reconciliation is a single knob
``K`` (``--boot-ranks-per-batch``): a batch is solved by a *group* of ``K``
ranks, and the ``R`` ranks are partitioned into ``G = R // K`` groups that run
concurrently, each solving its share of the batches in sequence, with the
results gathered to global rank 0.

The two endpoints are the same mechanism:

* ``K = 1`` -> ``G = R``: every rank is its own group and a batch is a direct
  extensive form. This reproduces the original standalone behavior exactly, so
  ``user_boot`` / ``simulate_boot`` are unaffected.
* ``K = R`` -> ``G = 1``: one group of all ranks solves the batches in sequence,
  each by a full wheel on the whole communicator. This is the development
  checkpoint for the ``K > 1`` cylinders path.

This module owns only the rank arithmetic and the collectives. The two solves a
batch needs -- the optimal (outer-bound) solve and the xhat evaluation -- are
supplied by the caller: the xhat evaluation is just an ``Xhat_Eval`` on the
group communicator (scenarios spread across the group's ranks), and the optimal
solve is a direct EF for ``K = 1`` or a wheel (via ``batch_optimal_solver``) for
``K > 1``.
"""

import numpy as np

import mpisppy.MPI as MPI


def slice_lens_for(nB, nslices):
    """Split ``nB`` items into ``nslices`` contiguous shares as evenly as possible.

    Returns a list of length ``nslices`` summing to ``nB`` (the historical
    ``boot_sp.slice_lens`` split, generalized to an arbitrary slice count so it
    can apportion batches over groups instead of over individual ranks).
    """
    avg = nB / nslices
    lens = [int((i + 1) * avg) - int(i * avg) for i in range(nslices)]
    assert sum(lens) == nB
    return lens


class BatchExecutor:
    """Partition ``comm`` into ``G = size // K`` groups of ``K`` ranks.

    Attributes (all valid on every rank unless noted):
        K (int): ranks per group (the group size).
        n_groups (int): G, the number of groups.
        group_index (int): this rank's group, 0..G-1.
        group_rank (int): this rank's position within its group, 0..K-1.
        is_group_leader (bool): group_rank == 0 (the rank that reports the
            group's batch results into the cross-group gather).
        is_root (bool): global rank 0 -- where the estimator does its analysis.
        groupcomm (MPI comm): the K ranks of this group; the communicator a
            batch solve / xhat evaluation runs on.
        leadercomm (MPI comm): the G group leaders (MPI.COMM_NULL on non-leaders
            under real MPI); the communicator the batch results are gathered on.
        batch_optimal_solver (callable or None): for K > 1, a callable
            ``(scenario_names, sample_mapping, groupcomm) -> outer_bound`` that
            runs a wheel on the group; None selects the K = 1 direct-EF path.
    """

    def __init__(self, K=1, comm=None, batch_optimal_solver=None):
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        R = comm.Get_size()
        rank = comm.Get_rank()

        if K is None or K < 1:
            raise ValueError(f"boot_ranks_per_batch (K) must be a positive integer, got {K}")
        if K > R:
            raise ValueError(
                f"boot_ranks_per_batch (K={K}) exceeds the number of MPI ranks "
                f"({R}); a group cannot have more ranks than the world.")
        # Leftover-rank handling (the R mod K ranks that would sit out) is a
        # scheduled refinement; for now the partition must be exact so no rank
        # is silently idle. The two checkpoints (K=1 and K=R) both divide.
        if R % K != 0:
            raise ValueError(
                f"boot_ranks_per_batch (K={K}) must divide the number of MPI "
                f"ranks ({R}) evenly; got remainder {R % K}. Choose K in "
                f"{[k for k in range(1, R + 1) if R % k == 0]}.")

        self.K = K
        self.n_groups = R // K
        self.group_index = rank // K
        self.group_rank = rank % K
        self.is_group_leader = (self.group_rank == 0)
        self.is_root = (rank == 0)
        self.batch_optimal_solver = batch_optimal_solver

        # The group communicator: K ranks cooperate on one batch here. At K=1
        # this is a private single-rank comm (the old rankcomm); at K=R it is
        # the whole world (the G=1 checkpoint).
        self.groupcomm = comm.Split(color=self.group_index, key=self.group_rank)

        # The leader communicator: one rank per group, used only to gather the
        # per-group batch results. Non-leaders are excluded (COMM_NULL under
        # real MPI) and never touch it. At K=1 every rank is a leader, so this
        # is the whole world -- exactly the old COMM_WORLD gather.
        leader_color = 0 if self.is_group_leader else MPI.UNDEFINED
        self.leadercomm = comm.Split(color=leader_color, key=self.group_index)

    @property
    def uses_cylinders(self):
        """True when a batch is solved by a wheel (K > 1) rather than an EF."""
        return self.K > 1

    def batch_share(self, nB):
        """This group's number of batches (all ranks in a group agree)."""
        return slice_lens_for(nB, self.n_groups)[self.group_index]

    def group_seed(self, seed_offset):
        """RNG seed for this group.

        All ranks in a group share it so they resample the *same* batches (and
        then cooperate on solving them); distinct groups get distinct seeds so
        they cover different batches. At K=1 this is ``seed_offset + rank``, the
        original per-rank seeding.
        """
        return seed_offset + self.group_index

    def gather(self, local, nB, item_len=1):
        """Gather each group's local results into the full array.

        ``local`` is this group's ``batch_share(nB) * item_len`` results
        (identical on all ranks of the group; only the leader contributes).
        ``item_len`` is the number of floats each batch contributes -- 1 for a
        scalar per batch, or e.g. ``sample_size`` for the bagging count rows.
        Returns the assembled length-``nB * item_len`` array on global rank 0
        and None elsewhere.
        """
        local = np.ascontiguousarray(local, dtype=np.float64)
        if not self.is_group_leader:
            return None
        lenlist = [ell * item_len for ell in slice_lens_for(nB, self.n_groups)]
        if self.is_root:
            full = np.empty(nB * item_len, dtype=np.float64)
        else:
            full = None
        self.leadercomm.Gatherv(sendbuf=local, recvbuf=(full, lenlist), root=0)
        return full
