###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Unequal-rank integration test for an *asynchronous* sender: the APH hub.

The earlier flexible-rank integration tests all use a synchronous hub (PH /
FWPH), where every hub rank finishes an iteration and writes its buffers in
lockstep, so the several remote ranks a spoke assembles a per-scenario field
from always carry the same write_id. An APH hub does not: each hub rank runs
its own iteration loop and calls sync_with_spokes (hence put_send_buffer, which
bumps that rank's write_id) at its own pace, so the sources of a multi-source
read can sit at different write_ids. This is the case the per-field coherence
policy was designed for, and Phase 5 of the flexible-rank work is to verify it
*composes* with an async sender rather than merely assert it.

Two regimes are exercised:

  * Deterministic (aph_frac_needed = 1.0). The projective step waits for every
    rank, so the APH math is reproducible and the assembled fields are the same
    regardless of the rank split. We can therefore pin both correctness and
    cross-split agreement, exactly as the synchronous tests do:
      - relaxed NONANTS_VALS (xhatshuffle spoke): the reassembled per-scenario
        nonants must drive the incumbent to the extensive-form optimum, and
        4+2 must agree with 2+4;
      - strict DUALS (Lagrangian spoke): the reassembled W must yield a valid
        (and, here, rank-split-independent) Lagrangian bound.

  * Genuinely asynchronous (aph_frac_needed = 0.5). The hub proceeds on a
    fraction of its ranks, so the sources of a read are routinely at mixed
    write_ids. We do not pin tightness or cross-split agreement (the accepted
    snapshots now depend on timing); we pin the two properties the coherence
    model promises under async and that a broken policy would violate:
      - no deadlock (the run terminates), even though strict reads reject and
        retry whenever their sources disagree; and
      - never an *invalid* result -- the Lagrangian bound never exceeds the
        optimum (a mixed-iteration W would break this), and the xhatshuffle
        incumbent is a genuine feasible point (>= the optimum). A strict read
        that never finds a coherent snapshot simply makes no progress (bound
        stays at the initial infinity), which is honest, not wrong.

mpiexec -np 6 python -m mpi4py -m pytest mpisppy/tests/test_flexible_rank_aph.py
"""

import unittest

from mpi4py import MPI

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver

comm = MPI.COMM_WORLD

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

# Extensive-form optimum for farmer NUM_SCENS=10 (see test_flexible_rank_cylinders).
EF_OPT = -122146.70


def _build_dicts(spoke_kind, num_scens, max_iterations, hub_ratio, spoke_ratio,
                 frac_needed):
    # Fresh cfg/dicts per run: WheelSpinner.run mutates the dicts (key
    # renaming) and may only run once, so each run needs its own.
    cfg = config.Config()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.xhatshuffle_args()
    cfg.lagrangian_args()
    cfg.solver_name = solver_name
    cfg.max_solver_threads = 2
    cfg.num_scens = num_scens
    cfg.default_rho = 1.0
    cfg.max_iterations = max_iterations
    cfg.rel_gap = 1e-6
    cfg.aph_frac_needed = frac_needed
    cfg.aph_sleep_seconds = 0.01

    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = farmer.scenario_names_creator(num_scens)
    scenario_creator_kwargs = farmer.kw_creator(cfg)
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    hub_dict = vanilla.aph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
    hub_dict["rank_ratio"] = hub_ratio

    if spoke_kind == "xhatshuffle":
        spoke = vanilla.xhatshuffle_spoke(
            *beans, scenario_creator_kwargs=scenario_creator_kwargs
        )
    elif spoke_kind == "lagrangian":
        spoke = vanilla.lagrangian_spoke(
            *beans, scenario_creator_kwargs=scenario_creator_kwargs
        )
    else:
        raise ValueError(spoke_kind)
    spoke["rank_ratio"] = spoke_ratio

    return hub_dict, [spoke]


def _run(spoke_kind, num_scens, max_iterations, hub_ratio, spoke_ratio,
         frac_needed):
    hub_dict, spoke_list = _build_dicts(
        spoke_kind, num_scens, max_iterations, hub_ratio, spoke_ratio, frac_needed
    )
    wheel = WheelSpinner(hub_dict, spoke_list)
    wheel.spin()
    if wheel.global_rank == 0:
        payload = (wheel.BestInnerBound, wheel.BestOuterBound)
    else:
        payload = None
    return comm.bcast(payload, root=0)


@unittest.skipUnless(comm.size == 6, "needs exactly 6 MPI ranks")
@unittest.skipUnless(solver_available, "no solver is available")
class TestFlexibleRankAPH(unittest.TestCase):

    NUM_SCENS = 10
    MAX_ITERS = 50

    # ---- deterministic regime (aph_frac_needed = 1.0) --------------------

    def test_relaxed_nonants_both_directions_agree_and_optimal(self):
        # APH hub + xhatshuffle spoke: only the relaxed field NONANTS_VALS
        # crosses cylinders (hub -> spoke, multi-source). The async hub's
        # per-rank write_id drift is absorbed by relaxed coherence (floor over
        # sources); the incumbent must still reach the EF optimum both ways.
        ib_42, _ = _run("xhatshuffle", self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5, 1.0)
        ib_24, _ = _run("xhatshuffle", self.NUM_SCENS, self.MAX_ITERS, 1.0, 2.0, 1.0)

        self.assertTrue(abs(ib_42) < float("inf"))
        self.assertTrue(abs(ib_24) < float("inf"))
        self.assertLess(
            abs(ib_42 - ib_24), 1e-3 * abs(ib_42),
            msg=f"4+2 inner bound {ib_42} disagrees with 2+4 {ib_24}",
        )
        for ib in (ib_42, ib_24):
            self.assertLess(
                abs(ib - EF_OPT), 1e-2 * abs(EF_OPT),
                msg=f"inner bound {ib} is not near the EF optimum {EF_OPT}",
            )

    def test_strict_duals_both_directions_agree_and_valid(self):
        # APH hub + Lagrangian spoke: DUALS (the W_s multipliers) crosses
        # cylinders and uses strict coherence. With frac_needed=1.0 the hub's W
        # trajectory is rank-split-independent, so the reassembled W -- and the
        # bound it yields -- must match across splits and be a valid lower
        # bound.
        _, ob_42 = _run("lagrangian", self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5, 1.0)
        _, ob_24 = _run("lagrangian", self.NUM_SCENS, self.MAX_ITERS, 1.0, 2.0, 1.0)

        self.assertTrue(abs(ob_42) < float("inf"))
        self.assertTrue(abs(ob_24) < float("inf"))
        self.assertLess(
            abs(ob_42 - ob_24), 1e-3 * abs(ob_42),
            msg=f"4+2 Lagrangian bound {ob_42} disagrees with 2+4 {ob_24}",
        )
        self.assertLessEqual(ob_42, EF_OPT + 1e-4 * abs(EF_OPT))
        self.assertLessEqual(ob_24, EF_OPT + 1e-4 * abs(EF_OPT))

    # ---- genuinely asynchronous regime (aph_frac_needed = 0.5) -----------

    def test_async_strict_duals_no_deadlock_and_valid(self):
        # The projective step proceeds on half the hub ranks, so the sources of
        # a DUALS read are routinely at mixed write_ids -- strict coherence then
        # rejects and the spoke retries. The run must still terminate (no
        # deadlock) and never produce an *invalid* bound. We do NOT assert the
        # bound is finite or matches the deterministic value: which coherent W
        # snapshots a read happens to catch is timing-dependent, and a strict
        # read that never finds one simply makes no progress (bound = -inf),
        # which is honest. A mixed-iteration W, by contrast, could exceed the
        # optimum -- that is what this guards against.
        _, ob = _run("lagrangian", self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5, 0.5)
        self.assertLessEqual(ob, EF_OPT + 1e-4 * abs(EF_OPT))

    def test_async_relaxed_nonants_no_deadlock_and_feasible(self):
        # Same async regime, relaxed field: the run must terminate and the
        # xhatshuffle incumbent must be a genuine feasible point, i.e. an upper
        # bound on the (minimizing) optimum. Relaxed coherence accepts the
        # mixed-iteration nonants; the spoke re-evaluates each candidate, so a
        # stale/blended candidate is still honestly evaluated.
        ib, _ = _run("xhatshuffle", self.NUM_SCENS, self.MAX_ITERS, 1.0, 0.5, 0.5)
        self.assertTrue(abs(ib) < float("inf"))
        self.assertGreaterEqual(ib, EF_OPT - 1e-4 * abs(EF_OPT))


if __name__ == "__main__":
    unittest.main()
