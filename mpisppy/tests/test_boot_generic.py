###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Tests for the generic_cylinders bootstrap CI integration (mpisppy/generic/
# boot.py: do_boot / boot_requested), using the deterministic schultz_data
# dataset example.
#
# Serial:
#   python -m pytest mpisppy/tests/test_boot_generic.py
# Parallel (exercises the Gatherv batch split across ranks):
#   mpiexec -np 2 python -m mpi4py mpisppy/tests/test_boot_generic.py
#
# schultz_data has integer data that is a function of the dataset row, so the
# extensive-form optima and (numpy-seeded) bootstrap draws are the same for
# every correct solver. Each rank seeds its own bootstrap stream, so the
# assembled rank-0 result depends on the number of ranks; the locked values
# below are keyed by comm size and compared with round_pos_sig.

import os
import sys
import tempfile
import unittest

import mpisppy.utils.config as config
import mpisppy.utils.sputils as sputils
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.bootsp.boot_utils as boot_utils
from mpisppy.tests.utils import get_solver, round_pos_sig
from mpisppy.generic.boot import do_boot, boot_requested, _estimator_cfg

sputils.disable_tictoc_output()

solver_available, solver_name, persistent_available, persistent_solver_name = get_solver()

comm = boot_utils.comm
n_proc = boot_utils.n_proc
my_rank = boot_utils.my_rank

module_dir = os.path.dirname(os.path.abspath(__file__))
data_example_dir = os.path.join(
    module_dir, "..", "..", "examples", "bootsp", "schultz_data")
if not os.path.exists(data_example_dir):
    raise RuntimeError(f"Directory not found: {data_example_dir}")
if data_example_dir not in sys.path:
    sys.path.insert(0, data_example_dir)

MODULE_NAME = "schultz_data"

# A bootstrap run now requires --boot-batch-config-file (a file of
# generic_cylinders flags for the batch solves). For the K=1 tests it need only
# name the solver. One file per rank avoids a write race under mpiexec.
_batch_cfg_fd, BATCH_CFG_PATH = tempfile.mkstemp(prefix=f"bootbatch{my_rank}", suffix=".txt")
with os.fdopen(_batch_cfg_fd, "w") as _f:
    _f.write(f"--solver-name {solver_name or 'gurobi'}\n")

# For the K>1 (wheel-per-batch) test: a subgradient hub is a single cylinder
# (works at any rank count) that yields an outer (dual) bound on each batch.
_wheel_cfg_fd, WHEEL_BATCH_CFG_PATH = tempfile.mkstemp(prefix=f"bootwheel{my_rank}", suffix=".txt")
with os.fdopen(_wheel_cfg_fd, "w") as _f:
    _f.write(f"--solver-name {solver_name or 'gurobi'}\n"
             "--subgradient-hub\n--max-iterations 5\n--default-rho 1.0\n")

# A fixed, feasible candidate solution so the CI depends only on the
# (deterministic) bootstrap draws over the dataset rows, not on which optimum a
# given solver returns.
XHAT = {"ROOT": [0.0, 3.0]}

# do_boot ci_gap (Classical_quantile, seed_offset=100), keyed by number of ranks.
# FILE path: M=0, pool = rows 0..99.  WHEEL path: M=5, disjoint pool = rows 5..104.
locked_ci_gap_file = {
    1: [0.5924999999999517, 3.396999999999958],
    2: [1.0749999999999584, 2.919499999999952],
}
locked_ci_gap_wheel = {
    1: [1.4749999999999504, 3.7364999999999475],
    2: [1.4749999999999504, 3.2804999999999445],
}
# center (point) estimate of the gap is rank-count independent
locked_center_gap_file = 2.139999999999958
locked_center_gap_wheel = 2.5799999999999557


class _FakeWheel:
    """Stand-in for a WheelSpinner that writes a known xhat (the configured
    solve) so the wheel path is deterministic in tests."""
    def write_first_stage_solution(self, path, first_stage_solution_writer=None):
        ciutils.write_xhat(XHAT, path=path)


def _make_cfg(method="Classical_quantile"):
    """Build a generic_cylinders-style cfg with the boot_* option group set."""
    cfg = config.Config()
    cfg.add_to_config("solver_name", description="solver", domain=str, default=None)
    cfg.add_to_config("branching_factors", description="bf", domain=None, default=None)
    cfg.boot_args()
    import schultz_data
    schultz_data.inparser_adder(cfg)  # adds num_scens and data_file
    cfg.solver_name = solver_name
    cfg.boot_method = method
    cfg.boot_sample_size = 100
    cfg.boot_subsample_size = 20
    cfg.boot_nB = 20
    cfg.boot_alpha = 0.1
    cfg.boot_seed_offset = 100
    cfg.boot_batch_config_file = BATCH_CFG_PATH
    cfg.data_file = "schultz_data.csv"
    return cfg


def _make_small_cfg(K, batch_cfg_path):
    """A cheap cfg for the wheel-vs-EF comparison: a small pool and few batches
    so the per-batch wheel solves stay fast."""
    cfg = _make_cfg("Classical_quantile")
    cfg.boot_sample_size = 8
    cfg.boot_subsample_size = 4
    cfg.boot_nB = 4
    cfg.boot_batch_config_file = batch_cfg_path
    cfg.boot_ranks_per_batch = K
    return cfg


#*****************************************************************************
class Test_boot_generic(unittest.TestCase):
    """Test do_boot and boot_requested through the generic_cylinders surface.

    The value tests call do_boot on all ranks (so the MPI collectives line up)
    and only assert on rank 0.
    """

    def _assert_gap(self, res, locked, locked_center):
        if my_rank == 0:
            ci_gap = list(res[2])
            center_gap = res[5]
            self.assertLessEqual(ci_gap[0], ci_gap[1])
            self.assertEqual(round_pos_sig(center_gap, 4),
                             round_pos_sig(locked_center, 4))
            if n_proc in locked:
                for g, e in zip(ci_gap, locked[n_proc]):
                    self.assertEqual(round_pos_sig(g, 4), round_pos_sig(e, 4))
        else:
            self.assertEqual(res, (None, None, None, None, None, None))

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_do_boot_file_value(self):
        # xhat read from a file (the no-wheel path); pool = the first N records.
        xf = tempfile.mkstemp(prefix=f"xhat{my_rank}", suffix=".npy")[1]
        ciutils.write_xhat(XHAT, path=xf)
        try:
            cfg = _make_cfg("Classical_quantile")
            cfg.boot_xhat_input_file_name = xf
            self.assertTrue(boot_requested(cfg))
            res = do_boot(MODULE_NAME, cfg)
            self._assert_gap(res, locked_ci_gap_file, locked_center_gap_file)
        finally:
            if os.path.exists(xf):
                os.remove(xf)

    @unittest.skipIf(not solver_available, "no solver is available")
    def test_do_boot_wheel_disjoint(self):
        # xhat from the (fake) wheel; the M candidate records are held disjoint
        # from the N-record resampling pool, so the CI differs from the file
        # path above (which pools rows 0..99).
        cfg = _make_cfg("Classical_quantile")
        cfg.boot_candidate_sample_size = 5
        cfg.num_scens = 5
        res = do_boot(MODULE_NAME, cfg, wheel=_FakeWheel())
        self._assert_gap(res, locked_ci_gap_wheel, locked_center_gap_wheel)

    def test_batch_solver_from_config_file(self):
        # the estimator's solver name and options come from the batch config
        # file, not the xhat-solve command line (PR-4 retired --boot-solver-*).
        from mpisppy.generic import boot_batch
        import schultz_data
        pool = schultz_data.scenario_names_creator(3)

        bf = tempfile.mkstemp(suffix=".txt")[1]
        with open(bf, "w") as f:
            f.write("--solver-name myboot_solver --solver-options mipgap=0.01\n")
        try:
            batch_cfg = boot_batch.parse_batch_config_file(bf, schultz_data)
            cfg = _make_cfg("Classical_quantile")
            boot_cfg = _estimator_cfg("schultz_data", schultz_data, cfg, batch_cfg, 3, 0, pool)
            self.assertEqual(boot_cfg.solver_name, "myboot_solver")
            self.assertIn("mipgap", boot_cfg.solver_options)
        finally:
            if os.path.exists(bf):
                os.remove(bf)

    def test_boot_requested_requires_batch_config(self):
        # a bootstrap run must name a batch config file
        cfg = _make_cfg("Classical_quantile")
        cfg.boot_batch_config_file = None
        with self.assertRaises(ValueError):
            boot_requested(cfg)

    @unittest.skipIf(not solver_available, "no solver is available")
    @unittest.skipIf(n_proc < 2, "the K>1 wheel path needs at least 2 ranks")
    def test_do_boot_g1_wheel_matches_ef(self):
        # The G=1 checkpoint (design 9.4): K = n_proc, so one group of all ranks
        # solves each batch by a wheel in sequence. Compare that wheel path to
        # the K=1 direct-EF path over the *same* pool (both deterministic, file
        # xhat). The value at xhat (center_upper) is solver-exact and does not
        # depend on K, so it must match exactly; the wheel's optimal is an outer
        # (dual) bound, so it must sit at or below the EF optimum, making the
        # reported gap conservative (>= the EF gap).
        xf = tempfile.mkstemp(prefix=f"xhatw{my_rank}", suffix=".npy")[1]
        ciutils.write_xhat(XHAT, path=xf)
        try:
            cfg_ef = _make_small_cfg(K=1, batch_cfg_path=BATCH_CFG_PATH)
            cfg_ef.boot_xhat_input_file_name = xf
            res_ef = do_boot(MODULE_NAME, cfg_ef)

            cfg_wheel = _make_small_cfg(K=n_proc, batch_cfg_path=WHEEL_BATCH_CFG_PATH)
            cfg_wheel.boot_xhat_input_file_name = xf
            res_wheel = do_boot(MODULE_NAME, cfg_wheel)

            if my_rank == 0:
                co_ef, cu_ef, cg_ef = res_ef[3], res_ef[4], res_ef[5]
                co_w, cu_w, cg_w = res_wheel[3], res_wheel[4], res_wheel[5]
                ci_gap_w = list(res_wheel[2])
                tol = 1e-4 * (1 + abs(cu_ef))
                # value at xhat is K-invariant -> exact match
                self.assertAlmostEqual(cu_w, cu_ef, delta=tol)
                # wheel optimal is an outer bound -> at or below the EF optimum
                self.assertLessEqual(co_w, co_ef + tol)
                # so the reported gap over-states (is conservative)
                self.assertGreaterEqual(cg_w, cg_ef - tol)
                # and the CI is ordered and finite
                self.assertLessEqual(ci_gap_w[0], ci_gap_w[1])
                self.assertTrue(all(abs(v) < float("inf") for v in ci_gap_w))
        finally:
            if os.path.exists(xf):
                os.remove(xf)

    def test_boot_requested_none(self):
        cfg = _make_cfg()
        cfg.boot_method = None
        self.assertFalse(boot_requested(cfg))

    def test_boot_requested_smoothed_raises(self):
        cfg = _make_cfg("Smoothed_bagging")
        with self.assertRaises(ValueError):
            boot_requested(cfg)

    def test_boot_requested_exclusivity_raises(self):
        cfg = _make_cfg("Classical_quantile")
        cfg.boot_xhat_input_file_name = "some_xhat.npy"
        cfg.boot_candidate_sample_size = 5
        with self.assertRaises(ValueError):
            boot_requested(cfg)

    def test_do_boot_rejects_mmw(self):
        cfg = _make_cfg("Classical_quantile")
        cfg.mmw_args()
        cfg.mmw_num_batches = 2
        cfg.mmw_batch_size = 10
        cfg.mmw_start = 5
        with self.assertRaises(ValueError):
            do_boot(MODULE_NAME, cfg, wheel=_FakeWheel())

    def test_do_boot_disjoint_overflow_raises(self):
        cfg = _make_cfg("Classical_quantile")
        cfg.boot_candidate_sample_size = 5
        cfg.boot_sample_size = 500   # 5 + 500 > 200 rows
        cfg.num_scens = 5
        with self.assertRaises(ValueError):
            do_boot(MODULE_NAME, cfg, wheel=_FakeWheel())

    def test_do_boot_bad_ranks_per_batch_raises(self):
        # K must divide the number of MPI ranks; serially (1 rank) any K > 1 is
        # invalid (K > R), so the executor rejects it.
        cfg = _make_cfg("Classical_quantile")
        cfg.boot_candidate_sample_size = 5
        cfg.num_scens = 5
        cfg.boot_ranks_per_batch = n_proc + 1
        with self.assertRaises(ValueError):
            do_boot(MODULE_NAME, cfg, wheel=_FakeWheel())


def tearDownModule():
    # remove the per-rank batch config files created at import time
    for path in (BATCH_CFG_PATH, WHEEL_BATCH_CFG_PATH):
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    unittest.main()
