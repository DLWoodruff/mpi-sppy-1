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
    cfg.data_file = "schultz_data.csv"
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

    def test_boot_solver_role_resolution(self):
        # the batch ("boot") solver role resolves its own name and options, and
        # falls back to the generic solver_name when --boot-solver-name is unset
        import schultz_data
        pool = schultz_data.scenario_names_creator(3)

        cfg = _make_cfg("Classical_quantile")
        cfg.boot_solver_name = "myboot_solver"
        cfg.boot_solver_options = "mipgap=0.01"
        boot_cfg = _estimator_cfg("schultz_data", schultz_data, cfg, 3, 0, pool)
        self.assertEqual(boot_cfg.solver_name, "myboot_solver")
        self.assertIn("mipgap", boot_cfg.solver_options)

        cfg2 = _make_cfg("Classical_quantile")   # no boot_solver_name
        boot_cfg2 = _estimator_cfg("schultz_data", schultz_data, cfg2, 3, 0, pool)
        self.assertEqual(boot_cfg2.solver_name, solver_name)

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

    def test_do_boot_ranks_per_batch_gt_1_raises(self):
        cfg = _make_cfg("Classical_quantile")
        cfg.boot_candidate_sample_size = 5
        cfg.num_scens = 5
        cfg.boot_ranks_per_batch = 2
        with self.assertRaises(ValueError):
            do_boot(MODULE_NAME, cfg, wheel=_FakeWheel())


if __name__ == '__main__':
    unittest.main()
