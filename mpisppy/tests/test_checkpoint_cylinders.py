###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The A/B checkpoint harness on cylinders (design section 11.1, phase 4).

The serial harness in ``test_checkpoint.py`` proves the mechanism. This proves
the thing people actually run: a hub with spokes, stopped and resumed as
separate MPI jobs.

Each leg is its own ``mpiexec`` job (see ``cylinders_ab_driver.py``), because
that is both what the design's acceptance gate asks for and what a stopped
study really does -- the resume is a new job tomorrow, sharing nothing with
today's but the checkpoint directory.

Three properties, and they are different claims:

* **The hub's primal trajectory is untouched by the stop.** Farmer is a
  deterministic LP and the hub's PH iterate does not depend on the spokes, so
  the resumed run must land bit-identically on the uninterrupted run's state.
* **The best solution survives.** That one does not live on the hub -- the
  xhat spoke holds it -- so it is preserved by the spoke's own incumbent file
  rather than by the hub checkpoint.
* **Bounds stay valid**, in the best-so-far sense the determinism contract
  (section 7) promises, rather than being reproduced exactly.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRIVER = os.path.join(_HERE, "cylinders_ab_driver.py")
#: generic_cylinders resolves --module-name with importlib, so this is the
#: dotted name rather than a path -- and being importable makes the legs
#: independent of the directory mpiexec starts in.
_FARMER = "mpisppy.tests.examples.farmer"

mpiexec_available = shutil.which("mpiexec") is not None


class _ResumeABMixin:
    """The three A/B legs and what they must show, per configuration."""

    #: One rank per cylinder. Multi-rank cylinders are refused at setup until
    #: phase 2, so this is the whole supported space for now.
    NP = 3
    N = 5
    STOP = 2
    #: Set by the concrete cases below.
    MODULE = None
    MODEL_ARGS = ()
    SPOKE_ARGS = ()
    INCUMBENT_SPOKE = None

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _leg(self, name, *extra_args):
        """Run one mpiexec job and return the hub's snapshot."""
        out_path = os.path.join(self._tmp.name, f"{name}.json")
        cmd = [
            "mpiexec", "-np", str(self.NP),
            sys.executable, "-m", "mpi4py", _DRIVER,
            "--out", out_path,
            "--module-name", self.MODULE,
            *self.MODEL_ARGS,
            "--solver-name", solver_name,
            *self.SPOKE_ARGS,
            # The comparison needs both legs to run the same iterations, so
            # every early exit the cylinders add has to be off: no
            # inter-cylinder convergence, no gap-based termination.
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            *extra_args,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=1800, check=False)
        self.assertEqual(
            result.returncode, 0,
            msg=f"leg {name!r} failed:\n{result.stdout[-4000:]}\n"
                f"{result.stderr[-4000:]}")
        self.assertTrue(
            os.path.exists(out_path),
            msg=f"leg {name!r} wrote no hub snapshot:\n{result.stdout[-4000:]}")
        # Checkpoint write failures warn and continue by design, so a broken
        # write does not fail a leg -- it just prints. A resume-only leg
        # attempting a write it has nowhere to put got all the way through
        # the harness this way once.
        self.assertNotIn(
            "could not write its incumbent", result.stdout,
            msg=f"leg {name!r} failed to write a spoke incumbent")
        self.assertNotIn(
            "WARNING: checkpoint write failed", result.stdout,
            msg=f"leg {name!r} failed to write a hub checkpoint")
        with open(out_path) as f:
            snapshot = json.load(f)
        # Whatever the spokes reported about their own restore.
        snapshot["spokes"] = []
        for fname in sorted(os.listdir(self._tmp.name)):
            if fname.startswith(f"{name}.json.spoke"):
                with open(os.path.join(self._tmp.name, fname)) as f:
                    snapshot["spokes"].append(json.load(f))
        return snapshot

    def _run_ab(self):
        reference = self._leg("A", "--max-iterations", str(self.N))
        stopped = self._leg("B1", "--max-iterations", str(self.STOP),
                            "--checkpoint-dir", self.ckpt_dir)
        # --max-iterations bounds this run, so leg B2 asks for the iterations
        # B1 did not do rather than for the study total.
        resumed = self._leg("B2", "--max-iterations", str(self.N - self.STOP),
                            "--resume-from", self.ckpt_dir)
        return reference, stopped, resumed

    def test_cylinders_resume_matches_an_uninterrupted_run(self):
        reference, stopped, resumed = self._run_ab()

        # Without this the rest proves only that farmer is deterministic: a
        # leg that ignored the checkpoint and ran all N iterations from
        # scratch would land on the same state and pass everything below.
        self.assertTrue(resumed["resumed"],
                        msg="the third leg started from scratch")
        self.assertEqual(resumed["resume_iteration"], self.STOP)
        self.assertEqual(stopped["iteration"], self.STOP)
        self.assertEqual(resumed["iteration"], self.N)

        want, got = reference["state"], resumed["state"]
        self.assertEqual(set(want), set(got))
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        self.assertEqual(
            worst, 0.0,
            msg=f"the resumed hub differs from the uninterrupted one by "
                f"{worst}; farmer is a deterministic LP and the hub's primal "
                f"trajectory does not depend on the spokes, so this should be "
                f"bit-identical")
        self.assertEqual(reference["trivial_bound"], resumed["trivial_bound"])

    def test_the_spoke_incumbent_survives_the_stop(self):
        """The answer lives on the xhat spoke, not in the hub checkpoint."""
        _, stopped, resumed = self._run_ab()

        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        written = os.listdir(spokes_dir)
        self.assertTrue(
            any(f.startswith(f"spoke_{self.INCUMBENT_SPOKE}") for f in written),
            msg=f"the xhat spoke checkpointed no incumbent: {written}")

        # The spoke has to say it restored. Comparing bounds alone proves
        # nothing here: farmer is deterministic, so a spoke that read nothing
        # would re-find the same incumbent within an iteration or two and the
        # comparison below would pass regardless.
        restored = [s for s in resumed["spokes"]
                    if s["restored_incumbent_obj"] is not None]
        self.assertTrue(
            restored,
            msg=f"no spoke restored an incumbent: {resumed['spokes']}")
        self.assertEqual(
            restored[0]["restored_incumbent_obj"], stopped["BestInnerBound"],
            msg="the spoke restored something other than the incumbent its "
                "own checkpoint held")

        # Minimization: a smaller inner bound is a better incumbent, and the
        # resumed run must not report a worse one than the leg it resumed.
        self.assertLessEqual(
            resumed["BestInnerBound"], stopped["BestInnerBound"],
            msg="the resumed run reports a worse incumbent than its "
                "checkpoint; the spoke's best xhat was lost")

    def test_bounds_stay_valid_after_a_resume(self):
        """Best-so-far, not reproduced: bound timing is spoke-dependent."""
        _, _, resumed = self._run_ab()
        self.assertLessEqual(
            resumed["BestOuterBound"], resumed["BestInnerBound"],
            msg="the resumed run's outer bound crossed its incumbent")
        self.assertLessEqual(
            resumed["best_bound_obj_val"], resumed["BestInnerBound"],
            msg="the hub's restored best bound crossed the incumbent")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerCylindersResumeAB(_ResumeABMixin, unittest.TestCase):
    """PH hub + lagrangian + xhatshuffle on the deterministic-LP baseline."""

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "3", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle")
    INCUMBENT_SPOKE = "XhatShuffleInnerBound"


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestStochAdmmCylindersResumeAB(_ResumeABMixin, unittest.TestCase):
    """The same harness on a stoch-ADMM configuration.

    Worth its own case because the ADMM wrapper is where checkpointing has
    the most to get wrong (design section 8.2): scenario names are wrapped,
    so they reach the checkpoint's file names; the wrapper holds its own
    references to the models a resume replaces; and the variable-probability
    mask and fixed-at-0 dummy vars have to survive the dill round-trip. FWPH
    is not in the spoke set because it does not support variable probability.
    """

    MODULE = "mpisppy.tests.examples.stoch_distr.stoch_distr"
    MODEL_ARGS = ("--stoch-admm", "--num-stoch-scens", "4",
                  "--num-admm-subproblems", "2", "--default-rho", "10")
    SPOKE_ARGS = ("--lagrangian", "--xhatxbar")
    INCUMBENT_SPOKE = "XhatXbarInnerBound"


if __name__ == "__main__":
    unittest.main()
