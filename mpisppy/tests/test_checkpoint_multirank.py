###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The A/B checkpoint harness on multi-rank cylinders (design phase 2).

Phase 1a checkpointed a single-rank hub and refused anything else; phase 4 put
that hub in a wheel with spokes, still one rank each. This is the phase that
lets a cylinder span ranks, which is how mpi-sppy is actually run on a cluster,
and it is where a checkpoint stops being one rank's business:

* **A generation spans every rank.** Each rank owns a different slice of the
  scenarios, so the checkpoint is the *set* of per-rank files and is resumable
  only if all of them are there. The manifest must therefore be written after
  the last rank's write, not after rank 0's -- and every rank must resume from
  its own slice. So these tests compare **every** hub rank's state across
  legs, not just rank 0's, which is where a single-rank harness stops seeing
  anything.
* **Failure is collective or it is a hang.** The design has a failed write warn
  and let the run continue; on several ranks that only works if the ranks agree
  about whether the write failed, or the one that gave up leaves the others
  waiting at a barrier forever.
* **The distribution has to be exactly reproduced.** Resuming with a different
  rank count, or with the scenarios landing on different ranks, is refused
  rather than half-restored (section 5.7).

Each leg is its own ``mpiexec`` job (``cylinders_ab_driver.py``), for the
reason the single-rank cylinders harness gives: the design's acceptance gate
asks for the resume to happen in a fresh process, and a stopped study really
does resume as tomorrow's job.

The instances are the ones section 11.1 assigns to this phase: farmer with the
scenarios spread evenly and unevenly, proper bundles (section 8.1), the
``sizes`` MIP, a full multi-rank wheel, and stoch-ADMM (section 8.2), whose
wrapper is where a resume has the most to get wrong.
"""

import json
import os
import pickle
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
_FAILURE_DRIVER = os.path.join(_HERE, "multirank_failure_driver.py")
_DEADLINE_DRIVER = os.path.join(_HERE, "multirank_deadline_driver.py")
#: generic_cylinders resolves --module-name with importlib, so these are dotted
#: names rather than paths -- which also makes the legs independent of the
#: directory mpiexec starts in.
_FARMER = "mpisppy.tests.examples.farmer"
_SIZES = "mpisppy.tests.examples.sizes.sizes"
_STOCH_DISTR = "mpisppy.tests.examples.stoch_distr.stoch_distr"

mpiexec_available = shutil.which("mpiexec") is not None


def _run_leg(tmpdir, name, np, module, model_args, spoke_args, extra_args,
             check=True):
    """Run one mpiexec job. Returns (CompletedProcess, out_path)."""
    out_path = os.path.join(tmpdir, f"{name}.json")
    cmd = [
        "mpiexec", "-np", str(np),
        sys.executable, "-m", "mpi4py", _DRIVER,
        "--out", out_path,
        "--module-name", module,
        *model_args,
        "--solver-name", solver_name,
        *spoke_args,
        # The comparison needs both legs to run the same iterations, so every
        # early exit has to be off: no inter-cylinder convergence, no
        # gap-based termination.
        "--intra-hub-conv-thresh", "-1",
        "--rel-gap", "0.0", "--abs-gap", "0.0",
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600, check=False)
    if check and result.returncode != 0:
        raise AssertionError(
            f"leg {name!r} failed:\n{result.stdout[-4000:]}\n"
            f"{result.stderr[-4000:]}")
    return result, out_path


def _hub_ranks(out_path):
    """Every hub rank's snapshot for a leg, ordered by rank."""
    directory = os.path.dirname(out_path)
    prefix = f"{os.path.basename(out_path)}.hubrank"
    snapshots = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(prefix):
            with open(os.path.join(directory, fname)) as f:
                snapshots.append(json.load(f))
    return sorted(snapshots, key=lambda s: s["cylinder_rank"])


def _spoke_ranks(out_path, cylinder):
    """Every rank's marker for one spoke cylinder, ordered by rank."""
    directory = os.path.dirname(out_path)
    prefix = f"{os.path.basename(out_path)}.cyl"
    markers = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(prefix):
            with open(os.path.join(directory, fname)) as f:
                marker = json.load(f)
            if marker["cylinder"] == cylinder:
                markers.append(marker)
    return markers


def _published_generation(ckpt_dir):
    with open(os.path.join(ckpt_dir, "manifest.json")) as f:
        return json.load(f)


class _MultiRankABMixin:
    """Three legs -- reference, stopped, resumed -- run once per class.

    Once per class rather than once per test: each leg is an mpiexec job, and
    running the whole A/B again for every assertion would multiply the CI cost
    of this file by the number of things it checks without checking anything
    new.
    """

    #: Total ranks. With no spokes the whole job is the hub, so this is the
    #: number of ranks *within* one cylinder -- which is what phase 2 is about.
    NP = 2
    #: Ranks the hub ends up with, once the wheel has split them by cylinder.
    HUB_RANKS = 2
    N = 4
    STOP = 2
    MODULE = None
    MODEL_ARGS = ()
    SPOKE_ARGS = ()
    #: False for MIP instances, where the section 7 contract promises a valid
    #: continuation rather than a reproducible trajectory under default solver
    #: settings.
    BIT_IDENTICAL = True
    #: Relative agreement required of the expected objective when
    #: BIT_IDENTICAL is False. A MIP with alternate optima can be resumed onto
    #: a different one of them, so the per-variable iterate is allowed to
    #: differ; the objective is not allowed to walk away.
    OBJECTIVE_RTOL = 1e-3

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ckpt_dir = os.path.join(cls._tmp.name, "ckpt")

        def leg(name, *extra):
            _, out_path = _run_leg(cls._tmp.name, name, cls.NP, cls.MODULE,
                                   cls.MODEL_ARGS, cls.SPOKE_ARGS, extra)
            return _hub_ranks(out_path)

        cls.reference = leg("A", "--max-iterations", str(cls.N))
        cls.stopped = leg("B1", "--max-iterations", str(cls.STOP),
                          "--checkpoint-dir", cls.ckpt_dir)
        # --max-iterations bounds this run, so leg B2 asks for the iterations
        # B1 did not do rather than for the study total.
        cls.resumed = leg("B2", "--max-iterations", str(cls.N - cls.STOP),
                          "--resume-from", cls.ckpt_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_rank_took_part(self):
        """A hub really did span ranks, and each owns a distinct slice.

        Without this the rest of the file could pass on a run that quietly
        put every scenario on rank 0 -- which is the single-rank case phase 1a
        already covered, wearing a multi-rank command line.
        """
        self.assertEqual(len(self.reference), self.HUB_RANKS)
        for leg in (self.reference, self.stopped, self.resumed):
            owned = [tuple(snap["scenario_names"]) for snap in leg]
            self.assertEqual(len(set(owned)), len(owned),
                             msg=f"ranks share scenarios: {owned}")
            for snap in leg:
                self.assertEqual(snap["n_proc"], self.HUB_RANKS)
                self.assertTrue(snap["scenario_names"],
                                msg="a hub rank owns no scenarios")

    def test_every_rank_resumed(self):
        for snap in self.resumed:
            self.assertTrue(
                snap["resumed"],
                msg=f"rank {snap['cylinder_rank']} started from scratch")
            self.assertEqual(snap["resume_iteration"], self.STOP)
            self.assertEqual(snap["iteration"], self.N)
        for snap in self.stopped:
            self.assertEqual(snap["iteration"], self.STOP)

    def test_the_generation_holds_every_rank_and_is_published_once(self):
        """The manifest names one generation, complete on every rank.

        This is the phase-2 write protocol stated as a file-system fact: a
        manifest published before the last rank finished would name a
        generation missing that rank's leaf file, and the resume above would
        have refused it.
        """
        manifest = _published_generation(self.ckpt_dir)
        self.assertEqual(manifest["generation"], self.STOP)
        self.assertEqual(manifest["n_proc"], self.HUB_RANKS)

        hub_dir = os.path.join(self.ckpt_dir, "hub")
        generations = [d for d in os.listdir(hub_dir) if d.startswith("gen_")]
        self.assertEqual(generations, [f"gen_{self.STOP:04d}"],
                         msg=f"expected exactly one live generation: "
                             f"{sorted(os.listdir(hub_dir))}")

        written = os.listdir(os.path.join(hub_dir, generations[0]))
        for rank in range(self.HUB_RANKS):
            self.assertIn(f"hub_rank_{rank:04d}.pkl", written)
            self.assertTrue(
                any(f.startswith(f"hub_rank_{rank:04d}_scen_")
                    for f in written),
                msg=f"no model files for rank {rank}: {sorted(written)}")

    def test_resume_matches_the_uninterrupted_run_on_every_rank(self):
        """The iterate itself, rank by rank.

        The key-set comparison runs for every instance and is not a formality:
        it is what would catch a resume that re-attached the W or proximal
        terms and so carries duplicated components. The *values* are compared
        only where the determinism contract promises they can be -- see
        ``test_the_objective_agrees_within_tolerance`` for what a MIP under
        default solver settings gets instead.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            rank = got_snap["cylinder_rank"]
            self.assertEqual(want_snap["scenario_names"],
                             got_snap["scenario_names"],
                             msg=f"rank {rank} owns different scenarios")
            want, got = want_snap["state"], got_snap["state"]
            self.assertEqual(set(want), set(got))
            if not self.BIT_IDENTICAL:
                continue
            worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
            self.assertEqual(
                worst, 0.0,
                msg=f"rank {rank} differs from the uninterrupted run by "
                    f"{worst}; this instance is deterministic, so a "
                    f"resume must land bit-identically")

    def test_the_objective_agrees_within_tolerance(self):
        """For a MIP this is the comparison; for an LP it is a corollary.

        A MIP with alternate optima can be resumed onto a different optimal
        solution than the uninterrupted run found, which moves the iterate
        without meaning anything went wrong. What would mean something went
        wrong is the objective walking away, so that is what is pinned.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            want, got = want_snap["objective"], got_snap["objective"]
            self.assertIsNotNone(want)
            scale = max(1.0, abs(want))
            self.assertLessEqual(
                abs(want - got), self.OBJECTIVE_RTOL * scale,
                msg=f"rank {got_snap['cylinder_rank']}: the resumed run's "
                    f"expected objective is {got}, the uninterrupted run's "
                    f"is {want}")

    def test_bounds_stay_valid_after_a_resume(self):
        for snap in self.resumed:
            bound = snap["best_bound_obj_val"]
            incumbent = snap["best_solution_obj_val"]
            if bound is not None and incumbent is not None:
                self.assertLessEqual(
                    bound, incumbent,
                    msg=f"rank {snap['cylinder_rank']}: the restored best "
                        f"bound crossed the incumbent")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """The baseline: a two-rank hub on a deterministic LP, evenly split."""

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerUnevenMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """Five scenarios over two ranks: 3 and 2.

    Worth its own case because an even split hides anything that assumes the
    ranks are interchangeable -- a per-rank file whose name or contents were
    derived from a scenario *count* rather than from the rank's own scenario
    list would still work when every rank holds the same number.
    """

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "5", "--default-rho", "1")

    def test_the_split_really_is_uneven(self):
        sizes = sorted(len(snap["scenario_names"]) for snap in self.reference)
        self.assertEqual(sizes, [2, 3])


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestBundlesMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """Proper bundles across ranks (design section 8.1).

    A proper bundle is a first-class subproblem -- its own entry in
    ``local_scenarios``, its own ``nonant_indices``, its own Pyomo model -- so
    the claim under test is that checkpointing needs no bundle-specific code
    at all. Validating it is also what retires the 2019 "will not work on
    bundles" warning on ``_restore_nonants``.
    """

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "8", "--scenarios-per-bundle", "2",
                  "--default-rho", "1")

    def test_the_subproblems_really_are_bundles(self):
        names = [n for snap in self.reference for n in snap["scenario_names"]]
        self.assertTrue(all(n.startswith("Bundle") for n in names),
                        msg=f"expected bundles, got {names}")

    def test_bundles_are_checkpointed_by_name(self):
        """Each bundle gets its own model file, named after the bundle.

        Bundle names are not scenario names and carry no usable scenario
        index, which is the case ``sputils.extract_num`` would have got wrong
        (section 10).
        """
        gen_dir = os.path.join(self.ckpt_dir, "hub", f"gen_{self.STOP:04d}")
        written = os.listdir(gen_dir)
        for snap in self.reference:
            rank = snap["cylinder_rank"]
            for bundle in snap["scenario_names"]:
                self.assertIn(f"hub_rank_{rank:04d}_scen_{bundle}.dill",
                              written)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestCvarMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """farmer + ``--cvar``: a model mutated after its creator returned.

    Everything else in this file checkpoints a model that its
    ``scenario_creator`` built and nobody touched afterwards. CVaR rewrites
    one: it deactivates the risk-neutral objective, adds an active
    ``WITH_CVAR`` alongside it, and appends the value-at-risk variable eta to
    the root nonants. All three have to come back through the dill, and the
    resume branch has to rebuild ``saved_objectives`` from the *active*
    objective -- picking the deactivated original instead would leave the run
    reporting a risk-neutral number for a risk-averse problem, with nothing
    raising anywhere.

    Originally a phase-1b instance; phase 1b was retired and this is the
    phase that absorbed it.
    """

    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1",
                  "--cvar", "--cvar-weight", "0.5", "--cvar-alpha", "0.8")

    def test_the_run_really_is_risk_averse(self):
        """Otherwise everything below is the plain farmer case again."""
        for snap in self.reference:
            for sname, objname in snap["active_objective_names"].items():
                self.assertIn("WITH_CVAR", objname,
                              msg=f"{sname} is not solving the CVaR objective")
        eta_nonants = [k for snap in self.reference for k in snap["state"]
                       if "|x|" in k and "eta" in k]
        self.assertTrue(eta_nonants,
                        msg="eta was not appended to the root nonants")

    def test_the_resume_resolves_to_the_active_objective(self):
        for want_snap, got_snap in zip(self.reference, self.resumed):
            self.assertEqual(want_snap["active_objective_names"],
                             got_snap["active_objective_names"],
                             msg="the resumed run reads a different objective "
                                 "than the uninterrupted one")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestSizesMultiRankHub(_MultiRankABMixin, unittest.TestCase):
    """The MIP target, across ranks.

    Under default solver settings section 7 promises a valid *continuation*,
    not a reproduced trajectory, so the inherited comparison runs to a
    tolerance here. What this case is really for is the things a MIP resume
    can lose outright: the warm start that rode back in the dill, and an
    incumbent that must never regress across the stop.
    """

    MODULE = _SIZES
    MODEL_ARGS = ("--num-scens", "3", "--default-rho", "1")
    BIT_IDENTICAL = False
    N = 3
    STOP = 1

    def test_the_incumbent_does_not_regress_across_the_stop(self):
        for stopped, resumed in zip(self.stopped, self.resumed):
            before = stopped["best_solution_obj_val"]
            after = resumed["best_solution_obj_val"]
            if before is None or after is None:
                continue
            self.assertLessEqual(
                after, before,
                msg=f"rank {resumed['cylinder_rank']}: the resumed run "
                    f"reports a worse incumbent than its checkpoint")

    def test_the_trivial_bound_is_carried_not_recomputed(self):
        """Resume skips iteration 0, so the bound has to come from the file.

        A resumed run that recomputed it would be computing it from the
        checkpointed (W-laden) iterate, which is not the same quantity -- and
        for a MIP it would also pay a full round of subproblem solves.
        """
        for stopped, resumed in zip(self.stopped, self.resumed):
            self.assertEqual(stopped["trivial_bound"],
                             resumed["trivial_bound"])


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestFarmerMultiRankCylinders(_MultiRankABMixin, unittest.TestCase):
    """The real shape: three cylinders, two ranks each.

    Phase 4 proved a wheel resumes; this proves it still does when each
    cylinder is itself parallel, which is the configuration a cluster run
    actually has. The hub's barriers are now over a strict subset of
    COMM_WORLD, and getting that comm wrong -- COMM_WORLD instead of the
    cylinder's -- deadlocks the job against spokes that never call it.
    """

    NP = 6
    HUB_RANKS = 2
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle")

    def test_the_spoke_incumbent_survives_the_stop(self):
        """The best solution lives on the spoke, one file per spoke rank."""
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        written = os.listdir(spokes_dir)
        self.assertTrue(
            any(f.startswith("spoke_XhatShuffleInnerBound") for f in written),
            msg=f"the xhat spoke checkpointed no incumbent: {written}")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestStochAdmmMultiRank(_MultiRankABMixin, unittest.TestCase):
    """stoch-ADMM across ranks (design section 8.2).

    The wrapper is where checkpointing has the most to get wrong, and most of
    it fails quietly rather than loudly: wrapped scenario names reach the
    checkpoint's file names, the probability mask and the fixed-at-0 dummy
    vars are built by identity at construction and only their *result* rides
    in the dill, and the wrapper keeps its own references to the models a
    resume replaces. A run with any of those broken still solves and still
    prints numbers.
    """

    NP = 4
    HUB_RANKS = 2
    MODULE = _STOCH_DISTR
    #: Three ADMM subproblems, not two: with two regions every consensus
    #: variable happens to appear in both, so no nonant gets probability zero
    #: and no dummy var is added -- the run would pass every check below
    #: without exercising either. The meta-assertions in those tests are there
    #: to keep that from going unnoticed again.
    MODEL_ARGS = ("--stoch-admm", "--num-stoch-scens", "4",
                  "--num-admm-subproblems", "3", "--default-rho", "10")
    SPOKE_ARGS = ("--xhatxbar",)

    def test_wrapped_names_reach_the_files_intact(self):
        """File discovery enumerates local_scenarios, never a name creator.

        ADMM's names come from the wrapper and collide on their trailing
        digits across subproblems, so anything derived from
        ``sputils.extract_num`` would map two distinct subproblems onto one
        file (section 8.2, item 1).
        """
        gen_dir = os.path.join(self.ckpt_dir, "hub", f"gen_{self.STOP:04d}")
        written = os.listdir(gen_dir)
        seen = set()
        for snap in self.reference:
            rank = snap["cylinder_rank"]
            for sname in snap["scenario_names"]:
                self.assertIn("ADMM", sname)
                fname = f"hub_rank_{rank:04d}_scen_{sname}.dill"
                self.assertIn(fname, written)
                self.assertNotIn(fname, seen)
                seen.add(fname)

    def test_the_probability_mask_survives_the_restore(self):
        """Variable probabilities are consumed once, at construction.

        After that the reloaded model's own ``_mpisppy_data`` masks are
        authoritative (section 8.2, item 3). If the dill lost them, W would be
        applied to nonants that this subproblem does not own and the run would
        converge to the wrong answer without complaining.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            self.assertEqual(want_snap["probability_mask"],
                             got_snap["probability_mask"],
                             msg=f"rank {got_snap['cylinder_rank']}: the "
                                 f"variable-probability mask changed")
            self.assertTrue(
                any(0.0 in values
                    for key, values in want_snap["probability_mask"].items()
                    if "prob0_mask" in key),
                msg="no nonant has zero probability, so this instance does "
                    "not actually exercise the mask")

    def test_the_dummy_variables_are_still_fixed_at_zero(self):
        """The wrapper's inline dummy vars are added after construction.

        They are not nonants, so nothing else in this file looks at them, and
        a resume that relaxed them would quietly drop the consensus structure.
        """
        for want_snap, got_snap in zip(self.reference, self.resumed):
            self.assertEqual(want_snap["fixed_variables"],
                             got_snap["fixed_variables"],
                             msg=f"rank {got_snap['cylinder_rank']}: fixed "
                                 f"variables changed across the resume")
            self.assertTrue(want_snap["fixed_variables"],
                            msg="this instance fixes nothing, so the check "
                                "above proves nothing")

    def test_the_wrapper_does_not_keep_the_replaced_models(self):
        """Otherwise a resumed ADMM run holds two copies of every scenario.

        The wrapper builds every scenario at startup whether or not the run is
        resuming -- it needs them to assemble consensus lists and node names --
        so the freshly built models stay reachable through it after the swap.
        Nothing in the run reads them again; they just occupy memory, which on
        a large MIP is the memory checkpointing exists to save (section 8.2,
        item 2).
        """
        for snap in self.resumed:
            self.assertTrue(
                snap["model_holder_is_current"],
                msg=f"rank {snap['cylinder_rank']}: the ADMM wrapper still "
                    f"points at the models the resume replaced")
        # The same probe on the uninterrupted run, so a True above cannot be
        # the probe failing to find the wrapper at all.
        for snap in self.reference:
            self.assertTrue(snap["model_holder_is_current"],
                            msg="the probe found no ADMM wrapper to check")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestMultiRankGeometryRefusal(unittest.TestCase):
    """Resuming into a different rank layout is refused, not half-restored.

    Cross-geometry resume is a stated non-goal (section 12), and the failure
    mode without a refusal is the bad kind: each rank would look for the file
    of a rank that no longer exists, or find one holding somebody else's
    scenarios, and either restore nothing or restore the wrong slice.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _run_leg(self._tmp.name, "write", 2, _FARMER,
                 ("--num-scens", "6", "--default-rho", "1"), (),
                 ("--max-iterations", "2", "--checkpoint-dir", self.ckpt_dir))

    def tearDown(self):
        self._tmp.cleanup()

    def test_resuming_on_fewer_ranks_is_refused(self):
        result, _ = _run_leg(
            self._tmp.name, "resume1", 1, _FARMER,
            ("--num-scens", "6", "--default-rho", "1"), (),
            ("--max-iterations", "4", "--resume-from", self.ckpt_dir),
            check=False)
        self.assertNotEqual(result.returncode, 0,
                            msg="a rank-count change was accepted")
        self.assertIn("rank count", result.stdout + result.stderr)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestOneRankFailingDoesNotHangTheOthers(unittest.TestCase):
    """A write that fails on one rank must not deadlock the cylinder.

    This is the failure mode the multi-rank protocol exists to prevent, and it
    is invisible to every other test in this file: with the ranks disagreeing
    about whether the write failed, the run does not crash, it *stops* -- one
    rank has returned to the PH loop while the others wait at a barrier it
    will never reach -- and the job burns its wall-clock allocation with no
    error in the log. So the test asserts the run finished at all, which is
    most of the point, and then that the failure was handled the way section 8
    promises: every rank warns, no generation is published for the failed
    write, and the previous checkpoint is still there to resume from.

    The last iteration is the one sabotaged, so the manifest is left naming
    the generation before it. A failure in the middle would be repaired by the
    next successful write and prove less.
    """

    N = 3
    FAIL_AT = 3
    FAIL_RANK = 1

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        cmd = [
            "mpiexec", "-np", "2",
            sys.executable, "-m", "mpi4py", _FAILURE_DRIVER,
            "--fail-on-rank", str(self.FAIL_RANK),
            "--fail-at-generation", str(self.FAIL_AT),
            "--module-name", _FARMER, "--num-scens", "6",
            "--default-rho", "1", "--solver-name", solver_name,
            "--max-iterations", str(self.N),
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--checkpoint-dir", self.ckpt_dir,
        ]
        # Generous, but it is a timeout on a farmer LP that finishes in under a
        # second when it finishes at all: what this is really measuring is
        # whether the job returns.
        self.result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=600, check=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_run_finishes(self):
        self.assertEqual(
            self.result.returncode, 0,
            msg=f"the run did not survive a one-rank write failure:\n"
                f"{self.result.stdout[-4000:]}\n{self.result.stderr[-4000:]}")
        self.assertIn("Reached user-specified limit", self.result.stdout,
                      msg="the run did not reach its iteration limit")

    def test_both_ranks_report_the_failure(self):
        """Every rank raises, so every rank warns -- and says where.

        The rank whose own write succeeded has no exception to describe, so it
        names the rank that does. Without that, a multi-rank failure prints
        one real diagnosis and n-1 misleading ones.
        """
        warnings = [line for line in self.result.stdout.splitlines()
                    if "WARNING: checkpoint write failed" in line]
        self.assertEqual(len(warnings), 2,
                         msg=f"expected one warning per rank: {warnings}")
        for rank in (0, self.FAIL_RANK):
            self.assertTrue(
                any(f"on rank {rank}" in line for line in warnings),
                msg=f"rank {rank} did not report the failure: {warnings}")
        # Rank 0's write succeeded, so its message must point at the rank that
        # holds the cause rather than implying its own write went wrong.
        self.assertIn("could not write its part of the checkpoint",
                      self.result.stdout)
        # And the failing rank's own message must carry the real cause, which
        # is the thing a rank-0-only warning would have thrown away.
        self.assertIn("No space left on device", self.result.stdout)

    def test_the_failed_generation_is_not_published(self):
        """All-or-nothing: rank 0's half of it is discarded too."""
        manifest = _published_generation(self.ckpt_dir)
        self.assertEqual(manifest["generation"], self.FAIL_AT - 1)

        hub_dir = os.path.join(self.ckpt_dir, "hub")
        self.assertEqual(
            sorted(d for d in os.listdir(hub_dir) if d.startswith("gen_")),
            [f"gen_{self.FAIL_AT - 1:04d}"],
            msg=f"the abandoned generation was left behind: "
                f"{sorted(os.listdir(hub_dir))}")

    def test_the_previous_checkpoint_is_still_resumable(self):
        _, out_path = _run_leg(
            self._tmp.name, "resume", 2, _FARMER,
            ("--num-scens", "6", "--default-rho", "1"), (),
            ("--max-iterations", str(self.N + 1),
             "--resume-from", self.ckpt_dir))
        for snap in _hub_ranks(out_path):
            self.assertTrue(snap["resumed"])
            self.assertEqual(snap["resume_iteration"], self.FAIL_AT - 1)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestDeadlineOnOneRankDoesNotHangTheOthers(unittest.TestCase):
    """``--checkpoint-before-seconds`` is the one trigger that is rank-local.

    Every other checkpoint point is a pure function of the iteration number, so
    the ranks arrive at the write together without being asked. Elapsed wall
    clock is not, and the ranks of a cylinder do not share a clock: a rank that
    believed its own and started writing would wait in the write's barrier for
    ranks that went on with the iteration, and the job would burn its
    allocation with nothing in the log.

    So the driver puts one rank a year past the deadline and leaves the other
    where it was. The run either agrees -- through ``allreduce_or`` -- and
    writes on both ranks, or it hangs; there is no third outcome, which is what
    makes the timeout below an assertion rather than a guess.

    The cadence is set past the iteration count so that nothing but the
    deadline (and the final iteration, which is always written) can produce a
    write, and the deadline is set to an hour that a farmer LP has no other way
    of reaching.
    """

    N = 4
    SKEW_AT = 2
    SKEW_RANK = 1
    DEADLINE = 3600.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        cmd = [
            "mpiexec", "-np", "2",
            sys.executable, "-m", "mpi4py", _DEADLINE_DRIVER,
            "--skew-on-rank", str(self.SKEW_RANK),
            "--skew-at-generation", str(self.SKEW_AT),
            "--module-name", _FARMER, "--num-scens", "6",
            "--default-rho", "1", "--solver-name", solver_name,
            "--max-iterations", str(self.N),
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--checkpoint-dir", self.ckpt_dir,
            "--checkpoint-every-iterations", "100",
            "--checkpoint-before-seconds", str(self.DEADLINE),
        ]
        # Generous, but it is a timeout on a farmer LP that finishes in under a
        # second when it finishes at all: what this is really measuring is
        # whether the job returns.
        self.result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=600, check=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_run_finishes(self):
        self.assertEqual(
            self.result.returncode, 0,
            msg=f"the run did not survive a one-rank deadline:\n"
                f"{self.result.stdout[-4000:]}\n{self.result.stderr[-4000:]}")
        self.assertIn("Reached user-specified limit", self.result.stdout,
                      msg="the run did not reach its iteration limit")

    def test_the_deadline_wrote_the_skewed_generation(self):
        """Off cadence and not the last iteration, so the deadline is the only
        thing that could have written it -- and one rank's clock was enough."""
        self.assertIn(f"Checkpoint written at iteration {self.SKEW_AT}",
                      self.result.stdout)
        self.assertIn("--checkpoint-before-seconds", self.result.stdout)

    def test_every_rank_wrote_its_slice(self):
        """A generation spans the ranks, so a write that only the skewed rank
        joined would leave a generation that cannot be resumed."""
        gen_dir = os.path.join(self.ckpt_dir, "hub",
                               f"gen_{self.SKEW_AT:04d}")
        # gen_0002 is retired by the final iteration's write, so the run's own
        # log is what says both ranks were in it; what remains checkable here
        # is that the generation that replaced it is whole.
        final_dir = os.path.join(self.ckpt_dir, "hub", f"gen_{self.N:04d}")
        self.assertFalse(os.path.isdir(gen_dir))
        leaves = sorted(f for f in os.listdir(final_dir)
                        if f.startswith("hub_rank_") and f.endswith(".pkl"))
        self.assertEqual(leaves,
                         ["hub_rank_0000.pkl", "hub_rank_0001.pkl"])

    def test_it_fires_once_and_not_on_every_later_iteration(self):
        """The skewed rank stays past the deadline for the rest of the run.
        Without the latch, every iteration after it would write."""
        written = [line for line in self.result.stdout.splitlines()
                   if "Checkpoint written at iteration" in line]
        self.assertEqual(len(written), 2, msg=f"expected the deadline write "
                                              f"and the final one: {written}")

    def test_the_deadline_checkpoint_is_resumable(self):
        """Resume from the deadline's own generation, not the one that
        replaced it: rerun with the iteration limit at the skew point so the
        final-iteration rule cannot write anything later."""
        cmd = [
            "mpiexec", "-np", "2",
            sys.executable, "-m", "mpi4py", _DEADLINE_DRIVER,
            "--skew-on-rank", str(self.SKEW_RANK),
            "--skew-at-generation", str(self.SKEW_AT),
            "--module-name", _FARMER, "--num-scens", "6",
            "--default-rho", "1", "--solver-name", solver_name,
            "--max-iterations", str(self.SKEW_AT),
            "--intra-hub-conv-thresh", "-1",
            "--rel-gap", "0.0", "--abs-gap", "0.0",
            "--checkpoint-dir", self.ckpt_dir,
            "--checkpoint-every-iterations", "100",
            "--checkpoint-before-seconds", str(self.DEADLINE),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       check=True)
        self.assertEqual(_published_generation(self.ckpt_dir)["generation"],
                         self.SKEW_AT)

        _, out_path = _run_leg(
            self._tmp.name, "resume", 2, _FARMER,
            ("--num-scens", "6", "--default-rho", "1"), (),
            ("--max-iterations", str(self.N),
             "--resume-from", self.ckpt_dir))
        for snap in _hub_ranks(out_path):
            self.assertTrue(snap["resumed"])
            self.assertEqual(snap["resume_iteration"], self.SKEW_AT)


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestMultiRankSpokeCursorAgreement(unittest.TestCase):
    """A multi-rank xhat spoke resumes onto one cursor, not one per rank.

    The ranks of an xhatshuffle spoke explore together: they pick the same
    scenario and ``_try_one`` broadcasts its nonants from the rank that owns
    it. Each rank writes its own checkpoint file at the bottom of its own
    pass, though, so a stop can land between two of those writes and leave
    files whose cursors disagree -- and a rank that never found an incumbent
    writes no file at all. Resuming each rank onto whatever its own file says
    makes the ranks pick different scenarios and broadcast from different
    roots, and the objective that reaches the hub is then a blend of several
    scenarios' solutions, reported as an ordinary feasible inner bound with
    no error and no warning.

    Both tests manufacture the disagreement by editing what the stopped leg
    wrote. Racing the two ranks' writes would produce it only sometimes,
    which is no way to guard against it.
    """

    NP = 6
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--lagrangian", "--xhatshuffle")
    STOP = 2
    RESUME_FOR = 2
    SPOKE = "XhatShuffleInnerBound"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _run_leg(self._tmp.name, "B1", self.NP, self.MODULE, self.MODEL_ARGS,
                 self.SPOKE_ARGS,
                 ("--max-iterations", str(self.STOP),
                  "--checkpoint-dir", self.ckpt_dir))

    def tearDown(self):
        self._tmp.cleanup()

    def _spoke_files(self):
        """This spoke's checkpoint files, one per rank, ordered by rank."""
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        names = sorted(f for f in os.listdir(spokes_dir)
                       if f.startswith(f"spoke_{self.SPOKE}"))
        self.assertGreater(
            len(names), 1,
            msg=f"expected one file per spoke rank, got {names}")
        return [os.path.join(spokes_dir, n) for n in names]

    def _resume(self):
        result, out_path = _run_leg(
            self._tmp.name, "B2", self.NP, self.MODULE, self.MODEL_ARGS,
            self.SPOKE_ARGS,
            ("--max-iterations", str(self.RESUME_FOR),
             "--resume-from", self.ckpt_dir))
        return result, _spoke_ranks(out_path, self.SPOKE)

    def test_the_ranks_adopt_one_cursor(self):
        paths = self._spoke_files()
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        self.assertIsNotNone(
            state["loop_state"],
            msg="the stopped leg checkpointed no cursor, so this test would "
                "pass without exercising anything")
        # Wind the last rank's file back a pass: what a stop that landed
        # between the two ranks' writes leaves behind.
        state["loop_state"]["xh_iter"] = int(state["loop_state"]["xh_iter"]) - 1
        cursor = state["loop_state"]["cursor"]
        cursor["cycle_idx"] = max(0, int(cursor["cycle_idx"]) - 1)
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        _, markers = self._resume()
        self.assertGreater(len(markers), 1,
                           msg=f"expected a marker per spoke rank: {markers}")
        adopted = [m["applied_loop_state"] for m in markers]
        for marker in markers:
            self.assertIsNotNone(
                marker["applied_loop_state"],
                msg="a spoke rank adopted no cursor at all")
        distinct = {json.dumps(a, sort_keys=True) for a in adopted}
        self.assertEqual(
            len(distinct), 1,
            msg=f"the spoke's ranks resumed onto {len(distinct)} different "
                f"cursors: {adopted}")

    def test_ranks_holding_different_incumbents_stop_the_whole_restore(self):
        """Half of one xhat beside half of another is not a solution.

        The cached values cannot be agreed by broadcast -- each rank owns
        different scenarios -- so what is agreed is whether they all came
        from the same pass. The objective of the cached solution says so:
        every rank reads it out of the same reduction.
        """
        paths = self._spoke_files()
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        self.assertIsNotNone(
            state["best_solution_obj_val"],
            msg="the stopped leg checkpointed no incumbent, so this test "
                "would pass without exercising anything")
        # An incumbent from a different pass: what a stop landing between the
        # two ranks' writes leaves when one of them has just improved.
        state["best_solution_obj_val"] = \
            float(state["best_solution_obj_val"]) - 1.0
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_incumbent_obj"],
                msg="a spoke rank restored values from a pass that another "
                    "rank of the same spoke did not checkpoint")
        self.assertIn("checkpointed different incumbents", result.stdout,
                      msg="the run declined to restore without saying so")

    def test_a_file_that_does_not_match_stops_every_rank(self):
        """A load that refuses on some ranks and not others.

        The refusal is per rank -- the file is named after the rank and
        checked against the scenarios that rank owns -- and the agreement
        that follows is collective, so a rank-local raise leaves the others
        waiting for a rank that has gone. The plainest way in is a resume
        with a different rank count; this manufactures the same split
        directly, which does not depend on how ranks divide scenarios.
        """
        paths = self._spoke_files()
        os.remove(paths[0])
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        state["format_version"] = 0
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        result, _ = _run_leg(
            self._tmp.name, "B2", self.NP, self.MODULE, self.MODEL_ARGS,
            self.SPOKE_ARGS,
            ("--max-iterations", str(self.RESUME_FOR),
             "--resume-from", self.ckpt_dir),
            check=False)
        self.assertNotEqual(result.returncode, 0,
                            msg="a checkpoint that does not match this run "
                                "was accepted")
        self.assertIn(
            "could not read their checkpoint", result.stdout + result.stderr,
            msg="the run died on one rank's traceback without the agreement "
                "saying how many ranks could not read theirs")

    def test_a_rank_without_a_file_stops_the_whole_restore(self):
        """Half an incumbent is not a solution the study ever found."""
        paths = self._spoke_files()
        os.remove(paths[-1])

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_incumbent_obj"],
                msg="a spoke rank restored an incumbent although another "
                    "rank of the same spoke had no file to restore from")
        self.assertIn("ranks of this spoke have a checkpointed incumbent",
                      result.stdout,
                      msg="the run declined to restore without saying so")


@unittest.skipIf(not solver_available, "no solver is available")
@unittest.skipIf(not mpiexec_available, "mpiexec is not available")
class TestMultiRankDualWeightAgreement(unittest.TestCase):
    """A multi-rank dual cylinder restores one iteration's W, not one each.

    W is per scenario and so per rank, and there is nothing to broadcast --
    but the iteration it belongs to is the cylinder's. Ranks that restore W
    from different iterations hand ``Compute_Xbar`` an allreduce over values
    from two points of the run, and under ``--ph-primal-hub`` that blended W
    is what the hub's W is built from. Like the xhat case, it costs no error
    and no warning.

    Each rank writes at the bottom of its own iteration and a failed write
    warns and carries on, so the files really can disagree; the tests
    manufacture that by editing what the stopped leg wrote.
    """

    NP = 4
    MODULE = _FARMER
    MODEL_ARGS = ("--num-scens", "6", "--default-rho", "1")
    SPOKE_ARGS = ("--relaxed-ph",)
    STOP = 2
    RESUME_FOR = 2
    CYLINDER = "RelaxedPHSpoke"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Removed here rather than in tearDown, which unittest does not call
        # when setUp raises -- and the leg below raises on a failed run.
        self.addCleanup(self._tmp.cleanup)
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        _run_leg(self._tmp.name, "B1", self.NP, self.MODULE, self.MODEL_ARGS,
                 self.SPOKE_ARGS,
                 ("--max-iterations", str(self.STOP),
                  "--checkpoint-dir", self.ckpt_dir))

    def _dual_files(self):
        """This cylinder's checkpoint files, one per rank, ordered by rank."""
        spokes_dir = os.path.join(self.ckpt_dir, "spokes")
        names = sorted(f for f in os.listdir(spokes_dir)
                       if f.startswith(f"spoke_{self.CYLINDER}"))
        self.assertGreater(
            len(names), 1,
            msg=f"expected one file per cylinder rank, got {names}")
        return [os.path.join(spokes_dir, n) for n in names]

    def _resume(self):
        result, out_path = _run_leg(
            self._tmp.name, "B2", self.NP, self.MODULE, self.MODEL_ARGS,
            self.SPOKE_ARGS,
            ("--max-iterations", str(self.RESUME_FOR),
             "--resume-from", self.ckpt_dir))
        markers = _spoke_ranks(out_path, self.CYLINDER)
        self.assertGreater(
            len(markers), 1,
            msg=f"expected a marker per cylinder rank: {markers}")
        return result, markers

    def test_the_ranks_restore_the_same_iteration(self):
        """The undisturbed case, so the tests below cannot pass vacuously."""
        _, markers = self._resume()
        generations = {m["restored_dual_generation"] for m in markers}
        self.assertEqual(
            len(generations), 1,
            msg=f"the cylinder's ranks restored W from {len(generations)} "
                f"different iterations: {generations}")
        self.assertNotIn(None, generations,
                         msg="no rank restored any dual weights at all")

    def test_ranks_holding_different_iterations_stop_the_whole_restore(self):
        paths = self._dual_files()
        with open(paths[-1], "rb") as f:
            state = pickle.load(f)
        self.assertIsNotNone(state["generation"])
        # W from the iteration before: what a stop landing between the two
        # ranks' writes leaves behind.
        state["generation"] = int(state["generation"]) - 1
        with open(paths[-1], "wb") as f:
            pickle.dump(state, f)

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_dual_generation"],
                msg="a rank restored W from an iteration another rank of the "
                    "same cylinder did not checkpoint")
        self.assertIn("dual weights from different iterations",
                      result.stdout,
                      msg="the run declined to restore without saying so")

    def test_a_rank_without_a_file_stops_the_whole_restore(self):
        paths = self._dual_files()
        os.remove(paths[-1])

        result, markers = self._resume()
        for marker in markers:
            self.assertIsNone(
                marker["restored_dual_generation"],
                msg="a rank restored W although another rank of the same "
                    "cylinder had no file to restore from")
        self.assertIn("ranks of this cylinder have checkpointed dual weights",
                      result.stdout,
                      msg="the run declined to restore without saying so")


if __name__ == "__main__":
    unittest.main()
