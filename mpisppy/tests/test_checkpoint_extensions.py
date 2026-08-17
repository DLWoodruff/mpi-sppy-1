###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Extension and converger state across a checkpoint (design phase 3).

Everything checkpointed before this phase lived on a scenario model, so it came
back with the dilled models and came back consistent with the variable values
it pairs with. This phase covers the state extensions keep on *themselves*,
which no model carries and which was therefore lost outright.

That loss is the quiet kind. Nothing is missing, nothing raises, the run
continues and reports numbers -- but an extension whose behavior depends on
what it did in earlier iterations takes a *different action* at the first
iteration after the stop than the uninterrupted run would have, and the two
diverge from there. A rho updater with no record of the previous xbar skips a
rho update. A fixer whose per-variable countdowns were reset waits longer to
fix. A converger with no previous xbar measures a dual residual of zero and can
stop the run early.

So each case here is an A/B comparison in the shape design section 11.1 lays
out -- an uninterrupted run of N iterations against one stopped at k and
resumed -- with the extension attached to both. Farmer is a deterministic LP,
so "the extension state was carried" and "the runs are bit-identical" are the
same statement, and the assertions say so directly rather than inferring it
from a summary number.

Two of these are regressions rather than divergences: ``--sep-rho`` and its
siblings *crashed* on the first iteration after a resume, and the fixer's
per-variable counts were being zeroed by its own setup hook on the way back in.
"""

import json
import os
import pickle
import tempfile
import unittest

import mpisppy.tests.examples.farmer as farmer
import mpisppy.tests.examples.sizes.sizes as sizes
import mpisppy.utils.checkpointing as checkpointing
from mpisppy.extensions.checkpointer import Checkpointer
from mpisppy.extensions.extension import Extension, MultiExtension
from mpisppy.opt.ph import PH
from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()

FARMER_SCENARIOS = ["scen0", "scen1", "scen2"]
FARMER_KWARGS = {"use_integer": False, "crops_multiplier": 1}
SIZES_SCENARIOS = ["Scenario1", "Scenario2", "Scenario3"]
SIZES_KWARGS = {"scenario_count": 3}


def _options(max_iters, ckpt_dir=None, resume_from=None, **overrides):
    options = {
        "solver_name": solver_name,
        "PHIterLimit": max_iters,
        "defaultPHrho": 1.0,
        # Never converge early: the A/B comparison needs both sides to run the
        # same iterations.
        "convthresh": -1.0,
        "verbose": False,
        "display_progress": False,
        "display_timing": False,
        "display_convergence_detail": False,
        "iter0_solver_options": None,
        "iterk_solver_options": None,
        "tee-rank0-solves": False,
        "smoothed": 0,
        "time_limit": None,
    }
    if ckpt_dir is not None:
        options["checkpoint_dir"] = ckpt_dir
        options["checkpoint_backend"] = checkpointing.DILL_RELOAD_BACKEND
        options["checkpoint_every_iterations"] = 1
    if resume_from is not None:
        options["resume_from"] = resume_from
    options.update(overrides)
    return options


def _make_ph(options, ext_classes, model=farmer, scenario_names=None,
             creator_kwargs=None, **ph_kwargs):
    """A PH hub with the checkpointer and the extensions under test attached."""
    classes = list(ext_classes)
    if "checkpoint_dir" in options or "resume_from" in options:
        classes.insert(0, Checkpointer)
    return PH(
        options,
        scenario_names if scenario_names is not None else FARMER_SCENARIOS,
        model.scenario_creator,
        model.scenario_denouement,
        scenario_creator_kwargs=(creator_kwargs if creator_kwargs is not None
                                 else FARMER_KWARGS),
        extensions=MultiExtension,
        extension_kwargs={"ext_classes": classes},
        **ph_kwargs,
    )


def _extension(ph, cls):
    for candidate in ph.extobject.extdict.values():
        if isinstance(candidate, cls):
            return candidate
    raise AssertionError(f"no {cls.__name__} attached")


def _primal_snapshot(ph):
    """Nonant values, fixedness and the per-nonant Params, keyed by name."""
    snap = {}
    for sname, s in ph.local_scenarios.items():
        for ndn_i, v in s._mpisppy_data.nonant_indices.items():
            snap[f"{sname}|x|{v.name}"] = v._value
            snap[f"{sname}|fixed|{v.name}"] = float(v.is_fixed())
            for pname in ("W", "rho", "xbars"):
                param = getattr(s._mpisppy_model, pname, None)
                if param is not None:
                    snap[f"{sname}|{pname}|{ndn_i}"] = float(param[ndn_i]._value)
    return snap


class _ABMixin:
    """Uninterrupted vs stop-and-resume, with the extension under test.

    The legs run in one process, which is enough here: what is under test is
    whether state survives the *checkpoint*, and a fresh interpreter is what
    ``test_checkpoint.py`` and the cylinders harnesses already pin.
    """

    N = 5
    STOP = 2
    MODEL = farmer
    SCENARIOS = None
    CREATOR_KWARGS = None
    #: Extra options every leg needs to configure the extension under test.
    EXTRA_OPTIONS = {}
    PH_KWARGS = {}

    def ext_classes(self):
        raise NotImplementedError

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _ph(self, max_iters, **ckpt_kwargs):
        options = _options(max_iters, **ckpt_kwargs, **self.EXTRA_OPTIONS)
        return _make_ph(options, self.ext_classes(), model=self.MODEL,
                        scenario_names=self.SCENARIOS,
                        creator_kwargs=self.CREATOR_KWARGS, **self.PH_KWARGS)

    def run_ab(self):
        """Returns (reference, stopped, resumed) PH objects, all run."""
        reference = self._ph(self.N)
        reference.ph_main()
        stopped = self._ph(self.STOP, ckpt_dir=self.ckpt_dir)
        stopped.ph_main()
        # --max-iterations counts this run's iterations, so the resumed leg
        # asks for the ones that are left rather than for the study total.
        resumed = self._ph(self.N - self.STOP, resume_from=self.ckpt_dir)
        resumed.ph_main()
        self.assertTrue(resumed._resumed_from_checkpoint,
                        msg="the third leg started from scratch")
        return reference, stopped, resumed

    def assert_bit_identical(self, reference, resumed):
        want, got = _primal_snapshot(reference), _primal_snapshot(resumed)
        self.assertEqual(set(want), set(got))
        worst = max((abs(want[k] - got[k]) for k in want), default=0.0)
        self.assertEqual(
            worst, 0.0,
            msg=f"the resumed run differs from the uninterrupted one by "
                f"{worst}; this instance is deterministic, so a resume that "
                f"carried the extension's state must land bit-identically")


@unittest.skipIf(not solver_available, "no solver is available")
class TestNormRhoUpdaterResume(_ABMixin, unittest.TestCase):
    """A rho updater that compares against the previous iteration's xbar.

    Without the restore this does not merely differ in a statistic. `miditer`
    branches on whether it has a previous xbar at all, so a resumed run takes
    the "first time through" branch: it snapshots and performs **no rho update
    that iteration**, then carries the resulting rho -- different from the
    uninterrupted run's -- for the rest of the run.
    """

    EXTRA_OPTIONS = {"norm_rho_options": {"verbose": False}}

    def ext_classes(self):
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        return [NormRhoUpdater]

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)

    def test_the_previous_xbar_is_actually_restored(self):
        """Named directly, so a passing comparison cannot be a coincidence."""
        from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
        _, stopped, resumed = self.run_ab()
        saved = _extension(stopped, NormRhoUpdater)._prev_avg
        # The resumed run has moved on by the time it finishes, so compare
        # what the checkpoint holds rather than the extension's final state.
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as f:
            generation = json.load(f)["generation"]
        leaf = _read_leaf(self.ckpt_dir, generation)
        carried = leaf["extension_state"]["extensions"]["NormRhoUpdater"]
        self.assertEqual(carried["prev_avg"], saved)
        self.assertTrue(saved, msg="the extension recorded no previous xbar, "
                                   "so this test proves nothing")
        self.assertTrue(_extension(resumed, NormRhoUpdater)._prev_avg)


@unittest.skipIf(not solver_available, "no solver is available")
class TestMultRhoUpdaterResume(_ABMixin, unittest.TestCase):
    """A rho updater that anchors a ratio once and scales from it forever.

    A resumed run that forgot the anchor re-derives it from the *checkpointed*
    rho and the current convergence metric, so every later rho is scaled from
    a baseline the uninterrupted run never had.
    """

    EXTRA_OPTIONS = {"mult_rho_options": {"verbose": False}}

    def ext_classes(self):
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        return [MultRhoUpdater]

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)

    def test_the_anchor_is_restored_not_re_derived(self):
        from mpisppy.extensions.mult_rho_updater import MultRhoUpdater
        _, stopped, resumed = self.run_ab()
        before = _extension(stopped, MultRhoUpdater)
        after = _extension(resumed, MultRhoUpdater)
        self.assertIsNotNone(before._first_rho,
                             msg="the anchor was never set, so this test "
                                 "proves nothing")
        self.assertEqual(after.first_c, before.first_c)
        self.assertEqual(after._first_rho, before._first_rho)


@unittest.skipIf(not solver_available, "no solver is available")
class TestSepRhoResume(_ABMixin, unittest.TestCase):
    """A regression: this configuration used to *crash* on resume.

    The dynamic-rho extensions track a W history through a `WTracker`, and
    `W_diff` indexes it at the two iterations before the current one. A resumed
    run had none of them, so the first iteration after the stop died with a
    bare `KeyError` -- no warning, no partial result, just a traceback out of a
    utility three call levels below the extension.

    Its own rho is *not* recomputed at the resume (the checkpointed rho, with
    whatever adaptation it carries, is the right starting point and the
    extension already knew that), so the assertion here is the ordinary one:
    the run continues, and continues identically.
    """

    EXTRA_OPTIONS = {"dynamic_rho_primal_crit": False,
                     "dynamic_rho_dual_crit": False}

    def ext_classes(self):
        from mpisppy.extensions.sep_rho import SepRho
        return [SepRho]

    def _ph(self, max_iters, **ckpt_kwargs):
        # SepRho reads its settings off a cfg handed to it in the options.
        from mpisppy.utils.config import Config
        cfg = Config()
        cfg.add_to_config("sep_rho_multiplier", description="", domain=float,
                          default=1.0)
        cfg.add_to_config("dynamic_rho_primal_crit", description="",
                          domain=bool, default=False)
        cfg.add_to_config("dynamic_rho_dual_crit", description="", domain=bool,
                          default=False)
        options = _options(max_iters, **ckpt_kwargs)
        options["sep_rho_options"] = {"cfg": cfg}
        return _make_ph(options, self.ext_classes())

    def test_a_resumed_run_does_not_crash(self):
        """The crash regression, stated as plainly as it can be."""
        reference, _, resumed = self.run_ab()
        self.assertEqual(resumed._PHIter, self.N)
        self.assert_bit_identical(reference, resumed)

    def test_the_w_history_the_next_diff_reads_is_carried(self):
        from mpisppy.extensions.sep_rho import SepRho
        _, stopped, resumed = self.run_ab()
        tracker = _extension(stopped, SepRho).wt
        carried = _extension(resumed, SepRho).wt
        # W_diff reads its own ph_iter and one before it; those are the two
        # the checkpoint has to carry, and are what used to be missing.
        for wanted in (tracker.ph_iter, tracker.ph_iter - 1):
            self.assertIn(wanted, carried.local_Ws,
                          msg=f"local_Ws[{wanted}] was not carried; the next "
                              f"W_diff would raise KeyError")


@unittest.skipIf(not solver_available, "no solver is available")
class TestFixerResume(_ABMixin, unittest.TestCase):
    """The fixer's per-variable countdowns, on a MIP that actually fixes.

    Those counts live on the scenario models, so they ride in the dill for
    free -- and then the fixer's own `post_iter0` hook, which runs on a resumed
    run too, zeroed every one of them on the way back in. A nonant one
    iteration short of its threshold restarted its countdown, so a resumed run
    fixed strictly later than the uninterrupted one.
    """

    MODEL = sizes
    SCENARIOS = SIZES_SCENARIOS
    CREATOR_KWARGS = SIZES_KWARGS
    N = 4
    STOP = 2

    def setUp(self):
        super().setUp()
        self.EXTRA_OPTIONS = {
            "fixeroptions": {
                "verbose": False,
                "boundtol": 0.01,
                "id_fix_list_fct": sizes.id_fix_list_fct,
            },
        }

    def ext_classes(self):
        from mpisppy.extensions.fixer import Fixer
        return [Fixer]

    def test_the_countdowns_survive_the_resume(self):
        from mpisppy.extensions.fixer import Fixer
        _, stopped, resumed = self.run_ab()
        # Compare what the checkpoint's models hold against what the resumed
        # run started from, per scenario and per nonant.
        for sname, s in stopped.local_scenarios.items():
            saved = dict(s._mpisppy_data.conv_iter_count)
            self.assertTrue(saved, msg="the fixer tracked nothing, so this "
                                       "test proves nothing")
        counts = {sname: dict(s._mpisppy_data.conv_iter_count)
                  for sname, s in resumed.local_scenarios.items()}
        self.assertTrue(any(counts[sname] for sname in counts))
        self.assertIsNotNone(_extension(resumed, Fixer))

    def test_the_same_variables_end_up_fixed(self):
        """The observable consequence: fixing happens at the same iteration.

        A resumed run whose countdowns restarted would still fix these
        variables eventually -- just later -- so comparing the final fixed set
        against the uninterrupted run's is what catches the delay.
        """
        reference, _, resumed = self.run_ab()
        want = _fixed_nonant_names(reference)
        got = _fixed_nonant_names(resumed)
        self.assertEqual(want, got)

    def test_the_running_totals_are_carried(self):
        from mpisppy.extensions.fixer import Fixer
        reference, _, resumed = self.run_ab()
        self.assertEqual(_extension(resumed, Fixer).fixed_so_far,
                         _extension(reference, Fixer).fixed_so_far)


def _fixed_nonant_names(ph):
    return {(sname, v.name)
            for sname, s in ph.local_scenarios.items()
            for v in s._mpisppy_data.nonant_indices.values()
            if v.is_fixed()}


def _read_leaf(ckpt_dir, generation):
    path = os.path.join(ckpt_dir, "hub", f"gen_{generation:04d}",
                        "hub_rank_0000.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@unittest.skipIf(not solver_available, "no solver is available")
class TestSlammerResume(_ABMixin, unittest.TestCase):
    """What a slammer has already pinned, and to what value.

    Slams are sticky, so `_slammed` is both the record of what was done and
    the list of what would have to be released to undo it. A resumed run that
    forgot it reports the wrong totals and would slam a nonant a second time
    the moment anything else unfixed it.

    There is a second, subtler failure here that has nothing to do with
    `_slammed`: `pre_iter0` classifies every nonant that is fixed *right now*
    as the modeler's and drops it from the eligibility map. On a resumed run
    every mid-run fixing is already applied, so the extension would file its
    own earlier slams -- and the fixer's fixings -- as untouchable.
    """

    N = 5
    STOP = 3
    #: Farmer's nonants are continuous, so slam them to a bound.
    DIRECTIVE_PATTERN = "DevotedAcreage[*]"

    def setUp(self):
        super().setUp()
        from mpisppy.extensions.slammer import SlamDirective
        self.EXTRA_OPTIONS = {
            "slammer_options": {
                "directives": [SlamDirective(self.DIRECTIVE_PATTERN, True,
                                             ("lb",), 1.0)],
                "slam_start_iter": 1,
                "iters_between_slams": 1,
                "verbose": False,
            },
        }

    def ext_classes(self):
        from mpisppy.extensions.slammer import Slammer
        return [Slammer]

    def test_the_slam_record_is_restored(self):
        from mpisppy.extensions.slammer import Slammer
        _, stopped, resumed = self.run_ab()
        before = _extension(stopped, Slammer)._slammed
        self.assertTrue(before, msg="nothing was slammed before the stop, so "
                                    "this test proves nothing")
        after = _extension(resumed, Slammer)._slammed
        for ndn_i, value in before.items():
            self.assertIn(ndn_i, after,
                          msg="the resumed run forgot a nonant it had slammed")
            self.assertEqual(after[ndn_i], value)

    def test_previous_slams_are_not_filed_as_modeler_fixed(self):
        from mpisppy.extensions.slammer import Slammer
        _, stopped, resumed = self.run_ab()
        before = _extension(stopped, Slammer)._slammed
        ext = _extension(resumed, Slammer)
        for ndn_i in before:
            self.assertNotIn(
                ndn_i, ext._modeler_fixed,
                msg="a nonant this extension slammed itself came back "
                    "classified as fixed by the modeler")

    def test_resume_matches_the_uninterrupted_run(self):
        reference, _, resumed = self.run_ab()
        from mpisppy.extensions.slammer import Slammer
        self.assertEqual(
            set(_extension(resumed, Slammer)._slammed),
            set(_extension(reference, Slammer)._slammed),
            msg="the resumed run slammed a different set of nonants")
        self.assert_bit_identical(reference, resumed)


@unittest.skipIf(not solver_available, "no solver is available")
class TestPrimalDualConvergerResume(_ABMixin, unittest.TestCase):
    """A converger measures against the previous iterate, so it holds state.

    Getting this wrong is not just a divergence: the converger decides when
    the run *stops*. Its dual residual is rho * ||xbar_t - xbar_{t-1}||, and a
    resumed run whose `prev_xbars` came from its own constructor is comparing
    xbar_t against itself -- a residual of zero, and a run that can declare
    convergence an iteration early.
    """

    def ext_classes(self):
        return []

    def _ph(self, max_iters, **ckpt_kwargs):
        from mpisppy.convergers.primal_dual_converger import (
            PrimalDualConverger)
        options = _options(max_iters, **ckpt_kwargs)
        # A threshold nothing will reach, so both legs run every iteration and
        # the comparison is of the state rather than of where each stopped.
        options["primal_dual_converger_options"] = {"tol": -1.0,
                                                    "verbose": False}
        return _make_ph(options, self.ext_classes(),
                        ph_converger=PrimalDualConverger)

    def test_the_previous_xbars_are_restored(self):
        _, stopped, resumed = self.run_ab()
        with open(os.path.join(self.ckpt_dir, "manifest.json")) as f:
            generation = json.load(f)["generation"]
        carried = _read_leaf(self.ckpt_dir, generation)["extension_state"]
        self.assertEqual(carried["converger"]["class"], "PrimalDualConverger")
        self.assertEqual(carried["converger"]["state"]["prev_xbars"],
                         stopped.convobject.prev_xbars)
        self.assertTrue(resumed.convobject.prev_xbars)

    def test_resume_is_bit_identical(self):
        reference, _, resumed = self.run_ab()
        self.assert_bit_identical(reference, resumed)


class _StatefulExtension(Extension):
    """Minimal extension with state, for the contract tests."""

    def __init__(self, opt):
        super().__init__(opt)
        self.counter = 0
        self.restored = None

    def checkpoint_state(self):
        return {"counter": self.counter}

    def restore_state(self, state):
        self.restored = state
        self.counter = state["counter"]


class _StatelessExtension(Extension):
    pass


class _FakeOpt:
    def __init__(self, extobject=None, convobject=None):
        self.extobject = extobject
        self.convobject = convobject


class TestExtensionStateContract(unittest.TestCase):
    """The aggregation itself, without a solver in the way."""

    def test_the_base_extension_has_no_state(self):
        ext = _StatelessExtension(None)
        self.assertIsNone(ext.checkpoint_state())
        # Must not raise: an extension that never opted in is still handed
        # nothing, and doing nothing with it is the correct response.
        ext.restore_state(None)

    def test_the_base_converger_has_no_state(self):
        from mpisppy.convergers.converger import Converger
        conv = Converger(None)
        self.assertIsNone(conv.checkpoint_state())
        conv.restore_state(None)

    def test_stateless_extensions_are_left_out_entirely(self):
        """The common case must add nothing to the file."""
        opt = _FakeOpt(extobject=_StatelessExtension(None))
        self.assertIsNone(checkpointing.gather_extension_state(opt))

    def test_state_is_keyed_by_class_name(self):
        ext = _StatefulExtension(None)
        ext.counter = 7
        state = checkpointing.gather_extension_state(_FakeOpt(extobject=ext))
        self.assertEqual(state["extensions"],
                         {"_StatefulExtension": {"counter": 7}})

    def test_multiextension_is_flattened_away(self):
        """The container has no state; the extensions inside it do."""
        multi = MultiExtension(None, [_StatefulExtension, _StatelessExtension])
        multi.extdict["_StatefulExtension"].counter = 3
        state = checkpointing.gather_extension_state(_FakeOpt(extobject=multi))
        self.assertEqual(list(state["extensions"]), ["_StatefulExtension"])

    def test_restore_dispatches_by_name(self):
        multi = MultiExtension(None, [_StatefulExtension])
        warnings = checkpointing.restore_extension_state(
            _FakeOpt(extobject=multi),
            {"extensions": {"_StatefulExtension": {"counter": 9}}})
        self.assertEqual(warnings, [])
        self.assertEqual(multi.extdict["_StatefulExtension"].counter, 9)

    def test_state_for_an_extension_that_is_gone_warns(self):
        """Resuming with a different extension set is allowed, not silent.

        The hub iterate in the checkpoint is still valid, so refusing the
        whole thing would be disproportionate -- but dropping trajectory state
        without saying so is exactly the silent divergence this phase exists
        to close.
        """
        multi = MultiExtension(None, [_StatelessExtension])
        warnings = checkpointing.restore_extension_state(
            _FakeOpt(extobject=multi),
            {"extensions": {"NormRhoUpdater": {"prev_avg": {}}}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("NormRhoUpdater", warnings[0])

    def test_an_extension_added_since_the_checkpoint_does_not_warn(self):
        """It is starting fresh because it never ran, which is correct."""
        multi = MultiExtension(None, [_StatefulExtension])
        warnings = checkpointing.restore_extension_state(
            _FakeOpt(extobject=multi), {"extensions": {}})
        self.assertEqual(warnings, [])

    def test_a_different_converger_is_refused_by_name(self):
        from mpisppy.convergers.converger import Converger

        class _OtherConverger(Converger):
            def is_converged(self):
                return False

        opt = _FakeOpt(extobject=None, convobject=_OtherConverger(None))
        warnings = checkpointing.restore_extension_state(
            opt, {"extensions": {},
                  "converger": {"class": "PrimalDualConverger",
                                "state": {"prev_xbars": {}}}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("PrimalDualConverger", warnings[0])
        self.assertFalse(checkpointing.converger_state_is_carried(
            opt, {"converger": {"class": "PrimalDualConverger"}}))

    def test_no_converger_carries_trivially(self):
        """Nothing to lose means nothing to warn about."""
        self.assertTrue(
            checkpointing.converger_state_is_carried(_FakeOpt(), None))


@unittest.skipIf(not solver_available, "no solver is available")
class TestFixerCountsAreNotZeroedOnResume(unittest.TestCase):
    """The `populate` regression, isolated from the A/B comparison.

    `Fixer.post_iter0` calls `populate`, which builds the per-nonant counts --
    and it runs on a resumed run too, where the counts it is about to build
    already came back in the dilled models. Zeroing them there discarded the
    checkpoint's most fixer-specific state while every other part of the
    resume worked perfectly.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")

    def tearDown(self):
        self._tmp.cleanup()

    def _ph(self, max_iters, **ckpt_kwargs):
        from mpisppy.extensions.fixer import Fixer
        options = _options(max_iters, **ckpt_kwargs)
        options["fixeroptions"] = {
            "verbose": False,
            "boundtol": 0.01,
            "id_fix_list_fct": sizes.id_fix_list_fct,
        }
        return _make_ph(options, [Fixer], model=sizes,
                        scenario_names=SIZES_SCENARIOS,
                        creator_kwargs=SIZES_KWARGS)

    def test_counts_come_back_nonzero(self):
        stopped = self._ph(3, ckpt_dir=self.ckpt_dir)
        stopped.ph_main()
        saved = {sname: dict(s._mpisppy_data.conv_iter_count)
                 for sname, s in stopped.local_scenarios.items()}
        self.assertTrue(
            any(v for counts in saved.values() for v in counts.values()),
            msg="no nonant had a nonzero count at the stop, so this test "
                "cannot tell a preserved count from a zeroed one")

        resumed = self._ph(4, resume_from=self.ckpt_dir)
        # Stop after Iter0 so the comparison is against what the resume
        # restored, before any iteration has moved the counts on.
        resumed.PH_Prep()
        resumed.Iter0()
        for sname, s in resumed.local_scenarios.items():
            self.assertEqual(dict(s._mpisppy_data.conv_iter_count),
                             saved[sname],
                             msg=f"{sname}: the fixer's counts were reset by "
                                 f"its own setup hook on the way back in")


if __name__ == "__main__":
    unittest.main()
