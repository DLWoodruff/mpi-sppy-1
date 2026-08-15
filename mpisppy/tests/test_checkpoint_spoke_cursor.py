###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""The xhatshuffle spoke's loop cursor across a checkpoint (design phase 5).

Phase 4 gave a spoke back its best solution; this gives it back its *place*.
An xhatshuffle spoke walks a shuffled list of scenarios, trying each as a
candidate xhat, and remembers which ones it has already tried this epoch. A
resumed spoke that started that walk over would re-try scenarios it had
already tried -- and every try is a subproblem solve, so on a large MIP that
is real time spent re-learning what the checkpoint already knew.

Two things make this cheaper than it sounds, and both are worth stating
because they are what the design leans on:

* **The scenario order is not carried.** The shuffle is seeded to a fixed
  value and drawn once, so a resumed spoke reproduces it exactly. Only the
  *position* in it is checkpointed -- and a position is only meaningful
  against the order it indexes, so the file carries a fingerprint of that
  order and the cursor is discarded if it no longer matches.
* **A cursor move costs a subproblem solve.** So writing the small spoke file
  whenever the cursor moves is negligible against what caused the move, while
  a pass that solves nothing writes nothing.
"""

import os
import tempfile
import unittest

from mpisppy.cylinders.xhatshufflelooper_bounder import ScenarioCycler
from mpisppy.tests.utils import get_solver

solver_available, solver_name, persistent_available, persistent_solver_name = \
    get_solver()


def _cycler(names, iter_step=1):
    """A two-stage cycler over ``names``, in that order.

    ``nonleaves={}`` is what makes it two-stage: the constructor looks for a
    'ROOT' entry with children and finds none, which is the same branch a real
    two-stage problem takes.
    """
    shuffled = list(enumerate(names))
    return ScenarioCycler(shuffled, {}, False, iter_step)


class TestScenarioCyclerState(unittest.TestCase):
    """The cursor round trip, without a solver or an MPI job in the way."""

    NAMES = ["scen0", "scen1", "scen2", "scen3", "scen4"]

    def test_a_fresh_cycler_starts_at_the_beginning(self):
        cycler = _cycler(self.NAMES)
        state = cycler.checkpoint_state()
        self.assertEqual(state["cycle_idx"], 0)
        self.assertEqual(state["cur_root_scen"], "scen0")
        self.assertEqual(state["scenarios_this_epoch"], [])

    def test_the_position_survives_a_round_trip(self):
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()
        saved = cycler.checkpoint_state()

        restored = _cycler(self.NAMES)
        self.assertEqual(restored.restore_state(saved), [])
        self.assertEqual(restored.checkpoint_state(), saved)

    def test_a_restored_cycler_continues_where_the_other_left_off(self):
        """The property that matters: the same scenarios come next.

        Comparing the saved dict to itself only proves the fields round-trip.
        This proves the cursor still *means* the same thing -- that the
        resumed spoke tries what the uninterrupted one would have tried next,
        rather than starting the walk again.
        """
        uninterrupted = _cycler(self.NAMES)
        for _ in range(2):
            uninterrupted.get_next()

        resumed = _cycler(self.NAMES)
        resumed.restore_state(_cycler_after(self.NAMES, 2))

        for _ in range(len(self.NAMES)):
            self.assertEqual(resumed.get_next(), uninterrupted.get_next())

    def test_the_tried_set_survives_so_scenarios_are_not_re_tried(self):
        """`get_next` returns None once the epoch is exhausted.

        A resumed spoke that forgot which scenarios it had tried would hand
        them all back a second time -- a subproblem solve each, for candidates
        already known to be no better.
        """
        cycler = _cycler(self.NAMES)
        tried = []
        while True:
            nxt = cycler.get_next()
            if nxt is None:
                break
            tried.append(nxt["ROOT"])
        self.assertEqual(len(tried), len(self.NAMES))

        restored = _cycler(self.NAMES)
        restored.restore_state(cycler.checkpoint_state())
        self.assertIsNone(
            restored.get_next(),
            msg="the resumed cycler offered a scenario the checkpointed one "
                "had already tried this epoch")

    def test_the_best_scenario_survives(self):
        """`best` decides where the next epoch starts, so it is trajectory."""
        cycler = _cycler(self.NAMES)
        cycler.get_next()
        cycler.best = "scen3"
        restored = _cycler(self.NAMES)
        restored.restore_state(cycler.checkpoint_state())
        self.assertEqual(restored.best, "scen3")
        restored.begin_epoch()
        self.assertEqual(restored.get_next()["ROOT"], "scen3")

    def test_a_changed_scenario_order_is_refused_and_reported(self):
        """A position means nothing against a different list.

        The cursor is an index. If the model's scenario list changed, index 3
        is a different scenario, and restoring it would send the spoke
        somewhere it never meant to go -- silently, since an index is always a
        valid index. So the file carries a fingerprint of the order it was
        taken against.
        """
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()
        saved = cycler.checkpoint_state()

        other = _cycler(self.NAMES + ["scen5"])
        warnings = other.restore_state(saved)
        self.assertEqual(len(warnings), 1)
        self.assertIn("different scenario order", warnings[0])

    def test_a_refused_cursor_leaves_a_usable_cycler(self):
        """Refusing the cursor must not break the spoke.

        The same file carries the incumbent, which is the part worth keeping.
        A cursor that no longer fits is a reason to explore from the start
        again, not a reason to fail the resume.
        """
        cycler = _cycler(self.NAMES)
        for _ in range(3):
            cycler.get_next()

        other = _cycler(self.NAMES + ["scen5"])
        other.restore_state(cycler.checkpoint_state())
        self.assertEqual(other.get_next()["ROOT"], "scen0")

    def test_the_fingerprint_is_order_sensitive(self):
        """The same names shuffled differently must not compare equal."""
        forward = _cycler(self.NAMES)
        backward = ScenarioCycler(
            list(enumerate(self.NAMES))[::-1], {}, False, 1)
        self.assertNotEqual(
            forward.checkpoint_state()["order_fingerprint"],
            backward.checkpoint_state()["order_fingerprint"])


def _cycler_after(names, advances):
    """The state of a fresh cycler advanced ``advances`` times."""
    cycler = _cycler(names)
    for _ in range(advances):
        cycler.get_next()
    return cycler.checkpoint_state()


class _CursorSpokeStub:
    """A spoke whose cursor the test moves by hand."""

    def __init__(self, loop_state=None):
        self.strata_rank = 2
        self.best_inner_bound = None
        self.loop_state = loop_state
        self.sent_bounds = []
        self.sent_xhats = 0

    def send_bound(self, value):
        self.sent_bounds.append(value)

    def send_best_xhat(self):
        self.sent_xhats += 1

    def checkpoint_loop_state(self):
        return self.loop_state


@unittest.skipIf(not solver_available, "no solver is available")
class TestSpokeWritesWhenTheCursorMoves(unittest.TestCase):
    """The write gate: an unchanged incumbent is no longer enough to skip.

    Before this phase the spoke wrote only when its incumbent improved, which
    is rare. The cursor moves far more often -- but only ever as the result of
    a subproblem solve, so the write is cheap against what caused it. What
    still has to cost nothing is a pass that solves nothing, and that is the
    case pinned here.
    """

    def setUp(self):
        from mpisppy.tests.test_checkpoint import _xhat_eval, _set_and_cache_solution
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = os.path.join(self._tmp.name, "ckpt")
        self.opt = _xhat_eval(ckpt_dir=self.ckpt_dir)
        self.spoke = _CursorSpokeStub()
        self.opt.spcomm = self.spoke
        self._set_and_cache_solution = _set_and_cache_solution

    def tearDown(self):
        self._tmp.cleanup()

    def _checkpointer(self):
        from mpisppy.extensions.checkpointer import Checkpointer
        return Checkpointer(self.opt)

    def _spoke_file(self):
        return os.path.join(
            self.ckpt_dir, "spokes",
            "spoke__CursorSpokeStub_strata_02_rank_0000.pkl")

    def test_a_pass_that_changes_nothing_writes_nothing(self):
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        ext.maybe_checkpoint()
        first = os.path.getmtime(self._spoke_file())

        # Same incumbent, same cursor: the spinning-loop case.
        ext.maybe_checkpoint()
        self.assertEqual(os.path.getmtime(self._spoke_file()), first)

    def test_a_cursor_move_alone_triggers_a_write(self):
        import pickle
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        ext.maybe_checkpoint()

        # The incumbent is unchanged; only the cursor moved.
        self.spoke.loop_state = {"xh_iter": 7, "cursor": {"cycle_idx": 3}}
        ext.maybe_checkpoint()
        with open(self._spoke_file(), "rb") as f:
            written = pickle.load(f)
        self.assertEqual(written["loop_state"],
                         {"xh_iter": 7, "cursor": {"cycle_idx": 3}})

    def test_a_spoke_with_no_cursor_carries_none(self):
        """Every xhatter but xhatshuffle, which is most of them."""
        import pickle
        ext = self._checkpointer()
        self._set_and_cache_solution(self.opt, 1.0)
        self.opt.best_solution_obj_val = 10.0
        ext.maybe_checkpoint()
        with open(self._spoke_file(), "rb") as f:
            self.assertIsNone(pickle.load(f)["loop_state"])


if __name__ == "__main__":
    unittest.main()
