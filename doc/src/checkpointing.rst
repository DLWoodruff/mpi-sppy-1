.. _checkpointing:

Checkpointing and Resuming a Run
================================

A long Progressive Hedging run can be stopped and picked up later. The intended
use is a planned stop: a multi-day study that ends each day and resumes the next
morning on the same cluster, losing at most the work done since the last
checkpoint.

A resumed run picks up from the last checkpoint that was **published**, which is
not always the iteration at which the run stopped. How far back that can be is
set by ``--checkpoint-every-iterations``; read `Choosing K`_ before relying on
this for anything expensive.

Checkpointing is entirely opt-in. With no ``--checkpoint-dir`` the machinery is
not attached at all, and a run that does not ask for it pays nothing.

.. note::
   The current implementation covers a **synchronous PH hub**, run on its own
   or with spokes, on any number of ranks per cylinder, with plain scenarios,
   proper bundles, or stoch-ADMM. Other hub types (notably ``--APH``) are
   refused at startup. See ``doc/designs/checkpointing_design.md`` for the full
   design and the phased rollout.

Writing a checkpoint
--------------------

Give a directory and the run keeps a checkpoint up to date as it goes::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --time-limit 28800 \
      --checkpoint-dir ./ckpt

A checkpoint always describes a **completed** PH iteration -- never a partial
one -- and only one is kept at a time, so the directory holds the most recent
iteration that was both completed and due a write. By default every completed
iteration is due one; ``--checkpoint-every-iterations`` changes that, and on a
real model you will want it to (`Choosing K`_).

``--time-limit`` is how you get a run to stop on its own before a scheduler
kills it, but be clear about what it does: the elapsed time is compared against
the limit **once per iteration, at the top**, and nowhere else. It is not
checked during a solve. So a run whose iterations take an hour can overshoot the
limit by nearly an hour, and you have to set the limit below your wall-clock
allocation by at least one iteration's worth of solve time for it to help at
all.

It also does not force a checkpoint to be written. That is what
``--checkpoint-before-seconds`` is for (`Guarding against a scheduler
timeout`_).

The options
~~~~~~~~~~~

There are five, and they are the whole interface:

``--checkpoint-dir DIR``
   Where checkpoints are written. Giving it is what turns checkpointing on;
   without it none of this machinery is attached.

``--checkpoint-every-iterations K``
   Write at every K-th completed iteration instead of every one. The default is
   ``1``, but on a real model you will want it higher -- see `Choosing K`_,
   which is the one decision here that repays some thought.

``--checkpoint-before-seconds S``
   Also write once, at the last iteration boundary expected to arrive within S
   seconds of the start of the run. For a run that will be stopped by a clock
   rather than by its iteration limit -- see `Guarding against a scheduler
   timeout`_.

``--resume-from DIR``
   Start from the checkpoint in ``DIR`` instead of from scratch.

``--checkpoint-backend``
   How model state is stored. ``dill-reload`` is the default and currently the
   only implemented value, so there is no reason to set it.

``--max-iterations`` is not a checkpoint option, but it interacts with them and
is the one people trip over: it bounds the **study**, not the leg. See
`Resuming`_.

.. warning::
   **Nothing is published until the first iteration completes.** A run killed
   during startup, or during the iteration-0 solve, leaves no checkpoint at all
   -- not an empty one, nothing to resume from. For the case this feature exists
   for, large MIP subproblems, that iteration-0 solve is often the longest in
   the whole run, so the window is not small.

Choosing K
~~~~~~~~~~

A checkpoint serializes every scenario model this rank owns. That cost scales
with the size of your models, and the only number that matters is how it
compares with a solve: if serializing a scenario takes a small fraction of the
time to solve it, checkpointing every iteration is free in practice, and if it
does not, it is not. Measure it on your own model rather than guessing -- the
bracketing ``toc`` lines around each write report it directly (`What it
costs`_).

``--checkpoint-every-iterations K`` writes less often::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 50 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --checkpoint-dir ./ckpt --checkpoint-every-iterations 10

**What K means.** A checkpoint is written at the end of a completed iteration
whose number is a multiple of K. The numbers are the PH iteration numbers you
see in the log, counted from the start of the study, so with ``K = 10`` the
checkpoints are iterations 10, 20, 30, and so on. K is not a countdown from
whenever the current run happened to begin: it does not restart at a resume,
and there is no drift. The default is ``1``, which writes at every iteration.

**What it costs you.** Only one checkpoint is kept, so at any moment the
directory holds the most recent multiple of K that completed. If the run stops
anywhere else -- a time limit, convergence, a crash -- the iterations after
that point are gone and the resumed run redoes them. That is at most K-1
iterations, and it is the whole trade: you are buying back write time with
work you are willing to repeat.

For example, with ``K = 10`` a run that stops at iteration 34 leaves a
checkpoint of iteration 30, and iterations 31 through 34 are lost. Resuming
picks up at 31 and the next checkpoint is iteration 40 -- not 41, because the
count follows the study, not the resumed leg.

**How to size it.** "At most K-1 iterations" assumes you reach a multiple of K
at all. If you never do, you lose everything: with ``K = 50`` and a run that
only ever completes 30 iterations, no checkpoint is written and there is
nothing to resume from. So pick K against the number of iterations you expect
to **complete**, not against the iteration limit you set -- and leave enough
margin that the run passes several multiples of K before any plausible
stopping point.

**The one exception.** The final iteration of an exhausted ``--max-iterations``
budget is always written, whatever K is. With ``--max-iterations 100`` and
``K = 30`` the checkpoints are iterations 30, 60, 90 and 100. Raising the limit
and resuming is a normal way to extend a study, and that last iterate is
known-good and already in memory, so it is not worth discarding to save one
write. No other kind of stop can be caught this way: a time limit, the
convergence threshold and a user converger are all tested partway through the
*next* iteration, by which point there is nothing coherent left to write.

Changing K between a stop and a resume is allowed; like the iteration limit,
it describes how the run is managed rather than what problem is being solved,
so it is not part of the check that decides whether a checkpoint may be
resumed.

Two cases worth calling out
"""""""""""""""""""""""""""

**Setting K equal to the iteration limit.** With ``--max-iterations 40
--checkpoint-every-iterations 40`` you get exactly one checkpoint, at iteration
40, and pay for exactly one write. If all you want is the option of extending
the study later by raising the limit, that is the cheapest way to get it.

The catch is that it only works when the iteration limit is what stops the run.
Anything else -- ``--rel-gap``, ``--abs-gap``, the convergence threshold, a user
converger, ``--time-limit``, a crash -- stops it before iteration 40, no write
has happened, and **there is nothing to resume from**. That failure does not
announce itself. The checkpoint directory exists, and in a cylinders run it
already holds a ``spokes/`` subdirectory, because the xhat spokes write their
incumbents on a different trigger. So it looks like checkpointing worked, and
you find out the next day::

  CheckpointMismatch: No checkpoint manifest at './ckpt/manifest.json'.

If you want the cheap single checkpoint, either turn the other stopping
criteria off (``--rel-gap 0.0 --abs-gap 0.0``) or accept that an early
convergence may leave you nothing. ``--time-limit`` is the one of them you can
cover outright -- see below.

**Guarding against a scheduler timeout.** On a batch system the thing to avoid
is the scheduler killing the job partway through a solve. Give the run a
``--time-limit`` comfortably below your wall-clock allocation and it stops
itself at an iteration boundary instead::

  ... --max-iterations 500 --time-limit 28800 \
      --checkpoint-dir ./ckpt --checkpoint-every-iterations 10

``--time-limit`` does not *force* a write. On its own, what you resume from is
the last multiple of K, exactly as for any other stop -- and you cannot predict
which iteration the limit will fall on, so at ``K = 10`` a run stopped this way
loses up to nine iterations, or all of them if it never reached a multiple of K.

``--checkpoint-before-seconds S`` closes that gap::

  ... --max-iterations 500 --time-limit 28800 \
      --checkpoint-dir ./ckpt --checkpoint-every-iterations 10 \
      --checkpoint-before-seconds 28800

At the end of each completed iteration it asks whether *another* iteration
would take the run past S seconds since it started, and writes now if it would.
The estimate is simply how long the last iteration took, so the option is worth
most when your iterations take similar amounts of time; where they vary wildly,
a short iteration followed by a long one can still overrun S.

It writes at most once. Past the deadline every later iteration would qualify
too, and writing at all of them is the cost K was set to avoid.

**S is yours to size, and nothing is added to it.** In particular the write
itself is not: it takes as long as any other checkpoint of your models, and it
starts when S has all but arrived. Read a write's cost off the ``toc`` lines in
your own log (`What it costs`_) and leave room for it, along with anything else
that must happen before the scheduler's axe falls. Setting S equal to
``--time-limit``, as above, is the usual choice: the write then lands at the
last iteration boundary before the time limit stops the run.

.. note::
   There is still no option to write a checkpoint every N seconds. Between K
   and this one, the cases that came up are covered.

Writing only at iteration boundaries is deliberate. PH computes xbar, updates
the dual weights, gives extensions their mid-iteration hook, and only then
solves; a run that stops on ``--time-limit`` or on convergence stops *before*
that solve, leaving the model describing half an iteration. Rather than try to
unwind that -- an open-ended problem, since any extension may have changed rho,
fixed variables or added cuts -- the checkpoint is simply taken at the last
point where everything agrees.

It is also why a run that ends before finishing iteration 1 publishes nothing,
as `The options`_ warns: no iteration completed, so there is no iterate to
write.

Each write is bracketed by a pair of timestamped ``toc`` lines, so the log shows
how long it took::

  [ 1234.56] Writing checkpoint at iteration 42 to ./ckpt
  [ 1261.03] Checkpoint written at iteration 42

A write that fails mid-run -- the disk filling up, a network filesystem
hiccup -- does not stop the optimization. The failure is reported loudly in
the log, the previously published checkpoint stays intact and resumable, and
the next iteration boundary tries again. Conditions detectable at setup (an
unwritable directory, a model that cannot be serialized) still stop the run
at startup, before any solving is done.

Resuming
--------

Point a new run at the directory::

  python -m mpisppy.generic_cylinders --module-name farmer --num-scens 3 \
      --solver-name cplex --max-iterations 100 --default-rho 1.0 \
      --resume-from ./ckpt

The resumed run continues from the checkpointed iterate rather than starting
over. It does **not** re-solve the subproblems at iteration 0 -- for large MIPs
that solve is often the most expensive in the run, and its answer would be
thrown away. Iteration numbering continues where it left off, so
``--max-iterations`` bounds the run as a whole rather than each leg of it.

.. warning::
   That last point is the one thing people get wrong. To resume a run that
   stopped at iteration 4 and take it to 6, pass ``--max-iterations 6``, not
   ``--max-iterations 2``. Passing 2 says the budget was already spent at
   iteration 4, and the resumed run does nothing.

Trying it out
-------------

The quickest way to convince yourself is to run the same problem twice -- once
straight through, once stopped and resumed -- and compare. On ``farmer``, a
deterministic LP, the two should agree exactly.

All three runs below share the same options; only the last line of each
differs::

  GC="mpiexec -np 3 python -m mpi4py -m mpisppy.generic_cylinders \
      --module-name farmer --num-scens 3 --default-rho 1 \
      --solver-name cplex --lagrangian --xhatshuffle \
      --rel-gap 0.0 --abs-gap 0.0"

Run them from ``examples/farmer``, or use ``--module-name
mpisppy.tests.examples.farmer`` from anywhere.

The two gap options are there because this example is small enough to converge
on the inter-cylinder gap at iteration **1**, which would leave nothing to stop
and resume. Turning them off makes the iteration limit the thing that ends the
run. That is a property of the example, not of checkpointing.

**1. A reference run** of six iterations, with no checkpointing at all -- this
is the answer to compare against::

  $GC --max-iterations 6 --solution-base-name ref_soln

**2. The same run, stopped after four iterations**, leaving a checkpoint::

  $GC --max-iterations 4 --checkpoint-dir ./ckpt

**3. Resume it** and let it finish the remaining two iterations::

  $GC --max-iterations 6 --resume-from ./ckpt --solution-base-name resumed_soln

What you should see
~~~~~~~~~~~~~~~~~~~

The resumed run reports what it restored -- the hub's iterate, the spoke's best
solution, and the spoke's place in its own scenario walk::

  Restored the checkpointed incumbent for XhatShuffleInnerBound (objective -108382.2222...)
  Resuming from checkpoint in ./ckpt (iteration 4)
  Restored the checkpointed xhatshuffle cursor (pass 250, next scenario scen0)

It then starts at iteration 5 -- there is no iteration-0 solve and no repeat of
iterations 1 through 4 -- and ends on the same numbers the reference run did:

=====================  ==============  ================  ==========  ==========
run                        Best Bound    Best Incumbent    Rel. Gap    Abs. Gap
=====================  ==============  ================  ==========  ==========
reference (6)            -108931.8045      -108382.2222      0.505%    549.5823
resumed (4 then 2)       -108931.8045      -108382.2222      0.505%    549.5823
=====================  ==============  ================  ==========  ==========

The written solutions match too::

  diff ref_soln.csv resumed_soln.csv
  diff -r ref_soln_soldir resumed_soln_soldir

Both are clean. Farmer is a deterministic LP, so here "very close" is in fact
"identical"; on a MIP under default solver settings expect agreement to a
tolerance instead, since a resumed run can land on a different optimal solution
(see `What resume guarantees`_).

.. note::
   In the resumed run the *Best Incumbent* column reads ``inf`` for an
   iteration or two before the restored value reappears. The spoke restores its
   incumbent immediately but publishes it to the hub at its first checkpoint
   point, so the hub's display lags briefly. Nothing is lost -- the column
   settles back to the restored value, and the final answer is unaffected.

To exercise the multi-rank path as well, give each cylinder more than one rank
-- ``-np 6`` with three cylinders gives each of them two::

  GC="mpiexec -np 6 python -m mpi4py -m mpisppy.generic_cylinders \
      --module-name farmer --num-scens 6 --default-rho 1 \
      --solver-name cplex --lagrangian --xhatshuffle \
      --rel-gap 0.0 --abs-gap 0.0"

``./ckpt/hub/gen_0004/`` then holds a ``hub_rank_NNNN.pkl`` and one ``.dill``
per scenario **for every rank**, and the manifest that publishes the generation
is written only once all of them have finished.

What must match, and what may change
------------------------------------

A checkpoint records the configuration it was written with and refuses to load
into a run that differs, rather than producing a subtly wrong answer. Resuming
requires the same number of MPI ranks with the same scenarios on each, and a
configuration that matches everywhere except a short list of settings a resume
may legitimately change:

* **the budget** -- ``--max-iterations``, ``--time-limit``, the gap and
  stalling thresholds;
* **solver choice and how it is driven** -- ``--solver-name``, solver options
  and thread counts, mipgaps, and every per-cylinder solver setting. Tightening
  a mipgap on day two continues the same problem rather than redefining it;
* **display, tracking and output destinations**, and the checkpoint options
  themselves;
* **which cylinders run.** The hub's primal trajectory does not depend on the
  spokes.

Everything else must match -- including options your own model module
registers. That is deliberate: checking by default is what stops a farmer
checkpoint from being resumed with ``--farmer-with-integers`` and quietly
answering the linear program.

The practical consequence is that a checkpoint is tied to the mpi-sppy and
model version that wrote it. Adding a new option to your model module will
cause existing checkpoints to be refused.

What resume guarantees
----------------------

For the target case -- large MIP subproblems -- a resumed run **continues
correctly**, and never loses or regresses the best solution found so far.

The reloaded models carry the recourse values from the last solve before the
stop, so the first resumed solve *can* warm-start from them -- but only with
``--warmstart-subproblems``, which is off by default and which mpi-sppy does
not turn on for you. Without it that first solve is cold, exactly as every
other solve in the run is. It is **not** bit-for-bit reproducible against a hypothetical
uninterrupted run: multi-threaded MIP solves are not deterministic and admit
multiple optima, so the resumed iterates may differ. That is expected, not a
bug.

Bounds and the incumbent are carried forward as valid best-so-far values. A
resumed run never reports a worse best-so-far than its checkpoint.

In a cylinders run the best solution does not live on the hub: the xhat spoke
that found it holds it. So each xhat spoke keeps its own small file under
``spokes/`` in the checkpoint directory, holding the best solution it has
found, written by variable name whenever that solution improves. A resumed
spoke reads it back and reports it to the hub, which is why a resumed
cylinders run starts from the answer it already had rather than from nothing.

Those files are deliberately not synchronised with the hub's: a spoke writes
when it has something new to record, the hub writes at iteration boundaries,
and neither waits for the other. A spoke whose file is missing -- because the
earlier run stopped before it found anything, or because you resumed with a
different set of spokes -- simply starts without an incumbent and says so in
the log.

An ``xhatshuffle`` spoke also records **where it had got to** in its walk
through the scenarios, so a resumed spoke carries on exploring rather than
re-trying candidates it has already tried. Each re-try it avoids is a
subproblem solve. The scenario order itself is not stored -- the shuffle is
seeded to a fixed value, so a resumed spoke reproduces it exactly -- only the
position in it, which is why a run whose scenario list has changed discards the
position (with a warning) and explores from the start again. The other xhat
spokes re-evaluate from scratch whenever the hub sends new values, so they have
no such position and store none.

On a deterministic LP or QP solve the primal trajectory can come back
bit-identical, but that is a bonus rather than the guarantee.

Disk usage
----------

Exactly one checkpoint is kept. Retaining older generations is not supported.
The new checkpoint is written in full before the old one is deleted, so the
**peak** on disk is two generations -- size disk quotas for that, not for one.
Under the default ``dill-reload`` backend a checkpoint holds a serialized copy
of every local scenario model, which for large MIPs is not small.

Publication is atomic: the new generation is staged, moved into place, and only
then does a manifest rewrite commit it. A run killed at any point leaves a
complete, resumable checkpoint referenced -- never a half-written one -- and
the next successful write reclaims anything the interrupted one left behind.

Use one checkpoint directory per run. Two runs sharing one share a manifest and
will overwrite each other.

Cylinders that span several ranks
---------------------------------

Each rank holds a different slice of the scenarios, so one checkpoint is the
whole set of per-rank files, and it is committed only once every rank has
written its own. If any rank cannot write, none of them publishes: the previous
checkpoint stays on disk as the resumable one and the run carries on to try
again at the next checkpoint point. There is no state in which some ranks have
advanced their checkpoint and others have not.

You get two log lines from such a failure: one from the rank that failed,
carrying the actual cause, and one from rank 0 naming that rank.

The cost is that the ranks wait for each other at each write, which the
bracketing ``toc`` lines include -- the slowest rank sets the pace.

Nothing in this coordination reaches beyond one cylinder: a hub and its spokes
never wait on each other, and neither do two spokes.

What it costs
-------------

By default a checkpoint is written at every completed iteration, so the cost is
one model serialization per iteration. Measured over ten iterations, against
the same run without ``--checkpoint-dir``:

===========================  ==========  ============  ==========
instance                     no ckpt     with ckpt     overhead
===========================  ==========  ============  ==========
farmer, 3 scenarios (LP)       0.62 s        0.88 s        43%
farmer, 50 scenarios (LP)      1.34 s        4.85 s       262%
sizes, 3 scenarios (MIP)       2.28 s        3.15 s        38%
sizes, 10 scenarios (MIP)      5.05 s        7.46 s        48%
===========================  ==========  ============  ==========

.. warning::
   **These are toy models, and the percentages do not carry over.** Farmer has
   three first-stage variables; ``sizes`` is a small MIP. Serializing one of
   their scenarios takes milliseconds, and a real model's can take seconds. Do
   not scale these figures by your scenario count to predict your own overhead
   -- the per-scenario cost is the part that changes, and it changes by orders
   of magnitude.

   What does carry over is the *shape* of the result: the overhead is
   serialization time against solve time. The 50-scenario farmer run is the
   pathological end -- solves so cheap that writing dominates and the run takes
   three times as long. A large MIP whose solves take minutes is the other end,
   where even a slow serialization is a small share.

The bracketing ``toc`` lines are what you should actually go by: their
difference is essentially the whole overhead, so one calibration run on your own
model tells you the cost directly. If it is too high,
``--checkpoint-every-iterations`` buys most of it back in exchange for repeating
some iterations after a stop; see `Choosing K`_.

Requirements and limitations
----------------------------

**dill is required.** ``--checkpoint-backend dill-reload`` is the default and
currently the only implemented backend, and it needs the optional ``dill``
package::

  pip install mpi-sppy[extras]

**Your scenario models must be serializable.** A model can be made
unserializable by what its ``scenario_creator`` closes over -- most commonly a
Pyomo rule written as a nested function that reads ``cfg`` directly, which pulls
the whole configuration object into the model. See :ref:`scenario_creator` for
the pattern and the fix. Checkpointing serializes every scenario on every rank
at setup rather than discovering the problem at the first write, and the error
names the offending rule and the scenario it was found in.

**The synchronous PH hub only.** ``--APH`` and the other hub types are refused
at startup when either ``--checkpoint-dir`` or ``--resume-from`` is given, as
are an unwritable directory (checked from every rank -- on a cluster a path can
be writable from some nodes and not others), an unimplemented backend, scenario
names that would collide once made filename-safe, and any configuration where
the checkpointing extension would not actually be attached. The intent is that
checkpointing either works or says so at startup, rather than running for hours
and writing nothing.

**Extension and converger state is part of a checkpoint.** Extensions that
accumulate their own state across iterations carry it across a stop: the rho
updaters (``--norm-rho``, ``--mult-rho``, ``--sep-rho``, ``--sensi-rho``,
``--grad-rho``), ``fixer``, ``slammer``, and the primal-dual converger. So a
resumed run using one of them follows the same trajectory an uninterrupted run
would, rather than merely continuing correctly from the right models.

The rho-setting extensions do not *recompute* rho at the resume itself: the
checkpointed rho -- including whatever adaptation had happened by the write --
carries over, and the extensions resume their per-iteration updates from there.

Two things this does not cover:

* **Your own extension carries nothing unless you say so.** If it keeps state
  on itself that decides what it does next -- a history, a counter, a record of
  what it has already changed -- implement ``checkpoint_state()`` and
  ``restore_state(state)`` on it. They are no-ops on the base class, so an
  extension that does not need them costs nothing. Return plain data keyed by
  variable *name* or by ``(node name, index)``, never Pyomo objects: a resume
  replaces every model, so a saved variable reference addresses something that
  no longer exists.
* **A converger with no such implementation still starts fresh**, and the resume
  says so in the log. That matters more than it sounds: a converger decides when
  the run stops, so one that accumulates history can terminate a resumed run at a
  different iteration than an uninterrupted one.

If you resume with a *different* set of extensions than the checkpoint was
written with, that is allowed -- the hub's iterate is still valid -- and the run
reports each piece of state it could not hand to anybody.

**The order you attach extensions in does not affect what is checkpointed.**
The write happens at a dedicated point in the iteration loop, after every
extension's end-of-iteration hook has run. So if your own extension uses that
hook to change rho, fix a variable, relax a domain or add a cut, the change is
part of that iteration's checkpoint and is there when you resume.
