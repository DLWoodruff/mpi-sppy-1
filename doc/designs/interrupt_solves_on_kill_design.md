# Interrupting in-flight spoke solves on the kill signal — design

**Status:** Draft. Nothing implemented. Branch `kill_running_solves` (off
Pyomo/mpi-sppy `main`).
**Author:** dlw (captured with Claude Code assistance)
**Last updated:** 2026-09-14

---

## 0. Problem

When the hub finishes it calls `Hub.send_terminate()`, which writes 1.0 to
the `SHUTDOWN` field. Spokes see it only when they next call
`Spoke.got_kill_signal()`, which they do between solves. A spoke that is in
the middle of a long subproblem solve (a hard MIP in an xhat spoke, a
Lagrangian subproblem) runs that solve to completion first, and the hub
waits for it at the `cylinder_comm.Barrier()` / `fullcomm.Barrier()` in
`WheelSpinner.run()`. With many scenarios per rank, "to completion" can mean
finishing every remaining solve in `solve_loop`.

Goal: a spoke solve that is running when the kill signal arrives stops
promptly, its result is discarded, and the spoke reaches its next
`got_kill_signal()` without raising and without breaking lockstep with the
other ranks of its cylinder. Solves must not get slower when no kill signal
arrives.

Consequence to accept: a bound the spoke would have delivered by finishing
its solve during the shutdown wait is not delivered.

## 1. What aph-fw does (commit b19eabb, PR #79)

- The controller thread does all MPI and decides termination. It sets a
  per-scenario `std::atomic<bool>` abort flag and then, from the controller
  thread, calls each solver's thread-safe interrupt on the solve running in
  a worker thread: Gurobi `model->terminate()` (only if a solve is in
  progress, because in its C++ interface a pending terminate carries into
  the next solve), CPLEX `IloCplex::Aborter::abort()` (cleared before each
  solve), Xpress `XPRSinterrupt(prob, XPRS_STOP_USER)`.
- No solver callback polls anything, and no MPI call happens during a solve.
- An interrupted solve is recognised by status (`GRB_INTERRUPTED`,
  `IloCplex::AbortUser`, `XPRS_STOP_USER`) and returns without reading the
  solution and without throwing; the caller discards the result. The FW
  loop and task dispatch also check the flag between solves.
- The abort flag is reset when a task is dispatched, so the final
  Lagrangian solves at shutdown run normally.
- `--abort-inflight-solves` (default on) turns it off.
- The commit reports shutdown on sslp_10_50_500_b10_50 (np=4, 15 s wall
  limit) dropping from about 10 s to 4.4 s with the same final bound.

## 2. Approaches measured

A spoke rank runs its solves on the main Python thread, so something has to
reach a solve that is blocking that thread. Two ways were measured on
2026-09-14: polling from a solver callback, and a watcher thread. All
measurements: Gurobi 13.0.2, CPLEX 22.2.0 (pip wheel), Open MPI 5.0.8,
16-core workstation. Probe scripts are kept out of the repo.

### 2.1 Polling from a solver callback — rejected

The callback would read `SHUTDOWN` and call `terminate()`/`abort()`/
`interrupt()` from inside the solve.

**Cost.** markshare2 (`utils/callbacks/termination/tests/markshare2.py`) on
`gurobi_persistent` with `WorkLimit=10` and `Seed=0`, so every variant does
the identical search (node counts match exactly); differences are wall-clock
overhead. Median of 3.

| Callback installed | Threads=1 | Threads=4 |
|---|---|---|
| none | 10.06 s | 6.30 s |
| no-op Python function | ×1.06 | ×1.06 |
| `time.monotonic()` rate-limit check (kill check minus the RMA read) | ×1.08 | ×1.09 |
| what `TimedMIPGapCB` does today (3 `cbGet` at MIP) | ×1.09 | ×1.11 |

Spare cores do not help (Threads=4 ran on a 16-core machine): the solver
waits for each Python callback to return, and they are serialized on one
thread. markshare2 has very cheap nodes and so a lot of callbacks per second
(about 745,000 in 5 s); models with more expensive nodes may pay less, which
has not been measured. This cost is paid by every spoke solve for the whole
run to shorten one shutdown, so the approach is rejected.

Other findings from this approach, kept because they matter for
`TimedMIPGapCB`:

- Callback threads (4 solver threads). Gurobi: every `where`, MIP and LP
  (simplex, barrier, concurrent), on the main thread only. CPLEX:
  `MIPInfoCallback` and LP callbacks on the main thread, except concurrent LP
  (`lpmethod=6`), where every call was on a worker thread; `NodeCallback` ran
  on worker threads under parallel B&B.
- A Gurobi `terminate()` from inside a callback never carried into the next
  solve (re-solve, `set_objective` between, callback removed, and on
  `gurobi_direct`). A `terminate()` issued after the solve had finished was
  ignored and the solve reported OPTIMAL.
- Pyomo's `gurobi_direct` honors `plugin._callback`.

### 2.2 Watcher thread — chosen

A Python thread sleeps, reads `SHUTDOWN`, and calls `model.terminate()` on
the running solve from outside it, as aph-fw does.

- **gurobipy releases the GIL during `optimize()`.** A watcher on a 0.1 s
  sleep ticked on schedule throughout a solve (largest gap between ticks
  0.10 s).
- **Interrupt latency.** `terminate()` at 2.00 s; `optimize()` returned
  `INTERRUPTED` at 2.00 s.
- **No carry-over.** `terminate()` called while no solve was running did not
  affect the next solve (it ran to its full 1 s time limit), also with a
  `set_objective` in between. So a watcher that calls `terminate()` just
  after a solve returns does no harm.
- **Cost.** Same fixed-work benchmark under `mpiexec -np 1`; the watcher does
  a real `Lock`/`Get`/`Unlock` on an MPI window
  (`MPI_THREAD_MULTIPLE` provided):

  | Threads | no watcher | read every 0.1 s | read every 0.001 s | no watcher, repeated |
  |---|---|---|---|---|
  | 1 | 10.20 s | ×0.994 | ×1.006 | ×0.993 / ×0.998 |
  | 4 | 16.63 s | ×1.004 | ×1.036 | ×1.012 / ×1.025 |

  Node counts identical in every row. The watcher's differences are within
  the spread of repeated runs without it, even reading every millisecond.
  (Threads=4 is slower than in §2.1 because `mpiexec` binds the rank to one
  core and the 4 Gurobi threads shared it. That affects any threaded solver
  in an mpi-sppy run and is separate from this design.)
- **Not measured:** CPLEX and Xpress — whether their Python APIs release the
  GIL during the solve, and whether `Aborter.abort()` / `interrupt()` from
  another thread behave as Gurobi's `terminate()` does.

The cost moves from the solver to MPI: an RMA read from a second thread
needs `MPI_THREAD_SERIALIZED` or better (§3.2).

## 3. Design

### 3.1 A non-collective read of `SHUTDOWN`

`got_kill_signal()` ends with `allreduce_or`, a collective, and it updates
the shared receive buffer's write-id bookkeeping. The watcher needs neither.
It reads only the `SHUTDOWN` value into its own one-element array:

```python
class Spoke:
    def _read_shutdown_value(self, dest):
        """Non-collective RMA read of SHUTDOWN into ``dest``. Touches no
        shared receive buffer, so it is safe from the watcher thread."""
        self.window.get(dest, 0, Field.SHUTDOWN, item_offset=0, item_count=1)
        return dest[0] == 1.0
```

`SPWindow.get` with `item_count` already supports a partial read. The
flexible-rank path addresses the hub by global rank through the overlap
maps; the equivalent single-value read there must be written and checked
(§7).

`got_kill_signal()` is unchanged.

### 3.2 `SolveInterrupter`

One per spoke rank, in `mpisppy/cylinders/solve_interrupter.py`.

State, all read and written under the GIL:

- `target` — the solver plugin currently solving, or `None`.
- `kill_seen` — set by the watcher the first time `SHUTDOWN` reads 1.0;
  never cleared.
- `fired_during_solve` — set by the watcher when it calls the interrupt on
  the current `target`; cleared when a solve starts.
- `mpi_lock` — a `threading.Lock` that keeps the two threads out of MPI at
  the same time.

**MPI rule.** The main thread holds `mpi_lock` at all times except while it
is inside a solver plugin's `solve()`. It releases the lock on entering a
solve and re-acquires it on leaving, which blocks until any watcher read in
progress finishes. The watcher takes `mpi_lock` with a non-blocking
`acquire` around each RMA read and skips the tick if it can't get it. So the
two threads are never in MPI at once, and `MPI_THREAD_SERIALIZED` suffices.
Nothing inside `plugin.solve()` makes MPI calls (agnostic guests are out of
scope, §5).

**Main thread, in `SPOpt.solve_one`:**

```python
with spcomm.interrupter.solving(s._solver_plugin):
    results = s._solver_plugin.solve(s, **solve_keyword_args, load_solutions=False)
```

`solving()` sets `target`, clears `fired_during_solve`, and releases
`mpi_lock`. On exit it sets `target = None` and re-acquires `mpi_lock`.

**Watcher loop:**

```python
while not stop.wait(interval):
    plugin = target
    if plugin is None:
        continue
    if not kill_seen:
        if not mpi_lock.acquire(blocking=False):
            continue
        try:
            kill_seen = spoke._read_shutdown_value(dest)
        finally:
            mpi_lock.release()
    if kill_seen and target is plugin:
        interrupt(plugin)            # per-solver, table below
        fired_during_solve = True
```

- The watcher reads MPI only while a solve is running. Between solves the
  main thread's own `got_kill_signal()` covers it.
- After `kill_seen`, the watcher calls the interrupt again on every tick
  while a solve is running. `gurobi_direct` builds a new Gurobi model inside
  `solve()`, so a `terminate()` that lands before the model exists hits the
  old one and has no effect; the next tick reaches the new model. Repeated
  and late calls are harmless (§2.2).

**Per-solver table**, keyed by plugin class:

| Plugin | `interrupt(plugin)` | `was_interrupted(plugin)` |
|---|---|---|
| `GurobiPersistent`, `GurobiDirect` | `plugin._solver_model.terminate()` | `plugin._solver_model.Status == GRB.INTERRUPTED` |
| CPLEX, Xpress | not yet (§7) | — |

A plugin not in the table gets no interrupt, but still benefits from
`kill_seen` in §3.4. If the option is on and the spoke's solver is not in
the table, rank 0 of the spoke says so once.

**Lifetime.** `WheelSpinner.run()` starts the interrupter on each spoke rank
immediately before `spcomm.main()` and stops and joins it immediately after.
The hub never has one. Solves after `main()` (a `post_everything`
extension, anything in `finalize`) run with no watcher, so the `SHUTDOWN`
value that stays at 1.0 can't interrupt them.

**Thread level.** At start, the interrupter checks `MPI.Query_thread()`. If
it is below `MPI_THREAD_SERIALIZED`, the option is refused with a message
naming the level provided. `WheelSpinner.run()` makes this check on every
rank, the hub included, before any `main()`, and combines the results with
an allreduce over `fullcomm`. Every rank then refuses together, not one rank
on its own.

### 3.3 Discard results from an interrupted solve

In `solve_one`, before the `outer_bound_only` / `not_good_enough_results`
branches:

```python
if interrupter.fired_during_solve and was_interrupted(s._solver_plugin):
    s._mpisppy_data.solution_available = False
    s._mpisppy_data.outer_bound = None      # already cleared before the solve
    s._mpisppy_data.inner_bound = None
    s._mpisppy_data.termination_condition = "interrupted_by_kill_signal"
    return
```

- Both conditions are needed. `fired_during_solve` alone is wrong when the
  `terminate()` landed after the solve finished; the solve reports OPTIMAL
  and keeps its result. The solver status alone is wrong when something
  else interrupted the solve (`TimedMIPGapCB`), because that result is
  wanted.
- Use Gurobi's status, not the Pyomo termination condition. Pyomo maps
  `INTERRUPTED` to `TerminationCondition.error`; only
  `set_gurobi_callback`'s `_postsolve` wrapper rewrites it to
  `userInterrupt`, and only on persistent plugins.
- No `load_vars`, and no raise whatever `need_solution` says. Without this,
  `LagrangianOuterBound` (`need_solution=not self.outer_bound_only`) would
  raise on the one rank whose solve was interrupted with no incumbent,
  leaving the rest of its cylinder in the next collective.
- Incumbents and best bounds from an interrupted solve are valid but weak;
  duals and reduced costs are not valid. The run is ending, so none of it
  is kept.

### 3.4 Skip the remaining solves on this rank

Before each solve in `solve_loop` (and in the spokes' own per-scenario solve
loops), if `interrupter.kill_seen`, mark the scenario as in §3.3 without
solving. This helps every solver, including those with no interrupt in the
table. The collectives after the loop (`Eobjective`, `Ebound`'s
missing-bound `Allreduce`) still run on every rank; `Ebound` already handles
`outer_bound is None`.

Spoke code between the solve and the next `got_kill_signal()` must not send
anything computed from a skipped or interrupted solve. The bound spokes send
only when `Ebound`/`Eobjective` succeed, so this is mainly an audit of
`reduced_costs_spoke`, `cross_scen_spoke` and `fwph_spoke`.

## 4. Configuration

- `--interrupt-solves-on-kill` (bool).
- `--kill-check-interval` (seconds): the watcher's sleep. Measured at 0.1 s
  and 0.001 s with no detectable cost (§2.2).

Default on or off is open (§7).

## 5. Out of scope

- Interrupting the hub's own solves. The hub decides when to stop.
- Agnostic guests: their solve runs through the guest's own code, which may
  make MPI calls, so the MPI rule in §3.2 does not hold.
- `appsi_gurobi` and the `pyomo.contrib.solver` Gurobi interface.
- The cost of `TimedMIPGapCB`'s callback (§2.1). Separate design.

## 6. Testing

- Unit, no MPI: construct a `SolveInterrupter` with its read function
  replaced by one that returns True after a short time. On markshare2 with
  `gurobi_persistent` and `gurobi_direct`: the solve returns early,
  `solution_available` is False, no raise with `need_solution=True`.
- No carry-over, as a test: interrupt a solve, then solve again with the
  read returning False; the second solve is not `INTERRUPTED`.
- Late fire: interrupter fires after the solve has finished; the result is
  kept.
- `kill_seen` before `solve_loop`: no solves run and nothing raises.
- Thread level: with `Query_thread` patched below `SERIALIZED`, the option is
  refused on every rank.
- MPI: a hub that stops at iteration 1 with a spoke on a slow MIP; the run
  finishes without waiting out the spoke's solve, and the spoke reports no
  bound from it. Existing cylinder tests pass with the option on.
- markshare2 is small enough for CI's community-edition solvers. Skip
  decorators are checked with the solver binary hidden.
- New test files go into the GitHub workflow and `run_coverage.bash`.

## 7. Open questions

1. The `MPI.Query_thread()` level the cluster MPIs provide (MVAPICH in
   particular), and what mpi4py's default `MULTIPLE` request costs there.
2. The single-value `SHUTDOWN` read on the flexible-rank path.
3. CPLEX and Xpress: GIL release during the solve, and thread-safe
   `Aborter.abort()` / `interrupt()`. Each is a new row in the §3.2 table.
4. Gurobi's documentation on calling `terminate()` from a non-callback
   thread. Measured to work (§2.2), but not confirmed against the docs.
5. Default on or off.
