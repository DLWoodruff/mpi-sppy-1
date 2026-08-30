# PH divergence study: farmer, three crops, large rho

Question: with a very large rho, does PH diverge on the three-crop farmer --
does its convergence metric go up?

Setup: serial PH hub only (no spokes), `examples/farmer/farmer.py` with
`crops_multiplier=1` (3 crops) and 3 scenarios, continuous, gurobi_direct,
100 iterations, `convthresh = -1` so nothing stops early.  The metric is
`phbase.convergence_diff()`, i.e. `||x_s - xbar||_1 / n`.

Run it with:

    python farmer_ph_divergence.py --itermax 100

## Result

    EF optimum -108390.00 at CORN0=80, SUGAR_BEETS0=250, WHEAT0=170

           rho    conv_last   rises    max jump    |xbar-x*|   drift/iter
           0.1       0.1017      38       4.829       0.3812       0.5197
             1    0.0002579      38       40.08    0.0009677    0.0007006
            10     0.003702      35   3.906e+05      0.01389     0.007888
           100    8.148e-09      46    5.42e+06        32.03         0.13
          1000    9.495e-08      39   3.533e+05        47.68        0.043
         10000    8.437e-08       2       1.168        56.63      0.01726
        100000     2.09e-08       0           1        58.16     0.001726
         1e+06    1.488e-08       0           1        58.32    0.0001726
         1e+08     5.55e-12       0           1        58.33    1.726e-06

`rises` counts iterations (after the first) where conv went up, `max jump` is
the largest single-iteration increase as a ratio, `|xbar-x*|` is the sup-norm
distance from the final xbar to the EF root solution.

## Answer

No -- at very large rho PH does not diverge.  The metric goes the other way:
it collapses to solver tolerance (1e-8) at iteration 2 and stays there, and
that is exactly the problem.  It reports convergence while xbar sits 58 acres
away from the optimum.

The mechanism is that the farmer's first-stage feasible set (`sum acres <=
500`) contains xbar, so once the proximal term dominates the linear cost, the
subproblem solution is just the projection of xbar onto that set, which is
xbar itself.  Every scenario returns the same x, the primal residual is zero,
and PH is frozen.  xbar still creeps toward the optimum at a rate exactly
proportional to 1/rho -- 0.1726/rho acres per iteration, visible in the
`drift` column across rho = 1e4 ... 1e8 -- so at rho = 1e5 you would need
roughly 30,000 iterations to close the gap.  Stalling, not divergence.

The metric *does* go up, but in the middle of the rho range, not at the top.
At rho = 100 and 1000 it repeatedly falls to 1e-10 and then jumps back to
order 1 -- a 5.4e6x single-iteration increase at rho = 100.  Those are the
spikes in the left panel of `farmer_divergence.png`.  Each spike is xbar
drifting far enough that some scenario's LP flips to a different vertex, all
three disagree for a few iterations, then the proximal term pins them back
together.  So the metric is non-monotone there but the run is still making
progress.

The headline for a divergence study: on this problem the convergence metric
is not a proxy for optimality.  Near-zero conv at large rho means "the
subproblems agree", not "we are at the answer" -- the right panel of the plot
puts conv and true distance to the optimum side by side.

## Files

- `farmer_ph_divergence.py` -- the driver; `ConvTracer` is a PH extension
  whose `miditer()` runs right after phbase sets `self.conv`, so the trace is
  exactly the sequence phbase compares against `convthresh`
- `farmer_divergence.csv` -- per-iteration rho, conv, ||W||_inf, |xbar-x*|, xbar
- `farmer_divergence.png` -- conv vs iteration, and true error vs iteration
