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
proportional to 1/rho -- 172.6/rho acres per iteration, visible in the
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

## FWPH

`farmer_fwph_divergence.py` runs the same instance and sweep through FWPH,
whose own check is Boland Algorithm 3 (`sum_s p_s ||x^QP_s - xbar||_2^2`, a
weighted SQUARED two-norm, not on PH's scale) and which also reports an outer
bound.

       rho     conv_last  rises   max jump   bound first    bound last   |xbar-x*|
       0.1       0.05493     28      104.3       -115406       -108391      0.4279
         1      1.24e-07     36       5723       -115406       -108390    0.000643
        10     5.784e-05     31  6.761e+12       -115406       -108392     0.01389
       100     1.011e-17      5   3.67e+13       -115406       -112258       32.03
      1000     1.603e-23      4  6.786e+12       -115406       -112978       47.68
     10000     3.102e-24      0          1       -115406       -115406       56.63
    100000     3.229e-24      0          1       -115406       -115406       58.16
     1e+06     3.701e-24      0          1       -115406       -115406       58.32
     1e+08     3.375e-25      0          1       -115406       -115406       58.33

Same answer, same numbers.  At rho >= 1e4 the metric never rises, the outer
bound never moves off its first value, and xbar ends the same 58.33 acres from
x*.  drift*rho is 172.5556, 172.5556, 172.556, 172.6 at rho = 1e4 .. 1e8,
matching PH to every digit either run resolves.

Why they coincide: the FW inner loop looks for the convex combination of
columns minimizing the subproblem objective, and when the proximal term
dominates that minimizer is xbar whatever the columns are.  The columns stop
mattering and the major iteration becomes the same 1/rho projected gradient
step.  Large rho makes the Frank-Wolfe machinery inert rather than merely
unhelpful.

`rises` and `max jump` ignore changes below conv = 1e-12.  Beneath that the
squared metric is machine zero and it wanders between 1e-23 and 1e-26 on
solver noise; counting those as rises would report a diverging metric for a
run that has frozen, which is the exact mistake this study is about.

The outer bound is a second witness PH does not have, and it says the same
thing: frozen at the iteration-1 value to every digit printed.

## The integer case

Same instance with the acreage variables declared general integer
(`--integer` on both drivers).  The EF answer is unchanged, (80, 250, 170) at
-108390, because the LP relaxation happens to be integral.

PH:

           rho    conv_last   rises    max jump    |xbar-x*|   drift/iter
             1            0      14        2.25            0            0
            10       0.2963      21           2       0.3333       0.6667
           100            0       2         1.5           47            0
          1000            0       0           1           58            0
         10000            0       0           1           59            0
        100000            0       0           1           59            0
         1e+06            0       0           1           59            0
         1e+08    9.474e-15       0           1           59    1.723e-06

FWPH ends at |xbar-x*| = 0.00064, 0.0128, 32.1, 47.7, 56.6, 58.2, 58.3, 58.3
over the same rhos, with its outer bound frozen at -115400 for rho >= 1e4 --
the continuous FWPH column to three figures.

Three findings.

1. Large rho freezes harder, not less.  For rho >= 1e3 the metric is exactly
   0.0, not merely below tolerance, while xbar is 58-59 acres off.  No
   threshold helps because no positive threshold is below zero.  drift is 0
   too: the 172.6/rho first-order correction is smaller than the one-acre
   grid, so it cannot move an integer variable and xbar is a true fixed point
   rather than a slow crawl.

2. rho=10 is a limit cycle, which the continuous run does not have.  conv
   floors at 0.2963 and cycles through {0.2963, 0.4444, 0.5185} with period
   about fourteen; xbar orbits x* at distance 0.333 to 1.0, still moving
   0.667/iteration at iteration 100.  The one place in the study where an
   algorithm neither converges nor freezes.  Still bounded, so still not
   divergence.

3. FWPH is unaffected by integrality, as expected: its QP is over the convex
   hull of the columns, a continuous set regardless of whether the MIP
   subproblems generating those columns are integral.

`rises` uses a 1e-5 floor here, not the 1e-12 used for the continuous runs: a
MIQP returns integers only to its integrality tolerance, so xbar wobbles in
the sixth decimal while the true iterate is constant.  Without the floor,
rho=1e8 reported 46 rises for a run that had not moved at all.  Both drivers
take `--conv-floor` to override.

## Files

- `farmer_ph_divergence.py` -- the PH driver; `ConvTracer` is a PH extension
  whose `miditer()` runs right after phbase sets `self.conv`, so the trace is
  exactly the sequence phbase compares against `convthresh`
- `farmer_divergence.csv` -- per-iteration rho, conv, ||W||_inf, |xbar-x*|, xbar
- `farmer_divergence.png` -- conv vs iteration, and true error vs iteration
- `farmer_fwph_divergence.py` -- the FWPH driver; same tracer hook, since
  fwph's `iterk_loop` also calls `miditer()` right after setting `self.conv`
- `farmer_fwph_divergence.csv` -- per-iteration conv, outer bound, error, xbar
- `farmer_divergence_int.csv`, `farmer_divergence_int.png`,
  `farmer_fwph_divergence_int.csv` -- the same for `--integer`
- `aph-fw/doc/divergence-report/ph_divergence.tex` -- the write-up, which
  lives in the aph-fw repo with the rest of the divergence investigation
