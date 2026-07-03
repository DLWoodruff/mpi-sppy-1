.. _boot_sp:

Bootstrap Confidence Intervals
==============================

The ``mpisppy.confidence_intervals.bootsp`` subpackage provides bootstrap
and bagging confidence intervals for the optimality gap (and for the optimal
value and the value at a candidate solution) of *data-based*, two-stage
stochastic programs. Unlike the other confidence-interval methods in
mpi-sppy, no distribution of the uncertain data is assumed: the estimators
work directly from sampled data. The methods and software are described in
[ChenWoodruff2023]_ and [ChenWoodruff2024]_.

The package has two families of estimators. The *empirical* methods
(classical, extended, subsampling, and bagging) resample the observed data
directly and need only numpy. The *smoothed* methods fit a univariate
distribution to the sampled data (using the bundled ``statdist`` library) and
resample from the fitted distribution; they need `scipy
<https://scipy.org>`_, which mpi-sppy treats as an optional dependency and
imports lazily. If scipy is not installed, the empirical methods still work
and a smoothed method fails with an informative import error.

Modes
-----

There are two modes, each runnable with ``python -m``:

*User mode* (``user_boot``) computes a confidence interval for one problem
instance. A long list of arguments is supplied on the command line, so users
usually put the command in a shell script:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.user_boot module arguments

Here ``module`` is the name of an importable Python module (without ``.py``)
that supplies the scenario creator and helper functions, and ``arguments`` is
the list of double-dash options described below.

*Simulation mode* (``simulate_boot``) estimates the coverage rate of a method
over many replications; it is aimed at researchers. All options come from a
json file:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.simulate_boot instance.json

The model module
----------------

The named module must provide the usual mpi-sppy scenario-creation contract
plus a few helpers used by the bootstrap code:

* ``scenario_creator(scenario_name, ...)`` — build a Pyomo model for one
  (data) scenario, annotated as usual for mpi-sppy;
* ``scenario_names_creator(num_scens, start=None)`` — the list of scenario
  names;
* ``kw_creator(cfg)`` — keyword arguments for the scenario creator;
* ``inparser_adder(cfg)`` — add any model-specific options;
* ``xhat_generator(scenario_names, solver_name=None, ...)`` — solve for a
  candidate solution ``xhat`` when none is supplied. The bootstrap code looks
  for this fixed name first and falls back to the legacy
  ``xhat_generator_<module_name>``. If a precomputed ``xhat`` file is given
  (``--xhat-fname``) the generator is not called.
* ``data_sampler(record_num, cfg)`` — return the data for one record (a scalar,
  or a dict keyed by variable name for multivariate data). This is used by the
  *smoothed* methods to build the sample that a distribution is fitted to; the
  empirical methods do not need it.

Methods
-------

The ``--boot-method`` (json ``boot_method``) option selects the estimator:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Token
     - Description
   * - ``Classical_gaussian``
     - Classical bootstrap, Gaussian confidence interval [eichhorn2007]_
   * - ``Classical_quantile``
     - Classical bootstrap, quantile confidence interval [eichhorn2007]_
   * - ``Extended``
     - Extended bootstrap [eichhorn2007]_
   * - ``Subsampling``
     - Subsampling bootstrap [eichhorn2007]_
   * - ``Bagging_with_replacement``
     - Bagging with replacement [lam2018]_
   * - ``Bagging_without_replacement``
     - Bagging without replacement [lam2018]_
   * - ``Smoothed_boot_epi``
     - Smoothed bootstrap, epi-spline fit, Gaussian interval [ChenWoodruff2024]_
   * - ``Smoothed_boot_kernel``
     - Smoothed bootstrap, kernel-density fit, Gaussian interval [ChenWoodruff2024]_
   * - ``Smoothed_boot_epi_quantile``
     - Smoothed bootstrap, epi-spline fit, quantile interval [ChenWoodruff2024]_
   * - ``Smoothed_boot_kernel_quantile``
     - Smoothed bootstrap, kernel-density fit, quantile interval [ChenWoodruff2024]_
   * - ``Smoothed_bagging``
     - Smoothed bagging, kernel-density fit [ChenWoodruff2024]_

The ``Smoothed_*`` tokens are the smoothed methods; the others are empirical.
The epi-spline fit builds a small Pyomo nonlinear program, so those two methods
additionally need a nonlinear solver (e.g. ``ipopt``); the kernel methods do
not.

Arguments
---------

Simulation and user modes use almost the same options; simulation mode reads
them from json (some with underscores), while user mode takes them on the
command line (with dashes). The main options are:

* ``max_count`` / ``--max-count`` — total sample size (integer).
* ``module_name`` — the module name (given as the first positional argument in
  user mode; a json key in simulation mode).
* ``candidate_sample_size`` / ``--candidate-sample-size`` — sample size used to
  compute ``xhat`` (M in the papers); ignored when an ``xhat`` file is given.
* ``sample_size`` / ``--sample-size`` — bootstrap/bagging sample size (N).
* ``subsample_size`` / ``--subsample-size`` — subsample size for bagging;
  ignored by the classical bootstrap methods.
* ``nB`` / ``--nB`` — number of bootstrap/bagging samples.
* ``alpha`` / ``--alpha`` — significance level (e.g. 0.05 for 95% confidence).
* ``seed_offset`` / ``--seed-offset`` — offset for the pseudo-random streams
  (enables replication); use 0 unless you have a reason not to.
* ``solver_name`` / ``--solver-name`` — solver (e.g. ``gurobi_direct``).
* ``xhat_fname`` / ``--xhat-fname`` — npy file with a precomputed ``xhat``, or
  the string ``"None"`` to compute it with ``xhat_generator``.
* ``optimal_fname`` (simulation only) — npy file with a (presumed) optimal
  value, or ``"None"`` to compute it from ``max_count`` scenarios.
* ``coverage_replications`` (simulation only) — number of coverage replications.
* ``boot_method`` / ``--boot-method`` — one of the tokens above.

The smoothed methods use two additional options (ignored, and not required in
the json, for the empirical methods):

* ``smoothed_center_sample_size`` / ``--smoothed-center-sample-size`` — number
  of points drawn from the fitted distribution to estimate the gap center.
* ``smoothed_B_I`` / ``--smoothed-B-I`` — number of outer replications for
  smoothed bagging.

There may also be model-specific options added by ``inparser_adder``.

Batch parallelism
-----------------

The bootstrap batches are split across MPI ranks and reassembled on rank 0
with ``Gatherv``, so a run can be accelerated with, e.g.,
``mpiexec -np 2 python -m mpi4py -m mpisppy.confidence_intervals.bootsp.user_boot ...``.
The estimate on rank 0 depends on the number of ranks because each rank seeds
its own bootstrap stream.

boot_general_prep
-----------------

``boot_general_prep`` writes the two npy files (a candidate ``xhat`` and a
presumed optimal value) that a simulation can reuse:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.boot_general_prep instance.json

Example
-------

The ``examples/bootsp/schultz`` directory has a small two-stage example whose
data is a deterministic function of the scenario number, so its results are
reproducible across solvers. From that directory:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.user_boot unique_schultz \
       --max-count 50 --candidate-sample-size 1 --sample-size 30 \
       --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 100 \
       --solver-name gurobi_direct --boot-method Classical_quantile

   $ python -m mpisppy.confidence_intervals.bootsp.simulate_boot unique_schultz.json

See ``examples/bootsp/schultz/schultz.bash`` for a serial run, a parallel run,
and a coverage simulation.

Working from a dataset file
---------------------------

The ``schultz`` example above generates its data arithmetically from the
scenario number. The companion example ``examples/bootsp/schultz_data`` shows
the more typical *data-based* setup: the same model, but each scenario reads
one observation (one row) from a committed dataset, ``schultz_data.csv``:

.. code-block:: text

   xi1,xi2
   11,11
   10,12
   12,9
   ...

The model's ``scenario_creator`` reads row ``scennum`` of the file (the file is
loaded once and cached), and ``inparser_adder`` exposes a ``--data-file``
option. The dataset defines the "population" the estimators work from:
``max_count`` is the number of rows, ``xhat`` is computed from
``candidate_sample_size`` of them (the scenarios
``[sample_size : sample_size + candidate_sample_size]``), and the bootstrap
draws a pool of ``sample_size`` rows from the whole dataset and resamples it
for its batches. (In this standalone mode the pool is drawn from all
``max_count`` rows, so it can overlap the candidate rows; the
``generic_cylinders`` integration below keeps the candidate and pool strictly
disjoint.) For example:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.user_boot schultz_data \
       --max-count 200 --candidate-sample-size 5 --sample-size 100 \
       --subsample-size 20 --nB 20 --alpha 0.1 --seed-offset 100 \
       --solver-name gurobi_direct --boot-method Bagging_with_replacement

``schultz_data.csv`` is produced by ``schultz_data_generator.py`` (a fixed seed
makes it reproducible); replace it with your own two-column dataset, or point
``--data-file`` at another file, to run the bootstrap on your own data.

In generic_cylinders
--------------------

The everyday driver ``generic_cylinders`` can report a data-based bootstrap
confidence interval as a first-class option, the data-based analog of its
``--mmw-*`` (MMW) group. Given a dataset, it finds a candidate solution
``xhat`` with whatever the command line configured (an extensive form for small
instances, or the PH cylinder system for large ones) and then reports a
bootstrap/bagging CI on the optimality gap of ``xhat``, computed from a part of
the data that is held **strictly disjoint** from the records that produced
``xhat``. That disjointness is a correctness requirement: a gap CI is only
meaningful when estimated on data that did not choose ``xhat``.

The workflow is a hold-out split over the *positions* in the dataset. The
driver treats ``scenario_names_creator(None)`` as the whole dataset (one
scenario name per record), reserves the first ``--boot-candidate-sample-size``
(``M``) records as the candidate block that ``xhat`` came from, and resamples
the next ``--boot-sample-size`` (``N``) records — disjoint from the candidate
block — for the CI. Because a model stays name-based while bootstrap resampling
is positional, the driver owns the position/name reconciliation: a model only
has to follow the usual mpi-sppy naming and map its own names to its own data.

The dataset is interpreted by the **model**, not the framework: the model owns
loading and any data-source option (e.g. ``--data-file``), and reports the
dataset by returning every implied scenario name from
``scenario_names_creator(None)``. There is no dataset-size or data-source
``--boot-*`` flag.

The ``--boot-*`` options are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``--boot-method``
     - the empirical estimator (the ``Smoothed_*`` methods are not yet
       available here; use the standalone ``user_boot`` for those)
   * - ``--boot-candidate-sample-size``
     - ``M``: leading records reserved for ``xhat``; omit or ``0`` when
       ``--boot-xhat-input-file-name`` is given (the two are mutually exclusive)
   * - ``--boot-sample-size``
     - ``N``: records resampled for the CI (the disjoint pool)
   * - ``--boot-subsample-size``
     - subsample size (subsampling and bagging methods)
   * - ``--boot-nB``
     - number of bootstrap/bagging batches
   * - ``--boot-alpha``
     - two-sided significance level
   * - ``--boot-seed-offset``
     - RNG offset for replication
   * - ``--boot-xhat-input-file-name``
     - optional precomputed ``xhat`` (the no-wheel path)
   * - ``--boot-solver-name``
     - solver for the batch solves (falls back to the generic ``--solver-name``)
   * - ``--boot-solver-options``
     - options string for the batch solver, e.g. ``mipgap=0.01``
   * - ``--boot-ranks-per-batch``
     - ``K``: ranks cooperating on one batch solve; only ``K = 1`` (a per-rank
       extensive form) is supported so far

Because it works from a fixed dataset, a bootstrap run is mutually exclusive
with the distribution-sampling CI methods (MMW and sequential sampling) and is
two-stage only. ``examples/bootsp/schultz_data/schultz_data_boot.bash`` runs
the whole workflow end to end:

.. code-block:: bash

   $ mpiexec -np 3 python -m mpisppy.generic_cylinders \
       --module-name schultz_data --num-scens 5 \
       --max-iterations 20 --default-rho 1.0 --solver-name gurobi_direct \
       --xhatshuffle --lagrangian \
       --boot-method Classical_quantile \
       --boot-candidate-sample-size 5 --boot-sample-size 100 \
       --boot-subsample-size 20 --boot-nB 20 --boot-alpha 0.1 \
       --boot-seed-offset 100

Here the main run finds ``xhat`` from the first 5 dataset records and the
bootstrap resamples the next 100 (disjoint) records for the gap CI.

Smoothed methods and statdist
-----------------------------

The smoothed methods (the ``Smoothed_*`` tokens) fit a univariate distribution
to the sampled data and then resample from the *fitted* distribution rather
than from the data directly. The distribution fitting is provided by the
bundled ``statdist`` library
(``mpisppy.confidence_intervals.bootsp.statdist``), a trimmed port of the
univariate distributions from the statdist package; ``statdist`` uses scipy,
which is imported lazily so that the empirical methods remain scipy-free.

To use a smoothed method the model module must supply ``data_sampler`` (see
above): the smoothed estimator calls it for each sampled record to assemble the
data that ``statdist`` fits. The kernel-density methods
(``Smoothed_boot_kernel``, ``Smoothed_boot_kernel_quantile``,
``Smoothed_bagging``) fit with a Gaussian kernel and need only scipy; the
epi-spline methods (``Smoothed_boot_epi``, ``Smoothed_boot_epi_quantile``) fit
by solving a small Pyomo nonlinear program and additionally need a nonlinear
solver such as ``ipopt``.

Three examples that need statdist ship in ``examples/bootsp``:

* ``farmer`` — the scalable farmer, with crop yields perturbed by a fitted
  (or, empirically, a uniform) distribution;
* ``cvar`` — a CVaR example (Lam & Qian) with standard-normal data;
* ``multi_knapsack`` — a multi-product knapsack (Vaagen & Wallace) whose
  deterministic data is read from a json file (``--deterministic-data-json``).

Each has an empirical json/bash and a ``smoothed_*.json``; for instance, from
``examples/bootsp/cvar``:

.. code-block:: bash

   $ python -m mpisppy.confidence_intervals.bootsp.user_boot cvar \
       --max-count 3000 --candidate-sample-size 10 --sample-size 75 \
       --subsample-size 10 --nB 20 --alpha 0.1 --seed-offset 0 \
       --solver-name cplex_direct --boot-method Bagging_with_replacement

   $ python -m mpisppy.confidence_intervals.bootsp.simulate_boot smoothed_cvar.json

References
----------

.. [ChenWoodruff2023] Chen, X. and Woodruff, D.L.: Software for data-based
   stochastic programming using bootstrap estimation. INFORMS Journal on
   Computing (2023).

.. [ChenWoodruff2024] Chen, X. and Woodruff, D.L.: Distributions and Bootstrap
   for Data-based Stochastic Programming. Computational Management Science
   (2024).

.. [eichhorn2007] Eichhorn, A. and Romisch, W.: Stochastic integer
   programming: Limit theorems and confidence intervals. Mathematics of
   Operations Research, 32(1), 118-135 (2007).

.. [lam2018] Lam, H. and Qian, H.: Assessing solution quality in stochastic
   optimization via bootstrap aggregating. Winter Simulation Conference,
   2061-2071 (2018).
