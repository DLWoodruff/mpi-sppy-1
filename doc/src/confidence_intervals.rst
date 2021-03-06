.. _Confidence intervals:

MMW confidence interval
=======================

If we want to assess the quality of a given candidate solution ``xhat_one`` 
(a first stage policy), we could try and evaluate the optimality gap, i.e. 
the gap between the value of the objective function
at ``xhat_one`` and the value of the solution to our problem.
The class ``MMWConfidenceIntervals`` compute an estimator of the optimality gap
as described in [mmw1999]_ (Section 3.2) and an asymptotic confidence interval for
this gap. 

We will document two steps in the process : finding a candidate solution 
``xhat_one``, and evaluating it.


Finding a candidate solution
----------------------------

Computing this confidence interval means that we need to find a solution to 
an approximate problem, and evaluate how good a solution to this approximate problem ``xhat_one`` is.
In order to use MMW, ``xhat_one`` must be written using one of two functions 
``ef_ROOT_nonants_npy_serializer`` or ``write_spin_the_wheel_first_stage_solution``.
These functions write ``xhat`` to a file and can be read using ``read_xhat``.

Evaluating a candidate solution
-------------------------------

To evaluate a candidate solution with some scenarios, one might
create a ``Xhat_Eval`` object and call its ``evaluate`` method 
(resp. ``evaluate_one`` for a single scenario). It takes as
an argument ``xhats``, a dictionnary of noon-anticipative policies for all 
non-leaf nodes of a scenario tree. While for a 2-stage problem, ``xhats`` is
just the candidate solution ``xhat_one``, for multistage problem the 
dictionnary can be computed using the function ``walking_tree_xhats`` 
(resp. ``feasible_solution``).


Computing a confidence interval
-------------------------------

The first step in computing a confidence interval is creating a ``MMWConfidenceIntervals`` object
that takes as an argument an ``xhat_one`` and options.
This object has a ``run`` method that returns a gap estimator and a confidence interval on the gap.

Example
-------

An example of use, with the ``farmer`` problem, can be found in the main of ``mmwci.py``.


Sequential sampling
===================

Similarly, given an confidence interval, one can try to find a candidate solution
``xhat_one`` such that its optimality gap has this confidence interval.
The class ``SeqSampling`` implements three procedures described in 
[bm2011]_ and [bpl2012]_. It takes as an input a method to generate
candidate solutions and options, and returns a ``xhat_one`` and a confidence interval on
its optimality gap.

Examples of use with the ``farmer`` problem and several options can be found in the main of ``seqsampling.py``.

.. Note::
   Unlike MMW, sequential samping does not run with multistage problems.
