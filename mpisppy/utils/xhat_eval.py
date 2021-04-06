# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# Code to evaluate a given x-hat, but given as a nonant-cache

import mpisppy.phbase
import shutil
import mpi4py.MPI as mpi
import mpisppy.utils.sputils as sputils
    
fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()

print("WHAT ABOUT MULTI-STAGE")

############################################################################
class Xhat_Eval(mpisppy.phbase.PHBase):
    """ PH. See PHBase for list of args. """

    #======================================================================
    def evaluate(self, nonant_cache):
        """ Compute the expected value.

        Args:
            nonant_cache(numpy vector): special numpy vector with nonant values (see ph)

        Returns:
            Eobj (float or None): Expected value (or None if infeasible)

        """
        verbose = self.PHoptions['verbose']

        self.subproblem_creation(verbose)
        self._create_solvers()
        self._fix_nonants(nonant_cache)

        solver_options = None  # ???
        
        self.solve_loop(solver_options=solver_options,
                        dis_prox=False, # Important
                        use_scenarios_not_subproblems=True,  # ???
                        gripe=True, 
                        tee=False,
                        verbose=verbose)

        Eobj = self.Eobjective(verbose)
        
        return Eobj


if __name__ == "__main__":
    #==============================
    # hardwired by dlw for debugging
    # ? xhat and right term at the same time?
    import mpisppy.tests.examples.farmer as refmodel
    import mpisppy.utils.amalgomator as ama
    ama_options = {"EF-2stage": True}   # 2stage vs. mstage
    ama_object = ama.from_module("mpisppy.tests.examples.farmer", ama_options)
    ama_object.run()
    print(f"inner bound=", ama_object.best_inner_bound)
    print(f"outer bound=", ama_object.best_outer_bound)

    # get the nonants
    nonant_cache = sputils.nonant_cache_from_ef(ama_object.ef)
    # NOTE: we probably should do an assert or two to make sure Vars match

    # create the eval object
    ScenCount = ama_object.args.num_scens
    all_scenario_names = ['scen' + str(i) for i in range(ScenCount)]
    scenario_creator = refmodel.scenario_creator
    scenario_denouement = refmodel.scenario_denouement
    solvername = ama_object.args.EF_solver_name

    # The options need to be re-done (and phase needs to be split up)
    PHopt = {"iter0_solver_options": None,
             "iterk_solver_options": None,
             "solvername": solvername,
             "verbose": False}
    # TBD: set solver options
    ev = Xhat_Eval(PHopt,
                   all_scenario_names,
                   scenario_creator,
                   scenario_denouement,
                   do_options_check=False)
    obj_at_xhat = ev.evaluate(nonant_cache)
    print(f"Expected value at xhat={obj_at_xhat}")
    
