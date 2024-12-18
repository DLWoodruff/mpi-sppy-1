###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# updated April 2020
import mpisppy.cylinders.spoke as spoke
from mpisppy.extensions.xhatlooper import XhatLooper
from mpisppy.utils.xhat_eval import Xhat_Eval
import logging
import mpisppy.log

# Could also pass, e.g., sys.stdout instead of a filename
mpisppy.log.setup_logger("mpisppy.cylinders.xhatlooper_bounder",
                         "xhatlp.log",
                         level=logging.CRITICAL)                         
logger = logging.getLogger("mpisppy.cylinders.xhatlooper_bounder")


class XhatLooperInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'X'

    def xhatlooper_prep(self):
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")

        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError("XhatShuffleInnerBound must be used with Xhat_Eval.")

        xhatter = XhatLooper(self.opt)

        ### begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()
        
        self.opt._lazy_create_solvers()  # no iter0 loop, but we need the solvers

        self.opt._update_E1()
        if abs(1 - self.opt.E1) > self.opt.E1_tolerance:
            if self.opt.cylinder_rank == 0:
                print("ERROR")
                print("Total probability of scenarios was ", self.opt.E1)
                print("E1_tolerance = ", self.opt.E1_tolerance)
            quit()
        ### end iter0 stuff

        xhatter.post_iter0()
        self.opt._save_nonants() # make the cache

        return xhatter

    def main(self):
        logger.debug(f"Entering main on xhatlooper spoke rank {self.global_rank}")

        xhatter = self.xhatlooper_prep()

        scen_limit = self.opt.options['xhat_looper_options']['scen_limit']

        xh_iter = 1
        while not self.got_kill_signal():
            if (xh_iter-1) % 10000 == 0:
                logger.debug(f'   Xhatlooper loop iter={xh_iter} on rank {self.global_rank}')
                logger.debug(f'   Xhatlooper got from opt on rank {self.global_rank}')

            if self.new_nonants:
                logger.debug(f'   *Xhatlooper loop iter={xh_iter}')
                logger.debug(f'   *got a new one! on rank {self.global_rank}')
                logger.debug(f'   *localnonants={str(self.localnonants)}')

                self.opt._put_nonant_cache(self.localnonants)
                # just for sending the values to other scenarios
                # so we don't need to tell persistent solvers
                self.opt._restore_nonants(update_persistent=False)
                upperbound, srcsname = xhatter.xhat_looper(scen_limit=scen_limit, restore_nonants=False)

                # send a bound to the opt companion
                self.update_if_improving(upperbound)
            xh_iter += 1
