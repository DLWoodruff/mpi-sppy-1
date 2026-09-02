###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################


import mpisppy.extensions.extension
import mpisppy.utils.w_utils.wtracker as wtracker


class Wtracker_extension(mpisppy.extensions.extension.Extension):
    """
        wrap the wtracker code as an extension
        
        Args:
            opt (PHBase (inherets from SPOpt) object): gives the problem that we bound

        Attributes:
          scenario_name_to_rank (dict of dict): nodes (i.e. comms) scen names
                keys are comms (i.e., tree nodes); values are dicts with keys
                that are scenario names and values that are ranks
    """

    def __init__(self, opt, comm=None):
        super().__init__(opt)
        self.cylinder_rank = self.opt.cylinder_rank
        self.verbose = self.opt.options["verbose"]
        self.wtracker = wtracker.WTracker(opt)
        self.options = opt.options["wtracker_options"]
        # TBD: more graceful death if options are bad
        self.wlen = self.options["wlen"]

    def pre_iter0(self):
        pass

    def post_iter0(self):
        pass
        
    def miditer(self):
        pass

    def enditer(self):
        self.wtracker.grab_local_Ws()

    def checkpoint_state(self):
        """The W sets the end-of-run report reads.

        `report_by_moving_stats` reads `local_Ws` at the last `wlen + 1`
        iterations (`compute_moving_stats` goes from `li - wlen` through
        `li`). A resumed run holds only the sets it grabbed itself, so its
        report reaches back past the stop whenever the resumed leg is shorter
        than the window -- and then died with a KeyError out of
        `post_everything`, after every solve had been paid for.

        Carry the last `wlen + 1` sets, not the whole history, which grows by
        scenarios times nonants of floats per iteration.
        """
        local_Ws = self.wtracker.local_Ws
        if not local_Ws:
            return None
        keep = sorted(local_Ws)[-(self.wlen + 1):]
        return {"local_Ws": {k: local_Ws[k] for k in keep},
                "ph_iter": self.wtracker.ph_iter}

    def restore_state(self, state):
        self.wtracker.local_Ws.update(state["local_Ws"])
        self.wtracker.ph_iter = state["ph_iter"]

    def post_everything(self):
        reportlen = self.options.get("reportlen")
        stdevthresh = self.options.get("stdevthresh")
        # "" rather than None when the option is unset, matching
        # report_by_moving_stats's own default: unset, the three reports were
        # named after a stringified None ("None_summary_iter5_rank0.txt").
        file_prefix = self.options.get("file_prefix") or ""
        self.wtracker.report_by_moving_stats(self.wlen,
                                             reportlen=reportlen,
                                             stdevthresh=stdevthresh,
                                             file_prefix=file_prefix)

        
