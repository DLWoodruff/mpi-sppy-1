# Diagnostic for issue #869; not for merging.
#
# Does a busy spoke stall the hub?  Farmer, a PH hub and one Lagrangian
# spoke, one rank each.  After every Lagrangian pass the spoke sleeps
# --spoke-sleep seconds without making MPI calls, standing in for a long
# subproblem solve.  The hub times each sync_bounds, which reads the
# spoke's window (Lock/Get/Unlock).  If reads need the target to make an
# MPI call, those times grow toward the sleep; otherwise they stay small.
#
# mpiexec -n 2 python -m mpi4py stall_probe.py --num-scens 3 \
#     --solver-name gurobi --max-iterations 10 --default-rho 1 \
#     --lagrangian --spoke-sleep 2
import statistics
import time

import mpisppy.utils.cfg_vanilla as vanilla
from mpisppy.cylinders.hub import PHHub
from mpisppy.cylinders.lagrangian_bounder import LagrangianOuterBound
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.utils import config

import farmer


class SleepyLagrangian(LagrangianOuterBound):
    def lagrangian(self, *args, **kwargs):
        bound = super().lagrangian(*args, **kwargs)
        time.sleep(self.opt.options["spoke_sleep"])
        return bound


class TimedPHHub(PHHub):
    def sync_bounds(self):
        t = time.perf_counter()
        super().sync_bounds()
        if not hasattr(self, "_bound_times"):
            self._bound_times = []
        self._bound_times.append(time.perf_counter() - t)

    def hub_finalize(self):
        super().hub_finalize()
        if self.cylinder_rank == 0:
            ts = getattr(self, "_bound_times", [])
            if ts:
                print(f"STALL n={len(ts)} median={statistics.median(ts):.3f} "
                      f"max={max(ts):.3f} total={sum(ts):.3f} "
                      f"all=[{', '.join(f'{t:.3f}' for t in ts)}]", flush=True)


def main():
    cfg = config.Config()
    farmer.inparser_adder(cfg)
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.lagrangian_args()
    cfg.add_to_config("spoke_sleep", description="seconds the spoke sleeps per pass",
                      domain=float, default=2.0)
    cfg.parse_command_line("stall_probe")

    names = farmer.scenario_names_creator(cfg.num_scens)
    beans = (cfg, farmer.scenario_creator, farmer.scenario_denouement, names)
    kw = farmer.kw_creator(cfg)

    hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=kw)
    hub_dict["hub_class"] = TimedPHHub
    spoke = vanilla.lagrangian_spoke(*beans, scenario_creator_kwargs=kw)
    spoke["spoke_class"] = SleepyLagrangian
    spoke["opt_kwargs"]["options"]["spoke_sleep"] = cfg.spoke_sleep

    t = time.perf_counter()
    ws = WheelSpinner(hub_dict, [spoke])
    ws.spin()
    if ws.global_rank == 0:
        print(f"STALL wall={time.perf_counter() - t:.3f}", flush=True)


if __name__ == "__main__":
    main()
