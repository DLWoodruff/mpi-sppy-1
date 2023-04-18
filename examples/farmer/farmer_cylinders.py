# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# general example driver for farmer with cylinders

import farmer
import mpisppy.cylinders

# Make it all go
from mpisppy.spin_the_wheel import WheelSpinner
import mpisppy.utils.sputils as sputils

from mpisppy.utils import config
import mpisppy.utils.cfg_vanilla as vanilla

from mpisppy.extensions.norm_rho_updater import NormRhoUpdater
from mpisppy.convergers.norm_rho_converger import NormRhoConverger

write_solution = True

def _parse_args():
    # create a config object and parse
    cfg = config.Config()
    
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()
    cfg.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)                
    cfg.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater extension",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use-norm-rho-converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("farmer_cylinders")
    return cfg

def _hack(xhatfile, ph_object):
    # This function should not be in this file (it should be somewhere else).
    # It is just to illustrated how to manipulate the ph object in some ways
    # It does not anything useful...
    import mpisppy.confidence_intervals.ciutils as ciutils
    xhat_one = ciutils.read_xhat(xhatfile)["ROOT"]

    # Loop over scenarios and compare the x values to xbar
    # (just for fun).
    for (sname, scenario) in ph_object.local_scenarios.items():
        for node in scenario._mpisppy_node_list:
            assert node.name == "ROOT", "only two stage for now"
            assert len(xhat_one) == len(node.nonant_vardata_list), "you must have a mismatched xhat file"
            for (ix,var) in enumerate(node.nonant_vardata_list):
                print(f"{sname =}, {ix =}, {var.name =}, value={var.value:.1f}, xhat={xhat_one[ix]:.1f}")


    
def main():
    
    cfg = _parse_args()

    num_scen = cfg.num_scens
    crops_multiplier = cfg.crops_mult

    rho_setter = farmer._rho_setter if hasattr(farmer, '_rho_setter') else None
    if cfg.default_rho is None and rho_setter is None:
        raise RuntimeError("No rho_setter so a default must be specified via --default-rho")

    if cfg.use_norm_rho_converger:
        if not cfg.use_norm_rho_updater:
            raise RuntimeError("--use-norm-rho-converger requires --use-norm-rho-updater")
        else:
            ph_converger = NormRhoConverger
    else:
        ph_converger = None
    
    scenario_creator = farmer.scenario_creator
    scenario_denouement = farmer.scenario_denouement
    all_scenario_names = ['scen{}'.format(sn) for sn in range(num_scen)]
    scenario_creator_kwargs = {
        'use_integer': False,
        "crops_multiplier": crops_multiplier,
    }
    scenario_names = [f"Scenario{i+1}" for i in range(num_scen)]

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    if cfg.run_async:
        # Vanilla APH hub
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter = rho_setter)
    else:
        # Vanilla PH hub
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter = rho_setter)

    ## hack in adaptive rho
    if cfg.use_norm_rho_updater:
        hub_dict['opt_kwargs']['extensions'] = NormRhoUpdater
        hub_dict['opt_kwargs']['options']['norm_rho_options'] = {'verbose': True}

    # FWPH spoke
    if cfg.fwph:
        fw_spoke = vanilla.fwph_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # Standard Lagrangian bound spoke
    if cfg.lagrangian:
        lagrangian_spoke = vanilla.lagrangian_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # Special Lagranger bound spoke
    if cfg.lagranger:
        lagranger_spoke = vanilla.lagranger_spoke(*beans,
                                              scenario_creator_kwargs=scenario_creator_kwargs,
                                              rho_setter = rho_setter)

    # xhat looper bound spoke
    if cfg.xhatlooper:
        xhatlooper_spoke = vanilla.xhatlooper_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = vanilla.xhatshuffle_spoke(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        
    list_of_spoke_dict = list()
    if cfg.fwph:
        list_of_spoke_dict.append(fw_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)
    if cfg.lagranger:
        list_of_spoke_dict.append(lagranger_spoke)
    if cfg.xhatlooper:
        list_of_spoke_dict.append(xhatlooper_spoke)
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)

    wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    if write_solution:
        xhatfile = 'farmer_cyl_nonants.npy'
        wheel.write_first_stage_solution('farmer_plant.csv')
        wheel.write_first_stage_solution(xhatfile,
                first_stage_solution_writer=sputils.first_stage_nonant_npy_serializer)
        wheel.write_tree_solution('farmer_full_solution')

    # Now call something to read the nonants and compute the gradient
    # This is a hack and will crash if write_solution is false
    if wheel.strata_rank == 0:  # don't do this for bound ranks
        ph_object = wheel.spcomm.opt
        _hack(xhatfile, ph_object)

if __name__ == "__main__":
    main()
