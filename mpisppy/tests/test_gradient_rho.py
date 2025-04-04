###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Author: Ulysse Naepels and D.L. Woodruff
"""
IMPORTANT:
  Unless we run to convergence, the solver, and even solver
version matter a lot, so we often just do smoke tests.
"""

import os
import unittest
import csv
from mpisppy.utils import config

import mpisppy.utils.cfg_vanilla as vanilla
import mpisppy.tests.examples.farmer as farmer
from mpisppy.spin_the_wheel import WheelSpinner
from mpisppy.tests.utils import get_solver
import mpisppy.utils.gradient as grad
import mpisppy.utils.find_rho as find_rho

from mpisppy.extensions.gradient_extension import Gradient_extension
from mpisppy.extensions.extension import MultiExtension

__version__ = 0.21

solver_available,solver_name, persistent_available, persistent_solver_name= get_solver()

def _create_cfg():
    cfg = config.Config()
    cfg.add_branching_factors()
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.dynamic_rho_args()
    cfg.solver_name = solver_name
    cfg.default_rho = 1
    return cfg

#*****************************************************************************

class Test_gradient_farmer(unittest.TestCase):
    """ Test the gradient code using farmer."""

    def _create_ph_farmer(self):
        self.cfg.num_scens = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        hub_dict['opt_kwargs']['options']['cfg'] = self.cfg                            
        list_of_spoke_dict = list()
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.strata_rank == 0:
            ph_object = wheel.spcomm.opt
            return ph_object

    def setUp(self):
        self.cfg = _create_cfg()
        self.cfg.xhatpath = './examples/rho_test_data/farmer_cyl_nonants.npy'
        self.cfg.grad_cost_file_out = '_test_grad_cost.csv'
        self.cfg.grad_rho_file_out = './_test_grad_rho.csv'
        self.cfg.grad_order_stat = 0.5
        self.cfg.max_iterations = 0
        self.ph_object = self._create_ph_farmer()

    def test_grad_cost_init(self):
        self.grad_object = grad.Find_Grad(self.ph_object, self.cfg)

    def test_compute_grad(self):
        self.grad_object = grad.Find_Grad(self.ph_object, self.cfg)
        k0, s0 = list(self.grad_object.ph_object.local_scenarios.items())[0]
        self.ph_object.disable_W_and_prox()
        grad_cost0 = self.grad_object.compute_grad(k0, s0)
        self.assertEqual(grad_cost0[('ROOT', 1)], -230)

    def test_find_grad_cost(self):
        self.grad_object = grad.Find_Grad(self.ph_object, self.cfg)
        self.grad_object.find_grad_cost()
        self.assertEqual(self.grad_object.c[('scen0', 'DevotedAcreage[CORN0]')], -150)

    def test_write_grad_cost(self):
        self.grad_object = grad.Find_Grad(self.ph_object, self.cfg)
        self.grad_object.write_grad_cost()
        try:
            os.remove(self.cfg.grad_cost_file_out)
        except Exception:
            raise RuntimeError('gradient.write_grad_cost() did not write a csv file')
    
    def test_find_grad_rho(self):
        self.cfg.grad_cost_file_in= './examples/rho_test_data/grad_cost.csv'
        self.grad_object = grad.Find_Grad(self.ph_object, self.cfg)
        rho = self.grad_object.find_grad_rho()
        self.assertAlmostEqual(rho['DevotedAcreage[CORN0]'], 3.375)
    
    def test_compute_and_write_grad_rho(self):
        self.cfg.grad_cost_file_in= './examples/rho_test_data/grad_cost.csv'
        self.grad_object = grad.Find_Grad(self.ph_object, self.cfg)
        self.grad_object.write_grad_rho()
        try:
            os.remove(self.cfg.grad_rho_file_out)
        except Exception:
            raise RuntimeError('gradient.compute_and_write_grad_rho() did not write a csv file')

    def test_grad_cost_and_rho(self):
        self.cfg.grad_cost_file_in= './examples/rho_test_data/grad_cost.csv'
        grad.grad_cost_and_rho('examples.farmer', self.cfg)
        with open(self.cfg.grad_cost_file_out, 'r') as f:
            read = csv.reader(f)
            rows = list(read)
            self.assertEqual(float(rows[1][2]), -150)
        with open(self.cfg.grad_rho_file_out, 'r') as f:
            read = csv.reader(f)
            rows = list(read)
            self.assertAlmostEqual(float(rows[3][1]), 2.0616161616161617)
        os.remove(self.cfg.grad_cost_file_out)
        os.remove(self.cfg.grad_rho_file_out)


###############################################
        
class Test_find_rho_farmer(unittest.TestCase):
    """ Test the find rho code using farmer."""

    def _create_ph_farmer(self):
        self.cfg.num_scens = 3
        self.cfg.max_iterations = 0
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans, scenario_creator_kwargs=scenario_creator_kwargs)
        list_of_spoke_dict = list()
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.strata_rank == 0:
            ph_object = wheel.spcomm.opt
            return ph_object

    def setUp(self):
        self.cfg = _create_cfg()
        self.ph_object = self._create_ph_farmer()
        self.cfg.grad_cost_file_in = './examples/rho_test_data/grad_cost.csv'
        self.cfg.grad_rho_file_out = './_test_rho.csv'
        self.cfg.grad_order_stat = 0.4

    def test_grad_rho_init(self):
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
    
    def test_w_denom(self):
        self.cfg.grad_cost_file_in= './examples/rho_test_data/grad_cost.csv'
        ### ?? self.cfg.grad_cost_file_in = self.cfg.grad_cost_file_out
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
        k0, s0 = list(self.rho_object.ph_object.local_scenarios.items())[0]        
        denom = {node.name: self.rho_object._w_denom(s0, node) for node in s0._mpisppy_node_list}
        self.assertAlmostEqual(denom["ROOT"][0], 25.0) 

    def test_prox_denom(self):
        self.cfg.grad_cost_file_out= './examples/rho_test_data/grad_cost.csv'
        self.cfg.grad_cost_file_in = self.cfg.grad_cost_file_out
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
        k0, s0 = list(self.rho_object.ph_object.local_scenarios.items())[0]
        denom = {node.name: self.rho_object._prox_denom(s0, node) for node in s0._mpisppy_node_list}
        self.assertAlmostEqual(denom["ROOT"][0], 1250.0)

    def test_grad_denom(self):
        self.cfg.grad_cost_file_out= './examples/rho_test_data/grad_cost.csv'
        self.cfg.grad_cost_file_in = self.cfg.grad_cost_file_out
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
        denom = self.rho_object._grad_denom()
        self.assertAlmostEqual(denom[0], 21.48148148148148) 

    def test_compute_rho(self):
        self.cfg.grad_cost_file_out= './examples/rho_test_data/grad_cost.csv'
        self.cfg.grad_cost_file_in = self.cfg.grad_cost_file_out
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
        rho = self.rho_object.compute_rho(indep_denom=True)
        self.assertAlmostEqual(rho['DevotedAcreage[CORN0]'], 6.982758620689655)
        rho = self.rho_object.compute_rho()
        self.assertAlmostEqual(rho['DevotedAcreage[CORN0]'], 8.163805471726114)

    def test_compute_and_write_grad_rho(self):
        pass
        """
        removed from rho object July 2024
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
        self.rho_object.compute_and_write_grad_rho()
        try:
            os.remove(self.cfg.grad_rho_file_out)
        except:
            raise RuntimeError('find_rho.compute_and_write_grad_rho() did not write a csv file')
        """

    def test_rho_setter(self):
        self.cfg.grad_rho = True
        self.cfg.rho_file_in = './examples/rho_test_data/rho.csv'
        self.rho_object = find_rho.Find_Rho(self.ph_object, self.cfg)
        self.set_rho = find_rho.Set_Rho(self.cfg)
        k0, s0 = list(self.rho_object.ph_object.local_scenarios.items())[0]
        rho_list = self.set_rho.rho_setter(s0)
        nlen = s0._mpisppy_data.nlens['ROOT']
        self.assertEqual(len(rho_list), nlen)
        id_var, rho = rho_list[0]
        self.assertIsInstance(id_var, int)
        self.assertAlmostEqual(rho, 6.982758620689654)
    

#*****************************************************************************

class Test_grad_extension_farmer(unittest.TestCase):
    """ Test the gradient extension code using farmer.
    See also: farmer_rho_demo.py
    writen by DLW Sept 2023 TBD: this code should be re-organized"""

    def setUp(self):
        print("test grad setup")
        self.cfg = _create_cfg()
        self.cfg.xhatpath = './examples/rho_test_data/farmer_cyl_nonants.npy'
        self.cfg.grad_cost_file_out = '_test_grad_cost.csv'
        self.cfg.grad_rho_file_out = './_test_grad_rho.csv'
        self.cfg.grad_order_stat = 0.5
        self.cfg.max_iterations = 0
    

    def _run_ph_farmer(self):
        ext_classes = [Gradient_extension]
        
        self.cfg.num_scens = 3
        scenario_creator = farmer.scenario_creator
        scenario_denouement = farmer.scenario_denouement
        all_scenario_names = farmer.scenario_names_creator(self.cfg.num_scens)
        scenario_creator_kwargs = farmer.kw_creator(self.cfg)
        beans = (self.cfg, scenario_creator, scenario_denouement, all_scenario_names)
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=MultiExtension,
                                  ph_converger = None)  # ??? DLW Oct 2023: is this correct?
        hub_dict["opt_kwargs"]["extension_kwargs"] = {"ext_classes" : ext_classes}


        hub_dict['opt_kwargs']['options']['gradient_extension_options'] = {'cfg': self.cfg}
        hub_dict['opt_kwargs']['extensions'] = MultiExtension
        
        list_of_spoke_dict = list()
        wheel = WheelSpinner(hub_dict, list_of_spoke_dict)
        wheel.spin()
        if wheel.strata_rank == 0:
            ph_object = wheel.spcomm.opt
            return ph_object

    def test_grad_extensions(self):
        print("** test grad extensions **")
        self.cfg.grad_rho_file_out = './_test_rho.csv'
        #self.cfg.grad_cost_file_in = './examples/rho_test_data/grad_cost.csv'
        self.cfg.xhatpath = './examples/rho_test_data/farmer_cyl_nonants.npy'
        self.cfg.max_iterations = 4
        self.ph_object = self._run_ph_farmer()
        self.assertAlmostEqual(self.ph_object.conv, 2.128968965420187, places=1)

    def test_dyn_grad_extensions(self):
        print("** test dynamic grad extensions **")
        self.cfg.grad_rho_file_out = './_test_rho.csv'
        #self.cfg.grad_cost_file_in = './examples/rho_test_data/grad_cost.csv'
        self.cfg.xhatpath = './examples/rho_test_data/farmer_cyl_nonants.npy'
        self.cfg.max_iterations = 10
        self.cfg.dynamic_rho_dual_crit = True
        self.cfg.dynamic_rho_dual_thresh = 0.1        
        self.ph_object = self._run_ph_farmer()
        self.assertAlmostEqual(self.ph_object.conv, 0.039598, places=1)


if __name__ == '__main__':
    unittest.main()
