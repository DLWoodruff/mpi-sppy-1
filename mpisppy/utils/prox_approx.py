# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

from enum import Enum
from pyomo.core.expr.numeric_expr import LinearExpression

class Side(Enum):
    LEFT = 1
    RIGHT = 2
    BOTH = 3

class ProxApproxManager:
    __slots__ = ()

    def __new__(cls, xvar, xvarsqrd, xsqvar_cuts, ndn_i):
        if xvar.is_integer():
            return ProxApproxManagerDiscrete(xvar, xvarsqrd, xsqvar_cuts, ndn_i)
        else:
            return ProxApproxManagerContinuous(xvar, xvarsqrd, xsqvar_cuts, ndn_i)

class _ProxApproxManager:
    '''
    A helper class to manage proximal approximations
    '''
    __slots__ = ()

    def __init__(self, xvar, xvarsqrd, xsqvar_cuts, ndn_i):
        self.xvar = xvar
        self.xvarsqrd = xvarsqrd
        self.var_index = ndn_i
        self.cuts = xsqvar_cuts
        self.lb = self.xvar.lb
        self.ub = self.xvar.ub
        self._verify_lb_ub()

    def _verify_lb_ub(self):
        if self.lb is None or self.ub is None:
            raise RuntimeError(f"linearize_nonbinary_proximal_terms requires all "
                                "nonanticipative variables to have bounds")

    def create_initial_cuts(self):
        '''
        create initial cuts at val
        '''
        pass

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val
        '''
        pass

    def check_tol_add_cut(self, tolerance, persistent_solver=None):
        '''
        add a cut if the tolerance is not satified
        '''
        measured_val = self.xvarsqrd.value
        actual_val = self.xvar.value**2

        if (actual_val - measured_val) > tolerance:
            self.add_cut(self.xvar.value, persistent_solver)
            return True
        return False

class ProxApproxManagerContinuous(_ProxApproxManager):

    def __init__(self, xvar, xvarsqrd, xsqvar_cuts, ndn_i):
        _ProxApproxManager.__init__(self, xvar, xvarsqrd, xsqvar_cuts, ndn_i)
        self.cut_index = 0

    def create_initial_cuts(self):

        lb, ub = self.lb, self.ub

        if lb == ub:
            # var is fixed
            self.create_cut(lb)
            return

        print(f"adding cut for lb {lb}")
        self.add_cut(lb)
        print(f"adding cut for ub {ub}")
        self.add_cut(ub)

    def add_cut(self, val, persistent_solver=None):
        '''
        create a cut at val using a taylor approximation
        '''
        # f'(a) = 2*val
        # f(a) - f'(a)a = val*val - 2*val*val
        f_p_a = 2*val
        const = -(val*val)

        ## f(x) >= f(a) + f'(a)(x - a)
        ## f(x) >= f'(a) x + (f(a) - f'(a)a)
        ## (0 , f(x) - f'(a) x - (f(a) - f'(a)a) , None)
        expr = LinearExpression( linear_coefs=[1, -f_p_a],
                                 linear_vars=[self.xvarsqrd, self.xvar],
                                 constant=-const )
        self.cuts[self.var_index, self.cut_index] = (0, expr, None)
        if persistent_solver is not None:
            persistent_solver.add_constraint(self.cuts[self.var_index, self.cut_index])
        self.cut_index += 1

def _compute_mb(val):
    ## [(n+1)^2 - n^2] = 2n+1
    ## [(n+1) - n] = 1
    ## -> m = 2n+1
    m = 2*val+1

    ## b = n^2 - (2n+1)*n
    ## = -n^2 - n
    ## = -n (n+1)
    b = -val*(val+1)
    return m,b

class ProxApproxManagerDiscrete(_ProxApproxManager):

    def create_initial_cuts(self):
        lb, ub = self.lb, self.ub

        if lb == ub:
            # var is fixed
            self.create_cut(lb, side=Side.BOTH)
            return

        print(f"adding cut for lb {lb}")
        self.add_cut(lb, side=Side.RIGHT)
        print(f"adding cut for ub {ub}")
        self.add_cut(ub, side=Side.LEFT)

    def add_cut(self, val, persistent_solver=None, side=Side.BOTH):
        '''
        create up to two cuts at val, exploiting integrality
        '''
        val = int(round(val))

        ## cuts are indexed by the x-value to the right
        ## e.g., the cut for (2,3) is indexed by 3
        ##       the cut for (-2,-1) is indexed by -1

        ## So, a cut to the RIGHT of the point 3 is the cut for (3,4),
        ## which is indexed by 4
        if side in (Side.BOTH, Side.RIGHT):
            if (self.var_index, val+1) not in self.cuts and val < self.ub:
                m,b = _compute_mb(val)
                expr = LinearExpression( linear_coefs=[1, -m],
                                         linear_vars=[self.xvarsqrd, self.xvar],
                                         constant=-b )
                print(f"adding cut for {(val, val+1)}")
                self.cuts[self.var_index, val+1] = (0, expr, None)
                if persistent_solver is not None:
                    persistent_solver.add_constraint(self.cuts[self.var_index, val+1])

        ## Similarly, a cut to the LEFT of the point 3 is the cut for (2,3),
        ## which is indexed by 3
        if side in (Side.BOTH, Side.LEFT):
            if (self.var_index, val) not in self.cuts and val > self.lb:
                m,b = _compute_mb(val-1)
                expr = LinearExpression( linear_coefs=[1, -m],
                                         linear_vars=[self.xvarsqrd, self.xvar],
                                         constant=-b )
                print(f"adding cut for {(val-1, val)}")
                self.cuts[self.var_index, val] = (0, expr, None)
                if persistent_solver is not None:
                    persistent_solver.add_constraint(self.cuts[self.var_index, val])

if __name__ == '__main__':
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    #m.x = pyo.Var(bounds = (-3, 3))
    m.x = pyo.Var(within=pyo.Integers, bounds = (-3, 3))
    m.xsqrd = pyo.Var(within=pyo.NonNegativeReals)

    zero = 4
    ## ( x - zero )^2 = x^2 - 2 x zero + zero^2
    m.obj = pyo.Objective( expr = m.xsqrd - 2*zero*m.x + zero**2 )

    m.xsqrdobj = pyo.Constraint([0], pyo.Integers)

    s = pyo.SolverFactory('gurobi_persistent')
    prox_manager = ProxApproxManager(m.x, m.xsqrd, m.xsqrdobj, 0)
    prox_manager.create_initial_cuts()
    s.set_instance(m)
    m.pprint()
    new_cuts = True
    while new_cuts:
        print("")
        s.solve(m,tee=False)
        print(f"x: {pyo.value(m.x)}")
        new_cuts = prox_manager.check_tol_add_cut(1e-6, persistent_solver=s)
        m.pprint()

    print(f"objval: {pyo.value(m.obj)}, x: {pyo.value(m.x)}")
