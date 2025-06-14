# --- makes project root importable ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from examples import *

from src.constrained_min import constrainedMinimizer



class TestConstrainedMinimizer(unittest.TestCase):
    
    
    
    def test_qp(self):
        const_min = constrainedMinimizer()
        
        x_star, f_star, flag = const_min.interior_pt(func=qp_objective, 
                                                     ineq_constraints=ineq_constraints, 
                                                     eq_constraints_mat=A_eq, 
                                                     eq_constraints_rhs=b_eq, 
                                                     x0=x0)
        
        print(x_star, f_star)
    
    def test_lp(self):
        
        const_min1 = constrainedMinimizer()
        
        x_star, f_star, flag = const_min1.interior_pt(func=lp_objective, 
                                                     ineq_constraints=ineq_constraints_lp, 
                                                     eq_constraints_mat=None, 
                                                     eq_constraints_rhs=None, 
                                                     x0=x0_lp)
        #TODO check if need -f 
        print(x_star, -f_star)
    
    
    
if __name__ == '__main__':
    unittest.main() 