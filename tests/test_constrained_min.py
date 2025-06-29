# --- makes project root importable ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from examples import *
import matplotlib.pyplot as plt

from src.constrained_min import constrainedMinimizer
from src.utils import (
    plot_feasible_and_path,
    plot_objective_vs_outer,
    print_final_candidate,
)



class TestConstrainedMinimizer(unittest.TestCase):
    
    
    
    def test_qp(self):
        const_min = constrainedMinimizer()
        
        x_star, f_star, _ = const_min.interior_pt(func=qp_objective, 
                                                     ineq_constraints=ineq_constraints, 
                                                     eq_constraints_mat=A_eq, 
                                                     eq_constraints_rhs=b_eq, 
                                                     x0=x0)
        
        hist = const_min.get_history()
        
        # (a) Feasible region + central path
        plot_feasible_and_path(
            ineq_constraints, A_eq, b_eq, hist,
            title="Central path (QP example)"
        )

        # (b) Objective value vs. outer iteration
        plot_objective_vs_outer(hist)

        # (c) Print final numbers
        print_final_candidate(qp_objective, x_star, ineq_constraints, A_eq, b_eq, label="QP - final")
        plt.show()
        
    
    def test_lp(self):
        
        const_min = constrainedMinimizer()
        
        x_star, f_star, _ = const_min.interior_pt(func=lp_objective, 
                                                     ineq_constraints=ineq_constraints_lp, 
                                                     eq_constraints_mat=None, 
                                                     eq_constraints_rhs=None, 
                                                     x0=x0_lp)
        
        hist = const_min.get_history()
        
        # (a) Feasible region + central path
        plot_feasible_and_path(
            ineq_constraints_lp,
            None, None,
            hist,
            title="Central path (LP example)"
        )
        
        # (b) Objective value vs. outer iteration
        plot_objective_vs_outer(hist)

        # (c) Print final numbers
        print_final_candidate(lp_objective, x_star, 
                              ineq_constraints_lp, 
                              None, None, label="LP - final")
        plt.show()
    
    
    
if __name__ == '__main__':
    unittest.main() 