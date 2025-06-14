import unittest
import examples

from src.constrained_min import constrainedMinimizer



class TestConstrainedMinimizer(unittest.TestCase):
    
    
    
    def test_lp():
        const_min = constrainedMinimizer()
        
        x_star, f_star, flag = const_min.interior_pt(qp_objective(), ineq_constraints, A_eq, b_eq, x0)
        
        print(x_star, f_star)
    
    def test_qp():
        pass
    
    
if __name__ == '__main__':
    unittest.main() 