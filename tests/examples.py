import numpy as np


def qp_objective(x: np.ndarray):
    """
    Quadratic program objective for

        minimise   f(x, y, z) = x**2 + y**2 + (z + 1)**2

    subject to    x + y + z = 1,
                  x >= 0, y >= 0, z >= 0.

    Parameters
    ----------
    x : np.ndarray
        A length-3 vector [x, y, z].

    Returns
    -------
    f_val : float
        Objective value at x.
    grad : np.ndarray
        Gradient of f at x.
    hess : np.ndarray
        Hessian of f (constant for this quadratic).
    """
    
    x, y, z = x
    f_val = x ** 2 + y ** 2 + (z + 1) ** 2
    grad = np.array([2 * x, 2 * y, 2 * (z + 1)], dtype=float)
    hess = 2.0 * np.eye(3)
    return f_val, grad, hess


# Equality constraint:  x + y + z = 1
A_eq = np.array([[1.0, 1.0, 1.0]])   # shape (1, 3)
b_eq = np.array([1.0])               # RHS vector of length 1

# Inequality constraints written as g_i(x) <= 0
def g1(x):
    # -x <= 0  ⇔  x >= 0
    return -x[0], np.array([-1.0, 0.0, 0.0]), np.zeros((3, 3))

def g2(x):
    # -y <= 0  ⇔  y >= 0
    return -x[1], np.array([0.0, -1.0, 0.0]), np.zeros((3, 3))

def g3(x):
    # -z <= 0  ⇔  z >= 0
    return -x[2], np.array([0.0, 0.0, -1.0]), np.zeros((3, 3))

ineq_constraints = [g1, g2, g3]


# A reasonable strictly‑feasible starting point for the interior‑point method
x0 = np.array([0.1, 0.2, 0.7]) #TODO fix
    



def lp_objective(x: np.ndarray):
    """
    Linear program objective for

        maximize  f(x, y) = x + y  ⇒  min -(x + y)

    subject to    y >= -x + 1
                  x <= 2, 0 <= y <= 1

    Parameters
    ----------
    x : np.ndarray
        A length-2 vector [x, y].

    Returns
    -------
    f_val : float
        Objective value at x.
    grad : np.ndarray
        Gradient of f at x.
    hess : np.ndarray
        Hessian of f
    """
    
    
   #Swap sign and minimize this function 
    
    x, y = x
   
    f_val = -(x+y)
    grad = np.array([-1, -1], dtype=float)
    hess = np.zeros(shape=(2,2))
    return f_val, grad, hess


#constraints remain the same

# Inequality constraints written as g_i(x) <= 0
def h1(x):
    #  -y -x + 1 <= 0
    return -x[0] - x[1] + 1, np.array([-1.0, -1.0]), np.zeros((2, 2))

def h2(x):
    # y - 1 <= 0  
    return x[1] - 1, np.array([0.0, 1.0]), np.zeros((2, 2))

def h3(x):
    # x - 2 <= 0 
    return x[1] - 2 , np.array([1.0, 0]), np.zeros((2, 2))

def h4(x):
    # -y <= 0 
    return -x[1] , np.array([0.0, -1.0]), np.zeros((2, 2))

# ineq_constraints = [h1, h2, h3, h4]

# # A reasonable strictly‑feasible starting point for the interior‑point method
# x0 = np.array([0.5, 0.75])
   