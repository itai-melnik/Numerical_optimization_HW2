import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional


# input: x, output: f(x), f'(x), hessian (optional)
FuncType = Callable[[np.ndarray],
                    Tuple[float, np.ndarray, Optional[np.ndarray]]]


@dataclass
class HistoryEntry:
    k: int
    x: np.ndarray
    f: float
    grad_norm: float


class constrainedMinimizer:
    """_summary_
    """

    def __init__(self,
                 obj_tol: float = 1e-12,
                 param_tol: float = 1e-8,
                 max_iter: int = 100,
                 t0: float = 1.0,
                 mu: float = 10.0,
                 rho: float = 0.5,
                 c1: float = 0.01,
                 epsilon: float = 1e-10) -> None:

        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter

        # barrier parameters
        self.t = t0
        self.mu = mu
        self.epsilon = epsilon

        # line search constants
        self.rho = rho
        self.c1 = c1

        # global x value
        self.x = None

        # previous values
        self.prev_x = None
        self.prev_f_val = None

        # internal storage
        self.history: List[HistoryEntry] = []

    def interior_pt(self, func: FuncType,
                    ineq_constraints: List[FuncType],
                    eq_constraints_mat: np.ndarray,
                    eq_constraints_rhs: np.ndarray,
                    x0: np.ndarray) -> Tuple[np.ndarray, float]:
        
        n = x0.size
        if eq_constraints_mat is None:
            A = np.empty((0, n))
            b = np.empty((0,))
        else:
            A = np.asarray(eq_constraints_mat, dtype=float)
            b = np.asarray(eq_constraints_rhs, dtype=float)
        
        #clear the history from previous runs
        self.history.clear() 

        # get augmented objective
        def aug_obj(x) -> FuncType:

            # values of original objective
            f_val, g_val, H_val = func(x)

            # values of phi
            phi, gphi, Hphi = self._log_barrier(x, ineq_constraints)

            return (f_val + phi, g_val + gphi, H_val + Hphi)

        bool_flag = False
     

            
       
        self.x = x0  # initial x
        m = len(ineq_constraints)         # number of inequalities

        outer_iterations = 0
        # outer iterations
        while (m/self.t >= self.epsilon):
            outer_iterations += 1
            
            # only save history for outer iterations
            f_val, g_val, _ = func(self.x)
            self._save_history(outer_iterations, f_val, g_val)
            
            # inner iterations
            for k in range(self.max_iter):
                f_val, g, h = aug_obj(self.x)

                # # print to console:
                # print('iteration number:', k + 1)
                # print('current location 𝑥𝑖:', self.x)
                # print('current objective value 𝑓(𝑥𝑖 ):', f_val)

                # break if stopping criteria met
                if self._is_converged(k):
                    bool_flag = True
                    break

                # get pk (primal) and w (dual)
                pk, w = self._solve_kkt(
                    H=h, A=A, g=g)

                # computer alpha_k
                alpha = self._backtracking(self.x, f_val, g, pk, f=aug_obj)

                # update
                x_new = self.x + alpha * pk

                # store previous values for stopping termination conditions

                self.prev_x = self.x
                self.prev_f_val = f_val

                # x_(k+1) = x_new
                self.x = x_new

        
            self.t = self.t * self.mu   # t -> ut
            
            
        # save history of last iteration
        f_val, g_val, _ = func(self.x)
        self._save_history(outer_iterations, f_val, g_val)

        # returns final location, final value and bool flag
        return self.x, self.history[-1].f, bool_flag

    ### helper methods ###

    def _solve_kkt(self,
                   H: np.ndarray,
                   A: np.ndarray,
                   g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the KKT system    [ H  Aᵀ ][ p ] = [ -g ]
                                [ A   0 ][ w ]   [  0 ]
        Returns search direction p and dual w.
        """
        
        m = A.shape[0]          # number of equalities

        if m == 0:
            # No equalities: solve H p = -g
            p = np.linalg.solve(H, -g)
            w = np.empty((0,))  # empty dual vector
            return p, w
        else:
            zeros = np.zeros(m)
            KKT_mat = np.block([[H, A.T],
                                [A, np.zeros((m, m))]])

            rhs = np.concatenate([-g, zeros])
            solution = np.linalg.solve(KKT_mat, rhs)
            n = g.size
            return solution[:n], solution[n:]

    def _log_barrier(self,
                     x: np.ndarray,
                     ineq_constraints: List[FuncType]
                     ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        φ_t(x)   = -(1/t) Σ_i log(-g_i(x))
        ∇φ_t(x)  =  (1/t) Σ_i ∇g_i(x) / g_i(x)
        ∇²φ_t(x) =  (1/t) Σ_i [ ∇g_i ∇g_iᵀ / g_i²  -  ∇²g_i / g_i ]
        """
        vals, grads, hess_list = [], [], []

        for gfun in ineq_constraints:
            val, grad, hess = gfun(x)
            if hess is None:
                hess = np.zeros((len(x), len(x)))
            vals.append(val)
            grads.append(grad)
            hess_list.append(hess)

        vals = np.asarray(vals, dtype=float)

        if np.any(vals >= 0):
        # Infeasible point → very high barrier value
            return np.inf, np.zeros_like(x), np.zeros((len(x), len(x)))
        
        inv_t = 1.0 / self.t
        phi = -inv_t * np.sum(np.log(-vals))
        gphi = inv_t * sum(grad / val for grad, val in zip(grads, vals))
        Hphi = inv_t * sum((np.outer(grad, grad) / (val ** 2) - hess / val)
                           for grad, hess, val in zip(grads, hess_list, vals))

        return phi, gphi, Hphi

    def _is_converged(self, k: int) -> bool:
        """check the stopping criteria"""

        if k == 0:
            return False

        param_change = np.linalg.norm(self.x - self.prev_x)
        obj_change = abs(self.history[-1].f - self.prev_f_val)

        if (param_change < self.param_tol or obj_change < self.obj_tol):
            return True

        return False

    def _backtracking(self,
                      x: np.ndarray,
                      f_val: float,
                      g: np.ndarray,
                      p: np.ndarray,
                      f: Callable[[np.ndarray],
                                  Tuple[float, np.ndarray, np.ndarray]]
                      ) -> float:
        """ used to compute alpha (the step size)"""

        alpha = 1.0  # initial value

        while True:
            new_val, *_ = f(x + alpha * p)

            if new_val <= f_val + self.c1 * alpha * g.dot(p):
                break
            alpha *= self.rho

            if alpha < 1e-12:
                break

        return alpha

    def _save_history(self, k, f_val, g) -> None:
        """save the history from previous iterations"""
        self.history.append(HistoryEntry(
            k=k, x=self.x.copy(), f=f_val, grad_norm=np.linalg.norm(g)))

    def get_history(self) -> List[HistoryEntry]:
        """ return the iteration history"""
        return self.history
