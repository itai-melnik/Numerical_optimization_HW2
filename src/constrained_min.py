import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional





FuncType = Callable[[np.ndarray], Tuple[float, np.ndarray, Optional[np.ndarray]]]  #input: x, output: f(x), f'(x), hessian (optional)

@dataclass
class HistoryEntry:
    k: int
    x: np.ndarray
    f: float
    grad_norm: float

class constrainedMinimizer:
    
    """
    
    
    """
    
    def __init__(self,
                 obj_tol=1e-12, 
                 param_tol=1e-8, 
                 max_iter=100, 
                 t=1,
                 mu=10, 
                 rho=0.5, 
                 c1=0.01,
                 epsilon=1e-10) -> None:
        
        
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        
        # interior‑point parameters
        self.t = t          # current barrier parameter
        self.mu = mu        # factor used to increase t
        
        
        #constants
        self.rho = rho #backtracking constant
        self.c1 = c1 #Wolfe condition
        
        
        # interior‑point parameters
        self.t = t          # current barrier parameter
        self.mu = mu        # factor used to increase t
        
        
        #tolerance for outer iterationa
        self.epsilon = epsilon
        
        
        #global x value
        self.x = None
        
        #previous values
        self.prev_x = None
        self.prev_f_val = None
        
        
        
        
        #internal storage
        self.history: List[HistoryEntry] = []
        
        
        
    def interior_pt(self, func: FuncType, 
                    ineq_constraints: List[FuncType], 
                    eq_constraints_mat: List[List[Callable]], 
                    eq_constraints_rhs: np.ndarray, 
                    x0: float) -> Tuple[np.ndarray, float]:
        
        
       
        
        #get augmented objective
        def aug_obj(x) -> FuncType:
                
            #values of original objective
            v0, g0, h0 = func(x)
            
            
            # ---- Log–barrier contribution ---------------------------------
            # Each inequality constraint g_i(x) must satisfy g_i(x) ≤ 0.
            # We use the scaled barrier  φ_t(x) = -(1 / self.t) * Σ log(-g_i(x)).
            vals      = []   # g_i(x) values
            grads     = []   # ∇g_i(x)
            hess_list = []   # ∇²g_i(x)   (may be None)

            for gfun in ineq_constraints:
                g_val, g_grad, g_hess = gfun(x)
                vals.append(g_val)
                grads.append(g_grad)
                # If the supplied constraint has no Hessian, treat it as zero.
                if g_hess is None:
                    g_hess = np.zeros((len(x), len(x)))
                hess_list.append(g_hess)

            vals = np.asarray(vals, dtype=float)

            # If any constraint is violated (g_i(x) ≥ 0) we return +∞ so that
            # the line‑search rejects this point.
            if np.any(vals >= 0):
                return np.inf, np.full_like(x, np.nan), np.full((len(x), len(x)), np.nan)

            inv_t = 1.0 / self.t          # 1 / t  (barrier parameter scaling)

            # scalar barrier term
            v1 = -inv_t * np.sum(np.log(-vals))

            # gradient of the barrier
            g1 = inv_t * np.sum([grad / val for grad, val in zip(grads, vals)], axis=0)

            # Hessian of the barrier
            h1 = inv_t * np.sum(
                [
                    (np.outer(grad, grad) / (val ** 2) - hess / val)
                    for grad, hess, val in zip(grads, hess_list, vals)
                ],
                axis=0,
            )
            
            
        
            #values of augmented function 
            
            val = v0 + v1
            
            grad = g0 + g1
            
            hessian = h0 + h1
            

            return val, grad, hessian
        
        
        bool_flag = False
        m = len(ineq_constraints) #number of inequality constraints 
        self.x = x0 #initial x
        

        #outer iterations
        
        while(m/self.t >= self.epsilon):
            
            #inner iterations 
            for k in range(self.max_iter):
                f_val, g, h = aug_obj(self.x)
                
                
                #save current iteration
                self._save_history(k, f_val, g)

                #print to console:
                print('iteration number:',k + 1 )
                print('current location 𝑥𝑖:',self.x)
                print('current objective value 𝑓(𝑥𝑖 ):',f_val )


                #break if stopping criteria met
                if self._is_converged(k):
                    bool_flag = True
                    break
                
                
                #TODO create the matrix

                KKT_mat = np.block([[h, eq_constraints_rhs.T],
                                [eq_constraints_rhs, np.zeros((eq_constraints_rhs.shape[0], eq_constraints_rhs.shape[0]))]
                        ])
                
                
                
                #TODO make sure shapes are correct 
                zeros = np.zeros(eq_constraints_rhs.shape[0])
                KKT_vec = np.concatenate([-g, zeros])
        
                
                
                #solve linear equation
                
                KKT_result = np.linalg.solve(KKT_mat,KKT_vec)
                
                #extract p and w (solutions)
                #TODO fix
                pk = KKT_result[:, :]
                w = KKT_result[:, :]
                
            
                #computer alpha_k
                alpha = self.backtracking(x, f_val, g, pk)
            

                #update
                x_new = self.x + alpha * pk


                #store previous values for stopping termination conditions 

                self.prev_x = self.x
                self.prev_f_val = f_val


                #x_(k+1) = x_new
                self.x = x_new

              
        
        #returns final location, final value and bool flag
        return self.x, self.history[-1].f, bool_flag
            
            
            
            
    
            
            
    ### helper methods ###
    
    
    def _is_converged(self, k: int) -> bool:
        """check the stopping criteria"""
        
        if k == 0:
            return False
        
        param_change = np.linalg.norm(self.x - self.prev_x)
        obj_change = abs(self.history[-1].f - self.prev_f_val)
        
        
        if(param_change < self.param_tol or obj_change < self.obj_tol):
            return True
        
        return False
        
    
    
    def _compute_direction(self, g, h) -> np.ndarray:
        """compute pk"""
        
      #newton method
      
        try:
            return -np.linalg.solve(h, g) #solves ax = b (gets x) h*p = -g since p = -g/h so we get p. AVOID computing inverse
        except np.linalg.LinAlgError:
            return -g
        
        
              
            
         
    
    def backtracking(self, x, f_val, g, p) -> float:
        """ used to compute alpha (the step size)"""
        
        alpha = 1.0 #initial value
        
        while self.f(x + alpha*p)[0] > f_val + self.c1*alpha*g.dot(p):
            alpha *= self.rho
            
            if alpha < 1e-12:
                break
            
            
        return alpha
        
    
    def _save_history(self, k, f_val, g) -> None:
        """save the history from previous iterations"""
        self.history.append(HistoryEntry(k=k,x=self.x.copy(),f=f_val,grad_norm=np.linalg.norm(g)))
        
        
        
    def get_history(self) -> List[HistoryEntry]:
        """ return the iteration history"""
        return self.history
        
        
        
        
    
        
        
        
    # def minimize(self) -> Tuple[np.ndarray, float]:
        
    #     bool_flag = False
       
            
    #     for k in range(self.max_iter):
            
    #         f_val, g, h = self.f(self.x)
            
            
    #         #save current iteration
    #         self._save_history(k, f_val, g)
            
    #         #print to console:
    #         print('iteration number:',k + 1 )
    #         print('current location 𝑥𝑖:',self.x)
    #         print('current objective value 𝑓(𝑥𝑖 ):',f_val )
            
    #         #break if stopping criteria met
    #         if self._is_converged(k):
    #             bool_flag = True
    #             break
            
            
    #         #compute p_k(the direction) changes depends on which method we use
    #         p = self._compute_direction(g, h)
            
            
    #         #computer alpha_k
    #         alpha = self.backtracking(self.x, f_val, g, p )
            
    #         #update
    #         x_new = self.x + alpha * p
            
            
    #         #store previous values for stopping termination conditions 
            
    #         self.prev_x = self.x
    #         self.prev_f_val = f_val
            
    
    #         #x_(k+1) = x_new
    #         self.x = x_new
            
              
        
    #     #returns final location, final value and bool flag
    #     return self.x, self.history[-1].f, bool_flag
            
            
    
        