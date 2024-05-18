import numpy as np
from scipy.optimize import line_search

class UnconstrainedMinimization:
    def __init__(self,f, grad_f, hessian_f=None, method='gradient_descent', max_iter=100, obj_tol=10**-12, param_tol=10**-8):
        self.method = method
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.path = []
        self.evaluations = []
        self.f = f
        self.grad_f = grad_f
        self.hessian_f = hessian_f

    def wolfe_condition(self, x, val, direction, alpha=0.01, beta=0.5):
        step_length = 1.0
        curr_val = self.f(x + step_length * direction)

        while curr_val > val + alpha * step_length * self.grad_f(x).dot(direction):
            step_length *= beta
            curr_val= self.f(x + step_length * direction)

        return step_length

    def minimize(self, x0):
        x = x0
        for i in range(self.max_iter):
            f_evaluation = self.f(x)
            self.path.append(x)
            self.evaluations.append(f_evaluation)
            print("Iteration {}: x = {}, f(x) = {}".format(i, x, f_evaluation))

            # Calculating step size using gradient method
            gradient_direction = - self.grad_f(x)
            if self.method == 'newton':
                if self.hessian_f is None:
                    print("Terminating due to non hessian matrix for newton")
                    return x, f_evaluation, False
                gradient_direction = np.linalg.solve(self.hessian_f(x), gradient_direction)
                
            step_size = self.wolfe_condition(x, f_evaluation, gradient_direction)

            # Stop iterating when step size is not defined
            if step_size == None:
                return x, f_evaluation, True

            # Caculate next X and it's evaluation
            next_x = x + step_size * gradient_direction
            next_f = self.f(next_x)

            if np.abs(next_f) == np.inf:
                print("Terminating iteration due to inf value in f(x)")
                return next_x, next_f, False 
            
            if np.any(np.isnan(next_x)):
                print("Iteration {}: x = {}, f(x) = {}".format(i, next_x, next_f))
                print("Terminating iteration due to NaN value in x")
                return next_x, next_f, False 
            
            if np.linalg.norm(next_x - x) < self.param_tol or np.abs(next_f - f_evaluation) < self.obj_tol:
                return next_x, next_f, True
            
            x = next_x
