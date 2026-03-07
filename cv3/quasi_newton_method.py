from cv1 import fibonacci_search, golden_section_search, quadratic_fit_search
from utils.vizialize import vizualize_func_levels

import numpy as np
from matplotlib import pyplot as plt

def quasi_newton_method(f_input: callable, grad: callable, x0: list, n: int = 100, threshold = 1e-2, bracketing_method: callable = fibonacci_search):
    x = np.array(x0, dtype=np.float64)
    traj = np.zeros([n, 3])
    Q = np.eye(x.size)
    
    count = 0
    
    while count < n:
        traj[count, :2] = x
        traj[count, 2] = f_input(x[0], x[1])
        
        grad_vec = np.array(grad(x[0], x[1]), dtype=np.float64)
        
        if count > 0:
            Q = davidon_fletcher_powell(grad_vec, grad_vec_prev, x, x_prev, Q)
            
        quasi_newton_grad = Q @ grad_vec
        grad_norm = np.linalg.norm(grad_vec)
            
        if grad_norm <= threshold: 
            break
        
        stepsize_interval = find_stepsize_interval(f_input, quasi_newton_grad, x)
        phi = lambda stepsize: f_input(x[0] - stepsize * quasi_newton_grad[0], x[1] - stepsize * quasi_newton_grad[1])
        a, b = bracketing_method(stepsize_interval, phi, 50)

        x_prev = x.copy()
        grad_vec_prev = grad_vec.copy()        
        
        stepsize = (a + b) / 2
        x -= stepsize * quasi_newton_grad
                
        count += 1

    return x, traj[:count+1]

def davidon_fletcher_powell(grad_vec_cur, grad_vec_prev, x_cur, x_prev, Q):
    gamma = np.array([grad_vec_cur - grad_vec_prev]).T
    sigma = np.array([x_cur - x_prev]).T
    
    return Q - Q @ gamma @ gamma.T @ Q / (gamma.T @ Q @ gamma) + sigma @ sigma.T / (sigma.T @ gamma)

def find_stepsize_interval(f_input: callable, grad_vec: np.array, x0: np.array):
    a1 = 0.0
    a2 = 1.0

    x_new_a1 = x0 - a1 * grad_vec
    f1 = f_input(x_new_a1[0], x_new_a1[1])

    x_new_a2 = x0 - a2 * grad_vec
    f2 = f_input(x_new_a2[0], x_new_a2[1])

    while f2 < f1:
        a1, f1 = a2, f2
        a2 *= 2
        x_new_a2 = x0 - a2 * grad_vec
        f2 = f_input(x_new_a2[0], x_new_a2[1])

    return [a1, a2]

def main():
    a, b = 1, 5
    n = 100
    threshold = 1e-2
    f_input = lambda x1, x2: (a - x1)**2 + b*(x2 - x1**2)**2
    f_grad = lambda x1, x2: [2*(a - x1)*(-1) + 2*b*(x2 - x1**2)*2*x1*(-1), 2*b*(x2 - x1**2)]

    x, traj = quasi_newton_method(f_input, f_grad, [-1, -1], n, threshold)
    print(f'Minimus is {x}')

    vizualize_func_levels(f_input, traj)
    
if __name__ == '__main__':
    main()