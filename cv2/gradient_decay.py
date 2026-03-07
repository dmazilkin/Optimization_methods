
from utils import vizualize_func_levels

import numpy as np
from matplotlib import pyplot as plt


def gradient_descent(f_input: callable, grad: callable, x0: list, n: int = 100, threshold = 1e-2):
    x = np.array(x0, dtype=np.float64)
    x_traj = np.zeros([n, 3])
    stepsize_decay = lambda k: 0.9**(k)
    
    count = 0
    
    while count < n:
        x_traj[count, :2] = x
        x_traj[count, 2] = f_input(x[0], x[1])
        
        grad_vec = np.array(grad(x[0], x[1]), dtype=np.float64)
        grad_norm = np.linalg.norm(grad_vec)
        direction = -grad_vec / grad_norm
        
        if grad_norm <= threshold: 
            break

        stepsize = stepsize_decay(count)
        x += stepsize * direction
        
        count += 1

    return x, x_traj[:count+1]

def main():
    a, b = 1, 5
    n = 100
    threshold = 1e-2
    f_input = lambda x1, x2: (a - x1)**2 + b*(x2 - x1**2)**2
    f_grad = lambda x1, x2: [2*(a - x1)*(-1) + 2*b*(x2 - x1**2)*2*x1*(-1), 2*b*(x2 - x1**2)]

    x, traj = gradient_descent(f_input, f_grad, [-1, -1], n, threshold)
    print(f'Minimus is {x}')
    
    vizualize_func_levels(f_input, traj)
    
if __name__ == '__main__':
    main()