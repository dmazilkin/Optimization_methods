from cv1 import fibonacci_search, golden_section_search, quadratic_fit_search
from utils.vizialize import vizualize_func_levels

import numpy as np

def cyclic_coordinate_search(f_input: callable, x0: list, n: int = 100, threshold = 1e-4, bracketing_method: callable = fibonacci_search):
    x = np.array(x0, dtype=np.float64)
    traj = np.zeros([n, 3])
    
    count = 0
    err = float('inf')
    
    while count < n :
        traj[count, :2] = x
        traj[count, 2] = f_input(x[0], x[1])
        
        if count > 0:
            err = np.linalg.norm(x - x_prev)
            
        if err <= threshold:
            break
        
        x_prev = x.copy()
        
        # update x1
        stepsize_interval = find_stepsize_interval(f_input, np.array([1, 0]), x)
        phi = lambda stepsize: f_input(x[0] + stepsize, x[1])
        a, b = bracketing_method(stepsize_interval, phi, 50)
        
        stepsize = (a + b) / 2
        x[0] += stepsize
        
        # update x2
        stepsize_interval = find_stepsize_interval(f_input, np.array([0, 1]), x)
        phi = lambda stepsize: f_input(x[0], x[1] + stepsize)
        a, b = bracketing_method(stepsize_interval, phi, 50)
        
        stepsize = (a + b) / 2
        x[1] += stepsize
        
        # acceleration step
        direction = x - x_prev
        
        stepsize_interval = find_stepsize_interval(f_input, direction, x)
        phi = lambda stepsize: f_input(x[0] + stepsize*direction[0], x[1] + stepsize*direction[1])
        a, b = bracketing_method(stepsize_interval, phi, 50)
        
        stepsize = (a + b) / 2
        x += stepsize * direction
        
        
        count += 1

    return x, traj[:count+1]

def find_stepsize_interval(f_input: callable, basis_vec: np.array, x0: np.array):
    a1 = 0.0
    a2 = 1.0

    x_new_a1 = x0 + a1 * basis_vec
    f1 = f_input(x_new_a1[0], x_new_a1[1])

    x_new_a2 = x0 + a2 * basis_vec
    f2 = f_input(x_new_a2[0], x_new_a2[1])

    while f2 < f1:
        a1, f1 = a2, f2
        a2 *= 2
        x_new_a2 = x0 + a2 * basis_vec
        f2 = f_input(x_new_a2[0], x_new_a2[1])

    return [a1, a2]

def main():
    a, b = 1, 5
    n = 100
    threshold = 1e-4
    f_input = lambda x1, x2: (a - x1)**2 + b*(x2 - x1**2)**2

    x, traj = cyclic_coordinate_search(f_input, [-1, -1], n, threshold)
    print(f'Minimus is {x}')

    vizualize_func_levels(f_input, traj)
    
if __name__ == '__main__':
    main()