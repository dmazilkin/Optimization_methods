from cv1 import fibonacci_search, golden_section_search, quadratic_fit_search

import numpy as np
from matplotlib import pyplot as plt


def gradient_descent(f_input: callable, grad: callable, x0: list, n: int = 100, threshold = 1e-2, bracketing_method: callable = fibonacci_search):
    x = np.array(x0, dtype=np.float64)
    grad_val = np.array(grad(x[0], x[1]), dtype=np.float64)

    count = 0
    err = float('inf')
    
    while err > threshold and count < n:
        stepsize_interval = find_stepsize_interval(f_input, grad, x)
        phi = lambda stepsize: f_input(x[0] - stepsize * grad_val[0], x[1] - stepsize * grad_val[1])
        a, b = bracketing_method(stepsize_interval, phi, 50)
        stepsize = (a + b) / 2
        x -= stepsize * grad_val
        grad_val_prev = grad_val
        grad_val = np.array(grad(x[0], x[1]), dtype=np.float64)
        err = np.linalg.norm(np.array(grad_val) - np.array(grad_val_prev))
        count += 1

    return x

def find_stepsize_interval(f_input: callable, grad: callable, x0: list):
    a1 = 0
    a2 = 1

    x1_new_a1 = x0[0] - a1 * grad(x0[0], x0[1])[0]
    x2_new_a1 = x0[1] - a1 * grad(x0[0], x0[1])[1]
    f1 = f_input(x1_new_a1, x2_new_a1)

    x1_new_a2 = x0[0] - a2 * grad(x0[0], x0[1])[0]
    x2_new_a2 = x0[1] - a2 * grad(x0[0], x0[1])[1]
    f2 = f_input(x1_new_a2, x2_new_a2)

    while f2 < f1: 
        a2 *= 2
        x1_new_a2 = x0[0] - a2 * grad(x0[0], x0[1])[0]
        x2_new_a2 = x0[1] - a2 * grad(x0[0], x0[1])[1]
        f2 = f_input(x1_new_a2, x2_new_a2)

    return [a1, a2]

def main():
    a, b = 1, 5
    f_input = lambda x1, x2: (a - x1)**2 + b*(x2 - x1**2)**2
    f_grad = lambda x1, x2: [2*(a - x1)*(-1) + 2*b*(x2 - x1**2)*2*x1*(-1), 2*b*(x2 - x1**2)]

    x = gradient_descent(f_input, f_grad, [-1, -1])
    print(x)

if __name__ == '__main__':
    main()