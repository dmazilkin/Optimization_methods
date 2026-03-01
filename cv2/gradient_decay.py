
from utils import plot_contours_with_traj

import numpy as np
from matplotlib import pyplot as plt


def gradient_descent(f_input: callable, grad: callable, x0: list, n: int = 100, threshold = 1e-2):
    x = np.array(x0, dtype=np.float64)
    x_traj = np.zeros([n, 3])
    stepsize_decay = lambda k: 0.9**(k+28)
    
    count = 0
    
    while count < n:
        x_traj[count, :2] = x
        x_traj[count, 2] = f_input(x[0], x[1])
        
        grad_vec = np.array(grad(x[0], x[1]), dtype=np.float64)
        grad_norm = np.linalg.norm(grad_vec)
        
        if grad_norm <= threshold: 
            break

        stepsize = stepsize_decay(count)
        x -= stepsize * grad_vec
        
        count += 1

    return x, x_traj[:count+1]

def main():
    a, b = 1, 5
    n = 100
    threshold = 1e-2
    f_input = lambda x1, x2: (a - x1)**2 + b*(x2 - x1**2)**2
    f_grad = lambda x1, x2: [2*(a - x1)*(-1) + 2*b*(x2 - x1**2)*2*x1*(-1), 2*b*(x2 - x1**2)]

    x, x_traj = gradient_descent(f_input, f_grad, [-1, -1], n, threshold)
    print(f'Minimus is {x}')
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x1 = np.linspace(-5, 5, n)
    x2 = np.linspace(-5, 5, n)
    X1, X2 = np.meshgrid(x1, x2)
    Y = f_input(X1, X2)
    ax.plot_surface(X1, X2, Y, alpha=0.6, linewidth=0)
    ax.plot(x_traj[:,0], x_traj[:,1], x_traj[:,2], marker='o')
    plt.show()
    
if __name__ == '__main__':
    main()