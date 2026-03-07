import numpy as np
from matplotlib import pyplot as plt

def vizualize_func_levels(f_input, traj) -> None:
    x1 = np.linspace(-2, 2, 600)
    x2 = np.linspace(-2, 2, 600)    
    X1, X2 = np.meshgrid(x1, x2)

    Y = f_input(X1, X2)
    levels = np.quantile(Y, np.linspace(0.01, 0.95, 30))
    
    plt.contour(X1, X2, Y, levels=levels)
    plt.plot(traj[:, 0], traj[:, 1], 'r')
    plt.grid()
    plt.show()