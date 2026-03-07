from utils.vizialize import vizualize_func_levels

import numpy as np


def hooke_jeeves(f_input: callable, x0: list, n: int = 100, alpha: float = 1.0, epsilon: float = 1e-6):
    x = np.array(x0, dtype=np.float64)
    traj = np.zeros([n, 3])

    count = 0

    while count < n:
        traj[count, :2] = x
        traj[count, 2] = f_input(x[0], x[1])

        x_base = x.copy()
        f_base = f_input(x[0], x[1])

        x_best = x.copy()
        f_best = f_base
        improved = False

        # exploratory search
        for i in range(2):
            for sgn in [-1, 1]:
                x_new = x.copy()
                x_new[i] += sgn * alpha
                f_new = f_input(x_new[0], x_new[1])

                if f_new < f_best:
                    x_best = x_new.copy()
                    f_best = f_new
                    improved = True

        if improved:
            # pattern move
            x = x_best + (x_best - x_base)
            f_pattern = f_input(x[0], x[1])

            if f_pattern >= f_best:
                x = x_best.copy()
        else:
            alpha *= 0.5

        if alpha <= epsilon:
            break

        count += 1

    return x, traj[:count + 1]

def main():
    a, b = 1, 5
    n = 100
    f_input = lambda x1, x2: (a - x1)**2 + b * (x2 - x1**2)**2

    x_hj, traj_hj = hooke_jeeves(f_input, [-1, -1], n=n, alpha=1.0, epsilon=1e-6)
    print(f'Hooke-Jeeves minimum is {x_hj}')
    vizualize_func_levels(f_input, traj_hj)

if __name__ == '__main__':
    main()