from utils.vizialize import vizualize_func_levels

import numpy as np

def nelder_mead(
    f_input: callable,
    n: int = 100,
    threshold: float = 1e-6,
    simplex: np.ndarray = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.5
):
    if simplex is None:
        simplex = np.array([
            [-1.0, -1.0],
            [-0.8, -0.8],
            [-0.5, -1.0]
        ], dtype=np.float64)
    else:
        simplex = np.array(simplex, dtype=np.float64)

    traj = np.zeros([n, 3])
    count = 0

    while count < n:
        values = np.array([f_input(p[0], p[1]) for p in simplex])
        order = np.argsort(values)

        simplex = simplex[order]
        values = values[order]

        xl = simplex[0]
        xh = simplex[-1]
        xs = simplex[-2]

        yl = values[0]
        yh = values[-1]
        ys = values[-2]

        traj[count, :2] = xl
        traj[count, 2] = yl

        if np.std(values) <= threshold:
            break

        xm = np.mean(simplex[:-1], axis=0)

        # reflection
        xr = xm + alpha * (xm - xh)
        yr = f_input(xr[0], xr[1])

        if yr < yl:
            # expansion
            xe = xm + beta * (xr - xm)
            ye = f_input(xe[0], xe[1])

            if ye < yr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr

        elif yr > ys:
            if yr <= yh:
                simplex[-1] = xr
                xh = xr
                yh = yr

            # contraction
            xc = xm + gamma * (xh - xm)
            yc = f_input(xc[0], xc[1])

            if yc > yh:
                # shrink
                for i in range(1, len(simplex)):
                    simplex[i] = (simplex[i] + xl) / 2.0
            else:
                simplex[-1] = xc

        else:
            simplex[-1] = xr

        count += 1

    best_idx = np.argmin([f_input(p[0], p[1]) for p in simplex])
    x = simplex[best_idx]

    return x, traj[:count + 1]

def main():
    a, b = 1, 5
    n = 100
    f_input = lambda x1, x2: (a - x1)**2 + b * (x2 - x1**2)**2

    simplex = np.array([
        [-1.0, -1.0],
        [-0.8, -0.8],
        [-0.5, -1.0]
    ])

    x_nm, traj_nm = nelder_mead(f_input, n=n, threshold=1e-6, simplex=simplex)
    print(f'Nelder-Mead minimum is {x_nm}')
    vizualize_func_levels(f_input, traj_nm)

if __name__ == '__main__':
    main()