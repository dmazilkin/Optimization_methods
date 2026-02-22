import numpy as np
from matplotlib import pyplot as plt

def quadratic_fit_search(bracket: list, univariate_func: callable, n: int = None):
    a, c = bracket
    b = (a + c) / 2
    fa = univariate_func(a)
    fb = univariate_func(b)
    fc = univariate_func(c)    
    
    # print(fb < fa, fb < fc)
    
    for _ in range(n - 3):
        num = fa * (b**2 - c**2) + fb * (c**2 - a**2) + fc * (a**2 - b**2)
        den = fa * (b - c) + fb * (c - a) + fc * (a - b)
        x_new = 0.5 * num / den
        f_new = univariate_func(x_new)
        
        if x_new > b:
            if f_new > fb:
                c = x_new
                fc = f_new
            else:
                a = b
                fa = fb
                b = x_new
                fb = f_new
        else:
            if f_new > fb:
                a = x_new
                fa = f_new
            else:
                c = b
                fc = fb
                b = x_new
                fb = f_new
                
    return [a, b, c]

def main():
    n = 5
    univariate_func = lambda x: 0.2 * np.exp(x-2) - x
    bracket = [-1, 5]
    
    a, b, c = quadratic_fit_search(bracket, univariate_func, n)    
    print(a, b, c)
    
    # visial check
    t = np.linspace(bracket[0], bracket[1])
    y = [univariate_func(t_val) for t_val in t]
    plt.plot(t, y)
    plt.scatter([a, b, c], [univariate_func(a), univariate_func(b), univariate_func(c)])
    plt.show()

if __name__ == '__main__':
    main()