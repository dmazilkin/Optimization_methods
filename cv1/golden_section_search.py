import numpy as np
from matplotlib import pyplot as plt

def golden_section_search(bracket: list, univariate_func: callable, n: int = None):
    phi = (np.sqrt(5) - 1) / 2
    ind = 0
    epsilon = 1e-3
    
    a, b = bracket
    interval_len = b - a
    
    x2 = a + interval_len * phi
    y2 = univariate_func(x2)
    
    x1 = b - interval_len * phi
    y1 = univariate_func(x1)
    
    while (n is None or ind < n - 2) and (np.abs(b - a) > epsilon):
        if y1 <= y2:
            b = x2
            x2 = x1
            y2 = y1
            # calculate new x1, y1
            interval_len = b - a
            
            x1 = b - interval_len * phi
            y1 = univariate_func(x1)
        else:
            a = x1
            x1 = x2
            y1 = y2
            # calculate new x2, y2
            interval_len = b - a
            
            x2 = a + interval_len * phi
            y2 = univariate_func(x2)
        ind += 1;
        
    return [a, b]

def main():
    n = 5
    univariate_func = lambda x: 0.2 * np.exp(x-2) - x
    bracket = [-1, 5]
    
    a, b = golden_section_search(bracket, univariate_func, n)    
    print(a, b)
    
    # visial check
    t = np.linspace(bracket[0], bracket[1])
    y = [univariate_func(t_val) for t_val in t]
    plt.plot(t, y)
    plt.scatter([a, b], [univariate_func(a), univariate_func(b)])
    plt.show()

if __name__ == '__main__':
    main()