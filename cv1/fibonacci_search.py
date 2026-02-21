import numpy as np
from matplotlib import pyplot as plt
import math

def fibonacci_search(bracket: list, univariate_func: callable, n: int):
    fib_numbers = find_fib_numbers(n+1)
    n_remaining = n
    
    a, b = bracket
    interval_len = b - a
    
    x2 = a + interval_len * fib_numbers[-2]/fib_numbers[-1]
    y2 = univariate_func(x2)
    n_remaining -= 1
    
    x1 = a + interval_len * fib_numbers[-3]/fib_numbers[-1]
    y1 = univariate_func(x1)
    n_remaining -= 1

        
    while n_remaining > 0:
        fib_numbers.pop()
        
        
        if y1 <= y2:
            b = x2
            x2 = x1
            y2 = y1
            # calculate new x1, y1
            interval_len = b - a
            
            x1 = a + interval_len * fib_numbers[-3]/fib_numbers[-1]
            y1 = univariate_func(x1)
        else:
            a = x1
            x1 = x2
            y1 = y2
            # calculate new x2, y2
            interval_len = b - a
            
            x2 = a + interval_len * fib_numbers[-2]/fib_numbers[-1]
            y2 = univariate_func(x2)
        
        n_remaining -= 1
        
    return [a, b]

def find_fib_numbers(n: int):
    fib_numbers = []
    i = 0
    
    while i < n:
        if i <= 1:
            fib_numbers.append(1)
        else:
            fib_numbers.append(fib_numbers[-1] + fib_numbers[-2])
        i += 1
        
    return fib_numbers

def main():
    n = 5
    univariate_func = lambda x: 0.2 * np.exp(x-2) - x
    bracket = [-1, 5]
    
    a, b = fibonacci_search(bracket, univariate_func, n)    
    print(a, b)
    
    # visial check
    t = np.linspace(bracket[0], bracket[1])
    y = [univariate_func(t_val) for t_val in t]
    plt.plot(t, y)
    plt.scatter([a, b], [univariate_func(a), univariate_func(b)])
    plt.show()

if __name__ == '__main__':
    main()