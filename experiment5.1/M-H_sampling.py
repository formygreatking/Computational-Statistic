# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

def pi_func(x, y):
#    a = 2.3
#    b = 1.2
    lamb_1 = 2
    lamb_2 = 0.3
#    pi = math.cos(math.exp(-lamb_1*x)/(y+1)**lamb_2 + stats.beta.pdf(z,a,b))
    pi = math.cosh(math.exp(-lamb_1*x)**(y+1)**lamb_2)
    return pi;

if __name__ == '__main__':
    max_N = 5000
    low = 0
    high = 10
    x = [0.0]*(max_N+1)
    y = [0.0]*(max_N+1)
#    z = [0.0]*(max_N+1)
    x[0] = np.random.uniform(low, high, size=1)
    y[0] = np.random.uniform(low, high, size=1)
#    z[0] = np.random.uniform(low, high, size=1)
    
    for i in range(1, max_N+1):
        x_new = np.random.uniform(low, high, size=1)
        y_new = np.random.uniform(low, high, size=1)
#        z_new = np.random.uniform(low, high, size=1)
        
        alpha = min(1, pi_func(x_new, y_new)/pi_func(x[i-1], y[i-1]))
        u = np.random.uniform(0, 1, size=1)
        if u < alpha:
            x[i] = x_new
            y[i] = y_new
#            z[i] = z_new
        else:
            x[i] = x[i-1]
            y[i] = y[i-1]
#            z[i] = z[i-1]

    fig1 = plt.figure()
    ax = Axes3D(fig1)
    X = np.arange(0, 10, 0.1)
    Y = np.arange(0, 10, 0.1)
    Z = []
    for i in range(100):
        Z.append(pi_func(X[i], Y[i]))
    X,Y = np.meshgrid(X,Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()