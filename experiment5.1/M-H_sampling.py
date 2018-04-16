# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

def pi_func(x, y):
    lamb_1 = 2.7
    lamb_2 = 1.3
    pi = math.cosh(math.exp(-lamb_1*x)+(y+1)**lamb_2)**lamb_1
#    var = np.array([x, y])
#    mu = np.array([4,10])
#    sigma = np.matrix('1 0;0 1')
#    pi = stats.multivariate_normal.pdf(var ,mu, sigma)
    return pi;

if __name__ == '__main__':
    max_N = 10000
    low = 0
    high = 10
    x = [0.0]*(max_N+1)
    y = [0.0]*(max_N+1)
    x[0] = np.random.uniform(low, high)
    y[0] = np.random.uniform(low, high)
    
    for i in range(1, max_N+1):
        x_new = np.random.uniform(low, high)
        y_new = np.random.uniform(low, high)
        
        alpha = min(1, pi_func(x_new, y_new)/pi_func(x[i-1], y[i-1]))
        u = np.random.uniform(0, 1, size=1)
        if u < alpha:
            x[i] = x_new
            y[i] = y_new
        else:
            x[i] = x[i-1]
            y[i] = y[i-1]

    plt.figure(1)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)        
    num_bins = 50
    plt.sca(ax1)
    plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
    plt.title('Histogram')
    plt.sca(ax2)
    plt.hist(y, num_bins, normed=1, facecolor='red', alpha=0.5)
    plt.title('Histogram')
    plt.show()