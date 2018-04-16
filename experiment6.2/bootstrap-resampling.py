# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def gen_data(mu, sigma, N):
    X = np.random.normal(mu, sigma, size = N)
    return X

if __name__ == '__main__':
    mu = 2
    sigma = 4
    N = 100
    X = gen_data(mu, sigma, N)
    iter_max = 1000
    mu_est = []
    sig_est = []
    err_mu = []
    err_sig = []
    for i in range(iter_max):
        resample = []
        for j in range(N):
            ind = np.random.randint(0,N)
            resample.append(X[ind])
        resample = np.array(resample)
        mu_est.append(resample.mean())
        sig_est.append(resample.std())
    
#    x = list(range(iter_max))
#    plt.plot(x, err_mu, '-', color = 'green')
#    plt.plot(x, err_sig, '-', color='red')
    mu_est = np.array(mu_est)
    sig_est = np.array(sig_est)
    print(mu_est.mean(), sig_est.mean())