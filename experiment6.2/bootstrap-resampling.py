# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def gen_data(mu, sigma, N):
    X = np.random.normal(mu, sigma, size = N)
    return X

if __name__ == '__main__':
    mu = 2
    sigma = 4
    N = list(range(100,1100,100))
    err_mu = []
    err_sig = []
    for n in N:
        X = gen_data(mu, sigma, n)
        iter_max = 1000
        mu_est = []
        sig_est = []
        for i in range(iter_max):
            resample = []
            for j in range(n):
                ind = np.random.randint(0,n)
                resample.append(X[ind])
            resample = np.array(resample)
        mu_est.append(resample.mean())
        sig_est.append(resample.std())
        mu_est = np.array(mu_est)
        sig_est = np.array(sig_est)
        err_mu.append(abs(mu_est.mean() - mu))
        err_sig.append(abs(sig_est.mean() - sigma))
        print(abs(mu_est.mean() - mu), abs(sig_est.mean()- sigma))
        
    plt.figure(1)
    plt.title('Result Analysis')
    plt.xlabel('Number of examples')
    plt.ylabel('absolute error')
    plt.plot(N, err_mu, color='green', label='mean')
    plt.plot(N, err_sig, color='red', label='sigma')
    plt.legend()
    plt.show()
