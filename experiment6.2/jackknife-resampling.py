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
        data = gen_data(mu, sigma, n)
        mu_bar = []
        sig_bar = []
        for i in range(n):
            data_i = np.delete(data, i, axis=0)
            mu_bar.append(data_i.mean())
            sig_bar.append(data_i.std())
        mu_bar = np.array(mu_bar)
        sig_bar = np.array(sig_bar)
        mu_est = mu_bar.mean()
        sig_est = sig_bar.mean()
        err_mu.append(abs(mu_est - mu))
        err_sig.append(abs(sig_est - sigma))
        print(abs(mu_est - mu), abs(sig_est - sigma))
        
    plt.figure(1)
    plt.title('Result Analysis')
    plt.xlabel('Number of examples')
    plt.ylabel('absolute error')
    plt.plot(N, err_mu, color='green', label='mean')
    plt.plot(N, err_sig, color='red', label='sigma')
    plt.legend()
    plt.show()
