# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def gen_data(mu, sigma, N):
    X = np.random.normal(mu, sigma, size = N)
    return X

if __name__ == '__main__':
    mu = 2
    sigma = 4
    N = 1000
    data = gen_data(mu, sigma, N)
    mu_bar = []
    sig_bar = []
    for i in range(N):
        data_i = np.delete(data, i, axis=0)
        mu_bar.append(data_i.mean())
        sig_bar.append(data_i.std())
    mu_bar = np.array(mu_bar)
    sig_bar = np.array(sig_bar)
    mu_est = mu_bar.mean()
    sig_est = sig_bar.mean()
    err_mu = abs(mu_est - mu)
    err_sig = abs(sig_est - sigma)
    print(err_mu, err_sig)