# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
import math

def gen_data(mu, sigma, N):
    
    X = np.zeros((N, 1))
    X = np.matrix(X)  
    from_1 = 0
    from_2 = 0
    for i in range(N):
        if np.random.random() < 0.6:
            X[i] = np.random.normal(mu[0], sigma[0], 1)
            from_1 += 1
        else:
            X[i] = np.random.normal(mu[1], sigma[1], 1) 
            from_2 += 1
    print(from_1, from_2)
    alpha = [0.5, 0.5]
    mu = np.random.random((2,1))
    mu = np.matrix(mu)
    sigma = np.random.random((2,1))
    sigma = np.matrix(sigma)
    expec = np.random.random((N,2)) 
    return X, mu, sigma, alpha, expec, from_1/N    
       
def e_step(alpha, mu, sigma, k, N):
    for i in range(N):
        denom = 0
        numer = 0
        for j in range(k):
            denom += (alpha[j]/sigma[j])*math.exp(-((X[i]-mu[j])/sigma[j])**2/2)
        for j in range(k):
            numer = (alpha[j]/sigma[j])*math.exp(-((X[i]-mu[j])/sigma[j])**2/2)
            expec[i,j] = numer/denom
#        print(numer, denom)        
    return expec
        
    
def m_step(mu, sigma, expec, k, N):
    for i in range(k):
        numer_mu = 0
        numer_sig = 0
        denom = 0
        for j in range(N):
            numer_mu += expec[j,i]*X[j]
            numer_sig += expec[j,i]*((X[j]-mu[i])**2)
            denom += expec[j,i]
        mu[i] = numer_mu/denom
        sigma[i] = math.sqrt(numer_sig/denom)
        alpha[i] = denom/N
    return alpha, mu, sigma
    
if __name__ == '__main__':
    mu = [10,1]
    sigma = [2,2]
    epsi = 1e-10
    k = 2
    N = 500
    iter_n = 200
#    true_alpha = [0.3, 0.7]
    [X, mu, sigma, alpha, expec, ratio] = gen_data(mu, sigma, N)
    print(ratio)
    for i in range(iter_n):
        mu_old = copy.deepcopy(mu)
        sigma_old = copy.deepcopy(sigma)
        alpha_old = copy.deepcopy(alpha)
        expec = e_step(alpha, mu, sigma, k, N)
        [alpha, mu, sigma] = m_step(mu, sigma, expec, k, N)
        print(i, alpha, mu, sigma)
        diff_mu = abs(mu - mu_old)
        diff_sigma = abs(sigma - sigma_old)
        if(diff_mu.mean() < epsi) and (diff_sigma.mean() < epsi):
            break;
    
    