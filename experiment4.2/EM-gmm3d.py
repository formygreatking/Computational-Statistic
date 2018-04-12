# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
from scipy import stats

def gen_data(mu_1, sigma_1, mu_2, sigma_2, N): 
    X = np.zeros((N, 2))
    X = np.matrix(X)  
    from_1 = 0
    for i in range(N):
        if np.random.random() < 0.4:
            X[i:] = np.random.multivariate_normal(mu_1, sigma_1, 1)
            from_1 += 1
        else:
            X[i:] = np.random.multivariate_normal(mu_2, sigma_2, 1) 
    print(from_1)
    return X, from_1/N    
       
def e_step(mu, cov_1, cov_2, alpha, X, N):
    ld_0 = stats.multivariate_normal.pdf(X, mu[0:], cov_1)
    ld_1 = stats.multivariate_normal.pdf(X, mu[1:], cov_2)
    denom = np.hstack((ld_0, ld_1))
    denom = np.dot(alpha.T, denom.T) 
    numer = np.repeat(alpha.T, N, axis=0)
    numer = np.multiply(numer, X)
    excep = numer/denom
#        print(numer, denom)        
    return excep
        
    
def m_step(mu, excep, N):
    denom = excep.sum(axis=0)
    numer_mu = np.dot(excep.T, X)
    alpha = denom/N
    mu[0,:] = numer_mu[0,:]/denom[0]
    mu[1,:] = numer_mu[1,:]/denom[1]
    X_temp = X - np.repeat(mu[0,:], N, axis=0)
    left = np.multiply(excep[:,0], X_temp[:,0])
    right = np.multiply(excep[:,0], X_temp[:,1])
    numer_cov = np.hstack((left, right))
    numer_cov = np.dot(X_temp.T, numer_cov)
    sigma_1 = numer_cov/denom[0]
    X_temp = X - np.repeat(mu[1,:], N, axis=0)
    left = np.multiply(excep[:,1], X_temp[:,0])
    right = np.multiply(excep[:,1], X_temp[:,1])
    numer_cov = np.hstack((left, right))
    numer_cov = np.dot(X_temp.T, numer_cov)
    sigma_1 = numer_cov/denom[1]
 
    return alpha, mu, sigma_1, sigma_2
    
if __name__ == '__main__':
    mu1_true = [10, 15]
    sigma_1 = np.matrix('10 4; 4 2')
    mu2_true = [40, 3]
    sigma_2 = np.matrix('2 3; 3 7')
    epsi = 1e-10
    k = 2
    N = 1000
    iter_n = 1
    #parameter initialion & data generation
    [X, ratio] = gen_data(mu1_true, sigma_1, mu2_true, sigma_2, N)
    alpha = np.matrix('0.5; 0.5')
    mu = np.array([[X[:,0].mean()+1],[X[:,1].mean()+1],[X[:,0].mean()-1],[X[:,1].mean()-1]])
    mu = np.matrix(mu)
    sigma_1 = np.random.random((2,2))
    sigma_1 = np.matrix(sigma_1)
    sigma_2 = np.random.random((2,2))
    sigma_2 = np.matrix(sigma_2)
    expec = np.random.random((N,2)) 
    print(ratio)
#    #running EM
    for i in range(iter_n):
#        mu_old = copy.deepcopy(mu)
#        sigma_old = copy.deepcopy(sigma)
#        alpha_old = copy.deepcopy(alpha)
        expec = e_step(mu, sigma_1, sigma_2, alpha, X, N)
        [alpha, mu, sigma_1, sigma_2] = m_step(mu, expec, N)
        print(i, alpha, mu, sigma_1, sigma_2)
#        diff_mu = abs(mu - mu_old)
#        diff_sigma = abs(sigma - sigma_old)
#        if(diff_mu.mean() < epsi) and (diff_sigma.mean() < epsi):
#            break;
    
    