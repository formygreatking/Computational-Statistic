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
        if np.random.random() < 0.3:
            X[i] = np.random.normal(mu[0], sigma[0], 1)
            from_1 += 1
        else:
            X[i] = np.random.normal(mu[1], sigma[1], 1) 
            from_2 += 1
    print(from_1, from_2)

    return X, from_1/N    
       
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
#            numer_sig += expec[j,i]*((X[j]-mu[i])**2)
            denom += expec[j,i]
        mu[i] = numer_mu/denom
        for j in range(N):
            numer_sig += expec[j,i]*((X[j]-mu[i])**2)
        sigma[i] = math.sqrt(numer_sig/denom)
        alpha[i] = denom/N
    return alpha, mu, sigma
    
if __name__ == '__main__':
    mu_true = [10,15]
    sigma_true = [5,2]
    epsi = 1e-5
    k = 2
    N = 500
    iter_n = 500
    #parameter initialion & data generation
    [X, ratio] = gen_data(mu_true, sigma_true, N)
    alpha = [0.5, 0.5]
    mu = np.array([[X.mean()+1],[X.mean()-1]])
    mu = np.matrix(mu)
    sigma = np.array([[X.std()+1],[X.std()-1]])
    sigma = np.matrix(sigma)
    expec = np.random.random((N,2)) 
    print(ratio)
    err_mu1 = []
    err_mu2 = []
    err_sig1 = []
    err_sig2 = []
    for i in range(iter_n):
        mu_old = copy.deepcopy(mu)
        sigma_old = copy.deepcopy(sigma)
        alpha_old = copy.deepcopy(alpha)
        expec = e_step(alpha, mu, sigma, k, N)
        [alpha, mu, sigma] = m_step(mu, sigma, expec, k, N)
        print(i, alpha, mu, sigma)
        diff_mu = abs(mu - mu_old)
        diff_sigma = abs(sigma - sigma_old)
        ermu1 = np.array(abs(mu[0] - mu_true[0])).tolist()
        ermu2 = np.array(abs(mu[1] - mu_true[1])).tolist()
        ersig1 = np.array(abs(sigma[0] - sigma_true[0])).tolist()
        ersig2 = np.array(abs(sigma[1] - sigma_true[1])).tolist()
        err_mu1.append(ermu1[0])
        err_mu2.append(ermu2[0])
        err_sig1.append(ersig1[0])
        err_sig2.append(ersig2[0])
        if(diff_mu.mean() < epsi) and (diff_sigma.mean() < epsi):
            break;
    
    x = list(range(i+1))
    plt.figure(1)
    plt.title('Result Analysis')
    plt.xlabel('iteration times')
    plt.ylabel('absolute error')
    plt.plot(x, err_mu1, color = 'green', label='mu_1')
    plt.plot(x, err_mu2, color = 'red', label='mu_2')
    plt.plot(x, err_sig1, color = 'blue', label='sigma_1')
    plt.plot(x, err_sig2, color = 'black', label='sigma_2')
    plt.legend()
    plt.show()