# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import stats

def gen_data(lamda_1, lamda_2, N):
    
    X = np.zeros((N, 1))
    X = np.matrix(X)
    from_1 = 0
    for i in range(N):
        if np.random.random() < 0.6:
            X[i] = stats.poisson.rvs(lamda_1, size=1)
            from_1 += 1
        else:
            X[i] = stats.poisson.rvs(lamda_2, size=1)
    print(from_1, N-from_1)
    
    return X, from_1/N    
       
def e_step(alpha, lamb, k, N):
    for i in range(N):
        denom = 0
        numer = 0
        for j in range(k):
            denom += alpha[j]*stats.poisson.pmf(X[i], lamb[j])
        for j in range(k):
            numer = alpha[j]*stats.poisson.pmf(X[i], lamb[j])
            expec[i,j] = numer/denom
#        print(numer, denom)        
    return expec
        
    
def m_step(lamb, expec, N):
    numer_p = 0
    denom = 0
    numer_q = 0
    for j in range(N):
        numer_p += expec[j,0]*X[j]
        numer_q += expec[j,1]*X[j]
        denom += expec[j,0]
    lamb[0] = numer_p/denom
    lamb[1] = (numer_q)/(N-denom)
    alpha[0] = denom/N
    alpha[1] = 1-alpha[0]
    return alpha, lamb
    
if __name__ == '__main__':
    lamb_true = [14, 5]
    n = 20
    epsi = 1e-10
    k = 2
    N = 500
    iter_n = 100
    #parameter initialion
    alpha = np.array([[0.5], [0.5]])
    alpha = np.matrix(alpha)
    lamb = np.random.random((2,1))
    lamb = np.matrix(lamb)
    expec = np.random.random((N,2)) 
    err_1 = []
    err_2 = []
#    true_alpha = [0.3, 0.7]
    [X, ratio] = gen_data(lamb_true[0], lamb_true[1], N)
    print(ratio)
    for i in range(iter_n):
        lamb_old = copy.deepcopy(lamb)
        alpha_old = copy.deepcopy(alpha)
        expec = e_step(alpha, lamb, k, N)
        [alpha, lamb] = m_step(lamb, expec, N)
        print(i, alpha, lamb)
        diff_lamb = abs(lamb - lamb_old)
        er1 = abs(lamb[0]-5)
        er1 = np.array(er1[0])
        er1 = er1.tolist()
        er2 = abs(lamb[1]-14)
        er2 = np.array(er2[0])
        er2 = er2.tolist()
        err_1.append(er1[0])
        err_2.append(er2[0])
        diff_alpha = abs(alpha - alpha_old)
        if(diff_alpha.mean() < epsi) and (diff_lamb.mean() < epsi):
            break;
    
    x = list(range(i+1))
    plt.figure(1)
    plt.title('Result Analysis')
    plt.xlabel('iteration times')
    plt.ylabel('absolute error')
    plt.plot(x, err_1, color = 'green', label='lambda_1')
    plt.plot(x, err_2, color = 'red', label='lambda_2')
    plt.legend()
    plt.show()
    
