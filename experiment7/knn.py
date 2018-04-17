# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def knn(tests, trains, train_tra, k):
    y_pri = []
    for test in tests:
        pri = 0
        dist = []
        test = np.array(test)
        for train in trains:
            train = np.array(train)
            temp = test - train
            temp = np.multiply(temp, temp)
            dist.append(math.sqrt(temp.sum()))
        union = np.array([dist, train_tar])
        union = union.T
        union = union[union[:,0].argsort()]
#        print(union)
        for i in range(k):
            pri += union[i,1]
        y_pri.append(pri/k)
        
    return y_pri
    

if __name__=='__main__':
    boston = load_boston()
    ind = list(range(506))
    ind_train = random.sample(ind, 350)
    err_mean = []
    K = list(range(1,10))
    for k in K:
        train = []
        train_tar = []
        test = []
        test_tar = []
        for i in range(506):
            if i in ind_train:
                train.append(boston.data[i,:])
                train_tar.append(boston.target[i])
            else:
                test.append(boston.data[i,:])
                test_tar.append(boston.target[i])
        y_pri = knn(test, train, train_tar, k)
        test_tar = np.array(test_tar)
        y_pri = np.array(y_pri)
        err = abs(y_pri-test_tar)
        err_mean.append(err.sum()/len(err))
    print(err_mean)
    plt.figure(1)
    plt.plot(K, err_mean, color='red')
    plt.title('Result Analysis')
    plt.xlabel('Number of k')
    plt.ylabel('error mean')
    plt.show()