# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == "__main__":  
    mu = np.array([1, 10, 20])  # 一维数组  
    sigma = np.matrix([[20, 10, 10], [10, 25, 1], [10, 1, 50]])   # 二维数组 协方差矩阵，该矩阵必须是对称而且是半正定的  
    data = np.random.multivariate_normal(mu, sigma, 1000)   # 产生样本点，第一个参数表示样本每维的均值，第二维表示维度之间的协方差，第三个表示产生样本的个数  
    values = data.T  
    kde = stats.gaussian_kde(values)  # 构造多变量核密度评估函数  
    density = kde(values)   # 给定一个样本点，计算该样本点的密度  

    # 可视化展现  
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))  
    x, y, z = values  
    ax.scatter(x, y, z, c=density)  
    plt.show()  

    print (data)