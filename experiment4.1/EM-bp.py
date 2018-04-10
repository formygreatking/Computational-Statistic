# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats

n = 20
p = 0.5
k = np.arange(0,50)  
binomial = stats.binom.pmf(k,n,p)  
rv = stats.poisson(20)
y_rv = []
for i in range(50):
    y_rv.append(rv.pmf(i))
print(binomial)
print(y_rv)
  
  
plt.plot(k, binomial, y_rv, 'o-')  
plt.title('binomial:n=%i,p=%.2f'%(n,p),fontsize=15)  
plt.xlabel('number of success')  
plt.ylabel('probalility of success', fontsize=15)  
plt.grid(True)  
plt.show()  
