# -*- coding: utf-8 -*-
# @Time : 2020/10/12 11:25
# @Author : fsf
# @File : multi-variate Linear regression.py
# @Project : PycharmProjects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'ex1data2.txt'
data2 = pd.read_csv(path,header=None,names=['size','bedrooms','Price'])

data2 = (data2 - data2.mean()) / data2.std()

def computeCost(X, y, theta):  # 代价函数
    inner = np.power((np.dot(X, theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data2.insert(0, 'ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

def gradientDescent(X, y, theta, alpha, iters): #梯度下降函数
    tmp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)
    m = X.shape[0]
    for i in range(iters):
        tmp = theta - (alpha / m)*(X * theta.T - y).T * X
        #X（97，2），theta.T（2，1），y（97，1）
        theta = tmp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

iters2 = 1000
alpha2 = 0.1
final_thata2, cost2 = gradientDescent(X2, y2, theta2, alpha2, iters2)
computeCost(X2, y2, final_thata2)  # 0.13068648053904197

# matrix([[-9.62057904e-17,   8.84765988e-01,  -5.31788197e-02]])

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters2), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


