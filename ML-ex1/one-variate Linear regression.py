# -*- coding: utf-8 -*-
# @Time : 2020/10/12 11:10
# @Author : fsf
# @File : one-variate Linear regression.py
# @Project : PycharmProjects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def computeCost(X, y, theta):  # 代价函数
    inner = np.power((np.dot(X, theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

data.insert(0, 'ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

computeCost(X, y, theta)


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

alpha = 0.01   # 设置 α 和 迭代次数
iters = 1000

final_theta, cost = gradientDescent(X, y, theta, alpha, iters)
computeCost(X,  y, final_theta)



# 以下为画图部分
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = final_theta[0, 0] + (final_theta[0, 1] * x)

fig, ax = plt.subplots(figsize=(8, 6))  # 调整画布的大小
ax.plot(x, f, 'r', label='Prediction')  # 画预测的直线 并标记左上角的标签
ax.scatter(data.Population, data.Profit, label='Traning Data')
# 画真实数据的散点图 并标记左上角的标签
ax.legend(loc=2)  # 显示标签
ax.set_xlabel('Population') #设置横轴标签
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs Population Size')  # 设置正上方标题
plt.show()  # 显示图像

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(iters), cost, color='r')
# 横坐标为0-1000的等差数组，纵坐标为cost
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs Training Epoch')
plt.show()
