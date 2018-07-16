# -*- coding: utf-8 -*-

"""
实现一个神经元的分类算法
https://www.imooc.com/learn/813
blog
http://blog.yunplus.io/%E4%BD%BF%E7%94%A8Python%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%9A%84%E5%8D%95%E4%B8%80%E7%A5%9E%E7%BB%8F%E5%85%83/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json, os, time

#分类器代码
class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        # 是否训练过的标识
        self._fited = self.load_model()
        """
        初始化向量为0
        加一是因为步调函数阈值
        """
        if not self._fited:
            self.w_ = np.zeros(1 + X.shape[1])
        
    def fit(self, X, y):
        """
        输入训练数据，培训神经元
        :param X: 输入样本向量
        :param y: 对应样本分类
         
        X:shape[n_samples, n_features]
        X:[[1,2,3],[4,5,6]]
        n_samples :2
        n_features:3
         
        y:[1,-1]
        """
        self.errors_ = []
 
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
 
                self.w_[1:] += update * xi
                self.w_[0] += update
 
                errors += int(update!= 0)
                self.errors_.append(errors)

        self.save_model()
 
    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])
        
    """
    将输入信息进行分类
    """
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def get_model(self):
        return self.w_

    def save_model(self):
        with open('a.model', 'w') as f:
            f.write(json.dumps(self.w_.tolist()))

    def is_fited(self):
        return self._fited

    def load_model(self):
        if os.path.exists('a.model'):
            with open('a.model', 'r', encoding="UTF-8") as f:
                data = json.load(f)
                self.w_ = np.array(data)
            return True
        else:
            return False
 

 
file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
 
df = pd.read_csv(file,header=None)
# 输出数据集的前10条数据，形式如下
"""
[[5.1 3.5 1.4 0.2 'Iris-setosa']
 [4.9 3.0 1.4 0.2 'Iris-setosa']
 [4.7 3.2 1.3 0.2 'Iris-setosa']
 [4.6 3.1 1.5 0.2 'Iris-setosa']
 [5.0 3.6 1.4 0.2 'Iris-setosa']
 [5.4 3.9 1.7 0.4 'Iris-setosa']
 [4.6 3.4 1.4 0.3 'Iris-setosa']
 [5.0 3.4 1.5 0.2 'Iris-setosa']
 [4.4 2.9 1.4 0.2 'Iris-setosa']
 [4.9 3.1 1.5 0.1 'Iris-setosa']
 [5.4 3.7 1.5 0.2 'Iris-setosa']]
"""
# print(df.loc[0:10,[0,1,2,3,4]].values)

# 使用前100行的作为训练样本
# type(y) is <class 'numpy.ndarray'>
y = df.loc[0:100,4].values
# 将矩阵中的 string 转换成 1/-1，便于运算
y = np.where(y=='Iris-setosa', -1, 1)

#根据整数位置选取单列或单行数据
# x 是一个 m*n 的矩阵，其中 m = 100, n = 2, 
# X11 对应的是数据集中第一行第一列，X12 是数据集中第一行第3列
X = df.loc[0:100,[0,2]].values

"""
# 绘制数据点
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label="setosa") 
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label="versicolor")

# 设置散点图的坐标和图示
plt.xlabel('huabanchangdu')
plt.ylabel('huajingchangdu')
plt.legend(loc='upper left')
"""

# 定义一个神经元，并进行训练
ppn = Perceptron(eta=0.1, n_iter=100)

# 如果存在已训练过的模型，则无需重复训练
if not ppn.is_fited():
    ppn.fit(X,y)
# print(ppn.get_model())

# 将训练好的简单神经元作为分类器进行输入的分类
def plot_decision_region(X, y, classifier, resolution = 0.02):
    markers=('s','x','o','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
 
    # 获取矩阵的两列的最小值和最大值
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    # 生成一个二维的矩阵
    xx1, xx2 = np.meshgrid(
                    np.arange(x1_min, x1_max, resolution),
                    np.arange(x2_min, x2_max, resolution)
               )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # print (xx1.ravel())
    # print (xx2.ravel())
    # print (Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl,1], alpha=0.8, 
                    c = cmap(idx), marker = markers[idx], label = cl)

    plt.xlabel('huajingchang')
    plt.ylabel('huabanchang')
    plt.legend(loc = 'upper left')
    plt.show()

# X = df.iloc[0:100,[0,2]].values
plot_decision_region(X, y, ppn, resolution = 0.03)
