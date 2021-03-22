#密度聚类
from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
np.set_printoptions(suppress=True)
def euclidean_distance(x, w):  # 欧式距离公式√∑（xi﹣wi）²
    return round(np.linalg.norm(np.subtract(x, w), axis=-1),8)

def find_neighbor(j, X, eps):
    N = list()
    for i in range(X.shape[0]):
        temp = euclidean_distance(X[i],X[j])  # 计算欧式距离
        print(str(j)+"到",str(i)+"的距离",'%.8f' % temp)
        if temp <= eps:
            N.append(i)
    return set(N)


def DBSCAN(X, eps, min_Pts,count_matrix):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) + int(count_matrix[i]) >= min_Pts:    #如果权值对应位置的数据样本数量和相似权值的数量之和大于一定的数
            omega_list.append(i)  # 将样本加入核心对象集合
    omega_list = set(omega_list)  # 转化为集合便于操作
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q])+ int(count_matrix[q]) >= min_Pts:
                delta = neighbor_list[q] & gama #set的交集
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta #去除包含的核心节点
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster

def main(eps,min_Pts):
    X = np.load("som3w.npy")
    count_matrix = np.load("count_matrix.npy")
    X = X.reshape((-1, 9))  # 修改维度
    count_matrix = count_matrix.reshape(-1)
    print(X)
    # eps = 0.000018
    #min_Pts = 20
    begin = time.time()
    flag = 1
    while flag==1:
        C = DBSCAN(X, eps, min_Pts,count_matrix)
        C = np.array(C)
        if max(C)>2:
            eps = eps *2
        elif max(C)<=0:
            eps = eps/2.0
        else:
            flag = 0
    np.save("classify.npy",C)
    print("C",C.reshape([8,8]))
    end = time.time()

    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=C)
    # plt.show()

