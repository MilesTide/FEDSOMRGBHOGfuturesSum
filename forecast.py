from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, pi, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply)
from numpy import sum as npsum
from numpy.linalg import norm
from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os
import math
from sklearn.preprocessing import MinMaxScaler

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import unittest
import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=True, random_generator=None):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instalce
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    iterations = arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    #return learning_rate / (1+t/(max_iter/2))原函数，此处做了更改
    decay = 0.49*(1-t/max_iter)+0.01 #值最小是0.01，初始学习率是0.5
    return decay


class MiniSom(object):
    def __init__(self,traindata, x, y, input_len,kernel,stride, Epsilon,sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

        random_seed : int, optional (default=None)
            Random seed to use.
        """
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)    #生成符合正态分布的随机数[0,1]
        self._learning_rate = learning_rate #初始学习率
        self._sigma = sigma #邻域函数初始值为1，代表100%覆盖，迭代的过程中，sigma的值会随迭代次数不断减小，sigma(t) = sigma / (1 + t/T)
        self._input_len = input_len #输入向量的维度，也是权值的维度
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1    #生成符合正态分布的权重矩阵
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True) #权值标准化，linalg.norm求范数，默认是二范数，axis=-1，按行向量求范数，keepdims保持二维特性

        self._activation_map = zeros((x, y))    #生成和网络一样大小的零矩阵，用来记录
        self._neigx = arange(x)     #得到一个列表[0,1,...,x-1]
        self._neigy = arange(y)  # 用来求邻域函数矩阵的值

        if topology not in ['hexagonal', 'rectangular']:    #判断是蜂窝网络还是矩阵网络
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology    #确定网络属性
        self._xx, self._yy = meshgrid(self._neigx, self._neigy) #生成网络坐标矩阵
        self._xx = self._xx.astype(float)   #坐标类型转换成浮点型
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        self._decay_function = decay_function   #学习率递减函数，learning_rate / (1+t/(max_iter/2))

        neig_functions = {'gaussian': self._gaussian,
                          }

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]   #确定领域函数

        distance_functions = {'euclidean': self._euclidean_distance,
                              }

        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]     #相似度计算函数，一般采用欧氏距离，这里如果采用模糊的话，要在这里更改，添加一个Fuzzy距离函数
        self.kernel = kernel  # 是一个滑动窗口的尺寸，如[3,3]
        self.stride = stride  # 滑动窗口的步长
        self.Epsilon = Epsilon  # 模糊训练截止的条件
        self.w_distance = 100   #记录权值的变化大小，用于终止训练
        self.spldata = []  # 里面是以图片为单位的data
        self.allfeature = []  # 里面是所有图片的所有特征，用于初始化权重和获取训练误差
        self.traindata = traindata
        self.processdata(self.traindata)
        self.x = x
        self.y = y

    def _activate(self, x):     #x是一个输入向量
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights)      #得到的是x到每个w的距离矩阵

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map     #得到x和每个向量的距离矩阵



    def _euclidean_distance(self, x, w):    #欧式距离公式√∑（xi﹣wi）²
        return linalg.norm(subtract(x, w), axis=-1)     #返回的是距离矩阵

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = self._input_len#self.kernel[0]*self.kernel[1]
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def _gaussian(self, c, sigma):  # 高斯核函数f(x) = e^[-(x-b)²]/2c
        """Returns a Gaussian centered in c."""
        """f(x)=ae^[-(x-b)²]/2c²
            a=1/sigma√2PI"""
        # c是坐标[x,y]
        # c是坐标（x,y）这里是二维高斯函数 f(x) = e^[-[ (x-x0)²/2sigma + (y-y0)²/2sigma ]]=e^[-[ (x-x0)²/2sigma]]  *  e^[-[ (y-y0)²/2sigma ]]
        d = 2 * pi * sigma * sigma  # 圆面积pi*r*r
        ax = exp(-power(self._xx - self._xx.T[c], 2) / d)  # power(x,y)函数，返回X的y次方，y可以是数组或者数字  e^[-[ (x-x0)²/2sigma]]
        ay = exp(-power(self._yy - self._yy.T[c], 2) / d)  # e^[-[ (y-y0)²/2sigma ]]
        # 这里是二维高斯函数 f(x) = e^[-[ (x-x0)²/2sigma + (y-y0)²/2sigma ]]=e^[-[ (x-x0)²/2sigma]]  *  e^[-[ (y-y0)²/2sigma ]]
        # print("neiborhood function",(ax * ay).T)
        return (ax * ay).T  # the external product gives a matrix 外部积得到一个矩阵,元素对应相乘
    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)
    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)       #self._activation_map = self._activation_distance(x, self._weights) #得到的是x到每个w的距离矩阵
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)        #找到最小的值，返回坐标[x,y]


    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        return self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]

    def forecast_weights_init(self,filename):
        """Initializes the weights of the SOM
        picking random samples from data."""
        #读取训练好的权值
        self._weights = np.load(filename)
    def processdata(self,data):
    #for iteration in data:
        #这里要先对数据提取特征
        features = self.FeatureExtraction(data)
        self.spldata.append(features)    # 以图片为单位，存储特征
        self.allfeature.extend(features)    #没有单位，只存储特征

    def kernel_random_weights_init(self):
        """Initializes the weights of the SOM
        picking random samples from data."""
        #从数据中随机选取数据作为权值

        self._check_input_len(self.allfeature)
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(self.allfeature))
            self._weights[it.multi_index] = self.allfeature[rand_i]
            it.iternext()

    #自己编写，用滑动窗口提取矩阵特征作为输入
    def FeatureExtraction(self,matrixs):#matrixs包含的是rgb矩阵
        features = []
        count = 0
        for matrix in matrixs:
            #matrix = np.array(matrix)
            row,loc = matrix.shape
            #print("matrix.shape",matrix.shape)
            #matrix = matrix.reshape(row,loc)
            for i in range(0,row-self.kernel[0]+1,self.stride):
                for j in range(0,loc-self.kernel[1]+1,self.stride):
                    feature = []  # 特征
                    for m in range(self.kernel[0]):
                        for n in range(self.kernel[1]):
                            pixel=matrix[i + m][j + n]
                            feature.extend([pixel])

                    #这里加入hog,计算梯度直方图
                    # 在这里设置参数
                    count = count + 1
                    image = np.array(feature).reshape(self.kernel[0],self.kernel[1])
                    size = tuple(self.kernel)
                    winSize = size
                    blockSize = size
                    blockStride = (1, 1)  # None#(1,1)
                    cellSize = size
                    nbins = self._input_len#9
                    # 定义对象hog，同时输入定义的参数，剩下的默认即可
                    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
                    winStride = None#(1,1)
                    # padding = (8,8)  , padding , winStride
                    image = np.uint8(image * 255)
                    hog_result = hog.compute(image).reshape(-1,)

                    #print("hog_reault",hog_result)
                    features.append(np.array(hog_result))
                    #print("梯度直方图",hog_result)
            # print("features",features)
            # print("features长度",len(features))
        #print("一张图片的特征数量", count)
        return features #返回数组，是一张图片的所有特征



    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours. Note that this method uses
        the euclidean distance."""
        #返回权值之间的距离矩阵
        um = zeros((self._weights.shape[0],
                    self._weights.shape[1],
                    8))  # 2 spots more for hexagonal topology

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]]*2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]]*2

        if self.topology == 'hexagonal':
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(self._weights.shape[0]):
            for y in range(self._weights.shape[1]):
                w_2 = self._weights[x, y]
                e = y % 2 == 0   # only used on hexagonal topology
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (x+i >= 0 and x+i < self._weights.shape[0] and
                            y+j >= 0 and y+j < self._weights.shape[1]):
                        w_1 = self._weights[x+i, y+j]
                        um[x, y, k] = fast_norm(w_2-w_1)

        um = um.sum(axis=2)
        return um/um.max()

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T)
        return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return norm(data-self.quantization(data), axis=1).mean()

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not ajacent conunts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        if self.topology == 'hexagonal':
            msg = 'Topographic error not implemented for hexagonal topology.'
            raise NotImplementedError(msg)
        total_neurons = prod(self._activation_map.shape)
        if total_neurons == 1:
            warn('The topographic error is not defined for a 1-by-1 map.')
            return nan

        t = 1.42
        # b2mu: best 2 matching units
        b2mu_inds = argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2my_xy = unravel_index(b2mu_inds, self._weights.shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
        dxdy = hstack([diff(b2mu_x), diff(b2mu_y)])
        distance = norm(dxdy, axis=1)
        return (distance > t).mean()

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

    def Forecast(self, random_order=False, verbose=False):
        """forecast

        Parameters
        ----------
        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.
        """

        '''  
        1、迭代w,计算每个x与w的距离关系，得到隶属度
        2、更新w，直到稳定
        3、找到每个X对应的获胜神经元，并记录
        '''
        #迭代完成后，计算位置特征映射
        features_map=[] #存储所有图片获胜神经元的坐标列表
        if random_order:
            random_generator = self._random_generator
        for pic in self.spldata:  #迭代数据
            #这里要先对数据提取特征
            feature_map = []  # 存储一张图片的所有特征对应的获胜节点，用于下一层映射
            for feature in pic:     #迭代每一张图片涵盖的特征
                win = self.winner(feature)
                num =win[0]*self.y+win[1]  #以坐标加和的方式记录特征
                feature_map.extend([int(num)])
            #print("获胜神经元的坐标映射",feature_map)
            features_map.append(feature_map)#存储这张图片的特征对应的获胜神经元
        if verbose:
            print('\n quantization error:', self.quantization_error(self.allfeature))
        return array(features_map)

class back_propagation:
    def __init__(self,modelname):

        self.model = tf.keras.models.load_model(str(modelname))  # 初始化模型
    def forcast(self,test_x,test_y):
        test_x = np.array(test_x).reshape(1,-1)
        test_y = np.array(test_y).reshape(1,-1)
        acc = self.model.evaluate(test_x,test_y)
        predict_x = self.model.predict(test_x)
        return acc,predict_x

#以下内容是自己编写

def classify(som,data,winmap):
    from numpy import sum as npsum
    default_class = npsum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def integrate(firstfeaturesmap,secondfeaturesmap):
    #先将两个特征进行归一化
    normfeature = MinMaxScaler()
    firstfeature = firstfeaturesmap.shape
    secondfeature = secondfeaturesmap.shape
    firstfeaturesmap = normfeature.fit_transform(firstfeaturesmap.reshape(-1, 1))
    firstfeaturesmap = firstfeaturesmap.reshape(firstfeature)
    secondfeaturesmap = normfeature.fit_transform(secondfeaturesmap.reshape(-1, 1))
    secondfeaturesmap = secondfeaturesmap.reshape(secondfeature)
    #整合特征
    featuresmap = []
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    x = list(map(lambda num: num * num, x))
    lg = len(firstfeaturesmap[0]) + len(secondfeaturesmap[0])
    # print(len(firstfeaturesmap[0]))
    # print(len(secondfeaturesmap[0]))
    # print("属性值",lg)
    if lg>max(x):
        print("属性过多")
    x = list(map(lambda num: num-lg , x))
    temp = []
    for i in range(len(x)):
        if x[i] >= 0:
            temp.extend([x[i]])
    print("min(temp)",min(temp))
    zo = np.zeros([min(temp)])
    zo = list(zo)
    for i in range(len(firstfeaturesmap)):
        tempmap = []
        tempmap.extend(firstfeaturesmap[i])
        tempmap.extend(secondfeaturesmap[i])
        tempmap.extend(zo)
        l = int(pow(len(tempmap), 0.5))
        tempmap = np.array(tempmap).reshape(l,l)
        featuresmap.append(tempmap)   #som中多了一个遍历，rgb图像有三个矩阵，所以加了一个[]
    # if not isinstance(pow(len(featuresmap[0]),0.5),int):  #这里输出的是浮点型
    #     print("列表填充有误")
    # print("featuresmap[0][0]",featuresmap[0][0][0])
    # print("int(pow(len(featuresmap[0][0]),0.5))",l)
    return featuresmap,l



def vec2matrix2vec(vector):
    '''将向量的变化趋势映射到矩阵中
    先找到向量元素对应的每个向量，将这些行向量组成矩阵后经过转置就是变化图'''
    vector = pre.minmax_scale(vector)
    matrix=[]
    for i in range(len(vector)):
        temp=[]
        location=vector[i]//0.1
        #print(location)
        for j in range(len(vector)):
            if j==location:
                temp.append(1.0)        #1.0代表占位，也可以是vector[i]
            else:
                temp.append(0)
        matrix.append(temp)   #转置之后才是上涨形态
    matrix = np.array(matrix).T
    vec = matrix.reshape(-1)
    #print(list(vec))
    #return np.mat(matrix)
    return vec

def statistics(origin, predicted):
    #compute RMSE
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(origin, predicted)#均方误差，
    rmse = np.sqrt(mse)
    mape = (sum(abs(origin - predicted))) / len(origin)
    return mse, rmse, mape

# normalize data set into [0, 1] or [-1, 1]
def normalize(ori_data,maxV,minV, flag='01'):
    data = ori_data.copy()
    # minV = np.min(data)
    # maxV = np.max(data)
    if np.abs(maxV - minV) > 0.00001:
        data = 2 * (data - minV) / (maxV - minV) - 1
    return data
# re-normalize data set from [0, 1] or [-1, 1] into its true dimension
def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if np.abs(maxV - minV) > 0.00001:
        if flag == '01':  # normalize to [0, 1]
            data = data * (maxV - minV) + minV
        else:
            data = (data + 1) * (maxV - minV) / 2 + minV
    return data

#处理数据
def processdata(way,max_price,min_price,max_volume,min_volume):
    data = pd.DataFrame(pd.read_csv(way, header=0))
    norm = MinMaxScaler()  # 为了归一化数据
    arrdata = np.array(data)
    # datetemp = list(arrdata[:, 0])
    # date = list(set(datetemp))
    # date.sort(key=datetemp.index)  # 集合数据按照原来顺序进行排序
    # 将所有数据按照30分钟进行切分，一天四个小时，240分钟
    list_data = []
    for i in range(0, len(data), 30):
        list_data.append(arrdata[i:i + 30, :])  # 数据的日期、时间、开、高、低、收、成交量、成交额
    # print(len(list_data))
    results = []
    # 切分的数据按照十个30分钟一组垂直组合
    for j in range(len(list_data) - 11):
        r = np.zeros([100, 10])  # 下跌，一行代表0.01
        g = np.zeros([100, 10])  # 上涨
        b = np.zeros([100, 10])  # 成交量
        # if j >= 10:
        #     break
        temp = np.concatenate((list_data[j], list_data[j + 1], list_data[j + 2], list_data[j + 3], list_data[j + 4],
                               list_data[j + 5], list_data[j + 6], list_data[j + 7], list_data[j + 8],
                               list_data[j + 9]), axis=0)  # 共10天，垂直组合
        # 组合后，归一化价格，绘制图像
        price = np.array(temp[:, 3:7], dtype='float')
        volume = np.array(temp[:, 7], dtype='int')
        # 这里放的十天的数据
        # norm_price = norm.fit_transform(price.reshape(-1, 1))  # 归一化价格
        # norm_volume = norm.fit_transform(volume.reshape(-1, 1))  # 归一化成交量
        # norm_price = norm_price.reshape(-1, 4)
        norm_price= normalize(price.reshape(-1, 1),max_price,min_price)  # 归一化价格
        norm_volume = normalize(volume.reshape(-1, 1),max_volume,min_volume)  # 归一化成交量
        norm_price = norm_price.reshape(-1, 4)
        # 切分归一化后的价格
        days_normprice = []
        days_normvolume = []
        days_volume = []
        for k in range(0, len(norm_price), 30):
            days_normprice.append(norm_price[k:k + 30, :])
            days_normvolume.append(norm_volume[k:k + 30, :])
            days_volume.append(volume[k:k + 30])
        # 按天绘制图像
        for l in range(len(days_normprice)):
            norm_open = float(days_normprice[l][0, 0])  # 开盘价
            norm_close = float(days_normprice[l][-1, 3])  # 收盘价
            norm_high = float(max(days_normprice[l][:, 1]))  # 最高价
            norm_low = float(min(days_normprice[l][:, 2]))  # 最低价
            print('open', norm_open, 'high', norm_high, 'low', norm_low, 'close', norm_close)
            loc_open = round(norm_open / 0.01)  # 开盘价的位置   ,四舍五入
            loc_high = round(norm_high / 0.01)  # 开盘价的位置
            loc_low = round(norm_low / 0.01)  # 开盘价的位置
            loc_close = round(norm_close / 0.01)  # 开盘价的位置
            body = loc_close - loc_open
            upper_shadow = loc_high - max([loc_open, loc_close])
            lower_shadow = min(loc_open, loc_close) - loc_low
            if loc_close == 100:
                loc_close = 99
            if loc_high == 100:
                loc_high = 99
            if loc_open == 100:
                loc_open = 99
            if loc_low == 100:
                loc_low = 99
            print('body', body, "upper_shadow", upper_shadow, "lower_shadow", lower_shadow)
            if body >= 0:  # 上涨，g矩阵改变
                g[loc_close:loc_high, l] = 0.5  # 上影线
                g[loc_open:loc_close, l] = 1  # 实体
                g[loc_low:loc_open, l] = 0.5  # 下影线
                if loc_open == loc_close:
                    g[loc_open, l] = 1  # 实体
                # print("g",g)
            else:  # 下跌,body<0
                r[loc_open:loc_high, l] = 0.5  # 上影线
                r[loc_close:loc_open, l] = 1  # 实体
                r[loc_low:loc_close, l] = 0.5  # 下影线
                if loc_open == loc_close:
                    r[loc_open, l] = 1  # 实体
                # print("r", r)
            # 成交量改变b矩阵
            for m in range(len(days_normprice[l])):
                begin_loc = int(round(days_normprice[l][m, 2] / 0.01))
                end_loc = int(round(days_normprice[l][m, 1] / 0.01))
                if begin_loc == 100:
                    begin_loc=99
                if end_loc == 100:
                    end_loc = 99
                # now_volume = days_normvolume[l][m]
                now_volume = days_volume[l][m]
                length = end_loc - begin_loc
                if length ==0:
                    b[begin_loc, l] = round(now_volume / 1) + b[begin_loc,l]
                else:
                    b[begin_loc:end_loc, l] = round(now_volume / length) + b[begin_loc:end_loc, l]  # 这一价格区间的成交量被均匀分布在这一区间
            # print('b',b)
            #print("最大成交量", max(max(b.reshape(1, -1))))
        b = norm.fit_transform(b.reshape(-1, 1))  # 归一化成交量
        b = b.reshape(100, 10)
        results.append([r,g,b])
    return results
def get_bpdata(path,loc,classies,max_price,min_price):
    # norm = MinMaxScaler()
    data = pd.DataFrame(pd.read_csv(path, header=None))
    arrdata = np.array(data)
    print(arrdata[loc])
    # maxV = max(np.array(arrdata[:,2:6]).reshape(-1,1))
    # minV = min(np.array(arrdata[:,2:6]).reshape(-1,1))
    # normarrdata = norm.fit_transform(arrdata[:,2:6])
    normarrdata = normalize(arrdata[:,2:6],max_price, min_price)
    adata = []
    alable = []
    normadata = []
    normlable = []
    if len(classies) == 1:  # len(arrdata)-11:
        for z in range(2, 6):
            for y in range(10):
                adata.extend([arrdata[loc + y][z]])
                normadata.extend([normarrdata[loc + y][z-2]])
            alable.extend([arrdata[loc + 11][z]])
            normlable.extend([normarrdata[loc + 11][z-2]])
    else:
        print("classiss长度不为1，结果为",classies)
    return adata,normadata, alable, normlable
def main(way1,way2,max_price,min_price,max_volume,min_volume):
    #EDSOM
    np.set_printoptions(suppress=True)
    count = 0
    way1 = "F:\论文实验\FEDSOMRGBHOGfutures9\mintesttest"
    arralldata = processdata(way1,max_price,min_price,max_volume,min_volume)  # 已经归一化后的rgb矩阵features
    #arralldata = np.array(features)
    data = []
    x_predict = []
    y = []
    norm_y = []
    unnorm = []
    sumvariance = 0
    summape = 0
    classify = np.load("classify.npy")
    for i in range(len(arralldata)):
        # 这里是rgb矩阵
        vec = arralldata[i] #里面有三个矩阵[r,g,b]
        Epsilon = 0.05  #用不到了


        som1 = MiniSom(vec,15, 15, 9,[10,10],1, Epsilon,sigma=3, learning_rate=0.5)
        som1.forecast_weights_init('som1w.npy')
        firstfeaturesmap = som1.Forecast(verbose=False)      #返回特征

        som2 = MiniSom(vec,15, 15, 9, [8, 8], 1, Epsilon,sigma=3, learning_rate=0.5)
        som2.forecast_weights_init('som2w.npy')
        secondfeaturesmap = som2.Forecast(verbose=False)        #返回特征

        # print("firstFeatu",firstfeaturesmap)
        # print("secondfeatu",secondfeaturesmap)
        featuresmap,l = integrate(firstfeaturesmap,secondfeaturesmap)     #整合两个特征
        featuresmap = array(featuresmap) #输入为数组形式,整合前两层训练结果
        #print("featuresmap",featuresmap)

        som3 = MiniSom(featuresmap,8, 8, 9,[l,l],1, Epsilon,sigma=3, learning_rate=0.5)
        som3.forecast_weights_init('som3w.npy')
        classies = som3.Forecast(verbose=False)#输入的是映射，没有经过归一化
        #classies中涵盖了分类信息，要取出和分类信息作比对，确定进入哪一个预测基
        loc = int(classies[0][0])
        loc = classify[loc]

        #处理数据放入BP网络中，之前的数据是分钟数据，现在是30分钟数据
        way2 = "F:\论文实验\FEDSOMRGBHOGfutures9\daytesttest"
        test_x,norm_test_x, test_y, norm_test_y = get_bpdata(way2,i,classies,max_price,min_price)
        y.append(test_y[-1])
        norm_y.append(norm_test_y[-1])
        if loc == -1:
            count = count + 1
            print("该点是异常点")
            predict = np.array(norm_test_y[-1])
            print("predict",predict)
            x_predict.append([predict])
            unnorm.append(predict)
            continue
        #print("类别",str(loc))
        modelname = str(loc)+'model.h5'
        try:
            bp = back_propagation(modelname)
            acc,predict = bp.forcast(norm_test_x, norm_test_y[-1])  #传入norm_test_y是为了计算精度
        except IOError as e:
            print('expect',e)
            print("没有该类")
            predict = np.array(norm_test_y[-1])
            print("predict", predict)
            x_predict.append([predict])
            unnorm.append(predict)
            continue
        print("model",modelname)
        print("input",test_x)
        print("x_predict",predict)
        print("acc",acc)
        print("norm_y", norm_test_y[-1])
        print("y",y[i])
        x_predict.append(predict)

        # print("accuracy:{:5.2f}%".format(100*acc))
        # print("acc",acc)
    norm_lable_data = np.array(norm_y, dtype='float').reshape(-1,1)
    x_predict = np.array(x_predict, dtype='float').reshape(1,-1)
    x_predict = x_predict.reshape(-1,1)
    plt.plot(norm_lable_data, '*', x_predict, 'r', )  # x_predict, 'r',
    plt.show()
    print("异常列表",unnorm)

    mse, rmse, mape = statistics(norm_lable_data, x_predict)
    print("mse1", mse)
    print("rmse1", rmse)
    print("mape2", mape)

    x_predict = re_normalize(x_predict,max_price,min_price)
    plt.plot( y, '*',x_predict, 'r',)#x_predict, 'r',
    plt.show()
    y = np.array(y,dtype='float').reshape(-1,1)
    mse, rmse, mape = statistics(y,x_predict)
    # 计算准确度
    accuracy_right = 0
    accuracy_wrong = 0
    hold = 0
    up = 0.0005
    down = -0.0005
    earn = 0
    fee = 0.0001 * 2
    for j in range(1, len(y)):
        if (x_predict[j] - x_predict[j - 1]) / x_predict[j - 1] < down and (y[j] - y[j - 1]) / y[j - 1] < down:  # 同时小于，抛出
            earn = earn - (y[j] - y[j - 1]) / y[j - 1] - fee # 值为负
            accuracy_right = accuracy_right + 1
        elif (x_predict[j] - x_predict[j - 1]) / x_predict[j - 1] >= up and (y[j] - y[j - 1]) / y[j - 1] >= up:  # 同时大于买入
            earn = earn + (y[j] - y[j - 1]) / y[j - 1]  - fee# 值为正
            accuracy_right = accuracy_right + 1
        elif (x_predict[j] - x_predict[j - 1]) / x_predict[j - 1] > up and (y[j] - y[j - 1]) / y[j - 1] < up:  #
            earn = earn + (y[j] - y[j - 1]) / y[j - 1] - fee # 值为负
            accuracy_wrong = accuracy_wrong + 1
        elif (x_predict[j] - x_predict[j - 1]) / x_predict[j - 1] < down and (y[j] - y[j - 1]) / y[j - 1] > down:
            earn = earn - (y[j] - y[j - 1]) / y[j - 1]  - fee # 值为正
            accuracy_wrong = accuracy_wrong + 1
        else:
            hold = hold + 1


    accuracy = accuracy_right / (accuracy_wrong + accuracy_right)

    print("mse2",mse)
    print("rmse2", rmse)
    print("mape2", mape)
    print("len_xp",len(x_predict))
    print("accuracy",accuracy)
    print("accuracy_right",accuracy_right,"accuracy_wrong",accuracy_wrong,"hold",hold)
    print("earn", earn)


