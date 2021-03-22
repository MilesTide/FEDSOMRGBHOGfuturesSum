#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: peidehao
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from time import time
def normalize(ori_data,maxV,minV,flag='01'):
    data = ori_data.copy()
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
class back_propagation:
    def __init__(self,inputdim,hide,output):
        '''

        :param inputdim:type->int,代表输入数据的维度
        :param hide: type->list,隐藏层可能有多个
        :param output: type->int,输出层个数
        '''
        self.inputdim  = inputdim
        self.hide = hide
        self.output = output
        self.model = tf.keras.Sequential()  # 初始化模型
        self.model.add(tf.keras.layers.Flatten(input_shape=(self.inputdim,)))  # 输入层
        # 批处理，归一化
        for i in range(len(self.hide)):  # 隐藏层
            self.model.add(tf.keras.layers.Dense(self.hide[i]))
            #self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))

        # 输出层
        #self.model.add(tf.keras.layers.AlphaDropout(rate=0.5))
        self.model.add(tf.keras.layers.Dense(self.output,activation='relu'))#'sigmoid'
        self.model.summary()
    def train(self,train,lable,epoch,name):
        '''

        :param train:训练集
        :param lable: 标签
        :param epoch: 整体样本的迭代次数
        :param name: 模型保存的名称
        :return: history[loss]
        '''
        self.model.compile(optimizer='adam', loss='mse',metrics=['acc'])
        print(len(train))
        history = self.model.fit(train,lable,epochs=epoch)  #,validation_data=(train[-30:-1],lable[-30:-1])
        self.model.save(str(name)+'model.h5')
        return history
def main(input,hidden,output,epoch,max_price,min_price):
    begin_time = time()
    np.set_printoptions(suppress=True)
    # 这里是数据的位置
    way = 'Traindatalocation.csv'
    alldata = pd.DataFrame(pd.read_csv(way, header=None))
    arralldata = np.array(alldata)
    #归一化所有数据
    # format_minmax = MinMaxScaler()
    ordata = arralldata[:,:-1]
    # normdata = format_minmax.fit_transform(ordata)
    normdata = normalize(ordata,max_price,min_price)


    # 这里是分类的位置
    classify = np.load("classify.npy")
    kind = set(classify)
    if -1 in kind:
        kind.remove(-1)  # 移除异常点坐标
    for one in kind:
        history = None
        data = []
        train = []  # 记录数据
        lable = []  # 记录标签
        for i in range(len(arralldata)):
            # 根据arralldata[i,-1]记录的位置，找到classify对应的分类
            try:
                print("class",classify[int(arralldata[i,-1])],i,arralldata[i,-1])
            except:
                print("异常数据",arralldata[i,-1],"类型",type(arralldata[i,-1]),i)
            # print("one",one)
            if classify[int(arralldata[i, -1])] == int(one):
                train.append(normdata[i][:-1])  #标签的个数 train.append(normdata[i][:-4])
                lable.append(normdata[i][-1])  #标签的个数 lable.append(normdata[i][-4:])
        if len(train) <= 1:
            print("元素不足1")
            if len(train) == 0:
                print(one, "类没有元素")
            continue
        #这里要修改
        bp = back_propagation(input,hidden,output)#跑出过0.5287356321839081
        train = np.array(train)
        lable = np.array(lable)
        # print(len(train))
        # print(train[0])
        # print(len(lable))
        # print(lable[0])
        history = bp.train(train, lable, epoch, str(one))
        # plt.plot(history.epoch,history.history.get('loss'),label='loss')
        # plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')
        # plt.show()
        end_time = time()
        runtime = end_time - begin_time
        print("程序开始时间", begin_time, "程序结束时间", end_time, "程序运行时间", runtime)



