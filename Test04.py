#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: peidehao
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt
import cv2
def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    minV = np.min(data)
    maxV = np.max(data)
    if np.abs(maxV - minV) > 0.00001:
        data = 2 * (data - minV) / (maxV - minV) - 1
    return data, maxV, minV
# re-normalize data set from [0, 1] or [-1, 1] into its true dimension
def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if np.abs(maxV - minV) > 0.00001:
        if flag == '01':  # normalize to [0, 1]
            data = data * (maxV - minV) + minV
        else:
            data = (data + 1) * (maxV - minV) / 2 + minV
    return data
def processdata(way,way3):
    data = pd.DataFrame(pd.read_csv(way, header=0))
    sum_data = pd.DataFrame(pd.read_csv(way3, header=0))
    norm = MinMaxScaler()  # 为了归一化数据
    arrdata = np.array(data)
    sum_arrdata = np.array(sum_data)
    # 将所有数据按照一30分钟进行切分
    list_data = []
    sum_list_data = []
    for i in range(0, len(data), 30):
        list_data.append(arrdata[i:i + 30, :])  #数据的 市场代码,合约代码,时间,开,高,低,收,成交量,成交额,持仓量
    for i in range(0, len(sum_data), 30):
        sum_list_data.append(sum_arrdata[i:i + 30, :])  # 数据的 市场代码,合约代码,时间,开,高,低,收,成交量,成交额,持仓量
    print(len(list_data))
    print(len(sum_list_data))
    results = []
    # 切分的数据按照十个30分钟一组垂直组合
    for j in range(len(sum_list_data) - 11):
        r = np.zeros([100, 10])  # 下跌，一行代表0.01
        g = np.zeros([100, 10])  # 上涨
        b = np.zeros([100, 10])  # 成交量
        # if j >= 20:
        #     break
        temp = np.concatenate((sum_list_data[j], sum_list_data[j + 1], sum_list_data[j + 2], sum_list_data[j + 3], sum_list_data[j + 4],
                               sum_list_data[j + 5], sum_list_data[j + 6], sum_list_data[j + 7], sum_list_data[j + 8],
                               sum_list_data[j + 9]), axis=0)  # 共10天，垂直组合
        # 组合后，归一化价格，绘制图像
        price = np.array(temp[:, 3:7], dtype='float')
        volume = np.array(temp[:, 7], dtype='int')
        # 这里放的十天的数据
        # norm_price = norm.fit_transform(price.reshape(-1, 1))  # 归一化价格
        # norm_volume = norm.fit_transform(volume.reshape(-1, 1))  # 归一化成交量
        # norm_price = norm_price.reshape(-1, 4)
        norm_price,max_price,min_price = normalize(price.reshape(-1, 1))  # 归一化价格
        norm_volume,max_volume,min_volume = normalize(volume.reshape(-1, 1))  # 归一化成交量
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
    print(len(list_data) - 11)
    return results[:len(list_data) - 10]
def main():
    mintrain = "mintrain.csv"
    minsumdata = "minsumdatat.csv"
    photos = processdata(mintrain,minsumdata)
    print(np.array(photos[2])*255)
    M = np.dstack(photos[2])*255
    im = Image.fromarray(np.uint8(M))
    im = im.convert('L')
    imdata = im.getdata()
    print('im',np.array(imdata).reshape((100,10)))
    print('imshape',np.array(imdata).shape)
    plt.imshow(M, origin='lower')
    plt.imshow(im,origin='lower',cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()
