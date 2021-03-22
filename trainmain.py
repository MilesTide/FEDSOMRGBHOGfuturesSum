#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: peidehao
'''根据股票、期货用processDataSlip.py切分数据
1、先运行FEDSOMCode_w_iterate.py，训练SOM网络，得到，每一层SOM对应的权值
2、运行DBSCAN.py对最后一层的预测结果进行密度聚类，得到每个坐标对应的BP预测基,对比count_matrix.npy中的情况
3、运行BpNetwork，根据密度聚类结果建立预测基
处理测试集，用processDataSlip.py切分
4、运行forcast,预测数据'''
import numpy as np
import pandas as pd
import datetime
import os

#切分数据
mintrain= "mintrain.csv"
daytrain = "daytrain.csv"
mintest = "mintest.csv"
daytest = "daytest.csv"
minsumdata = "minsumdatat.csv"
daysumdata = "daysumdata.csv"
'''
import processDataSlip as pds
#滑动窗口放在切分数据这里 按照日期进行切分
dataway = "rb.csv"
oneday = '2020-01-02 09:01:00'  #开始的时间
oneday = datetime.datetime.strptime(oneday,"%Y-%m-%d %H:%M:%S")
train_period = 60 #训练数据的周期，单位是天
test_period = 15    #测试数据的周期
win_size = test_period
train_end_day =oneday+datetime.timedelta(days=train_period) #两个月
test_end_day = train_end_day + datetime.timedelta(days=test_period)    #两周

#按30分钟切分，正常交易日一天有11个30分钟，还有的有8个30分钟，10:00-10:15的数据被删掉了
data = pd.DataFrame(pd.read_csv(dataway,header=0,encoding='utf-8'))
train_loc = []
test_loc = []
sumdata = []
for i in range(len(data)):
    if oneday<=datetime.datetime.strptime(data['时间'].iloc[i],"%Y-%m-%d %H:%M:%S")<=train_end_day:
        #记录到训练数据
        train_loc.extend([i])
    if train_end_day<=datetime.datetime.strptime(data['时间'].iloc[i],"%Y-%m-%d %H:%M:%S")<=test_end_day:
        # 记录到测试数据
        test_loc.extend([i])
train_data = data[min(train_loc):max(train_loc)]
test_data = data[min(test_loc):max(test_loc)]
sumdata= data[min(train_loc):max(test_loc)]
train_data.to_csv(mintrain,index=False,header=True)
test_data.to_csv(mintest, index=False, header=True)
sumdata.to_csv(minsumdata, index=False, header=True)
if os.path.exists(daytrain):
    os.remove(daytrain)
if os.path.exists(daytest):
    os.remove(daytest)
pds.Slip(mintrain,daytrain)
pds.Slip(mintest,daytest)
pds.Slip(minsumdata,daysumdata)
'''
#训练SOM
import FEDSOMCode_w_iterate as som
Epsilon = 0.001
max_price,min_price,max_volume,min_volume = som.main(Epsilon,mintrain,daytrain,minsumdata)
#保存最大值最小值
maxmin = {'max_price':max_price,'min_price':min_price,'max_volume':max_volume,'min_volume':min_volume}
np.save('maxmin.npy',maxmin)
#密度聚类
import DBSCAN as db
eps = 0.00002
min_Pts = 50
db.main(eps,min_Pts)
#训练BP
import BPNetwork as bp
input = 40
hidden = [10,10]
output = 1
epoch = 10000
bp.main(input,hidden,output,epoch,max_price,min_price)

#预测
import forecast
forecast.main(mintest,daytest )
