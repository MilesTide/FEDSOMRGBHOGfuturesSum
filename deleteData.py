#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# @author: peidehao
import datetime
import pandas as pd

data = pd.DataFrame(pd.read_csv('rb主力连续.csv',header=0,encoding='utf-8'))
start = '2020-01-02 10:01:00'
start = datetime.datetime.strptime(start,"%Y-%m-%d %H:%M:%S")
final = data['时间'].iloc[-1]
final = datetime.datetime.strptime(final,"%Y-%m-%d %H:%M:%S")
end = start+datetime.timedelta(minutes=15)
cols = []
period = (final-start).days
for d in range(0,period+1):
    for i in range(len(data)):
        if start<=datetime.datetime.strptime(data['时间'].iloc[i],"%Y-%m-%d %H:%M:%S")<end:
           cols.extend([i])
    start = start + datetime.timedelta(days=1)
    end = start + datetime.timedelta(minutes=15)
    print(start,end)
print(cols)
data2 = data.drop(cols,axis=0)
data2.to_csv('rb.csv',index=False,header=True)




