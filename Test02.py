#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: peidehao
import datetime
import pandas as pd

data = pd.DataFrame(pd.read_csv('rb.csv',index_col='时间',header=0,encoding='utf-8'))
start = '2020-01-02 10:01:00'
start = datetime.datetime.strptime(start,"%Y-%m-%d %H:%M:%S")
final = '2020-12-31 15:00:00'
final = datetime.datetime.strptime(final,"%Y-%m-%d %H:%M:%S")
end = start+datetime.timedelta(minutes=15)
cols = []
period = (final-start).days
for d in range(period):
    for i in range(15):
        cols.extend([start+datetime.timedelta(minutes=i)])
    start = start + datetime.timedelta(days=1)
    end = start + datetime.timedelta(minutes=15)
    print(start,end)
print(cols)
data2 = data.drop(cols,axis=1)
data2.to_csv('test.csv',index=False,header=True)




