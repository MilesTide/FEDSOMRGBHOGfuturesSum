import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from math import ceil
import csv
def Slip(way,filename):
    data = pd.DataFrame(pd.read_csv(way,header=0))
    norm = MinMaxScaler()   #为了归一化数据
    arrdata = np.array(data)
    #将所有数据按照30条进行切分
    list_data = []
    for i in range(0,len(data),30):
        list_data.append(arrdata[i:i+30,:])    #数据的 市场代码,合约代码,时间,开,高,低,收,成交量,成交额,持仓量
    #切分的数据按照十天一组垂直组合
    for j in range(len(list_data)):
        now_data = list_data[j]
        now_date = now_data[0][2]
        now_time = now_data[-1][2]
        now_volume = sum(np.array(now_data[:,7],dtype=float))
        openp = now_data[0][3]
        high = max(now_data[:,4])
        low = min(now_data[:,5])
        close = now_data[-1][6]
        date_today = datetime.strptime(now_date, '%Y-%m-%d %H:%M:%S')
        date_today = date_today.strftime('%Y-%m-%d %H:%M:%S')
        time_today = datetime.strptime(now_time, '%Y-%m-%d %H:%M:%S')
        time_today = time_today.strftime('%Y-%m-%d %H:%M:%S')
        text = [date_today,time_today,openp,high,low,close,now_volume]
        print(text)
        with open(filename, 'a',encoding='utf-8', newline='') as f:  # 条件满足，记录数据
            csv.writer(f).writerow(text)
        f.close()
    print("写入成功")