import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #引入划分数据集函数
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed,Dense,MaxPooling1D,Dropout,Activation,CuDNNLSTM,GaussianDropout
from keras.models import load_model

file_name ='BTCUSDH.csv'
f=open(file_name) 
df=pd.read_csv(f)     #读入股票数据
df.eval('average=(open+close)/2',inplace=True)
data=np.array(df[['open','close','volumefrom','high','low','average']])   #data 原始数据


#超参数
rnn_units=256       #神经元个数
time_step = 50    #时间步，一次输入50个数据
output_step=10     #输出步，一次预测10个数据
input_dim= 5             #输入数据与输出数据的维度
output_dim =1
dropout_rate=0.3   #dropout比例，随机断开多少神经元
stride = 5        #池化比例，根据时间步决定，一般不改

#将数据标准化
def normalize(data):
    std_data = 1/10000*data
    print("数据标准化完成")
    return std_data

def create_trainlist(data):
    a,b = [],[]
    for i in range(1,len(data)-time_step-output_step+1):
        x = data[i-1:i+time_step-1,:5]
        y = data[i+time_step-1:i+time_step+output_step-1,-1]
        a.append(x)
        b.append(y)
    x= np.array(a)
    y =np.array(b).reshape(-1,10,1)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
    print("划分训练/测试集完成")
    return x_train,x_test,y_train,y_test

#创建预测数据
def create_pred(data):
    x = []
    for i in range(1,len(data)-time_step+1):
        a = data[i-1:i+time_step-1,:5]
        x.append(a)
    return np.array(x)

# 创建神经网络
model = Sequential()
def create_LSTM():
    #第一层LSTM
    model.add(CuDNNLSTM(units=rnn_units,
        input_shape=(time_step,input_dim),
        return_sequences=True,
        ))
    #Dropout 防止过拟合
    model.add(Dropout(rate=dropout_rate))
    
    #第二层LSTM
    model.add(CuDNNLSTM(units=rnn_units,
        return_sequences=True))
    #第二层Dropout
    model.add(Dropout(rate=dropout_rate))
    #第三层LSTM
    model.add(CuDNNLSTM(units=128,
                   return_sequences=True))
    #第三层Dropout
    model.add(GaussianDropout(rate=dropout_rate))
    #池化
    model.add(MaxPooling1D(pool_size=time_step,
        strides = stride,
        padding = 'same'))
    #全连接层，降低维度
    model.add(TimeDistributed(Dense(128,activation='relu')))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(output_dim,activation='relu'))
    #编译神经网络
    model.compile(loss='mean_squared_error',optimizer='adam')
    print(model.summary())
    return 0

# train LSTM
def train_LSTM():
    model.fit(train_x, train_y, epochs=500, batch_size=60, verbose=2)
    score = model.evaluate(test_x,test_y,verbose=1)
    print(score)
    model.save('BTC_H_model.h5')
    print('模型储存完成')
    return 0
#预测模型
def predict_LSTM(train_x):
    model = load_model('BTC_H_model.h5')
    print('模型加载完成')
    result = model.predict(train_x, batch_size=60, verbose=0)
    return result
#转换回原数据
def transform_origin(data):
    origin_data = 10000*data
    return origin_data


#预测直接运行这里
set_data=normalize(create_pred(data))
print(set_data.shape)
result = predict_LSTM(set_data)
z = result[-140:,-1,:].reshape(140,1)
w = transform_origin(z)
print(w)

import plotly.plotly
import plotly.graph_objs as go
trace0 = go.Candlestick(x=np.arange(1,131),
                        open=data[-130:,0],
                        high=data[-130:,3],
                        low= data[-130:,4],
                        close = data[-130:,1]
                        )

trace1 = go.Scatter(x = np.arange(1,141),
                    y = w[:,-1])
data = [trace0,trace1]
plotly.offline.plot(data,filename='BTC-10H.html')