import requests
import json
import pandas as pd
import datetime
r = requests.get('https://min-api.cryptocompare.com/data/histohour', params={'fsym':'BTC','tsym':'USD','limit':'2000'})

a = r.json()['Data']
#print(df)
data1 = pd.DataFrame(a)
to = data1.iloc[0,4]

r1=requests.get('https://min-api.cryptocompare.com/data/histohour', params={'fsym':'BTC','tsym':'USD','limit':'2000','toTs':str(to)})
b = r.json()['Data']
data2 = pd.DataFrame(b)

df=pd.concat([data2,data1])
col = 'time'
if col in df.columns:
    df[col] = df[col].apply(lambda x : datetime.datetime.fromtimestamp(x))
df.to_csv('BTCUSDH.csv')
print('BTC完成')

r = requests.get('https://min-api.cryptocompare.com/data/histohour', params={'fsym':'ETH','tsym':'USD','limit':'2000'})

a = r.json()['Data']
#print(df)
data1 = pd.DataFrame(a)
to = data1.iloc[0,4]

r1=requests.get('https://min-api.cryptocompare.com/data/histohour', params={'fsym':'ETH','tsym':'USD','limit':'2000','toTs':str(to)})
b = r.json()['Data']
data2 = pd.DataFrame(b)

df=pd.concat([data2,data1])
col = 'time'
if col in df.columns:
    df[col] = df[col].apply(lambda x : datetime.datetime.fromtimestamp(x))
df.to_csv('ETHUSDH.csv')
print('ETH完成')

r = requests.get('https://min-api.cryptocompare.com/data/histohour', params={'fsym':'BCH','tsym':'USD','limit':'2000'})

a = r.json()['Data']
#print(df)
data1 = pd.DataFrame(a)
to = data1.iloc[0,4]

r1=requests.get('https://min-api.cryptocompare.com/data/histohour', params={'fsym':'BCH','tsym':'USD','limit':'2000','toTs':str(to)})
b = r.json()['Data']
data2 = pd.DataFrame(b)

df=pd.concat([data2,data1])
col = 'time'
if col in df.columns:
    df[col] = df[col].apply(lambda x : datetime.datetime.fromtimestamp(x))
df.to_csv('BCHUSDH.csv')
print('BCH完成')
