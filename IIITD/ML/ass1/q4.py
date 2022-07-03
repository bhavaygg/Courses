import json
from itertools import groupby
import pandas as pd
import numpy as np
'''
with open("Q4_Dataset.txt") as fp:
    data=fp.read().splitlines()
    d=[]
    for i in data:
        temp=i.split(" ")
        temp2=[]
        for q in temp:
            if q!="":
                temp2.append(int(float(q)))
        d.append(temp2)
    data=d

df=pd.DataFrame(data)
df.to_csv("q4.csv")
'''
df=pd.read_csv("q4.csv")
df=df.drop(columns=['Unnamed: 0'])
ones=[]
for i in range(0,33):
    ones.append(1)
df.insert(3,"3",ones)
X=df[['1','2','3']].to_numpy()
y=df[['0']].to_numpy()
b=((np.linalg.inv(X.T.dot(X)).dot(X.T)).dot(y))
print(df)
print(b)
#print(y,X.dot(b),b)
#print(np.exp(b[0]),np.exp(b[1]))
print(np.exp(0.19)/(1+np.exp(0.19)))
print(np.exp(7.869)/(1+np.exp(7.869)))
#print(X,y)