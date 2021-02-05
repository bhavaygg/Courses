import json
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,accuracy_score
from sklearn.utils import shuffle
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scratch import MyLinearRegression, MyLogisticRegression,MyPreProcessor


df= pd.read_csv("weight-height.csv")
df=df.replace("Male",100)
df=df.replace("Female",200)
print(df)
X=df[["Gender","Height"]].to_numpy(dtype=np.float64)
y=df[["Weight"]].to_numpy(dtype=np.float64)

from sklearn.linear_model import LinearRegression
acmean= np.mean(X)
bsmean=[]
bs=[]
for j in range(0,100):
    rs = [random.randint(0, X.shape[0]-1) for x in range(0,X.shape[0])]
    t=[]
    t_y=[]
    for i in rs:
        t.append(X[i])
        t_y.append(y[i])
    bsmean.append(np.mean(t))
    bs.append(np.mean(t)-acmean)
bestimate = np.mean(bsmean)
#print(np.sqrt(np.sum((np.array(bsmean)-bestimate)**2)/(len(bsmean)-1)))
bvar = (np.sqrt(np.sum((np.array(bsmean)-bestimate)**2)/(len(bsmean)-1)))
print((bestimate-acmean)**2,bvar,np.mean(bs)**2)
bias2=(bestimate-acmean)**2
reg = LinearRegression().fit(X, y)
t=np.array(t).reshape((X.shape[0],X.shape[1]))
t_y=np.array(t_y).reshape((y.shape[0],y.shape[1]))
yp = reg.predict(t)
mse=np.sum((yp-t_y)**2)/X.shape[0]
print(mse,mse-bias2-bvar)
#linear = MyLinearRegression()
#linear.fit(X,y)
#linear.boot(X,y)
#print(linear.predict(X[0]),y[0])