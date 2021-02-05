import json
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,accuracy_score
from sklearn.utils import shuffle
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scratch import MyLinearRegression, MyLogisticRegression,MyPreProcessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import math
from sklearn.model_selection import train_test_split

#Dataset1
f = h5py.File("part_A_train.h5","r")
X=list(f["X"])
y=list(f["Y"])
t = []
for i in X:
    t.append(i.tolist())
X=t
t=[]
for i in y:
    t.append(i.tolist())
y=t

t=[]
for i in y:
    index = i.index(1.0)
    t.append(index)
y=t

labels=set(y)
d2=pd.DataFrame(data=X)


def gaussian_density_function(X, mean, std):
    eps=1e-6
    c = -1/2 * np.log(2*np.pi) - 0.5 * np.sum(np.log(std + eps))
    p = 0.5 * np.sum((X - mean)**2/(std + eps), 1)
    return c - p
    #return (1 / ((np.sqrt(2*np.pi)*stdev)+ 0.01)) * np.exp(-((X-mean)**2) / (2*(stdev**2) + 0.01))

def gen(dat):
    means={}
    stds={}
    priors={}
    num_sam = len(dat)
    for n1,i in dat.groupby(["y"]):
        temp1=[]
        temp2=[]
        means[n1]=[]
        stds[n1]=[]
        for column in i:
            if column!="y":
                mean = np.mean(i[column].values)
                std = np.std(i[column].values)
                temp1.append(mean)
                temp2.append(std)
        means[n1]=temp1
        stds[n1]=temp2
        priors[n1]=len(i)/num_sam
    #print(means)
    #print(stds)
    dat=dat.drop("y",axis=1)
    return dat,means,stds,priors


def probs(da,me,st,prs,n):
    pps=[]
    #for n,j in enumerate(X_train):
    probs = np.zeros((len(da), n))
    for i in range(0,n):
        pp = (gaussian_density_function(np.asarray(da), np.asarray(me[i]), np.asarray(st[i])))+np.log(prs[i])
        #print(pp,pp.shape)
        probs[:,i] =pp
    #print(np.argmax(probs, 1))
    pps= np.argmax(probs, 1)
    return pps

def acc(pps,y_train):
    count=0
    for n,i in enumerate(pps):
        if i == y_train[n]:
            #print(i)
            count+=1
    return(count/len(pps))

print("\n Dataset1")
X_train, X_test, y_train, y_test = train_test_split(d2, y, test_size=0.3)
X_train["y"]=y_train
#X_test["y"]=y_test
X_train,means,stds,priors = gen(X_train)
pps=probs(X_train,means,stds,priors,10)
ppz=probs(X_test,means,stds,priors,10)

#print(len(pps))
print("train",acc(pps,y_train))
print("test",acc(ppz,y_test))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
print("sklearn acc train",clf.score(X_train, y_train))
print("sklearn acc test",acc(clf.predict(X_test),y_test))


#print(clf.score(X_test, y_test))
print("\n Dataset2")
#Dataset2
f = h5py.File("part_B_train.h5","r")
X=list(f["X"])
y=list(f["Y"])
t = []
for i in X:
    t.append(i.tolist())
X=t
t=[]
for i in y:
    t.append(i.tolist())
y=t

t=[]
for i in y:
    index = i.index(1.0)
    t.append(index)
y=t

labels=set(y)
d2=pd.DataFrame(data=X)

X_train, X_test, y_train, y_test = train_test_split(d2, y, test_size=0.3)
X_train["y"]=y_train
#X_test["y"]=y_test
X_train,means,stds,priors = gen(X_train)
pps=probs(X_train,means,stds,priors,2)
ppz=probs(X_test,means,stds,priors,2)

#print(len(pps))
print("train",acc(pps,y_train))
print("test",acc(ppz,y_test))

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
print("sklearn acc train",clf.score(X_train, y_train))
print("sklearn acc test",acc(clf.predict(X_test),y_test))
