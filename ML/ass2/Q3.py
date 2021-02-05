import json
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from sklearn import metrics
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
import pickle 

class kfer():
    df=0
    size=0
    train_s_1=0
    train_s_2=0
    train_f_1=0
    train_f_2=0
    test_s=0
    test_f=0
    val_s=0
    val_f=0
    def __init__(self):
        pass
    def Kfolder(self,df,k):
        if k==0:
            self.df=shuffle(df)
            self.size = len(df)
            self.train_s_1=int(0.2*self.size)
            self.train_s_2=int(0*self.size)
            self.train_f_1=int(0.8*self.size)
            self.train_f_2=int(0*self.size)
            self.val_s=int(0*self.size)
            self.val_f=int(0.2*self.size)
            self.test_s=int(0.8*self.size)
            self.test_f=int(1*self.size)
        elif k==1:
            self.train_s_1=int(0)
            self.train_s_2=int(0.4*self.size)
            self.train_f_1=int(0.2*self.size)
            self.train_f_2=int(0.8*self.size)
            self.val_s=int(0.2*self.size)
            self.val_f=int(0.4*self.size)
            self.test_s=int(0.8*self.size)
            self.test_f=int(1*self.size)
        elif k==2:
            self.train_s_1=int(0)
            self.train_s_2=int(0.6*self.size)
            self.train_f_1=int(0.4*self.size)
            self.train_f_2=int(0.8*self.size)
            self.val_s=int(0.4*self.size)
            self.val_f=int(0.6*self.size)
            self.test_s=int(0.8*self.size)
            self.test_f=int(1*self.size)
        elif k==3:
            self.train_s_1=int(0.2*self.size)
            self.train_s_2=int(0*self.size)
            self.train_f_1=int(0.8*self.size)
            self.train_f_2=int(0*self.size)
            self.val_s=int(0.8*self.size)
            self.val_f=int(1*self.size)
            self.test_s=int(0*self.size)
            self.test_f=int(0.2*self.size)
        df_train = df.iloc[self.train_s_1:self.train_f_1]
        df_train=df_train.append(df.iloc[self.train_s_2:self.train_f_2])
        df_val = df.iloc[self.val_s:self.val_f]
        df_test = df.iloc[self.test_s:self.test_f]
        y_train = df_train[["y"]]
        y_test = df_test[["y"]]
        y_val = df_val[["y"]]
        #print(df_train,df_val,df_test)
        X_train = df_train.drop("y",axis=1)
        X_test = df_test.drop("y",axis=1)
        X_val = df_val.drop("y",axis=1)
        return X_train,X_val,X_test,y_train,y_val,y_test


def evall(model,xt,yp,yt,nam):
    cm = conf(yp,yt)
    ac=acc(cm)
    pr,prl=pres(cm)
    re,rel=reca(cm)
    f1l= []
    print("accuracy",ac)
    print("precision",pr)
    print("recall",re)
    print("","Precision\t","Recall\t","F1")
    for n,i in enumerate(prl):
        f1 = 2*(prl[n]*rel[n])/(prl[n]+rel[n])
        f1l.append(f1)
        print(n,prl[n],"\t\t",rel[n],"\t",f1)
    
    print("Macro Avg Precision",np.mean(prl))
    print("Macro Avg Recall",np.mean(rel))
    print("Macro Avg F1",np.mean(f1l))
    X=[]
    yy=[]
    print(cm)
    abc=(model.predict_proba(xt)).astype(np.float32)
    if nam=="b":
        for i in np.arange(0,1,0.001):
            yp=[]
            for j in abc:
                if np.argmax(j)>=i:
                    yp.append(np.argmax(j))
                else:
                    if np.argmax(j)==0:
                        yp.append(1)
                    else:
                        yp.append(0)
            cm = conf(yp,yt)
            ac=acc(cm)
            re,rel=reca(cm)
            #FP = cm.sum(axis=0) - np.diag(cm)  
            #FN = cm.sum(axis=1) - np.diag(cm)
            #TP = np.diag(cm)
            #TN = cm.sum() - (FP + FN + TP)
            TN=cm[0][0]
            TP = cm[1][1]
            FN=cm[1][0]
            FP = cm[0][1]
            X.append(np.sum(TN)/np.sum(TN+FP))
            yy.append(1-(np.sum(TP)/np.sum(TP+FN)))
        plt.plot(yy,X)
        plt.show()


def acc(mat):
    tp=0
    div=0
    for n,i in enumerate(mat):
        tp+=mat[n][n]
        for n1,j in enumerate(mat):
            div+=mat[n1][n]
        for n1,j in enumerate(mat):
            div+=mat[n][n1]
        div-= mat[n][n]
    return tp/div

def pres(mat):
    tp=0
    div=0
    classwise=[]
    for n,i in enumerate(mat):
        tp+=mat[n][n]
        temp=0
        for n1,j in enumerate(mat):
            temp+=mat[n1][n]
        div+=temp
        classwise.append(round(mat[n][n]/temp,2))
    return tp/div,classwise

def reca(mat):
    tp=0
    div=0
    classwise=[]
    for n,i in enumerate(mat):
        tp+=mat[n][n]
        temp=0
        for n1,j in enumerate(mat):
            temp+=mat[n][n1]
        div+=temp
        classwise.append(round(mat[n][n]/temp,2))
    return tp/div,classwise

def tn(mat):
    tp=0
    div=0
    classwise=[]
    for n,i in enumerate(mat):
        tp+=mat[n][n]
        temp=0
        for n1,j in enumerate(mat):
            temp+=mat[n][n1]
        classwise.append(temp)
    temp=[]
    for n,i in enumerate(classwise):
        temp.append(i/(tp-mat[n][n]))
    return temp

def conf(y1,y2):
    max = np.max(y2)+1
    mat=np.zeros((max,max))
    for n,i in enumerate(y2):
        mat[y2[n][0]][y1[n]]+=1
    return mat
    
def graph(X,y,z):
    #plot graph
    plt.plot(X,y,label=z)

def gs(z):
    #display and save graph
    plt.legend(loc='upper left')
    #plt.savefig((z+".png"))
    plt.show() #uncomment to display grph


#kf = kfold()
def worker_dt(d2,nam):
    print("Decision Tree,",nam)
    k=4
    max_depth=[1,10,50,100,200,500,1000]
    train_scores=[0,0,0,0,0,0]
    val_scores=[0,0,0,0,0,0]
    tr=[]
    vl=[]
    bxt=0
    byt=0
    bestmod=0
    kf = kfer()
    for d in max_depth:
        temp1=0
        temp2=0
        for i in range(0,k):
            #kf = kfold()
            X_train,X_val,X_test,y_train,y_val,y_test=kf.Kfolder(d2,i)
            clf = DecisionTreeClassifier(random_state=0,max_depth=d)
            clf.fit(X_train,y_train)
            if clf.score(X_train,y_train) >temp1:
                temp1=clf.score(X_train,y_train)
            if clf.score(X_val,y_val) >temp2:
                temp2=clf.score(X_val,y_val)
            y_pred = clf.predict(X_test)
            if clf.score(X_val,y_val) > val_scores[0]:
                val_scores[0]=clf.score(X_val,y_val)
                val_scores[1]=d
                val_scores[2]=i
                val_scores[3]=clf.score(X_train,y_train)
                val_scores[4]=d
                val_scores[5]=i
                bestmod=clf
                bxt=X_test
                byt=y_test
            if clf.score(X_train,y_train) > train_scores[0]:
                train_scores[0]=clf.score(X_val,y_val)
                train_scores[1]=d
                train_scores[2]=i
                train_scores[3]=clf.score(X_train,y_train)
                train_scores[4]=d
                train_scores[5]=i
        tr.append(temp1)
        vl.append(temp2)
    filename = 'weights/f_model_dt_'+nam+'+.sav'
    #pickle.dump(bestmod, open(filename, 'wb'))
    bmodel = pickle.load(open(filename, 'rb'))
    graph(max_depth,tr,"training")
    graph(max_depth,vl,"validation")
    gs("zz")
    y_pred = bmodel.predict(bxt)
    evall(bmodel,bxt,y_pred,byt.values,nam)
    #print(train_scores)
    #print(val_scores)


def worker_nb(d2,nam):
    print("naive bayesian,",nam)
    k=3
    max_depth=[1,10,50,100,200,500,1000]
    train_scores=[0,0,0,0]
    val_scores=[0,0,0,0]
    tr=[]
    vl=[]
    bestmod=0
    bxt=0
    byt=0
    kf = kfer()
    temp1=0
    temp2=0
    for i in range(0,k):
        #kf = kfold()
        X_train,X_val,X_test,y_train,y_val,y_test=kf.Kfolder(d2,i)
        clf = GaussianNB()
        clf.fit(X_train,y_train)
        if clf.score(X_train,y_train) >temp1:
            temp1=clf.score(X_train,y_train)
        if clf.score(X_val,y_val) >temp2:
            temp2=clf.score(X_val,y_val)
        y_pred = clf.predict(X_test)
        if clf.score(X_val,y_val) > val_scores[0]:
            val_scores[0]=clf.score(X_val,y_val)
            val_scores[1]=i
            val_scores[2]=clf.score(X_train,y_train)
            val_scores[3]=i
            bestmod=clf
            bxt=X_test
            byt=y_test
        if clf.score(X_train,y_train) > train_scores[0]:
            train_scores[0]=clf.score(X_val,y_val)
            train_scores[1]=i
            train_scores[2]=clf.score(X_train,y_train)
            train_scores[3]=i
    tr.append(temp1)
    vl.append(temp2)
    filename = 'weights/f_model_nb_'+nam+'+.sav'
    #pickle.dump(bestmod, open(filename, 'wb'))
    bmodel = pickle.load(open(filename, 'rb'))
    y_pred = bmodel.predict(bxt)
    evall(bmodel,bxt,y_pred,byt.values,nam)
    #print(train_scores)
    #print(val_scores)

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
d2["y"]=y
worker_dt(d2,"a")
worker_nb(d2,"a")

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
d2=pd.DataFrame(data=X)
d2["y"]=y
worker_dt(d2,"b")
worker_nb(d2,"b")

