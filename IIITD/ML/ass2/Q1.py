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


def ss(dff,y):
    #shuffle and shit the dataset 
    dff=shuffle(dff)
    dtemp_a = pd.DataFrame()
    dtemp_b = pd.DataFrame()
    for n,i in enumerate(y):
        temp = dff.loc[dff['y'] == i]
        tsize = len(temp)
        divider = int(tsize*0.8)
        if n==0:
            dtemp_a = dff.iloc[:divider]
            dtemp_b = dff.iloc[divider:]
        else:
            dtemp_a.append(dff.iloc[:divider])
            dtemp_b.append(dff.iloc[divider:])
    y_train = dtemp_a[["y"]]
    X_train=dtemp_a.drop("y",axis=1)
    y_test = dtemp_b[["y"]]
    X_test=dtemp_b.drop("y",axis=1)
    return X_train,X_test,y_train,y_test

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


def ts(d2,y):
#t-SNE
    df_subset=pd.DataFrame()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(d2)
    df_subset['t1'] = tsne_results[:,0]
    df_subset['t2'] = tsne_results[:,1]
    df_subset["y"]=y

    plt.figure(figsize=(16,10))
    sns.scatterplot(x="t1", y="t2",hue="y",data=df_subset,legend="full",alpha=0.3, palette=sns.color_palette('deep'))
    #plt.savefig("plots/tsne")
    plt.show()

    #X_train,X_test,y_train,y_test=ss(df_subset,labels)
    #clf = LogisticRegression(max_iter=1000,random_state=0).fit(X_train,y_train)
    #y_pred = clf.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    #print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))


#PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(d2)
pDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
op = pd.Series(y)
ts(pDf,op)
pDf["y"]=op
sns.scatterplot(x="pc1", y="pc2",hue="y",data=pDf,legend="full",alpha=0.3, palette=sns.color_palette('deep'))
#plt.savefig("plots/pca")
plt.show()

X_train,X_test,y_train,y_test=ss(pDf,labels)
clf = LogisticRegression(max_iter=10000,random_state=0).fit(X_train,y_train)
y_pred = clf.predict(X_test)
#print(accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))



#SVD
d3=d2

d3["y"]=pd.Series(y)
X_train,X_test,y_train,y_test=ss(d3,labels)
svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
X_train=svd.fit_transform(X_train)
X_test = svd.transform(X_test)
clf = LogisticRegression(max_iter=10000,random_state=0).fit(X_train,y_train)
y_pred = clf.predict(X_test)

X_temp = np.append(X_train,X_test,axis=0)
pDf = pd.DataFrame(data = X_temp, columns = ['pc1', 'pc2'])
#print(y_train.values)
#op = pd.Series(y_train[["y"]].values)

ts(pDf,y_train.append(y_test))
pDf = pd.DataFrame(data = X_train, columns = ['pc1', 'pc2'])
pDf["y"]=y_train.values
sns.scatterplot(x="pc1", y="pc2",hue="y",data=pDf,legend="full",alpha=0.3, palette=sns.color_palette('deep'))
#plt.savefig("plots/svd")
plt.show()

#print(accuracy_score(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))


