import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv 
import seaborn as sns
import random
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset
        
        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        # np.empty creates an empty array only. You have to replace this with your code.
        X = np.empty((0,0))
        y = np.empty((0))

        if dataset == 0:
            # Implement for the abalone dataset
            with open("Dataset.data") as fp:
                dat=fp.read().splitlines()
                temp=[]
                for i in dat:
                    k=i.split(" ")
                    temp2=[]
                    for j in k: #converting gender to integer
                        if j=="M":
                            temp2.append(0)
                        elif j=="F":
                            temp2.append(1)
                        elif j=="I":
                            temp2.append(3)
                        else:
                            temp2.append(float(j))
                    temp.append(temp2)
                dat=temp
                df=pd.DataFrame(dat,columns=["a","b","c","ab","bc","d","ad","e","y"])
            df = shuffle(df)
            X=df[["a","b","c","ab","bc","d","ad","e"]]
            y=df[["y"]]
            X=(X.to_numpy())
            y=(y.to_numpy()).astype(int)
            pass
        elif dataset == 1:
            # Implement for the video game dataset
            df = pd.read_csv (r'vg.csv')
            for n,i in df.iterrows():
                if i["Critic_Score"]!=i["Critic_Score"] or i["User_Score"]!=i["User_Score"] or i["User_Score"]=="tbd" or i["Critic_Score"]=="tbd":
                    df.drop(n,inplace=True)
            df = shuffle(df)
            X=df[["Critic_Score","User_Score"]]#.astype(str).astype(float).astype(int)
            y=df[["Global_Sales"]]#.astype(str).astype(float).astype(int)
            X.iloc[:,1]*=10
            X=(X.to_numpy()).astype(float)/10
            y=np.ceil(y.to_numpy()).astype(int)
            pass
        elif dataset == 2:
            # Implement for the banknote authentication dataset
            with open("data_banknote_authentication.txt") as fp:
                stripped = (line.strip() for line in fp)
                lines = (line.split(",") for line in stripped if line)
                with open('data_bank.csv', 'w') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(("a","b","c","d","e"))
                    writer.writerows(lines)
            df = pd.read_csv("data_bank.csv")
            df = shuffle(df)
            X=df[["a","b","c","d"]]
            y=df[["e"]]
            X=(X.to_numpy(dtype="float64"))
            y=(y.to_numpy(dtype="float64"))
            #sns.boxplot(data=pd.melt(df[["a","b","c","d"]]),x="variable", y="value")
            #plt.show()
            #print(np.sum(df[["e"]].to_numpy())))
            #print(df)
            pass

        return X, y


#dataset is shuffled at launch for different validation splits everytime
class MyLinearRegression():
    """
    My implementation of Linear Regression.
    """
    phi=0
    theta=0
    yp=0
    c=0
    costs=0
    def __init__(self):
        pass

    def cost_mae(self,X,y,n):
        #cost function for MAE
        return ((1/n)*np.sum(abs(X.dot(self.theta)-y+self.c)))
    
    def cost_rmse(self,X,y,n):
        #cost function for RMSE
        return (np.sqrt(np.sum(((1/n)*(X.dot(self.theta)-y+self.c))**2)))
    
    def grad_rmse(self,X,y,n,epoch,alpha):
        #gradient descent for RMSE
        temp=0
        for i in range(0,epoch):
            temp = alpha*(X.T.dot(((1/n)*(X.dot(self.theta)-y+self.c))/(np.sqrt(((1/n)*(X.dot(self.theta)-y+self.c))**2))))
            self.c -= alpha*((((1/n)*(X.dot(self.theta)-y+self.c))/(np.sqrt(((1/n)*(X.dot(self.theta)-y+self.c))**2))))
            self.theta -= temp
            #print(self.theta)
            self.costs.append(self.cost_rmse(X,y,n))
        return self

    def grad_mae(self,X,y,n,epoch,alpha):
        #gradient descent for MAE
        for i in range(0,epoch):
            temp = (1/n)*np.sum((X*(np.sign(X.dot(self.theta)+self.c-y))),axis=0).reshape((X.shape[1],1))
            self.c -= alpha*np.sign(X.dot(self.theta)+self.c-y)*(1/n)
            self.theta-= alpha*temp
            #print(self.theta,self.c)
            #print(self.theta)
            self.costs.append(self.cost_mae(X,y,n))
        return self
    
    def normal_form(self,X,y):
        #normal equation form
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        return self

         
    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        n=X.shape[0]
        k=3
        self.yp = np.zeros(shape=(X.shape[0],1))
        self.theta = np.zeros((X.shape[1],1))
        self.c = 0
        self.costs=[]
        flag=1
        if flag==0:#random testing
            self.grad_mae(X,y,n,100,0.1)
            self.graph(100,self.costs,"a")
            self.gs()
        elif flag==2:#normal equation from
            self.normal_form(X,y)
        else:#k-fold cross validation
            for i in range(0,k):
                self.yp = np.zeros(shape=(X.shape[0],1))
                self.theta = np.zeros((X.shape[1],1))
                self.c = 0#np.zeros(shape=(X.shape[0],1))
                self.costs=[]
                start=int(i*n/k)
                end=int((i+1)*n/k)
                X_val=X[start:end]
                y_val=y[start:end]
                X_train=np.append(X[:start],X[end:],axis=0)
                y_train=np.append(y[:start],y[end:],axis=0)
                self.grad_rmse(X_train,y_train,X_train.shape[0],500,0.001) # comment this to run MAE
                #self.grad_mae(X_train,y_train,X_train.shape[0],1000,1) #uncomment this to run MAE
                train_cost=self.costs[-1]
                self.graph(500,self.costs,"Train")
                self.c=float(self.c[0])
                self.costs=[]
                self.grad_rmse(X_val,y_val,X_val.shape[0],500,0.001)  # comment this to run MAE
                #self.grad_mae(X_val,y_val,X_val.shape[0],1000,1)   # uncomment this to run MAE
                self.graph(500,self.costs,"Validate")
                self.gs("dataset2"+"_"+str(i))
                #print(i,train_cost,self.costs[-1])
            return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        y = X.dot(self.theta)+self.c
        # return the numpy array y which contains the predicted values
        return np.round(y)

    def graph(self,X,y,z):
        #plot graph
        plt.plot(range(X),y,label=z)


    def gs(self,z):
        #display and save graph
        plt.legend(loc='upper left')
        #plt.savefig((z+".png"))
        #plt.show() #uncomment to display grph

class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """
    theta=0
    c=0
    costs=0

    def __init__(self):
        pass
    
    def sigmoid(self,X):
        #sigmoid fucntion
        return (1/(1+np.exp(-(X.dot(self.theta)+self.c))))

    def los(self,X,y,n):
        #loss function
        sig=self.sigmoid(X)
        loss = np.sum(((1/n)*(-y.T.dot(np.log(sig+1e-7))))-((1-y).T.dot(np.log(1-sig+1e-7))))
        return loss

    def grad_loss(self,X,y,n,epoch,alpha):
        #gradeint descent for sgd
        for i in range(0,epoch):
            ra=random.randint(0,X.shape[0]-1)#select random number from the input batch
            temp = alpha*(1/n)*X[ra]*(self.sigmoid(X[ra])-y[ra])
            temp.reshape((X.shape[1],1))
            self.c-= alpha*(1/n)*(self.sigmoid(X[ra])-y[ra])
            self.costs.append(self.los(X[ra],y[ra],n))
            self.theta-= temp[0]
    
    def bat_grad_loss(self,X,y,n,epoch,alpha):
        #gradeint descent for bgd
        for i in range(0,epoch):
            temp = alpha*(1/n)*X.T.dot(self.sigmoid(X)-y)
            self.c-= alpha*(1/n)*(self.sigmoid(X)-y)
            self.costs.append(self.los(X,y,n))
            self.theta-= temp
            #print(self.theta.shape,self.c.shape,self.theta)

    def fit(self, X, y):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        self.theta = np.zeros((X.shape[1],1))
        self.c = 0
        self.costs=[]
        n=X.shape[0]
        k=10
        for i in range(0,k):
                self.yp = np.zeros(shape=(X.shape[0],1))
                self.theta = np.zeros((X.shape[1],1))
                self.c = 0#np.zeros(shape=(X.shape[0],1))
                self.costs=[]
                start=int(i*n/k)
                end=int((i+1)*n/k)
                X_val=X[start:end]
                y_val=y[start:end]
                X_train=np.append(X[:start],X[end:],axis=0)
                y_train=np.append(y[:start],y[end:],axis=0)
                self.bat_grad_loss(X_train,y_train,X_train.shape[0],200,0.001) #comment this for sgd
                #self.grad_loss(X_train,y_train,X_train.shape[0],200,10)  #uncomment this for sgd
                train_cost=self.costs[-1]
                self.graph(200,self.costs,"Train")
                self.c=float(self.c[0])
                yp=np.round(self.predict(X_train))
                self.accu(yp,y_train)
                self.costs=[]
                self.bat_grad_loss(X_val,y_val,X_val.shape[0],200,0.001)  #comment this for sgd
                #self.grad_loss(X_val,y_val,X_val.shape[0],200,10)  #uncomment this for sgd
                self.graph(200,self.costs,"Validate")
                self.gs("dataset3_10"+"_"+str(i))
                self.c=float(self.c[0])
                yp=self.predict(X_val)
                self.accu(yp,y_val)
                from sklearn.linear_model import LogisticRegressionCV
        clf = LogisticRegressionCV(cv=10,max_iter=200).fit(X, y)
        print(clf.score(X, y))
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        y=self.sigmoid(X)
        # return the numpy array y which contains the predicted values
        return np.round(y)
    
    def accu(self,a,y):
        #prints accuracy of predicitons
        xc=0
        for n,i in enumerate(a):
            if a[n]==y[n]:
                xc+=1  
        print(xc/(n+1))

    
    def graph(self,X,y,z):
        #plot graph
        plt.plot(range(X),y,label=z)


    def gs(self,z):
        #display and save graph
        plt.legend(loc='upper left')
        #plt.savefig((z+".png"))
        #plt.show()#uncomment to display grph