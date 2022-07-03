import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """
    weight_inp = 0
    weights= 0
    weight_out = 0
    num_layers=0
    num_units=0
    lr=0
    epochs=0
    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']
    act=""
    weig=""
    tweight=[]
    tbias=[]
    btsize=0
    layer_s=[]
    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """
        #implement batch
        self.tweight=[]
        self.tbias=[]
        self.num_layers=n_layers
        self.num_units=layer_sizes
        self.lr=learning_rate
        self.epochs=num_epochs
        self.btsize = batch_size
        self.layer_s=layer_sizes
        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        self.weig=weight_init
        self.act=activation
        pass

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        #leaky relu
        z=np.copy(X)
        z[z<0]=z[z<0]*0.01
        return z

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        z=np.ones_like(X)
        z[X<0]=0.1
        return z

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return .5 * (1 + np.tanh(.5 * X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        temp= self.sigmoid(X)
        return temp*(1-temp)

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.array(1)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        
        return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))#np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1-self.tanh(X)**2

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        temp = np.exp(X-X.max())
        return temp/np.sum(temp)

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        #print(X)
        temp=self.softmax(X)
        sm = temp.reshape((-1,1))
        t2 = np.diagflat(sm) - np.dot(sm, sm.T)
        return t2

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros((shape))

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.rand(shape)*0.1

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.normal(0,1,size=(shape))*0.1


    def forward(self,arr):
        z=[]
        a=[arr]
        for i in range(0,self.num_layers):
            Bias = np.array(self.tbias[i] * self.btsize).T
            temp_z = np.matmul(a[-1],self.tweight[i]) + Bias
            #sigmoid activation for the output layer
            if i!= self.num_layers-1:
                temp_a = self.activator(temp_z)
            else:
                temp_a=self.sigmoid(temp_z)
            z.append(temp_z)
            a.append(temp_a)
        return z,a

    def backward(self,z_arr,acts,input_arr,out):
        del_W = [np.zeros(W.shape) for W in self.tweight]
        del_B = [np.zeros(b.shape) for b in self.tbias]
        del_a = acts[-1]-out #output loss derivative
        delta = del_a*self.sigmoid_grad(z_arr[-1]) #sigmoid activation for the output layer
        del_B[-1] = np.sum(delta, axis = 0).reshape(self.tbias[-1].shape)
        del_W[-1] = np.dot(acts[-2].T,delta)
        delta=delta.T

        for l in range(2,self.num_layers):
            delta = np.dot(self.tweight[-l+1],delta)*self.activatorgrad(z_arr[-l]).T
            del_W[-l] = np.dot(delta,acts[-l-1]).T
            del_B[-l] = np.sum(delta, axis = 1).reshape(self.tbias[-l].shape)
        return del_W,del_B

    def fit(self, X, y,Xt=False,yt=False,customw=0,tin=0,tsne=0):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Xt,yt : Test data
        tin : 0 if test/validation data is not provided, 1 otherwise
        tsne : 1 to print tsne of final hidden layer
        Returns
        -------
        self : an instance of self
        """
        num_inputs= X.shape[1]
        num_outputs = y.shape[1]
        print(num_inputs,num_outputs,self.epochs,self.num_layers)
        self.weight_init(self.weig,num_inputs,num_outputs)

        err_epoch_train=[]
        err_epoch_test=[]
        
        for ep in range(0,self.epochs):    
            self.btsize=min(self.btsize,X.shape[0])
            batch_size=self.btsize
            batches = [zip(X[b:b+batch_size], y[b:b+batch_size]) for b in range(0, len(y), batch_size)]
            error=0
            for i in range(len(batches)-1):      
                X_train_batch, y_train_batch = zip(*batches[i])
                z,activations=self.forward(X_train_batch)
                y_train_batch=np.array(y_train_batch)
                #error += np.sum(np.power((y_train_batch-activations[-1]),2)) #mse loss
                error += np.sum((-(y_train_batch*np.log(activations[-1]+1e-7))-((1-y_train_batch)*np.log(1-activations[-1]+1e-7)))) #cross entropy loss
                delta_W,delta_B=self.backward(z,activations,X_train_batch,y_train_batch)
                self.tweight = [W - (self.lr * nW/batch_size) for W, nW in zip(self.tweight, delta_W)] #batch weight update
                self.tbias = [W - (self.lr * nW/batch_size) for W, nW in zip(self.tbias, delta_B)] #batch bias update
            err_epoch_train.append(error/len(X))
            if tin==1:
                z,activations=self.forward(Xt)
                error = np.sum(np.power((yt-activations[-1]),2))
                err_epoch_test.append(error/len(Xt))
                print("epoch",ep+1,"train error:",err_epoch_train[-1],"test error:",err_epoch_test[-1])
        plt.plot(range(self.epochs),err_epoch_train,label="Train")
        #print(self.tweight)
        if tin==1:
            plt.plot(range(self.epochs),err_epoch_test,label="Test")
        name="plots/"+self.act+"_q2.png"
        plt.legend(loc="upper right")
        plt.savefig(name)
        plt.close()
        #plt.show()
        if tsne==1:#to plot tsne for final hidden layer activations
            z,activations=self.forward(Xt)
            acts=activations[-2]
            df_subset=pd.DataFrame()
            tsne_results = TSNE(n_components=2).fit_transform(acts)
            df_subset['tsne-2d-one'] = tsne_results[:,0]
            df_subset['tsne-2d-two'] = tsne_results[:,1]
            df_subset["y"] = [np.argmax(x,axis=0) for x in yt]
            sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",hue="y",data=df_subset,legend="full",alpha=0.3,palette=sns.color_palette("tab10"))
            name="plots/"+self.act+"_q2_tsne.png"
            plt.savefig(name)
            plt.close()
            #plt.show()
        name="weights/"+self.act+".npy" #save weights
        np.save(name, self.tweight)
        name="weights/"+self.act+"_bias.npy" #save biases
        np.save(name, self.tweight)
        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def weight_init(self,q,x,y):
        '''
        function to initialise weights and biases based on input array.
        '''
        tem=[]
        layer_s=self.layer_s
        #Input Layers
        for i in range(0,x):
            tem.append(self.weight_helper(q,layer_s[0]))
        self.tweight.append(np.array(tem))
        self.tbias.append(np.zeros((layer_s[0],1)))
        #Hidden Layers
        for j in range(0,2):
            tem=[]
            for i in range(0,layer_s[j]):
                tem.append(self.weight_helper(q,layer_s[j+1]))
            self.tweight.append(np.array(tem))
            self.tbias.append(np.zeros((layer_s[j+1],1)))
        #Output Layer
        tem=[]
        for i in range(0,layer_s[-1]):
            tem.append(self.weight_helper(q,y))
        self.tweight.append(np.array(tem))
        self.tbias.append(np.zeros((y,1)))
        print([x.shape for x in self.tweight])
        print([x.shape for x in self.tbias])

    def weight_helper(self,q,num):
        '''
        Initialises weights based on the input choice
        '''
        if q==self.weight_inits[0]:
            return self.zero_init(num)
        elif q == self.weight_inits[1]:
            return self.random_init(num)
        else:
            return self.normal_init(num)

    def activator(self,arr):
        '''
        Activation helper
        '''
        if self.act=="relu":
            return self.relu(arr)
        elif self.act=="sigmoid":
            return self.sigmoid(arr)
        elif self.act=="linear":
            return self.linear(arr)
        elif self.act=="tanh":
            return self.tanh(arr)
        else:
            return self.softmax(arr)
    
    def activatorgrad(self,arr): 
        '''
        Activation Gradient helper
        '''
        if self.act=="relu":
            return self.relu_grad(arr)
        elif self.act=="sigmoid":
            return self.sigmoid_grad(arr)
        elif self.act=="linear":
            return self.linear_grad(arr)
        elif self.act=="tanh":
            return self.tanh_grad(arr)
        else:
            return self.softmax_grad(arr)

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """
        z,act= self.forward(X)
        probs = self.softmax(act[-1])
        # return the numpy array y which contains the predicted values
        return probs

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
        z,activations=self.forward(X)
        yp = [np.argmax(x,axis=0) for x in activations[-1]]
        # return the numpy array y which contains the predicted values
        return yp

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        yp=self.predict(X)
        score=0
        for n,i in enumerate(yp):
            if i==np.argmax(y[n],axis=0):
                score+=1
        # return the numpy array y which contains the predicted values
        return score/len(y)
