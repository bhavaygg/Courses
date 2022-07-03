from keras.datasets import mnist
import numpy as np
from Q1 import MyNeuralNetwork
from sklearn.neural_network import MLPClassifier
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

(X_train, y_train), (X_test, y_test) = mnist.load_data()
y_train=y_train.reshape((y_train.shape[0],1))
y_test=y_test.reshape((y_test.shape[0],1))
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
X_train = X_train.astype('float32')/255#normalize
X_test = X_test.astype('float32')/255
y_train = np_utils.to_categorical(y_train, 10)#encode to one hot
y_test = np_utils.to_categorical(y_test, 10)

print(X_train.shape,X_test.shape)

for i in ["sigmoid","relu","tanh","linear"]:
    trying = MyNeuralNetwork(4,[256,128,64],i,0.1,'normal',200,100)
    trying.fit(X_train, y_train,X_test,y_test,tin=1,tsne=1)
    score=trying.score(X_test,y_test)
    print(i,"accuracy",score)
    print(trying.predict_proba(X_test[0]))