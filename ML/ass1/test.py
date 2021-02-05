from scratch import MyLinearRegression, MyLogisticRegression,MyPreProcessor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
Xtrain = np.array([[1,2,3], [4,5,6]])
ytrain = np.array([1,2])

Xtest = np.array([[7,8,9]])
ytest = np.array([3])


#uncomment linear.fit to run code for dataset 2
pp =MyPreProcessor()
X,y=pp.pre_process(1)
print('Linear Regression')
linear = MyLinearRegression()
#linear.fit(X, y)

#uncomment linear.fit to run code for dataset 1
X,y=pp.pre_process(0)
linear.fit(X,y)


#uncomment linear.fit to run code for dataset 3
X,y=pp.pre_process(2)
log =MyLogisticRegression()
#log.fit(X,y)
