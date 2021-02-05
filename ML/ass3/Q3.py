# -*- coding: utf-8 -*-
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# Create random Tensors to hold inputs and outputs

df=pd.read_csv("largeTrain.csv")
y=torch.from_numpy(df.iloc[:, 0].to_numpy())
x=torch.from_numpy(df.iloc[:,1:].to_numpy())
x=x.view(x.size(0),-1).float()
y=y.double()

df=pd.read_csv("largeValidation.csv")
y_val=torch.from_numpy(df.iloc[:, 0].to_numpy())
x_val=torch.from_numpy(df.iloc[:,1:].to_numpy())
x_val=x_val.view(x_val.size(0),-1).float()
y_val=y_val.double()

print(x.shape,y.shape)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight)

loss_fn = torch.nn.CrossEntropyLoss()

eps=100
nhs=[5,20,50,100,200]
loss_arr=[]
loss_arr_val=[]

for i in nhs:
    N, D_in, H, D_out = 8999, 128, i, 10
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Softmax(1)
    )
    model.apply(init_weights)
    learning_rate = 0.01
    for t in range(eps):
        for mode in ['train','val']:
            if mode=="train":
                y_pred = model(x)
                loss = loss_fn(y_pred, y.long())
                if t ==eps-1:
                    loss_arr.append(loss)
                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad
            else:
                y_pred = model(x_val)
                loss = loss_fn(y_pred, y_val.long())
                if t == eps-1:
                    loss_arr_val.append(loss)

plt.plot(nhs,loss_arr,label="training")
plt.plot(nhs,loss_arr_val,label="validation")
plt.xlabel("Hidden Units")
plt.ylabel("Average Loss")
plt.legend(loc="upper right")
#plt.savefig("plots/plot_hu.png")
#plt.close()
plt.show()



lrs=[0.1,0.01,0.001]
for i in lrs:
    loss_arr=[]
    loss_arr_val=[]
    N, D_in, H, D_out = 8999, 128, 4, 10
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Softmax(1)
    )
    model.apply(init_weights)
    learning_rate = i
    optimizer = optim.SGD(model.parameters(), lr=i)
    for t in range(eps):
        for mode in ['train','val']:
            if mode=="train":
                y_pred = model(x)
                loss = loss_fn(y_pred, y.long())
                loss_arr.append(loss)
                model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad
            else:
                y_pred = model(x_val)
                loss = loss_fn(y_pred, y_val.long())
                loss_arr_val.append(loss)
    plt.plot(range(eps),loss_arr,label=("training_"+str(i)))
    plt.plot(range(eps),loss_arr_val,label=("validation_"+str(i)))
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.legend(loc="upper right")
    name="plots/plot_lr_"+str(i)+".png"
    #plt.savefig(name)
    #plt.close()
    plt.show()
