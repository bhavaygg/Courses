import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import torch.nn as nn
from keras.datasets import cifar10
import tensorflow as tf
from torch import optim
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve
#import data
AlexNet_Model = models.alexnet(pretrained=True)
print(AlexNet_Model.eval())

from torchvision import transforms
import torchvision
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


net = AlexNet_Model
net.cpu() #model.to("cuda") #for cuda


batch_size=200
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=0)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)


criterion = nn.CrossEntropyLoss()

new_data=[]
new_labels=[]
#custom model
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
    torch.nn.Softmax(0)
)

learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-3


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
for t in range(100):
  print("epoch",t)
  for i, data in enumerate(trainloader, 0):
    print(i)
    inputs, labels = data
    #inputs, labels = inputs.cuda(), labels.cuda()  #for cuda
    optimizer.zero_grad()
    outputs = net(inputs)
    y_pred = model(outputs)
    loss = loss_fn(y_pred, labels.long())
    model.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

t=[]
sc=0
conf=np.zeros((10,10))
#forward prop for test data
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    outputs = net(inputs)
    yp=model(outputs)
    yp=[np.argmax(i.detach().numpy()) for i in yp] #yp=[np.argmax(i.cpu().detach().numpy()) for i in yp] #for cuda
    for n,i in enumerate(yp):
      if i==labels[n].item():
        sc+=1
      conf[i][labels[n].item()]+=1
print(sc/10000)
print(conf)