
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
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,roc_curve
import torchvision
#class distribution
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sns.countplot(y_train.ravel())
plt.savefig("plots/Q4_train_dist.png")
plt.close()
sns.countplot(y_test.ravel())
plt.savefig("plots/Q4_test_dist.png")
plt.close()
plt.show()



class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),  # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool1
            nn.Conv2d(24, 96, 3, padding=1),  # Conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool2
            nn.Conv2d(96, 192, 3, padding=1),  # Conv3
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),  # Conv4
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 96, 3, padding=1),  # Conv5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pool3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(96 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 96 * 4 * 4)
        x = self.classifier(x)
        return x

transform = transforms.Compose(
    [
    
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

net = AlexNet(10)
net.cpu()
learning_rate = 1e-3
momentum = 0.9
weight_decay = 1e-3

batch_size=4
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=False,num_workers=0)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=0)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

new_data=[]
new_labels=[]

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    for k,tens in enumerate(outputs):
        new_data.append(tens)
        new_labels.append(labels[k])
X_train=new_data
y_train=new_labels
new_data=[]
new_labels=[]

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    for k,tens in enumerate(outputs):
        new_data.append(tens)
        new_labels.append(labels[k])
X_test=new_data
y_test=new_labels
print(new_labels[-1],labels[-1])
torch.save(X_train,"x_train_q4")
torch.save(y_train,"y_train_q4")
torch.save(X_test,"x_test_q4")
torch.save(y_test,"y_test_q4")

X_train=torch.load("x_train_q4")
y_train=torch.load("y_train_q4")
X_test= torch.load("x_test_q4")
y_test= torch.load("y_test_q4")

X_train=torch.stack([x for x in X_train])

y_train=torch.stack([x for x in y_train])
X_test=torch.stack([x for x in X_test])
y_test=torch.stack([x for x in y_test])

loss_fn = torch.nn.CrossEntropyLoss()
#custom model
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 512),
    torch.nn.Sigmoid(),
    torch.nn.Linear(512, 256),
    torch.nn.Sigmoid(),
    torch.nn.Linear(256, 10),
    torch.nn.Softmax(0)
)
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate) 
for t in range(100):
    print("epoch",t)
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train.long())
    model.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

yp=model(X_test)

sc=0
yp=[np.argmax(i.detach().numpy()) for i in yp]
yr=[i.item() for i in y_test]

#match with output
for n,i in enumerate(yp):
    if i==y_test[n].item():
        sc+=1
print(sc/len(yp))
print(confusion_matrix(yr, yp))

