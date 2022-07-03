# CSE 101 - IP HW4
# K-Map Minimization 
# Name: Bhavay Aggarwal
# Roll Number: 2018384
# Section: B
# Group: 1
import math as m
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.ion() # makes the plot interactive
inp=input()
X=[]
Y=[]
num=0

if inp == "polygon":
    X = list(map(int, input().split()))
    Y = list(map(int, input().split()))
    q=X[0]
    w=Y[0]
    X.append(q)
    Y.append(w)
    plt.plot(X,Y)
    plt.show()
    mat=[[0,0],[0,0]]
    trans = input()
    while trans!="quit":
        if trans[0]=='S'or trans[0]=='s':
            x1=trans.index(" ")
            y1=trans.index(" ",x1+1)
            mat[0][0]=int(trans[2:y1])
            mat[1][1]=int(trans[y1+1:])
            for i in range(0,len(X)):
                X[i]=mat[0][0]*int(X[i])+mat[0][1]*int(Y[i])
                Y[i]=mat[1][1]*int(Y[i])+mat[1][0]*int(X[i])
            plt.plot(X,Y)    
            plt.show()
            print(*X,sep=' ')
            print(*Y,sep=' ')
        elif trans[0]=='T'or trans[0]=='t':
            x1=trans.index(" ")
            y1=trans.index(" ",x1+1)
            mat[0][0]=int(trans[2:y1])
            mat[1][1]=int(trans[y1+1:])
            for i in range(0,len(X)):
                X[i]=mat[0][0]+X[i]
                Y[i]=mat[1][1]+Y[i]
            plt.plot(X,Y)    
            plt.show()
            print(*X,sep=' ')
            print(*Y,sep=' ')
        elif trans[0]=='R' or trans[0]=='r':
            x1=trans.index(" ")
            X1=int(trans[x1+1:])
            rrad =m.radians(X1)       
            for i in range(0,len(X)):
                a=X[i]
                mat[0][0]=m.cos(rrad)
                mat[0][1]=m.sin(rrad)
                mat[1][0]=-m.sin(rrad)
                mat[1][1]=m.cos(rrad)
                X[i]=round(a*mat[0][0])+round(Y[i]*mat[0][1])
                Y[i]= round(a*mat[1][0])+round(Y[i]*mat[1][1])
            plt.plot(X,Y)
            plt.show()
            print(*X,sep=' ')
            print(*Y,sep=' ')
        trans=input()
elif inp =="disc":
    plt.figure()
    plt.ylim(bottom=-5,top=5)
    plt.xlim(left=-5,right=5)
    ax = plt.gca()
    c1,c2,r1=map(int,input().split())
    r2=r1
    theta=0
    ellipse=Ellipse(xy=(c1,c2), width=r2, height=r1)
    ellipse.set_fill(False)
    ax.add_patch(ellipse)
    trans = input()
    while trans!="quit":
        if trans[0]=='S'or trans[0]=='s':
            x1=trans.index(" ")
            y1=trans.index(" ",x1+1)
            X1=int(trans[2:y1])
            Y1=int(trans[y1+1:])
            r1*=X1
            r2*=Y1
            ellipse=Ellipse(xy=(c1,c2), width=r2, height=r1, angle=theta)
            ellipse.set_fill(False)
            ax.add_patch(ellipse)
            print(c1,c2,r1,r2)
        elif trans[0]=='T'or trans[0]=='t':
            x1=trans.index(" ")
            y1=trans.index(" ",x1+1)
            X1=int(trans[2:y1])
            Y1=int(trans[y1+1:])
            c1+=X1
            c2+=Y1
            ellipse=Ellipse(xy=(c1,c2), width=r2, height=r1, angle=theta)
            ellipse.set_fill(False)
            ax.add_patch(ellipse)
            print(c1,c2,r1,r2)
        elif trans[0]=='R' or trans[0]=='r':
            x1=trans.index(" ")
            X1=int(trans[x1+1:])
            rrad =m.radians(X1)
            theta+=X1      
            ellipse=Ellipse(xy=(c1,c2), width=r2, height=r1, angle=theta)
            ellipse.set_fill(False)
            ax.add_patch(ellipse)
            print(c1,c2,r1,r2)
        trans=input()
plt.show()
