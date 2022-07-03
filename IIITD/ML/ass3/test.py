import numpy as np
from Q1 import MyNeuralNetwork

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidg(x):
    temp= sigmoid(x)
    #print(temp)
    return temp*(1-temp)

ots =[0,1,2]
input_arr= np.array([[10,20,30],[20,30,40],[40,50,60]])
weight = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],dtype=np.float32)
optt = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)

'''
for i in range(0,100):
    err=[]
    for n,k in  enumerate(input_arr):
        inp = input_arr[n]
        op=optt[n]
        z1 = np.dot(inp,weight[0].T) + 1
        o1 = sigmoid(z1)
        zo = np.dot(inp,weight[1].T) + 1
        o2 = sigmoid(zo)
        #print(o2)
        error = np.sum(np.power((o2-op),2))
        #error = (np.dot(op,-np.log(o2+1e-7))+np.dot((1-op),-np.log(1-o2+1e-7)))
        err.append(error)
        dcost_aout = o2-op
        daout_dzout =  sigmoidg(zo)
        dzo_do = o1
        print(dcost_aout,daout_dzout,dzo_do)
        dcost_wo = np.dot(dcost_aout,np.dot(daout_dzout,dzo_do))
        #print(daout_dzout,dzo_do,dcost_wo)
        dcost_dzo = dcost_aout*daout_dzout
        dzo_da1 = weight[1]
        dcost_da1 = np.dot(dcost_dzo,dzo_da1.T)
        da1_dz1 = sigmoidg(z1)
        dz1_dw1 = inp
        dcost_w1 = np.dot(dz1_dw1.T,np.dot(da1_dz1,dcost_da1))

        weight[0]-= 0.05*dcost_w1
        weight[1]-= 0.05*dcost_wo
        print(dcost_wo)
        break
    break
        #break
    #break
    print(np.mean(np.array(err)))
'''
num_layers=2
num_units=5
num_outputs=3
num_inputs =3


weight_inp = np.zeros((num_inputs,num_units))
weights= np.zeros((num_layers-1,num_units,num_units))
weight_out = np.zeros((num_units,num_outputs))


lr=0.05
'''
def forward(arr):
    z=[]
    a=[]
    costs=[]
    for i in range(0,num_layers+1):
        if i==0:
            temp_z = np.dot(weight_inp.T,inp) + 1
        elif i==num_layers:
            temp_z = np.dot(weight_out.T,a[len(a)-1]) + 1
        else:
            temp_z = np.dot(weights[i-1].T,a[len(a)-1]) + 1
        temp_a = sigmoid(temp_z)
        z.append(temp_z)
        a.append(temp_a)
    return z,a

def backward(z_arr,a_arr,input_arr,out):
    global weights,weight_inp,weight_out
    dcost_da=[]
    da_dz=[]
    dz_dw=[]
    dcost_dw=[]
    dcost_dz=[]
    dz_da=[]
    dcosts=[]
    temp_dcost_dz=np.zeros(5)
    temp_dz_da=np.zeros(5)
    for layer in range(num_layers,-1,-1):
        lr=0.05
        if layer==num_layers:
            temp_dcost_da=activations[len(activations)-1]-out
        else:
            temp_dcost_da = np.dot(temp_dz_da,temp_dcost_dz)
        temp_da_dz=sigmoidg(z_arr[layer])
        if layer!=0:
            temp_dz_dw=activations[layer-1]
            temp_dcost_dz = temp_dcost_da*temp_da_dz
            if layer==num_layers:
                temp_dz_da = weight_out
            else:
                temp_dz_da = weights[layer-1]
        else:
            temp_dz_dw=input_arr
        dcost_da.append(temp_dcost_da)
        da_dz.append(temp_da_dz)
        dz_dw.append(temp_dz_dw)

        temp1=np.array(temp_dz_dw).reshape((1,temp_dz_dw.shape[0]))
        temp2=np.array(temp_da_dz*temp_dcost_da).reshape((1,np.array(temp_da_dz*temp_dcost_da).shape[0]))
        temp_dcost_dwo=np.dot(temp1.T,temp2)
        #print(temp_dcost_dwo)
        dcosts.append(temp_dcost_dwo)
        if layer==num_layers:
            #print(weight_out,temp_dcost_dwo)
            weight_out-= lr*temp_dcost_dwo
        elif layer==0:
            #print(weight_inp,temp_dcost_dwo)
            weight_inp -= lr*temp_dcost_dwo
        else:
            #print(weights[layer-1],temp_dcost_dwo)
            weights[layer-1]-= lr*temp_dcost_dwo
        #print("w",weight_out)
        #print(layer,temp_dcost_da,temp_da_dz,temp_dcost_dz,temp_dz_dw,temp_dz_da)

def backward(self,z_arr,acts,input_arr,out):
        dcost_da=[]
        da_dz=[]
        dz_dw=[]
        dcost_dw=[]
        dcost_dz=[]
        dz_da=[]
        dcosts=[]
        temp_dcost_dz=np.zeros(5)
        temp_dz_da=np.zeros(5)
        for layer in range(self.num_layers,-1,-1):
            if layer==self.num_layers:
                temp_dcost_da=acts[len(acts)-1]-out
            else:
                temp_dcost_da = np.dot(temp_dz_da,temp_dcost_dz)

            temp_da_dz=self.activatorgrad(z_arr[layer])

            if layer!=0:
                temp_dz_dw=acts[layer-1]
                if layer!=0:
                    temp_dcost_dz = temp_dcost_da*temp_da_dz
                #if layer==self.num_layers:
                #    temp_dz_da = self.weight_out
                #else:
                #    temp_dz_da = self.weights[layer-1]
                temp_dz_da=self.tweight[layer]
            else:
                temp_dz_dw=input_arr
            
            dcost_da.append(temp_dcost_da)
            da_dz.append(temp_da_dz)
            dz_dw.append(temp_dz_dw)
            
            temp1=np.array(temp_dz_dw,dtype=np.float64).reshape((1,temp_dz_dw.shape[0]))
            temp2=np.array(temp_da_dz*temp_dcost_da,dtype=np.float64).reshape((1,np.array(temp_da_dz*temp_dcost_da).shape[0]))
            temp_dcost_dwo=np.dot(temp1.T,temp2)

            dcosts.append(temp_dcost_dwo)
            #if layer==self.num_layers:
            #    self.weight_out      -= (self.lr)*temp_dcost_dwo         
            #elif layer==0:
            #    self.weight_inp      -= self.lr*temp_dcost_dwo
            #else:
            #    self.weights[layer-1]-= self.lr*temp_dcost_dwo
#self.weight_inp = np.zeros((x,self.num_units))
#self.weights    = np.zeros((self.num_layers-1,self.num_units,self.num_units))
#self.weight_out = np.zeros((self.num_units,y))

epochs = 10
for i in range(0,epochs):
    errors=[]
    for n,inp in enumerate(input_arr):
        z,activations=forward(inp)

        error = np.sum(np.power((optt[n]-activations[len(activations)-1]),2))
        backward(z,activations,input_arr[n],optt[n])
        errors.append(error)
    print(np.mean(np.array(errors)))
'''
##input_arr= np.array([[10,20,30],[20,30,40],[40,50,60]])
optt = np.array([[1.5,2.5,0.5],[0.5,1.5,0.8],[3,4,1]],dtype=np.float32)
#trying = MyNeuralNetwork(3,5,'sigmoid',0.05,'zero',0,1)
#trying.fit(input_arr,optt)
input_arr= np.array([[0,0],[0,1],[1,0],[1,1]])
optt = np.array([[0],[1],[1],[0]],dtype=np.float32)


trying = MyNeuralNetwork(3,2,'tanh',0.1,'normal',200,100)
trying.fit(input_arr,optt)
print(trying.predict(input_arr))
#trying = MyNeuralNetwork(3,5,'sigmoid',0.05,'normal',0,100)
#trying.fit(input_arr,optt)
#trying = MyNeuralNetwork(3,5,'relu',0.05,'zero',0,100)
#trying.fit(input_arr,optt)
#trying = MyNeuralNetwork(3,5,'tanh',0.05,'normal',0,100)
#trying.fit(input_arr,optt)
#trying = MyNeuralNetwork(3,5,'linear',0.05,'zero',0,1)
#trying.fit(input_arr,optt)
#
#from keras.datasets import mnist
#
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#y_train=y_train.reshape((y_train.shape[0],1))
#X_train=X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
#print(X_train.shape)
## rescale the data, use the traditional train/test split
#
#trying = MyNeuralNetwork(3,100,'sigmoid',0.05,'zero',0,100)
#trying.fit(X_train, y_train)
'''

            '''
'''
def forward(self,arr):
        z=[]
        a=[]
        for i in range(0,self.num_layers+1):
            if i==0:
                temp_z = np.dot(self.tweight[i].T,arr)
            else:
                temp_z = np.dot(self.tweight[i].T,a[len(a)-1])

            temp_a = self.activator(temp_z)
            z.append(temp_z)
            a.append(temp_a)
        return z,a

    def backward(self,z_arr,acts,input_arr,out):
        dcost_da=[]
        da_dz=[]
        dz_dw=[]
        dcost_dw=[]
        dcost_dz=[]
        dz_da=[]
        dcosts=[]
        temp_dcost_dz=np.zeros(5)
        temp_dz_da=np.zeros(5)
        for layer in range(self.num_layers,-1,-1):
            if layer==self.num_layers:
                temp_dcost_da=acts[len(acts)-1]-out
            else:
                temp_dcost_da = np.dot(temp_dz_da,temp_dcost_dz)

            temp_da_dz=self.activatorgrad(z_arr[layer])

            if layer!=0:
                temp_dz_dw=acts[layer-1]
                temp_dcost_dz = temp_dcost_da*temp_da_dz
                temp_dz_da=self.tweight[layer]
            else:
                temp_dz_dw=input_arr
            
            dcost_da.append(temp_dcost_da)
            da_dz.append(temp_da_dz)
            dz_dw.append(temp_dz_dw)
            
            temp1=np.array(temp_dz_dw,dtype=np.float64).reshape((1,temp_dz_dw.shape[0]))
            temp2=np.array(temp_da_dz*temp_dcost_da,dtype=np.float64).reshape((1,np.array(temp_da_dz*temp_dcost_da).shape[0]))
            temp_dcost_dwo=np.dot(temp1.T,temp2)

            dcosts.append(temp_dcost_dwo)
            self.tweight[layer]      -= self.lr*temp_dcost_dwo
            '''
'''
    def forward(self,arr):
        z=[]
        a=[]
        for i in range(0,self.num_layers+1):
            if i==0:
                #temp_z = np.dot(self.tweight[i].T,arr)
                temp_z = np.matmul(self.tweight[i].T,arr.T)
            else:
                #temp_z = np.dot(self.tweight[i].T,a[len(a)-1])
                temp_z = np.matmul(self.tweight[i].T,a[len(a)-1])

            temp_a = self.activator(temp_z)
            z.append(temp_z)
            a.append(temp_a)
        return z,a

    def backward(self,z_arr,acts,input_arr,out):
        dcost_da=[]
        da_dz=[]
        dz_dw=[]
        dcost_dw=[]
        dcost_dz=[]
        dz_da=[]
        dcosts=[]
        temp_dcost_dz=np.zeros(5)
        temp_dz_da=np.zeros(5)
        delta=acts[-1]-out.T
        temp_W = [np.zeros(W.shape) for W in self.tweight]
        for layer in range(self.num_layers,-1,-1):
            
            temp_da_dz=self.activatorgrad(z_arr[layer])
            print("zulul",temp_da_dz.shape,acts[layer-1].shape)

            if layer==self.num_layers:
                temp_dcost_da=acts[-1]-out.T
            else:
                print(temp_dcost_dz.T.shape,temp_dz_da.T.shape)
                print(temp_dcost_dz.T,temp_dz_da)
                temp_dcost_da = np.matmul(temp_dcost_dz.T,temp_dz_da.T)

            if layer!=0:
                temp_dz_dw=acts[layer-1]
                print(temp_da_dz,temp_dcost_da,temp_dz_dw)
                print(temp_da_dz.shape,temp_dcost_da.shape,temp_dz_dw.shape,temp_dcost_da.shape)
                temp_dcost_dz = np.dot(temp_da_dz,temp_dcost_da.T)
                temp_dz_da=self.tweight[layer]
            else:
                temp_dz_dw=input_arr
            
            dcost_da.append(temp_dcost_da)
            da_dz.append(temp_da_dz)
            dz_dw.append(temp_dz_dw)
            #print(temp_da_dz.shape,temp_dcost_da.shape,temp_dz_dw.shape)
            temp1=np.array(temp_dz_dw,dtype=np.float64)#.reshape((len(input_arr),temp_dz_dw.shape[0]))
            temp2=np.array(np.dot(temp_da_dz.T,temp_dcost_da),dtype=np.float64)
            #print(temp1,temp1.shape,temp2.shape,temp2)
            #.reshape((np.array(temp_da_dz*temp_dcost_da).shape[1],np.array(temp_da_dz*temp_dcost_da).shape[0]))
            temp_dcost_dwo=np.dot(temp1,temp2)
            print(temp_dcost_dwo.shape)
            dcosts.append(temp_dcost_dwo)
            #print(self.tweight[layer].shape,np.mean(self.lr*temp_dcost_dwo,axis=1),np.mean(self.lr*temp_dcost_dwo,axis=1).shape)
            
            if layer==self.num_layers:
                temp_dcost_da=acts[len(acts)-1]-out.T
            else:
                temp_dcost_da = np.matmul(temp_dcost_da,temp_da_dz)
            temp_dcost_dz= temp_dcost_dz.reshape((temp_dcost_dz.shape[0],1))
            print(input_arr.shape,temp_dcost_da.shape,temp_dcost_dz.shape,np.array(self.tweight[layer]).shape)
            print(np.matmul(np.dot(temp_dcost_dz,temp_dcost_da),np.array(self.tweight[layer]).T))
            temp_dcost_dwo =input_arr
            self.tweight[layer]      -= np.mean(self.lr*temp_dcost_dwo,axis=1).reshape((np.mean(self.lr*temp_dcost_dwo,axis=1).shape[0],1))
            

            print("layer",layer)
            print(delta.shape,self.tweight[layer].shape)
            delta = np.dot(delta.T,self.tweight[layer].T)
            print(delta.shape,self.activatorgrad(z_arr[layer-1]).shape,)
            delta = np.dot(delta.T,self.activatorgrad(z_arr[layer]).T)
                #delta = np.dot(delta,d2.T)
            print(delta,delta.shape)
        
            self.tweight[layer]      -= delta#np.mean(delta,axis=1).reshape((self.tweight[layer].shape[0],self.tweight[layer].shape[1]))
            

            #self.tweight[layer]      -= np.mean(self.lr*temp_dcost_dwo,axis=1).reshape((np.mean(self.lr*temp_dcost_dwo,axis=1).shape[0],1))
'''