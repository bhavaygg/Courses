import subprocess
import matplotlib.pyplot as plt
name= "SRR390728"
#bashCommand = "fastq-dump --split-3 "+name
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()
filename=name+"_1.fastq"
with open(filename) as fp:
	fp=fp.read().splitlines()
	min_len=10000000
	for i in range(0,len(fp),4):		
		length = int(fp[i][fp[i].index("length=")+7:])
		if length<min_len:
			min_len=length
	length=min_len
	print(fp[:10],fp[-10:])
	print(length)
	print(len(fp)/4)
	A=[]
	T=[]
	C=[]
	G=[]
	dic={}
	for i in range(0,int(length)):
		dic[i]={"A":0,"T":0,"C":0,"G":0,"N":0}
	for i in range(1,len(fp),4):
		#print(fp[i])
		for n,j in enumerate(fp[i]):
			dic[n][j]+=1
	for i in dic:
		A.append(dic[i]["A"])
		T.append(dic[i]["T"])
		C.append(dic[i]["C"])
		G.append(dic[i]["G"])
	plt.plot(range(len(A)),A,"b.-",label="A")
	plt.plot(range(len(T)),T,"g.-",label="T")
	plt.plot(range(len(C)),C,"y.-",label="C")
	plt.plot(range(len(G)),G,"r.-",label="G")
	plt.legend(loc="upper left")
	plt.show()