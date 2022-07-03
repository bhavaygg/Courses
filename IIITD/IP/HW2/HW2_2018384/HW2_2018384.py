# CSE 101 - IP HW2
# K-Map Minimization 
# Name: Bhavay Aggarwal
# Roll Number: 2018384
# Section: B
# Group: 1

listd=[]
#above list stores dont cares as they are modified in various functions
#converting string input to lsit of binary
def converttolist(x,n):
	list3=[]
	list4=[]
	flist=[]
	x1 = x.index('(')
	x2 = x.index(')')
	x5 = x.index('d')
	if x[x5+2]!='-':
		x3 = x.index('(' , x1+1)
		x4 = x.index(')' , x2+1)
		d = x[x3+1:x4]
		d = d.replace(',' ,' ')
		list2 = d.split()
		for i in list2:
			z = bin(int(i))[2:]
			while len(z)< n:
				z = '0'+str(z)
			list4.append(z)
	s = x[x1+1:x2] 
	s= s.replace(',' ,' ')
	list1 =s.split()
	for i in list1:
		z = bin(int(i))[2:]
		while len(z)< n:
			z = '0'+str(z)
		list3.append(z)
	flist = list3+list4
	return flist
#returns only dont cares
def converttolist2(x,n):
	list4=[]
	x5 = x.index('d')
	x1 = x.index('(')
	x2 = x.index(')')
	if x[x5+2]!='-':
		x3 = x.index('(' , x1+1)
		x4 = x.index(')' , x2+1)
		d = x[x3+1:x4]
		d = d.replace(',' ,' ')
		list2 = d.split()
		for i in list2:
			z = bin(int(i))[2:]
			while len(z)< n:
				z = '0'+str(z)
			list4.append(z)
	return list4		
#returns binary which aernt dont care
def converttolist1(x,n):
	list3=[]
	x1 = x.index('(')
	x2 = x.index(')')
	s = x[x1+1:x2] 
	s= s.replace(',' ,' ')
	list1 =s.split()
	for i in list1:
		z = bin(int(i))[2:]
		while len(z)< n:
			z = '0'+str(z)
		list3.append(z)
	return list3
#divide binary into groups based on number of zeroes
def group0(x,n):
	count=0
	gr0=[]
	gr4=[]
	gr1=[]
	gr2=[]
	gr3=[]
	for i in x:
		for j in i:
			if j=='1':
				count +=1
		if count==0:
			gr0.append(i)
		elif count==1:
			gr1.append(i)
		elif count==2:
			gr2.append(i)
		elif count==3 and n>=3:
			gr3.append(i)
		elif count==4 and n==4:
			gr4.append(i)			
		count=0		
	grb=[]
	if n==4:	
		grb.extend((gr0,gr1,gr2,gr3,gr4))
	elif n==3:
		grb.extend((gr0,gr1,gr2,gr3))
	elif n==2:
		grb.extend((gr0,gr1,gr2))		
	return grb	
#group binary based on difference in terms 
def prime(g,h,n):
	global listd
	count=0
	pos=0
	gr0=[]
	gr4=[]
	gr1=[]
	gr2=[]
	gr3=[]
	for i in g[0]:
		for j in g[1]:
			for z in range(0,n):
				if i[z]!=j[z]:
					count+=1
					pos=z
			if count==1:
				gr0.append(i[:pos]+'-'+i[pos+1:])
				if i in h:
					listd.append(i[:pos]+'-'+i[pos+1:])
				if j in h:
					listd.append(i[:pos]+'-'+i[pos+1:])	
			count=0	
	for i in g[1]:
		for j in g[2]:
			for z in range(0,n):
				if i[z]!=j[z]:
					count+=1
					pos=z
			if count==1:
				gr1.append(i[:pos]+'-'+i[pos+1:])
				if i in h:
					listd.append(i[:pos]+'-'+i[pos+1:])
				if j in h:
					listd.append(i[:pos]+'-'+i[pos+1:])	
			count=0
	if n>=3:
		for i in g[2]:
			for j in g[3]:
				for z in range(0,n):
					if i[z]!=j[z]:
						count+=1
						pos=z
				if count==1:
					gr2.append(i[:pos]+'-'+i[pos+1:])
					if i in h:
						listd.append(i[:pos]+'-'+i[pos+1:])
					if j in h:
						listd.append(i[:pos]+'-'+i[pos+1:])	
				count=0						
	if n==4:
		for i in g[3]:
			for j in g[4]:
				for z in range(0,n):
					if i[z]!=j[z]:
						count+=1
						pos=z
				if count==1:
					gr3.append(i[:pos]+'-'+i[pos+1:])
					if i in h:
						listd.append(i[:pos]+'-'+i[pos+1:])
					if j in h:
						listd.append(i[:pos]+'-'+i[pos+1:])	
				count=0					
	if n ==4:
		gr4.extend((gr0,gr1,gr2,gr3))
	elif n==3:
		gr4.extend((gr0,gr1,gr2))	
	elif n==2:
		gr4.extend((gr0,gr1))	
	return gr4 					

#group binary based on difference in terms again
def prime2(g,h,n):
	global listd
	count=0
	pos=0
	gr0=[]
	gr1=[]
	gr2=[]
	gr3=[]
	gr4=[]
	gr5=[]
	for i in g[0]:
		for j in g[1]:
			for z in range(0,n):
				if i[z]!=j[z]:
					count+=1
					pos=z
			if count==1:
				gr0.append(i[:pos]+'-'+i[pos+1:])
				gr5.append(i)
				gr5.append(j)
				if i in listd:
					listd.append(i[:pos]+'-'+i[pos+1:])
				if j in listd:
					listd.append(i[:pos]+'-'+i[pos+1:])	
			count=0	
	if n>=3:
		for i in g[1]:
			for j in g[2]:
				for z in range(0,n):
					if i[z]!=j[z]:
						count+=1
						pos=z
				if count==1:
					gr1.append(i[:pos]+'-'+i[pos+1:])
					gr5.append(i)
					gr5.append(j)
					if i in listd:
						listd.append(i[:pos]+'-'+i[pos+1:])
					if j in listd:
						listd.append(i[:pos]+'-'+i[pos+1:])	
				count=0
	if n==4:
		for i in g[2]:
			for j in g[3]:
				for z in range(0,n):
					if i[z]!=j[z]:
						count+=1
						pos=z
				if count==1:
					gr2.append(i[:pos]+'-'+i[pos+1:])
					gr5.append(i)
					gr5.append(j)
					if i in listd:
						listd.append(i[:pos]+'-'+i[pos+1:])
					if j in listd:
						listd.append(i[:pos]+'-'+i[pos+1:])				
				count=0										
	if n==4:
		gr3.extend((gr0,gr1,gr2))
	elif n==3:
		gr3.extend((gr0,gr1))	
	for i in gr3:
		for j in i:
			if j not in gr4:
				gr4.append(j)
	for i in g:
		for j in i:
			if j not in gr5:
				gr4.append(j)									
	return gr4							
#group binary based on difference in terms again
def prime3(g,h):
	count=0
	pos=0
	count1=0
	gr0=[]
	gr1=[]
	gr2=[]
	gr3=[]
	for i in g:
		for j in g:
			if i!=j:
				for z in range(0,4):
					if i[z]!=j[z] and i[z]!=j[z]!='-':
						count+=1
						pos=z
				if count==1:
					gr0.append(i[:pos]+'-'+i[pos+1:])
					gr3.append(j)
					gr3.append(i)
					for k in gr0[-1]:
						if k == '-':
							count1+=1
						if count1 == 3:                
							gr2.append(i)
							gr2.append(j)
				count=0
				count1=0
	for i in gr0:
		if i not in gr1:
			gr1.append(i)
	for i in gr2:
		if i in gr1:
			gr1.remove(i)
	for i in g:
			if i not in gr3:
				gr1.append(i)							
	return gr1
#convert final answer to string format
def answer(b,n):
	gr=[]
	h=['W','X','Y','Z']
	ha=['W`','X`','Y`','Z`']
	ch=''
	for i in b:
		for j in range(0,n):
			if i[j]=='1':
				ch+=h[j]
			elif i[j]=='0':
				ch+=ha[j]
		ch+='+'	
	if not b:
		ch='11'		
	return ch[:-1]				
#finding essential primes by seeing the minterms occupied
def eprime(t,y,u,p):
	n=len(y)
	egr=[]
	t1=[]
	t2=[]
	t3=[]
	dont=[]
	do=[]
	num=[]
	count=0
	count1=0
	for i in t:
		for j in i:
			if j=='-':
				count+=1
		if count==1:
			x = i.replace('-','0')+i
			egr.append(x)
			x = i.replace('-','1')+i
			egr.append(x)
		elif count==2:
			x = i.replace('-','0')+i
			egr.append(x)
			x = i.replace('-','1')+i
			egr.append(x)
			q=i.index('-')
			w=i.index('-',q+1)
			x=i[:q]+'0'+i[q+1:w]+'1'+i[w+1:]+i
			egr.append(x)
			x=i[:q]+'1'+i[q+1:w]+'0'+i[w+1:]+i
			egr.append(x)
		elif count==3:
			x = i.replace('-','0')+i
			egr.append(x)
			x = i.replace('-','1')+i
			egr.append(x)
			x = i.replace('-','1',2)
			x = i.replace('-','0',-1)
			egr.append(x)
			x = i.replace('-','0',2)
			x = i.replace('-','1',-1)
			egr.append(x)
			x = i.replace('-','0',1)
			x = i.replace('-','1',-2)
			egr.append(x)
			x = i.replace('-','1',1)
			x = i.replace('-','0',-2)
			egr.append(x)
			x = i.replace('-','0',1)
			x = i.replace('-','1',-2)
			x = i.replace('-','0',-1)
			egr.append(x)
			x = i.replace('-','1',1)
			x = i.replace('-','0',-2)
			x = i.replace('-','1',-1)
			egr.append(x)	
		else:
			egr.append(i)	
		count=0		
	for i in egr:
		if i[0:p] not in t1:
			t1.append(i)	
	for j in t1:
		if j[p:] in listd:
			dont.append(j)
		else:
			do.append(j)			
	for k in y:
		for j in do:
			if j[0:p]==k:
				t2.append(j[p:])
				count1+=1
		if count1==0:
			num.append(k)
		count1=0
	for k in num:
		for l in dont:
			if l[0:p]==k:
				t2.append(l[p:])
				count1+=1
			if count1==1:
				break
		count1=0											
	for k in t2:
		if k not in t3:
			if k!='':
				t3.append(k)		
	return t3			
def minFunc(numVar, s):
	"""
		This python function takes function of maximum of 4 variables
		as input and gives the corresponding minimized function(s)
		as the output (minimized using the K-Map methodology),
		considering the case of Don’t Care conditions.

	Input is a string of the format (a0,a1,a2, ...,an) d(d0,d1, ...,dm)
	Output is a string representing the simplified Boolean Expression in
	SOP form.

		No need for checking of invalid inputs.
		
	Do not include any print statements in the function.
	"""
	n = numVar
	no=[]
	lista = converttolist(s,numVar)
	listo = converttolist1(s,numVar)
	listb = converttolist2(s,numVar)
	if(numVar==4):
		grp = group0(lista,numVar)
		grp = prime(grp,listb,numVar)
		grp1 = prime2(grp,listb,numVar)
		if not grp1:
			for i in grp:
				for j in i:
					no.append(j)
				no = eprime(no,listo,listb,numVar)
				ans = answer(no,numVar)
		else:	
			grp2 = prime3(grp1,listb)
			if not grp2:
				grp1 = eprime(grp1,listo,listb,numVar)
				ans = answer(grp1,numVar)
			else:
				grp2 = eprime(grp2,listo,listb,numVar)
				ans = answer(grp2,numVar)
		return ans
	elif numVar==3:
		grp = group0(lista,numVar)
		grp = prime(grp,listb,numVar)
		grp1 = prime2(grp,listb,numVar)
		grp1 = eprime(grp1,listo,listb,numVar)
		ans = answer(grp1,numVar)
		return(ans)
	elif numVar==2:
		grp = group0(lista,numVar)
		grp = prime(grp,listb,numVar)
		grp1 = prime2(grp,listb,numVar)
		grp1 = eprime(grp1,listo,listb,numVar)
		ans = answer(grp1,numVar)	
		return ans
	else:
		return("Wrong input!!!!")

	
	
	

	
