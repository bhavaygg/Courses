def Computation(n,a,b):
	i=0
	sum=0
	x=0
	num=0
	z=0
	if n != int(n) or a != int(a) or b != int(b):
		print("INV")
		z=1
	elif a/10>=1 or b/10>=1:
		print("INV")
		z=1	
	while i<n and z!=1:
		num = a*(10**(x+1)) + b*(10**x) + num
		sum += num
		x = x+2
		i +=1
	if z!= 1:
		print(sum)
n = float(input("n : "))
a = float(input("a : "))
b = float(input("b : "))
Computation(n,a,b)			 