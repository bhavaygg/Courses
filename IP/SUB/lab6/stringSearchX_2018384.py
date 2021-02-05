def Search(s1,s2):
	S1 = s1.lower()
	S2 = s2.lower()
	x=0
	y=0
	i=1
	n = len(s2)
	n1= len(s1)
	if(S1[:n]==S2 and s1[0].isupper()):
		x+=1
		y=1
		i=n
	
	while i<n1 and s2[0].isnumeric() == False:
		if S1[i:i+n] == S2 and y ==1:
			x+=1
			i+=n
		elif S1[i:i+n] == S2:	
			x+=1
			i+=n
		else:
			i+=1	
	print(x)
print("Searching S2 in S1!")
s1 = input("String 1 : ")
s2 = input("String 2 : ")
Search(s1,s2)			 