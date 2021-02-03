s1,s2 = input().split()
n1= len(s1)
n2 = len(s2)
S1= s1.upper()
S2= s2.upper()
if(S1[n1-n2:]==S2 or S2[n2-n1:]==S1):
	print(True)
else: 
	print(False)