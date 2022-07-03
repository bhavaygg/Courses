Num = input("Enter Hexadecimal number: ")
n= len(Num)
list =[]
for i in range (0,n):
	if Num[i]=='A':
		list.append(10)
	elif Num[i]=='B':
		list.append(11)
	elif Num[i]=='C':
		list.append(12)
	elif Num[i]=='D':
		list.append(13)
	elif Num[i]=='E':
		list.append(14)
	elif Num[i]=='F':
		list.append(15)
	else:
		list.append(Num[i])	
Dec=0
for j in list:
	Dec += int(j)*(16**(n-1))
	n-=1
if(Dec%7==0):
        print("Divisible by 7")
        Q = int(Dec//7)
        print("Decimal Number: "+str(Dec))
        S=''
        while Q!=0:
                T=Q%16
                if T==10:
                    T='A'
                elif T==11:
                    T='B'
                elif T==12:
                    T='C'
                elif T==13:
                    T='D'
                elif T==14:
                    T='E'
                elif T==15:
                    T='F'
                else:
                    T=str(T)
                S+=T
                k=k//16
    print('Quotient is',g[::-1])
else:
	print("Not divisible by 7")    
