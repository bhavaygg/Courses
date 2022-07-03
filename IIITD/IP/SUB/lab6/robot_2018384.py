import math
x=0
y=0
Ask = input("Want to enter new position?(y/n) :")
if(Ask == 'y'):
	x = int(input("New X: "))
	y = int(input("New Y: "))
x1=x
y1=x	
Com = input("Direction and Distance : ")
while Com!='STOP':
	z = Com.index(' ')
	Dir = Com[:z]
	Num = float(Com[z+1:])
	if(Num != int(Num)):
		print("TRY AGAIN!")
	elif(Dir != "UP" and Dir != "DOWN" and Dir != "RIGHT" and Dir != "LEFT"):
		print("TRY AGAIN!")	
	elif Dir == "UP":
		y = y+ Num
	elif Dir == "DOWN":
		y = y - Num
	elif Dir == 'LEFT':
		x = x - Num
	elif Dir == 'RIGHT':
		x = x+ Num	
	Com = input("Direction and Distance : ")	
print("Initial x = " + str(x1),"Initial y = "+str(y1))
print("Final x = "+ str(x),"Final y = "+str(y))
print("Distance :"+str(math.sqrt((x-x1)**2+(y-y1)**2)))			