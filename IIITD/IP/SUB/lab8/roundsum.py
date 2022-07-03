x,y,z = input().split()
x=int(x)
y=int(y)
z=int(z)
if x%10>=5:
	x = x + (10-(x%10))
else:
	x = x - (x%10)
if y%10>=5:
	y = y + (10-(y%10))
else:
	y = y - (y%10)
if z%10>=5:
	z = z + (10-(z%10))
else:
	z = z - (z%10)
print(x+y+z)	
