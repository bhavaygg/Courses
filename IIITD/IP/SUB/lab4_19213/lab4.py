#Name - Bhavay Aggarwal	
#Roll Number - 2018384
#Section - B
#Group - 1

def second_max(a,b,c,d):
    e = max(a,b,c,d)
    if(e == a):
        return max(b,c,d)
    elif(e == b):
        return max(a,c,d)
    elif(e == c):
        return max(b,a,d)
    else :
        return max(a,b,c) 

def admission(p,c,m,avg):
    if(p>=80 and c>=80 and m>=80 and avg>=80):
        return 1
    else: 
        return -1     

def triangleType(s1,s2,s3):
    if(s1==s2==s3):
        return 3
    elif(s1!=s2 and s2!=s3 and s1!=s3):
        return 1
    else: 
        return 2    

def validSides(a,b,c):
    if(a+b>c and a+c>b and b+c>a):
        return True
    else :
        return False

def characterCheck(a):                                          
    if(a.isalpha()):
        return 3       
    elif (a.isnumeric()):   
        return 2
    else:
        return 1    