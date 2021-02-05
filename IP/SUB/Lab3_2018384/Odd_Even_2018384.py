# Name - Bhavay Aggarwal
# Roll No - 2018384
# Section - B
# Group - 1

""" Inserts hyphens into a non-empty input string as follows:
The hyphen splits the first and second halves. 
"""
## We've added a number of so-called "debugging" print statements here

s = input('Enter an string: ')
n = len(s)
print ('n is',n)                
if n==0:
    # Null String
    # final output
    print ('-')
elif n%2 == 0:                           # Line A
    # s has even length
    m = n//2
    print ('even case. m is',m)
    first = s[0:m]                     # Line B
    print ('even case. first is',first)
    second = s[m:]                     # Line C
    print ('even case. second is',second)
    h = first + '-' + second
    # final output
    print (s,'becomes',h)

else :
    # s has odd length
    m=n//2
    print ('odd case. m is',m)
    first = s[:m+1]
    print ('odd case. first is',first)
    second = s[m+1:]
    print ('odd case. second is',second)
    h=first +'-'+second
    # final output
    print (s,'becomes',h)  

