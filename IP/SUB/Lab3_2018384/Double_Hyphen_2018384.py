# Hyphenator_broken.py
# the CS 1110 profs (cs-1110profs-L@cornell.edu)
# Feb 2016

""" Inserts hyphens into a non-empty odd-length input string as follows:
A hyphen is inserted on either side of the middle character.

Example: "abcde" becomes "ab-c-de"

"""
### This program intentionally has at least one error in it!

s = input('Enter an odd-length string: ')

n = len(s)

if n==0:
  # Null String
  print("Null String")
elif n==1 :
  # Single Character String
  print("Single Character String")
elif n%2!=0 :
  # s has odd length
  m = int(n/2)
  print(m)
  print ('odd case. m is',m) 
  first = s[0:m]
  print ('odd case. first is',first)
  middle = s[m]
  print ('odd case. middle is',middle)
  second = s[m+1:]
  print ('odd case. second is',second)

  h = first+'-'+middle+'-'+second

  # final output
  print (s,'becomes',h)
else:
   print('Even Character String')