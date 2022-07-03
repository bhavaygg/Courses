# DemoMath.py
# CS 1110 (cs-1110profs-L@cornell.edu)
# February, 2016
""" Examines a function that computes approximate square roots."""

import math


def sqrt(x):
    """Returns an approximate square root of x as float.

    Performs five steps of rectangle averaging.

    Precondition: The value of x is a positive number."""
    # As explained in lecture, imagine you have an x-by-1 rectangle,
    # which will thus have area x.
    length = float(x)
	
    # As explained in lecture, we change the length of our rectangle
    # (and then, implicitly, the width to keep the area the same), to make a
    # "more square" rectangle with the same area.
    length = (length + x/length)/2
    length = (length + x/length)/2
    length = (length + x/length)/2
    length = (length + x/length)/2
    length = (length + x/length)/2

    # If the "rectangle" with area x were now a square, then the length
    # of its side would be the sqrt.
    return length


def fourth_root(x):
    """Returns an approximate fourth root of x as float.

    Precondition: The value of x is a positive number."""
    return x # change this for question 6!

# Application Script
if __name__ == '__main__':
    """ A keyboard input framework for checking out sqrt.
    """

    x = input('Enter a number whose square root you want: ')
    x=float(x)
    y1 = math.sqrt(x)
    y2 = sqrt(x) # question 2: change this to "y2 = DemoMath.sqrt(x)"
    
    print ('\n\n           x = %5.2f' % x) # question 7: unindent this
    print ('math.sqrt(x) = %15.12f' %y1)
    print ('     sqrt(x) = %15.12f' %y2)
    z1 = math.sqrt(math.sqrt(x))
    z2 = fourth_root(x)
    print ('\n  True 4th Root = %16.12f' %z1)
    print ('  fourth_root(x) = %15.12f' %z2)
