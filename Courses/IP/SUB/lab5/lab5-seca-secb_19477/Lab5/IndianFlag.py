# IndianFlag.py
# CSE 101 (vikram@iiitd.ac.in)
# August, 2016
""" Contains a function for drawing the Indian
flag and an Application Script  that can be used to check it out
"""

from SimpleGraphics import *
import math


def DrawIndianFlag(x,y,W):
	""" Draws the Indian Flag.  W is the vertical dimension
	The center of the large flag is at (x,y).

	Precondition: x,y, and W are numbers and W>0.
	"""
	#Complete this


# Application Script
if __name__ == '__main__':
     # Display the Indian flag on a black background.
    MakeWindow(10,bgcolor=BLACK,labels=False)
    DrawIndianFlag(0,0,8)
    ShowWindow()

