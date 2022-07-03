# Bhavay Aggarwal
# Roll No - 2018384
# Section B
# Group 1
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
    DrawRect(x,y+W/3,1.5*W,W/3,FillColor=[0.738, 0.0860, 0.102],EdgeWidth=1)
    DrawRect(x,y,1.5*W,W/3,FillColor=[0.9, 0.9, 0.9],EdgeWidth=1)
    DrawRect(x,y-W/3,1.5*W,W/3,FillColor=[0.288, 0.395, 0.317],EdgeWidth=1)
    DrawDisk(x,y,W/6,EdgeColor=BLUE)
    a = 0 
    while a < 2*math.pi:
        DrawLineSeg(0,0,W/6*math.cos(a),W/6*math.sin(a), LineWidth=1,LineColor=BLUE)
        a+= math.pi/12
    #Complete this


# Application Script
if __name__ == '__main__':
     # Display the Indian flag on a black background.
    MakeWindow(10,bgcolor=BLACK,labels=False)
    DrawIndianFlag(0,0,8)
    ShowWindow()

