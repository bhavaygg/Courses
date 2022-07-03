# Bhavay Aggarwal
# Roll No - 2018384
# Section B
# Group 1
""" Draws a design with squares and a design
with rings."""

from SimpleGraphics import *
i=0
tilt=0
l=6
# First Figure
MakeWindow(6,bgcolor=DARKGRAY,labels=False)
while i<6:
    DrawRect(0,0,l,l,FillColor=WHITE,EdgeColor=BLUE,EdgeWidth=5,theta=tilt)
    tilt -= 5
    i += 1
    l-= 0.5
# Add more squares...
j=0
x=0
y=1
# Second Figure
MakeWindow(10,bgcolor=WHITE,labels=False)
# Rings
while j<3:
    m=x
    n=y
    global a
    for l in range(j+1):
        DrawDisk(m,n,2,EdgeWidth=1,FillColor=DARKGRAY)
        m+=4
    x -= 2
    y -= 6.928/2
    j+=1
# Add more rings...

ShowWindow()
