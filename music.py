#!/usr/bin/python

#****************************************************************
# Python-OpenCV Based Gesture Detection and Music Player Control
# Author: Alankar Kotwal <alankarkotwal13@gmail.com>
# Main File
#****************************************************************

# Imports
import numpy as np
from config import *
import cv2 as c
import os

# Fancy stuff
os.system("clear")

# Setup video and other stuff
cap = c.VideoCapture(capNo)
frame = np.fliplr(cap.read()[1])

imageY, imageX, _ = frame.shape

leftCenterY, leftCenterX = int(imageY/2), int(imageX/4)
rightCenterY, rightCenterX = int(imageY/2), int(3*imageX/4)

# Collect train data for the left hand
print "[INFO] Put your left hand into the white circle as well as possible and press c"
leftMask = np.zeros((imageX, imageY), dtype="uint8")
c.ellipse(leftMask, (leftCenterX, leftCenterY), (100*ellipseScale, 200*ellipseScale), 0, 0, 360, (1), -1)
while c.waitKey(30) != 'c':
	frame = np.mat(np.fliplr(cap.read()[1]))
	c.ellipse(frame, (leftCenterX, leftCenterY), (100*ellipseScale, 200*ellipseScale), 0, 0, 360, (255, 255, 255), 2)
	c.imshow('Left',frame)
print "[INFO] Left done."

c.destroyAllWindows()

# Collect train data for the right hand		
print "[INFO] Put your right hand into the white circle as well as possible and press c."
rightMask = np.zeros((imageX, imageY), dtype="uint8")
c.ellipse(rightMask, (rightCenterX, rightCenterY), (100*ellipseScale, 200*ellipseScale), 0, 0, 360, (1), -1)
while c.waitKey(30) != 'c':
	frame = np.fliplr(cap.read())
	c.ellipse(frame, (rightCenterX, rightCenterY), (100*ellipseScale, 200*ellipseScale), 0, 0, 360, (255, 255, 255), 2)
	c.imshow('Left',frame)
print "[INFO] Right done."

#while cap.isOpened():
#	pass

cap.release()
c.destroyAllWindows()
