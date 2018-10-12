#!/usr/bin/env python
import cv2
import numpy as np
import glob
#### Function to load non-object images and clip them to snippets of size 64x128 to create training images for the object detection SVM ####

def loadImages(path):
    # Return an array of images localized at the given path
    images = [cv2.imread(file) for file in glob.glob(path + '/*jpg')]
    return images


path = '/home/ras/robot_images/hsv_data'

images = loadImages(path)

for image in images:
    print('New Object!')
    cv2.imshow('display',image)
    cv2.waitKey(0)
    blurr = cv2.GaussianBlur(image,(11,11),0)
    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
    cv2.imshow('display',hsv)
    cv2.waitKey(0)
    channels = cv2.split(hsv)
    for channel in channels:
        min_Val,max_Val,min,max = cv2.minMaxLoc(channel)
        print("Min Value: " +str(min_Val)+"\nMax Value: " + str(max_Val))
