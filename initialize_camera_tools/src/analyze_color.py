#!/usr/bin/env python
import cv2
import numpy as np
import glob
#### Function to load non-object images and clip them to snippets of size 64x128 to create training images for the object detection SVM ####

def loadImages(path):
    # Return an array of images localized at the given path
    filenames = [ name for name in glob.glob(path + '/*png')]
    filenames.sort()
    print(filenames)
    images = [cv2.imread(file) for file in filenames]
    return images


path = '/home/ras16/maze_images/hsv_value_clipped/yellow_cube'

images = loadImages(path)
file = open("yellow_cube.txt","w")

for image in images:
    print('New Object!')
    file.write("New Clipped Image \n")
    cv2.imshow('display',image)
    #cv2.waitKey(0)
    blurr = cv2.GaussianBlur(image,(11,11),0)
    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
    cv2.imshow('display',hsv)
    #cv2.waitKey(0)
    channels = cv2.split(hsv)
    for channel in channels:
        min_Val,max_Val,min,max = cv2.minMaxLoc(channel)
        print("Min Value: " +str(min_Val)+"\nMax Value: " + str(max_Val))
        file.write("Min Value: " +str(min_Val)+"\t Max Value: " + str(max_Val)+"\n")
print("Done")
file.close()
