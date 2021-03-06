#!/usr/bin/env python
import cv2
import numpy as np
import glob
import time
#### Function to load non-object images and clip them to snippets of size 64x128 to create training images for the object detection SVM ####

def loadImages(path):
    # Return an array of images localized at the given path
    images = [cv2.imread(file) for file in glob.glob(path + '/*png')]
    return images


path = '/home/ras/robot_images/images'

# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(0, 169, 84), 'green':(37, 150, 60), 'blue':(80, 114, 60), 'yellow':(17, 150, 115), 'orange':(5, 190, 130), 'purple':(100,32,81)}
upper = {'red':(10,255,175), 'green':(70,255,190), 'blue':(110,255,170), 'yellow':(25,255,230), 'orange':(18,255,215), 'purple':(180,150,185)}

colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'purple':(210,255,128)}

images = loadImages(path)
#print(images)

for image in images:
    print('New Image')
    print(image)

    #image = cv2.resize(image,(0,0),fx = 0.25,fy = 0.25)
    start = time.time()
    # Blur image
    blurred_image = cv2.GaussianBlur(image,(11,11),0)
    print('Blurred')
    print(blurred_image)
    hsv = cv2.cvtColor(blurred_image,cv2.COLOR_BGR2HSV)
    for key,value in upper.items():

        print(key)
        print(value)
        # Mask the image and use open/close to have smoother conturs
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv,lower[key],upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours and initialize center (x,y) of the object
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts)>0:
            # Find biggest contur:
            c = max(cnts,key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            # Only draw bounding boxes that have a goodsize
            area = w * h
            if area >= 4000 and area <= 35000:
                print('Found: ' + key )
                cv2.rectangle(image,(x,y),(x+w,y+h),colors[key],2)

    end = time.time()
    print(end-start)

    cv2.imshow('display',image)
    cv2.waitKey(0)
