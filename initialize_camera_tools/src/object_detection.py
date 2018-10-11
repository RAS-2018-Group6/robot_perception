#!/usr/bin/env python
import cv2
import numpy as np
import glob
import random
from sklearn import svm
import pickle
import time

### Test code to evaluate object detection on static pictures

def loadImages(path):
    images = [cv2.imread(file) for file in glob.glob(path + '/*png')]
    return images

path = '/home/ras16/catkin_ws/src/robot_perception/initialize_camera_tools/src/test_picutres'
images = loadImages(path)
height = (images[0].shape)[0]
width = (images[0].shape)[1]
color_space = (images[0].shape)[2]
crop_width = 124
crop_height = 248

cropped_image = np.zeros((crop_height,crop_width,color_space))
left_side  = 62
right_side = 62
up_side = 124
down_side = 124

start_width = left_side+1
end_width = width - right_side #- 1
start_height = up_side+1
end_height = height - down_side #- 1


hog = cv2.HOGDescriptor()
filename = 'object_detection_svm.sav'
obj_svm = pickle.load(open(filename,'rb'))

binary_pictures = []

for image in images:
    start_time = time.time()
    binary_picture = np.zeros((height,width))
    # Itereate over the height in 20px steps
    for i in range(start_height,end_height,50):
        # Iterate over the width in 20 px steps
        for j in range(start_width,end_width,50):
            # Cropp the image
            cropped_image[:,:,:] = image[i-up_side:i+down_side,j-left_side:j+right_side,:]
            # Downsize the image
            res_image = cv2.resize(image,(64,128))
            #Compute features
            feature = hog.compute(res_image)
            feature = (feature.ravel()).reshape(1,-1)
            #Calculate result
            result = obj_svm.predict(feature)
            if result == 1:
                print(result)
                binary_picture[i,j] = 1
    binary_pictures.append(binary_picture)
    end_time = time.time()
    print(end_time-start_time)

counter = 0
for image in binary_pictures:
    cv2.imshow('dsplay',image)
    cv2.imshow('display_2',images[counter])
    cv2.waitKey(0)
    counter += 1
