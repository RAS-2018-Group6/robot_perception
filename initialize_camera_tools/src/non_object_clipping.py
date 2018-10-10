#!/usr/bin/env python
import cv2
import numpy as np
import glob
#### Function to load non-object images and clip them to snippets of size 64x128 to create training images for the object detection SVM ####

def loadImages(path):
    # Return an array of images localized at the given path
    images = [cv2.imread(file) for file in glob.glob(path + '/*png')]
    return images


path = '/home/ras/robot_images/non_object_images'
path_save =  '/home/ras/robot_images/clipped_non_objects/cropped_image_'

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
#print(height)
#print(width)
#print type(images[0])
counter = 0

for image in images:
    # Itereate over the height in 20px steps
    for i in range(start_height,end_height,20):
        # Iterate over the width in 20 px steps
        for j in range(start_width,end_width,20):
            cropped_image[:,:,:] = image[i-up_side:i+down_side,j-left_side:j+right_side,:]
            save = path_save + str(counter) + '.png'
            print(save)
            cv2.imwrite(save,cropped_image)
            counter += 1
