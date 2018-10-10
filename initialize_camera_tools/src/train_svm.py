#!/usr/bin/env python
import cv2
import numpy as np
import glob
import random
from sklearn import svm
#### Script to train the SVM ####

def loadImages(path):
    # Return an array of images localized at the given path
    images = [cv2.imread(file) for file in glob.glob(path + '/*png')]
    resized_images = []
    for image in images:
        res_image = cv2.resize(image,(64,128))
        resized_images.append(res_image)
    return resized_images


path_objects = '/home/ras/robot_images/clipped_objects'
path_non_objects =  '/home/ras/robot_images/clipped_non_objects/'

images_objects = loadImages(path_objects)
random.shuffle(images_objects)
train_images_object = images_objects[0:50]
test_images_object = images_objects[50:]

cv2.imshow("display",test_images_object[0])
cv2.waitKey(0)
cv2.imshow("display",test_images_object[1])
cv2.waitKey(0)
label_train_obj = np.ones(len(train_images_object)).tolist()
label_test_obj = np.ones(len(test_images_object)).tolist()

images_non_objects = loadImages(path_non_objects)
random.shuffle(images_non_objects)
train_images_non_object = images_non_objects[0:50]

test_images_non_object = images_non_objects[58:63]
label_train_non_obj = (np.ones(len(train_images_non_object)) * (-1)).tolist()
label_test_non_obj = (np.zeros(len(train_images_non_object)) * (-1)).tolist()

train_images = train_images_object + train_images_non_object
label_train = label_train_obj + label_train_non_obj

test_images = test_images_object + test_images_non_object
label_test = label_test_obj + label_test_non_obj

#images = list(zip(images_objects,images_non_objects))
#random.shuffle(images)
#images_objects, images_non_objects = zip(*images)

#Calculate the HOG features

hog = cv2.HOGDescriptor()

features = []
print('pass')
for image in train_images:
    feature = hog.compute(image)
    feature = feature.ravel()
    features.append(feature)
print(features[0])
#Shuffle the data for training
perm = list(zip(features,label_train))
random.shuffle(perm)
features, label_train = zip(*perm)

#for feature in features:
#    print(len(feature))

obj_svm = svm.SVC()
obj_svm.fit(features, label_train)

#Testing the SVM
for image in test_images:
    feature = hog.compute(image)
    feature = (feature.ravel()).reshape(1,-1)
    result = obj_svm.predict(feature)
    print(result)
