#!/usr/bin/env python
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import time


model = load_model('vgg16_1.h5')
print('Model loaded')
img = image.load_img('purple_star_1.png', target_size = (224,224))
print(type(img))

x = image.img_to_array(img)
x = x.reshape((1,)+x.shape  )
#print(type(x))
#print(x.shape)

#plt.imshow(x/255.)
#plt.savefig('test.pdf')
start_time = time.time()
y = model.predict(x)
print(y)
print(time.time()-start_time)
