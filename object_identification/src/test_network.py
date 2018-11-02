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


model = load_model('/home/ras/test_data/my_network.h5')
print('Model loaded')

test_img_path = '/home/ras/test_data/'

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
                    test_img_path,
                    target_size = (224,224),
                    batch_size = 1,
                    class_mode = "categorical",
                    shuffle = False)

predicts = model.predict_generator(test_generator,steps = 70)

for predict in predicts:
    max_idx = np.argmax(predict)
    print('Class: '+ str(max_idx))
