#!/usr/bin/env python
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


#Set global variables
train_img_path = '/home/ras/robot_images/test'
validation_img_path = 'path_to_val'
img_width = 224
img_height = 224
n_train_samples = 16
n_validation_samples = 0
batch_size = 16
epochs = 50

#Setup Network model
model = applications.VGG16(include_top = False, weights='imagenet', input_shape = (img_width,img_height,3))

for layer in model.layers[:5]:
    layer.trainable = False     #Freeze First 5 layers

#Add custom layers
x = model.output
x = Flatten()(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1,activation='sigmoid')(x)
#Create final model
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "binary_crossentropy", optimizer = 'rmsprop', metrics=["accuracy"])

#Load the data
train_datagen = ImageDataGenerator(rescale = 1./255,
                            rotation_range = 40,
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True,
                            fill_mode = 'nearest')

train_generator = train_datagen.flow_from_directory(
                    train_img_path,
                    target_size = (img_width,img_height),
                    batch_size = batch_size,
                    class_mode = "categorical")

validation_generator = train_datagen.flow_from_directory(
                    validation_img_path,
                    target_size = (img_width,img_height),
                    class_mode = "categorical")

# Train the model
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
model_final.fit_generator(train_generator,
                        samples_per_epoch = n_train_samples,
                        epochs = epochs,
                        validation_data = validation_generator,
                        nb_val_samples = n_validation_samples,
                        callbacks = [checkpoint,early])
print('Done')
