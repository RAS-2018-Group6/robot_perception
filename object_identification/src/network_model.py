#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

#Smaller version of the VGG net

class IdentificationNetwork:
    @staticmethod

    def build(height, width, depth, classes, finalAct="softmax"):
        # Initalize the model
        model = Sequential()
        # Input shape to be "channels last"
        inputShape = (height,width,depth)
        chanDim = -1

        #If "channels first", update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        # Convolution => RELU Activation => POOL

        model.add(Conv2D(32,(3,3), padding ="same", input_shape = inputShape)) #32 layer with 3x3 kernel
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.25)) #Randomly disconnecting nodes from the current layer to the next layer -> reduce overfitting

        # Convolution => RELU => Convolution => RELU => Pool

        model.add(Conv2D(64,(3,3),padding="same")) #64 layers with 3x3 kernel
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding="same")) #64 layers with 3x3 kernel
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Convolution => RELU => Convolution => RELU => Pool

        model.add(Conv2D(128,(3,3),padding="same")) #128 layers with 3x3 kernel
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128,(3,3),padding="same")) #128 layers with 3x3 kernel
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # Flatten => Dense => RELU

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        # "softmax" : single label classification
        # "sigmoid" : multi label classification
        model.add(Activation(finalAct))

        return model
