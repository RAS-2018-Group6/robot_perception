#!/usr/bin/env python
import matplotlib
matplotlib.use("Agg")

from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import IdentificationNetwork
import numpy as np
import random
import pickle
import cv2
import os
