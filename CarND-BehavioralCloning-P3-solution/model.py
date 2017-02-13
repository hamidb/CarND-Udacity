#################################################################
# File: model.py
#
# Created: 12-12-2016 by Hamid Bazargani <hamidb@google.com>
# Last Modified: Mon Dec 12 23:08:25 2016
#
# Description:
#
#
#
# Copyright (C) 2016, Google Inc. All rights reserved.
#
#################################################################

from keras.applications.vgg16 import VGG16 as vgg16
from keras.layers import Dense, Input, AveragePooling2D
from keras.layers import Dropout, Flatten, BatchNormalization
from keras.models import Model
import json
import h5py

class SteerModel(object):
    def __init__(self):
        self.model = None
        self.build()

    def build(self):
        base_model = vgg16(weights='imagenet', input_tensor=Input((64, 64, 3)), include_top=False)

        # freeze all convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = AveragePooling2D((2, 2))(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation="relu")(x)
        x = Dense(2048, activation="relu")(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(1, activation="linear")(x)
        # this is the model we will train
        self.model = Model(input=base_model.input, output=x)
        return self

    def save(self):
        with open('model.json', 'w') as f:
            json.dump(self.model.to_json(), f)
            self.model.save_weights('model.h5')
        print("model h5/json save!")
