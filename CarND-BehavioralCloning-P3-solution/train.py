#################################################################
# File: train.py
#
# Created: 12-12-2016 by Hamid Bazargani <hamidb@google.com>
# Last Modified: Mon Dec 12 23:04:32 2016
#
# Description:
#
#
#
# Copyright (C) 2016, Google Inc. All rights reserved.
#
#################################################################
from keras.optimizers import Adam

from model import SteerModel
from process import *

batch_size = 64
nb_epoch = 1

data = load_from_pickle('sim-data.pickle')
features = data['features']
labels = data['labels']
print("features shape is ", features.shape)
print("labels shape is ", labels.shape)

# Get randomized datasets for training and test
X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.10,
        random_state=43)

# Get randomized datasets for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.10,
        random_state=43)

# Print out shapes of new arrays
train_size = X_train.shape[0]
test_size = X_test.shape[0]
valid_size = X_valid.shape[0]
input_shape = X_train.shape[1:]

print("train size:", train_size)
print("valid size:", valid_size)
print("test size:", test_size)
print("input shape:", input_shape)

steer = SteerModel()
model = steer.model

print("Model summary:")
model.summary()

model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['mean_absolute_error'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_valid, y_valid))
#validation_split=0.15

steer.save()
