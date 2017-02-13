#################################################################
# File: process.py
#
# Created: 12-12-2016 by Hamid Bazargani <hamidb@google.com>
# Last Modified: Mon Dec 12 22:31:16 2016
#
# Description:
#
#
#
# Copyright (C) 2016, Google Inc. All rights reserved.
#
#################################################################

import numpy as np
import glob
import sys
import os.path
import fnmatch
import pandas as pd
import pickle
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

def read_raw_data(data_dir):
    headers = ["center", "left", "right", "steer_angle",
               "throttle", "break", "speed"]
    logs = []
    for setpath in sorted(glob.glob(data_dir+"/*")):
        setname = os.path.basename(setpath)
        for contents in os.listdir(setpath):
            if fnmatch.fnmatch(contents,'*.csv'):
                log_file = os.path.join(setpath, contents)
                log = pd.read_csv(log_file, header=None, names=headers)
                logs.append(log)

    logs = pd.concat(logs, axis=0, ignore_index=True)
    return logs

def process_raw_data(logs):
    features = []
    labels = []

    for item in zip(logs['center'], logs['steer_angle']):

        img = image.load_img(item[0], target_size=(124, 64))
        x = image.img_to_array(img)
        x = x[60:,:,:]
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(x[0])
        labels.append(item[1])

    features = np.array(features)
    labels = np.array(labels)
    return [features, labels]

def load_from_pickle(filename):
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    return data

def save_to_pickle(pickle_file, features, labels):
    with open(pickle_file, 'wb') as pfile:
        pickle.dump({'features': features,
                     'labels': labels},
                    pfile, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":

    logs = read_raw_data('../../data')

    features, labels = process_raw_data(logs)

    save_to_pickle('sim-data.pickle', features, labels)


