#################################################################
# File: Traffic_Signs_Recognition.py
#
# Created: 27-11-2016 by Hamid Bazargani <hamidb@google.com>
# Last Modified: Sun Nov 27 02:53:19 2016
#
# Description:
#
#
#
# Copyright (C) 2016, Google Inc. All rights reserved.
#
#################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime, timedelta
from six.moves import xrange
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import tensorflow as tf
import os

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

# Helper functions

# normalization function
def tensor_normalize(x):
    # new_value = value − min_value / (max_value − min_value)
    with tf.name_scope('normalize') as scope:
        t_min = tf.reduce_min(x, name='min')
        t_max = tf.reduce_max(x, name='max')

        tensor = tf.div(tf.sub(x, t_min, name='sub'),
                        tf.sub(t_max, t_min, name='sub'), name='div')
        tensor = tf.to_float(tensor, name='to_float')
    return tensor

def preprocess(inputs):
    # grayscale conversion
    gray = tf.image.rgb_to_grayscale(inputs, name='rgb2gray')
    # normalizing input data to fall in [0.0 - 1.0]
    #normal = tf.image.per_image_whitening(inputs)
    normal = tensor_normalize(gray)
    return normal

def shuffle_data(features, labels):
    idxs = np.arange(0, len(features))
    np.random.shuffle(idxs)
    return [features[idxs], labels[idxs]]

def get_variable(name, shape, dtype=tf.float32, trainable=True,
                 initializer=None, stddev=0.01):
    var = tf.get_variable(name, shape, dtype=dtype, trainable=trainable,
                           initializer=initializer)
    return var

def data_iterator(features, labels, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
    # shuffle labels and features
        idxs = np.arange(0, len(features))
        np.random.shuffle(idxs)
        shuf_features = features[idxs]
        shuf_labels = labels[idxs]
        for batch_idx in range(0, len(features), batch_size):
            images_batch = shuf_features[batch_idx:batch_idx+batch_size,:,:,0:1]
            labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield images_batch, labels_batch

def show_images(inputs, labels, labels_to_show):
    index = []
    for label in labels_to_show:
        y = np.where(labels==label)
        if len(y[0]) == 0:
            continue
        rnd_index = random.randint(0, len(y[0]))
        index.append(y[0][rnd_index])

    # display one random sample of each given label
    rows = (len(index)-1)//8
    fig = plt.figure(figsize=(8, rows+1.5))
    for i in range(len(index)):
        sample = inputs[index[i]]
        ax = fig.add_subplot(rows+1, 8, i+1)
        ax.imshow(sample)
        plt.title(str(labels[index[i]]), fontsize=10)
        ax.axes.get_xaxis().set_visible(False) # hide x tick labels
        ax.axes.get_yaxis().set_visible(False) # hide y tick labels
# Load pickled data
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

n_train = X_train.shape[0]
n_test =  X_test.shape[0]
image_shape =  X_train.shape[1:4]
n_classes =  len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

labels_to_show = [1,   3,  7, 11, 15, 19, 21, 25,
                  26, 27, 29, 31, 33, 35, 37, 41]  # specifies which labels to display
show_images(X_train, y_train, labels_to_show)

### Test the input range
[print("Input shape:{} - Input Range:[{}-{}]".format(i.shape, i.min(), i.max())) for i in X_train[0:10]]

### Generate data additional (if you want to!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
sh_features, sh_labels = shuffle_data(X_train, y_train)

### Define your architecture here.

# Parameters
learning_rate = 0.001
image_width = 32
image_height = 32
image_depth = 1

# Network Parameters
dropout = 0.9 # Dropout, probability to keep units

def inference(inputs):

    # Store layers weight & bias
    with tf.name_scope('weights') as scope:
        weights = {
            # 5x5 conv, 1 input, 1 outputs
            'wc1': get_variable('wc1', [3, 3, 1, 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.01)),
            # fully connected, 32*32*1 inputs, 64 outputs
            'wd1': get_variable('wd1', [32*32*1, 64],
                                initializer=tf.truncated_normal_initializer(stddev=0.01)),
            # 64 inputs, n_classes outputs
            'out': get_variable('wo', [64, n_classes],
                                initializer=tf.truncated_normal_initializer(stddev=0.01)),
        }

    with tf.name_scope('biases') as scope:
        biases = {
            'bc1': get_variable('bc1', [1], initializer=tf.constant_initializer(0)),
            'bd1': get_variable('bd1', [64], initializer=tf.constant_initializer(0)),
            'out': get_variable('bo', [n_classes], initializer=tf.constant_initializer(0)),
        }

    with tf.name_scope('preprocess') as scope:
        processed = preprocess(inputs)

    # Convolution Layer
    with tf.name_scope('conv1') as scope:
        conv1 = tf.nn.conv2d(processed, weights['wc1'], strides=[1, 4, 4, 1],
                             padding='SAME', name='conv1_1')
        conv1 = tf.nn.bias_add(conv1, biases['bc1'], name='conv1_2')
        conv1 = tf.nn.relu(conv1, name='relu1')

    # Max Pooling
    with tf.name_scope('maxpool_1') as scope:
        max1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
                              padding='SAME', name='max1')

    # Fully connected layer
    with tf.name_scope('fully_connected') as scope:
        # Reshape max1 output to fit fully the connected layer input
        fc1 = tf.reshape(processed, [-1, weights['wd1'].get_shape().as_list()[0]],
                         name='reshape')
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'], name='fc1')
        fc1 = tf.nn.relu(fc1, name='relu')
    #fc1 = tf.nn.dropout(fc1, dropout)

    # Logits
    with tf.name_scope('output') as scope:
        logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'], name='out')

    return logits

# Loss fucntion and optimizer
def loss_op(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits,labels, name='cross_entropy')
        loss_function = tf.reduce_mean(cross_entropy, name='mean')
        tf.scalar_summary("loss", loss_function)
    return loss_function

def trainer(loss):
    with tf.name_scope('train') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           name='AdamOpt').minimize(loss)
    return optimizer

def predict_op(logits):
    with tf.name_scope('predict') as scope:
        # Soft-max logits to a hot vector [None, 43].
        soft = tf.nn.softmax(logits, name='sofmax')
        # Select maximal value of softmax layer as predicted value
        predicted = tf.argmax(soft, 1, name='argmax')
    return predicted

def accuracy_op(predicted, labels):
    with tf.name_scope('accuracy') as scope:
        # Evaluate model with its accuracy
        true_positive = tf.equal(tf.cast(predicted, tf.int32), labels, name='equal')
        acc = tf.reduce_mean(tf.cast(true_positive, tf.float32), name='mean')
    return acc

def initializer():
    with tf.name_scope('initializer') as scope:
        # Initializing weights and biases
        init = tf.initialize_all_variables()
    return init

### Train your model here.
# Create a graph
graph = tf.Graph()
# Define the net in graph
with graph.as_default():

    # Placeholders for inputs and labels.
    inputs = tf.placeholder(tf.float32, [None, image_width, image_height, image_depth])
    labels = tf.placeholder(tf.int32, [None,])

    logits = inference(inputs)

    loss = loss_op(logits, labels)

    predict = predict_op(logits)

    accuracy = accuracy_op(predict, labels)

    train = trainer(loss)

    init = initializer()

    saver = tf.train.Saver(tf.all_variables())

    merged_summary_op = tf.merge_all_summaries()

sess = tf.Session(graph=graph)

# Set the logs writer to the folder /tmp/tensorflow_logs
summary_writer = tf.train.SummaryWriter('./logs/', sess.graph)

# initializing weights
sess.run(init)

training_iters = 50000
batch_size = 500
display_step = 10

# Train with 5000 smaples for faster training time
# We also kept the graph reasonably deep not to over/underfit.
input_dict = {inputs:sh_features[0:batch_size, :, :, 0:image_depth],
              labels:sh_labels[0:batch_size]}

step = 1


print("Optimization started...")
start = datetime.now()
iter_ = data_iterator(X_train, y_train, batch_size)
for step in xrange(training_iters):
    batch_image, batch_label = iter_.__next__()
    input_dict = {inputs:batch_image, labels:batch_label}
    # Run optimization op (backprop)
    summary_str, _ = sess.run([merged_summary_op, train], feed_dict=input_dict)
    summary_writer.add_summary(summary_str, step)
    if step % display_step == 0:
        # Calculate loss and accuracy
        loss_value, acc = sess.run([loss, accuracy], feed_dict=input_dict)
        time = int((datetime.now() - start).total_seconds())
        print("Iter:{}, Elapsed {:02d}:{:02d}:{:02d}, Loss: {:.6f}, accuracy: %{:.2f}".format(
                step , (time//3600), (time%3600)//60, time%60, loss_value, acc*100))
        if acc > 0.98:
            saver.save(sess, checkpoint_path, global_step=step)
            break
    step += 1
print("Optimization Finished!")

### Load the images and plot them here.
### Similar to what we have done for the training set

labels_to_show = [1,   3,  7, 11, 15, 19, 21, 25,
                  26, 27, 29, 31, 33, 35, 37, 41]  # specifies which labels to display
show_images(X_test, y_test, labels_to_show)

### Run the predictions here.
# Pick 15 random test samples
num_test = 15
test_features, test_labels = shuffle_data(X_test, y_test)

input_dict = {inputs:test_features[:,:,:, 0:image_depth], labels:test_labels}

predicted, acc = sess.run([predict, accuracy], feed_dict=input_dict)

print("Accuracy over the test set: %{:.2f}".format(acc*100))
print("Labels:    ", test_labels[0:num_test])
print("Predicted: ", predicted[0:num_test])

with open("signnames.csv", 'r') as f:
    sign_name = {row.split(',')[0] : row.split(',')[1] for row in f}

def visualize_prediction(inputs, labels, pred_labels):
    # display one random sample of each given label
    rows = (len(inputs)-1)//3
    fig = plt.figure(figsize=(10, rows+10))
    col = 1
    for i in range(len(inputs)):
        ax = fig.add_subplot(rows+3, 3, col)
        col += 1
        ax.imshow(inputs[i])
        text = sign_name[str(labels[i])]
        # limit the text length up to 25 chars and remove null-terminating end
        text = text[0:20] + "..." if len(text)>25 else text[0:len(text)-1]
        plt.title(str(labels[i]) + ": " +  text, fontsize=10,
                  color='green' if labels[i] == pred_labels[i] else 'red')
        ax.axes.get_xaxis().set_visible(False) # hide x tick labels
        ax.axes.get_yaxis().set_visible(False) # hide y tick labels

visualize_prediction(test_features[0:num_test], test_labels[0:num_test], predicted[0:num_test])

### Visualize the softmax probabilities here.
k = 3
num_test = 15

soft = tf.nn.softmax(logits)
probability = sess.run(soft, feed_dict=input_dict)
prob = np.squeeze(probability[0:num_test])

rows = (2*num_test-1)//4
col = 0
fig = plt.figure(figsize=(10, rows+4))
for i in range(num_test):
    prob_idx = prob.argsort()[i][::-1]
    top_k_idx = prob_idx[0:k]

    ax = fig.add_subplot(rows+1, 8, col+1)
    ax.imshow(test_features[i])
    plt.title(str(test_labels[i]), fontsize=10,
              color='green' if test_labels[i] == predicted[i] else 'red')
    ax.axes.get_xaxis().set_visible(False) # hide x tick labels
    ax.axes.get_yaxis().set_visible(False) # hide y tick labels

    col += 1
    ax = fig.add_subplot(rows+1, 8, col+1)
    ind = np.arange(k)
    width = 0.3
    barlist = ax.barh(width+ind, prob[i][top_k_idx],width)
    ax.set_aspect(0.35)
    ax.axis([0, 1, 0, k])
    barlist[0].set_color('g' if test_labels[i] == predicted[i] else 'r')
    plt.yticks(ind + 1.5*width, top_k_idx)
    fig.subplots_adjust(wspace=0.35)

    ax.axes.get_xaxis().set_visible(False) # hide x tick labels
    col += 1
