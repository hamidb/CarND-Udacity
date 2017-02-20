# Behavioral cloning with Keras

The goal of this project to build and train a deep nueral network to learn from human driving. This algorithm will be used in a simulator environment with a vehicle equipped with 3 cameras and can get steering angle as a control signal.
The objective here is to train the model to use these 3 cameras output and feed the correct steering angle to the vehicle.

## Dataset
I used the dataset collected by Udacity.
Here are the steps you'll take to build the network:
+ First load the data.
+ Append all recordings into a dataFormat structure.
+ Split all features and labels into two sets. 1- Training (80%) 2- Validation (20%)

The following is the property of our loaded dataset:
+ center image
+ left image
+ right image
+ steer angle
+ throttle
+ break
+ speed

## Load the Data

Start by importing the data from the pickle file.

# Data Augmentation
For better learning and ability to generalize, we needed to augment the correct dataset.
Our augmentation is not so complex since I did not have huge compute power to process all. I kept the network simple accordingly.

### Using left and right images
For each frame, we augment the center image with both left and right image to emulate deviation of the vehicle to the left and right sides of the road. Therefore, we also shifted the steering angle respectively. The amount we chose was '0.2' as the shift value to the steering angle.

### Flipping each frame
After some experiments, we found out that the trained network tends to steer more to either right or left side. We addressed this issue by balancing our dataset with even distribution of positive and negative steering angles.
A simple strategy for this is to augment the dataset with the flipped version of each frame along with it's negated steer angle.

### Crop and Resize
To discard useless parts of each frame such as sky, landscape, etc, we cropped header and footer pixels of all images.
In addition we reserved an option to resize each frame in order to bring down the complexity of our network.

The bellow figure, represents 6 different kinds of our augmentation:

![Augmentation](img1.png)
Since we will give a 3-channel input to the network, there is no need for BGR to RGB conversion.

# Batch generator
Since our dataset will be huge after augmentation, we will run out of memory and for higher efficiency we need to use a batch_generator function.

# Network topology
It is clear in the bellow network that we used 2 convolutional layer and 3 fully connected layers. 2x2 maxpooling is also used after each conv layer.
We have also included cropping and normalization steps as tensors inside our network.

![Graph](img2.png)
## Train the Network
Compile and train the network for 3 epochs. We used [`Adam`](https://keras.io/models/sequential/) optimizer, with `mse` loss function. The following parameters are used:
```
lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
```
We then used Keras's `.fit_generator()` method to have batch training with batch size of 64 sampes.

## Testing
Testing can be performed by running `drive.py` script given saved h5 files as follows:

```
$python drive.py model.h5
```

