# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  


# **Vehicle Detection** 
***
The goal is to develop a pipeline to detect vehicle driving on the road in the given video sequences. Like many other computer vision tasks, feature extraction is required for detecting cars from non-cars.
Histogram of Oriented Gradients (HOG) features extracted from a labeled datasetare used to train a classifier. 
Linear SVM classifier is chosen to perform this classification. Not only HOG features, but we also employed color information as an augmented feature to perform this task.
A preprocessing step is taken into account to normalize and reduce the dimension of the feature vector.

## Loading and exploring dataset
The first step is to explore our dataset. We read all **cars** and **non-cars** and store them into positive and negative samples. We also shuffle our samples for the sake of randomness.
After reading the files, total of **8792** positive samples and **8968** negative samples were stored. Since both classes are balanced in terms of numbers, no extra trimming is required.

![Alt][data]
[data]: img1.png "Cars and non-cars"

## Histogram of Oriented Gradients (HOG)
### Color space
After grouping both classes, we used **YCrCb** color space to extract HOG features. We chose this color space by examining different color space and displaying all 3 channels with their HOG features. Comparing to the other spaces such as RGB, we found more variation between three channels of the converted image. We also looked at the output of HOG features to determine which color space has more distinctive power between two classes. 
### HOG parameters
HOG method of `skimage` library is used to detect HOG features and we used the following parameters:
+ orient = 9
+ pix_per_cell = 8
+ cell_per_block = 2

This parameters are chosen by seeing the output of each channels. We found some parameters not so crucial like orientation parameters (i.e. $8 <= \text{orient} <= 10$ is a good choice). 

### RGB 
Color channels and HOG features for RGB.
![Alt][rgb]
[rgb]: rgb.png "HOG on RGB color space"

### HSV
Color channels and HOG features for HSV.
![Alt][hsv]
[hsv]: hsv.png "HOG on HSV color space"

### YCrCb
Color channels and HOG features for YCrCb.
![Alt][YCrCb]
[YCrCb]: YCrCb.png "HOG on YCrCb color space"

### Color features
We augment HOG features by adding histogram of color for all three channels. 
We first normalized each sample to a 32x32 patch and applied histogram of 32 bins on them. Then we concatenated both set of features and flatten the results to feed our classifier.
Ultimately, each feature contains total of 8460 elements including:
+ 3168 color elements
+ 5292 HOG elements.

We applied a normalization step to avoid a set of features dominating the response of our classifier. 

## Learning SVM classifier
We chose linear SVM for this classification task. Both car and non-car classes are labeld with `1` and `0` labels respectively. We shuffled and normalized out sample features and fed them to our SVC classifier. 
To avoid overfitting problem, we splited the whole set into training and test group containing 80% training data and 20% test data. 

The test accuracy of our classifier reached 99% accuracy. 

## Test on images
To test the classifier on images of different sizes, we used a sliding window function to scan all possible sub-windows that might contain vehicle. 
+ For not spending too many cycles, we limited our search area in the range of Y = [400, 650]. 
+ Also we used stride equal to half of each subwindow.  
+ Two different scale is selected to achieve multi-scale detection. 

To reject as many as possible false positives, we performed a grouping strategy for detected boxes. A threshold is used to tune the algorithm. 
Not only it helps discarding false positives, but also it will merge multiple detections into a single bounding box. 
The method is called **`non_max_suppress()`**
![Alt][detect]
[detect]: detect.png "Result of Classifier on images"

## Test on Videos
Once we are happy with our test on images, we move on to test on a video sequence.
To use temporal consistency from the video sequence, we added temporal window of **5** frames to extend our non max supression technique. So, we accumulated detected boxes in 5 consecutive frames and merge them together. 

`VideoProcessor()` is inspired from [hkorre](https://github.com/hkorre/CarND-Vehicle-Detection/blob/master/notebook.ipynb) implementation for the above purpose.

We stored the result of our algorithm on `project_video.mp4` in a file named `output.mp4`.

## Reflections

### Drawbacks and sources of improvement
1. This current algorithm is prone to miss some true positives and detect false positives. We can improve the classification by simply use more samples in the training time. 
2. Data augmentation is one choice to improve the accuracy.
3. Better investigation of different color space as well as interactive threshold finding is helpful
4. Hard negative mining can be applied to reduce the number of false positives.
5. For increasing the frame rate, we can avoid detection on regions where we expect already detected cars.
6. Better tracking approach can be useful.
7. Instead of resizing each subwindow, we can first down-sample our input image and adopt the sliding window search.
