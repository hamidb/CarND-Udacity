#################################################################
# File: P1.py
#
# Created: 31-10-2016 by Hamid Bazargani <hamidb@google.com>
# Last Modified: Tue Nov  1 17:42:18 2016
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

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import fnmatch

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

canny_low_threshold = 50
canny_high_threshold = 150
blur_kernel_size = 5

# Define the Hough transform parameters
rho = 1               # distance resolution in pixels of the Hough grid
theta = 1*np.pi/180   # angular resolution in radians of the Hough grid
threshold = 35        # minimum number of votes (intersections in Hough grid cell)
min_line_length = 60  # minimum number of pixels making up a line
max_line_gap = 150    # maximum gap in pixels between connectable line segments

filter_threshold_low = 20   # degree
filter_threshold_high = 70  # degree

low_pass_weight = 0.5
prev_left_line = []
prev_right_line = []

# Define mask parameters
offset_from_left = 100
offset_from_right = 10
offset_from_bottom = 0
trapezoid_height_ratio = 0.4
trapezoid_width_ratio1 = 1
trapezoid_width_ratio2 = 0.1

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def fit_line_LSE(points):
    # get mean, min and max of y1 and y2 for the left line
    point_mean = np.mean(np.array(points), axis = 0)[0]
    point_max = np.max(np.array(points), axis=0)[0]
    point_min = np.min(np.array(points), axis=0)[0]
    y_max = max(point_max[1], point_max[3])
    y_min = min(point_min[1], point_min[3])
    x_mean = (point_mean[0] + point_mean[2]) / 2
    y_mean = (point_mean[1] + point_mean[3]) / 2

    nom = denom = 0
    for point in points:
        for x1, y1, x2, y2 in point:
            nom += (x1-x_mean)*(y1-y_mean)
            nom += (x2-x_mean)*(y2-y_mean)
            denom += (x1-x_mean)*(x1-x_mean)
            denom += (x2-x_mean)*(x2-x_mean)
    m_lse = nom/denom
    b_lse = y_mean - m_lse*x_mean
    x1 = int((y_min - b_lse)/m_lse)
    x2 = int((y_max - b_lse)/m_lse)

    return [x1, y_min, x2, y_max]

def group_lines(lines):
    left_new = []
    right_new = []
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            rho = (y2 - y1) / (x2 - x1)
            if rho < 0 :
                left_lines.append(line)
            else:
                right_lines.append(line)

    if len(left_lines) > 0:
        left_new = [fit_line_LSE(left_lines)]
    if len(right_lines) > 0:
        right_new = [fit_line_LSE(right_lines)]
    return [np.array(left_new), np.array(right_new)]

def filter_lines(lines, rho_low, rho_high):
    lines_new = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            rho = abs(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if rho > rho_low and rho < rho_high :
                lines_new.append(line)
    return lines_new

def low_pass_filter(lines):
    global prev_left_line
    global prev_right_line
    left_line = lines[0]
    right_line = lines[1]
    if len(prev_left_line) > 0 and len(left_line) > 0:
        left_line =  np.floor(prev_left_line*low_pass_weight + left_line*(1-low_pass_weight))

    if len(prev_right_line) > 0 and len(right_line) > 0:
        right_line =  np.floor(prev_right_line*low_pass_weight + right_line*(1-low_pass_weight))
    left_line = left_line.astype(int)
    prev_left_line = left_line
    right_line = right_line.astype(int)
    prev_right_line = right_line
    return [left_line, right_line]

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    lines = filter_lines(lines, filter_threshold_low, filter_threshold_high)

    lines = group_lines(lines)

    lines = low_pass_filter(lines)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if len(lines) > 0:
        draw_lines(line_img, lines, [255, 0, 0], 8)

    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    imshape = image.shape

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # convert color
    # HSV color space is more robust to light luminance
    gray = hsv[:, :, 2]

    # smooth image
    blur = gaussian_blur(gray, blur_kernel_size)

    # apply canny edge detection
    edges = canny(blur, canny_low_threshold, canny_high_threshold)

    # This time we are defining a four sided polygon to mask
    height = trapezoid_height_ratio * imshape[0]
    width_large = trapezoid_width_ratio1 * imshape[1]
    width_small = trapezoid_width_ratio2 * imshape[1]

    vertices = np.array([[(offset_from_left, imshape[0] - offset_from_bottom ),
                          (width_large/2 - width_small/2, imshape[0] - height),
                          (width_large/2 + width_small/2, imshape[0] - height),
                          (width_large - offset_from_right, imshape[0] - offset_from_bottom)]], dtype=np.int32)

    # mask out undesired regions
    masked_edges = region_of_interest(edges, vertices)

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    # cv2.polylines(line_img, vertices, 1, [255, 255, 0], 2)

    line_img = weighted_img(line_img, image)

    return line_img

if __name__=="__main__":
    test_dir = os.getcwd() + '/test_images'
    for file in os.listdir(test_dir):
        if not fnmatch.fnmatch(file, "*.jpg"):
            continue
        filename = os.path.join(test_dir, file)
        image = mpimg.imread(filename)
        result  = process_image(image)
        plt.imshow(result)
        filename = filename.split('.')[0] + "_out.png"
        print("saved result in %s" % filename)
        plt.savefig(filename, format='png')
        prev_right_line = []
        prev_left_line = []

    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    prev_right_line = []
    prev_left_line = []
    white_output = 'yellow.mp4'
    clip1 = VideoFileClip("solidYellowLeft.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    # for the challenge sequence
    offset_from_left = 170
    offset_from_right = 50
    offset_from_bottom = 50
    trapezoid_height_ratio = 0.4
    trapezoid_width_ratio1 = 1
    trapezoid_width_ratio2 = 0.08

    prev_right_line = []
    prev_left_line = []
    white_output = 'challenge-out.mp4'
    clip1 = VideoFileClip("challenge.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

