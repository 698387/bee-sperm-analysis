"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: It extracts all the lines from a skeleton image. The image
             may have information of 2 layers (overlap and normal)
"""

import numpy as np
import math as m
import cv2 as cv
import Graph from graph

eight_neighbor_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype="float32")

"""
Returns the skeleton of the layers image
"""
def __extract_skeleton(layers_img):
    # The first layer is consider background
    binary_img = np.where(layers_img == 0, 0, 255).astype("ubyte")
    # Extracts the skeleton
    return cv.ximgproc.thinning(binary_img, 
                                thinningType = cv.ximgproc.THINNING_GUOHALL)\
                                    .astype("float32").clip(0,1)


"""
Performs a 8-neighbor convolution, and returns the number of neighbors of
to each pixel
"""
def __extract_num_neighbors(skeleton):
    return cv.filter2D(skeleton, -1, eight_neighbor_kernel) * skeleton

"""
Computes the local direction of the pixel, given the 8-neighbor
"""
def __extract_local_direction(skeleton):
    return np.arctan2(cv.filter2D(skeleton, -1, i_vector_kernel) * skeleton, 
                    cv.filter2D(skeleton, -1, j_vector_kernel) * skeleton)

"""
Find the extrems and the cross points
"""
def __find_interest_points(skeleton):
    # Count the number of neighbor pixels
    num_neighbors = __extract_num_neighbors(skeleton)
    # Eliminate points not connected
    skeleton[num_neighbors == 0] = 0
    # If only one neighbor, it is an extrem
    extrem_points = num_neighbors == 1
    # If more than 2 neighbors, it is an intersection
    intersection_point = num_neighbors > 2
    return extrem_points, intersection_point

"""
Line follower returns the line that exists from the point origin <<point_o>> 
to an extrem, or to a cross
"""
def __line_follower(skeleton, extrems, intersections, point_o):


"""
Extracts the lines asociated with the spermatozoon from the different layers
@param layers_img is the image with each pixel predicted to belong each class
@param overlapping indicates if there is an overlapping layer
@param debug if true, it will show debug messages and images
@return A list with all the lines found. Each line is an array of points
"""
def extractLines(layers_img, overlapping = False, debug = False):
    # Extracts the skeleton
    skeleton = __extract_skeleton(layers_img)
    # Detect points of interest
    extrems, intersections = __find_interest_points(skeleton)
    # Set intersection areas instead of points
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    intersection_area = cv.dilate(intersections.astype("ubyte"), kernel).astype("bool")

    if debug:
        font = cv.FONT_HERSHEY_SIMPLEX
        img2show = cv.cvtColor((skeleton*255).astype("ubyte"), cv.COLOR_GRAY2BGR)
        cv.putText(img2show, "Skeleton", (2,15), font, .5,(255,255,255),1,cv.LINE_AA)
        img2show[extrems] = [0, 255, 0]
        cv.putText(img2show, "Extrem points", (2,30), font, .5,(0, 255, 0),1,cv.LINE_AA)
        img2show[intersection_area] = [0, 0, 255]
        cv.putText(img2show, "Crosses", (2,45), font, .5,(0,0,255),1,cv.LINE_AA)
        cv.imshow("line extractor debug", img2show)




