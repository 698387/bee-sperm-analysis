import cv2 as cv
import numpy as np
import math

""" Extracts the skeleton of a binary image. It performs iterations until it 
finds the skeleton, or it reaches max_it iterations. If there is not limit for 
the iterations, max_it = 0"""
def find_skeleton(original_image, max_it = 0):
    skeleton = np.zeros(original_image.shape,np.uint8)
    eroded = np.zeros(original_image.shape,np.uint8)
    temp = np.zeros(original_image.shape,np.uint8)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    thresh = original_image.copy()

    iters = 0
    while(True):
        cv.erode(thresh, kernel, eroded)
        cv.dilate(eroded, kernel, temp)
        cv.subtract(thresh, temp, temp)
        cv.bitwise_or(skeleton, temp, skeleton)
        thresh = eroded.copy()
        iters += 1
        if cv.countNonZero(thresh) == 0:
            return (skeleton,iters)
