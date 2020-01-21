# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:42:38 2019

@author: Pablo Luesia Lahoz
@description: file that implements the preprocess of microscopic images to
              perform the analysis for the detection
"""

import cv2 as cv

def preprocess(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blur_image = cv.GaussianBlur(gray_image, (3,3), 0)  # Eliminates noise 
    blur_image = gray_image
    cv.equalizeHist(blur_image)                        # Histogram equalization
    return blur_image
