"""
Author: Pablo Luesia Lahoz
Date: February 2020
Description: It preprocess the input image, and returns a normalized image
"""

import numpy as np
import scipy.stats as st
import cv2 as cv

class Preprocess(object):
    """
    It preprocess the image, and returns a normalized image for the 
    bee-sperm tracking problem. It inverts the image if needed, and normalize
    it, so all the images returned will be in the same parameters
    """

    """
    Init function
    """
    def __init__(self, max_area):
        self.mean = 0
        self.stdev = 0
        self.min = 0
        self.max = 0
        self.inverted = False
        self.max_area = max_area

    """
    It returns a linear normalization of the image
    """
    def __linear_normalization(self, img):
        return np.multiply( 
            np.subtract( img.astype('float32'), self.min), 
            255 / self.max)\
            .astype('ubyte')

    """
    It returns a z-score normalization of the image
    """
    def __z_score_normalization(self, img):
        # If an image is a normal distribution, mean would be 128, and the 99%
        #  of the image (3*sigma) would be between the values 0 and 256. So 
        #  sigma is 128 / 3 = 43
        transformed_img = np.divide(
                np.subtract(img.astype('float32'), self.mean),
               self.stdev/43)
        transformed_img[transformed_img >= 256] = 255
        transformed_img[transformed_img < 0] = 0
        return transformed_img.astype('ubyte')
    
    """
    Invert the image
    """
    def __invert(self, img):
        return np.subtract(255, img)

    """
    It extract the new parameters from the image given
    @param img. Image to extract the parameters from
    """
    def ext_param(self, img):
        # Invert needed
        self.inverted = st.skew(img.ravel()) > 0
        if self.inverted:
            aux_img = self.__invert(img)
        # Extract the parameters
        self.mean = np.mean(img)
        self.stdev = np.std(img)
        self.min = np.min(img)
        self.max = np.max(img)        

    """
    Preprocess the image with the given method
    @param img. Image to preprocess
    @param linear. Performance linear normalization
    """
    def apply(self, img):
        # Normalize the image. It inverts if needed
        if not self.inverted:
            norm_img = self.__z_score_normalization(self.__invert(img))
        else:
            norm_img = self.__z_score_normalization(img)

        # Filter the particles
        if self.max_area > 0:
            # Binary image otsu method
            _, thres_img = cv.threshold(norm_img, 200, 255, cv.THRESH_BINARY)
            # Extract the contours
            contours, hierarchy = cv.findContours(thres_img,
                                                  cv.RETR_TREE,
                                                  cv.CHAIN_APPROX_SIMPLE)
            # Filter the contours by the area
            non_valid_contours = list(filter(
                lambda cnt:  cv.contourArea(cnt) > self.max_area, contours ))
            # Delete the selected contours from the image
            cv.fillPoly(norm_img, non_valid_contours, (0))
            cv.dilate(norm_img,np.ones((5,5), dtype = "ubyte"))

        return norm_img



