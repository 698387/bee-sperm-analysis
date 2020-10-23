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
        self.local_norm = False
        self.norm_mask = None

    """
    It returns a linear normalization of the image
    """
    def __linear_normalization(self, img):
        return np.multiply( 
            np.subtract( img.astype('float32'), self.min), 
            255 / self.max)\
            .astype('ubyte')

    """
    It returns a local normalization in the color of the image
    """
    def __local_normalization(self, img):
        norm_img = (img*self.norm_mask)
        norm_img[norm_img >= 256] = 255
        return norm_img.astype("ubyte")


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
        aux_img = img
        # Invert needed
        self.inverted = st.skew(img.ravel()) > 0
        if not self.inverted:
            aux_img = self.__invert(img)
        
        if self.local_norm:
            # Extract the mask to normalize the image locally
            self.norm_mask = np.mean(aux_img) / cv.blur(aux_img, (21,21))
            aux_img = self.__local_normalization(aux_img)

        # Extract the parameters to the z-score norm
        self.mean = np.mean(aux_img)
        self.stdev = np.std(aux_img)
        self.min = np.min(aux_img)
        self.max = np.max(aux_img)


    """
    Preprocess the image with the given method
    @param img. Image to preprocess
    @param local_norm. Performance local normalization iff true
    """
    def apply(self, img, local_norm = False):
        # Invert if needed
        norm_img = img
        if not self.inverted:
            norm_img = self.__invert(img)
        if self.local_norm:
            # Normalize the color in the image
            norm_img = self.__local_normalization(norm_img)
        # z-score normalize
        norm_img = self.__z_score_normalization(norm_img)

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



