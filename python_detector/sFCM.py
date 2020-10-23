"""
Author: Pablo Luesia Lahoz
Date: February 2020
Description: Class for performance a spatial fuzzy c-means
"""

import numpy as np
import cv2 as cv
import random as r
from scipy.spatial.distance import cdist

import time


class sFCM(object):
    """
    Performance the fuzzy c-means with an image, with the posibility of using
    the spatial component
    """

    """
    Init function.
    @param c is the number of clusters 
    @param m controls the fuzzyness
    @param p is the parameter to power u
    @param q is the parameter to power h
    @param NB is the window size of the spatial part
    @param init_points are the optional initial points for the clusters 
    """
    def __init__(self, c=2, m = 2, p=1, q=1, NB = 3, init_points = np.array([])):
        self.c = c
        self.m = m
        self.p = p
        self.q = q
        self.NB = NB
        self.init_points = init_points
        if len(init_points) > 0 and len(self.init_points.shape) < 2:
            self.init_points = np.reshape(self.init_points, (self.c, -1))
        r.seed()

    """
    Initialize the centroids based on the k-means++.
    @param image is the original data
    """
    def __init_centroids(self, data, data_shape):
        # The initial points are not defined by the user
        if len(self.init_points) == 0:

            # Index of the data
            idx_data = np.arange(data.shape[0])

            # Distance from each point to the nearest center
            D = np.full(data.shape[0], np.inf)
            # Probability of selecting each point
            p = np.full(data.shape[0], 1.0/data.shape[0])

            self.init_points = np.zeros((self.c, data.shape[1]))
            # Select the rest of the centroids
            for i in range(1, self.c):                
                # Select a next center
                v = data[np.random.choice(idx_data, p = p)]   
                self.init_points[i] = v
                # Compute distance
                D_aux = cdist(data, [v], metric='sqeuclidean').flatten()
                D = np.array([D, D_aux]).min(axis = 0)
                # Probability depending on the squared distance
                p = D / sum(D)

        return self.init_points

    """
    Returns the membership value of the data in data
    """
    def __get_membership(self, data, data_shape):
        # Matrix distance from each centroid to each point
        # m_dist = np.power(cdist(data, self.v), 2 / (self.m - 1))
        m_dist = cdist(data, self.v, metric = 'sqeuclidean')
        # Returns 1/(sum_(j=1)^(c)(v_dist_i/v_dist_j)^(2/(m-1)))
        # Nan values are converted to 1
        return np.nan_to_num(
            np.divide(1, np.divide(np.repeat(m_dist, self.c, axis = 0),
                          m_dist.reshape(-1,1) ).reshape(-1, self.c, self.c)\
                .reshape(-1, self.c, self.c).sum(axis=1) ),
            1)

    """
    Calculate the membership, depending on the spatial information
    """
    def __spatial_membership(self, data_shape, kernel, u):
        # Reshape u to store it in the spatial form
        sp_u = u.reshape((data_shape[0], data_shape[1], self.c))
        # Calculates h power to q, and its multiplication by u^p
        up_hq = np.multiply( np.power( 
            cv.filter2D(sp_u, -1, kernel)\
                .reshape((data_shape[0]*data_shape[1], -1)), self.q
            ), np.power(u, self.p) )
        return np.nan_to_num( 
            np.divide( up_hq, up_hq.sum(axis=-1).reshape(-1,1) ), 1)

    """
    Update the centroids v with the new values
    """
    def __update_centroids(self, data, data_shape, u):
        # v_j = (sum_(i=0)^n (u_ij^m*x_i) ) / sum_(i=0)^n (u_ij^m)
        p_u = np.power(u, self.m)
        return np.matmul(p_u.transpose(), data) / p_u.sum(axis = 0).reshape(-1,1)

    """
    Fit the cluster to the data
    @param data is a data representation. It has to be a 2d vector
    @param spatial if true, spatial component will be used
    """
    def fit(self, raw_data, spatial = True):
        kernel = np.ones((self.NB, self.NB))
        # Data information extraction
        data_shape = raw_data.shape
        data = raw_data.reshape(data_shape[0]*data_shape[1], -1)\
                       .astype('float32')
        # Sets the data to 3d
        #if len(data_shape) < 3:                
        #    data = np.expand_dims(raw_data, axis=-1).astype(np.float32)
        #    data_shape = data.shape
        #    if len(data_shape) < 3:             # Checks the shape of the data
        #        raise TypeError("The data has to be 2d or 3d")

        # Initial centroids
        self.v = self.__init_centroids(data, data_shape)
        # Difference between old and new centroids
        diff_v = np.inf
        # Iterate until converges
        while diff_v > 0.5:
            # Membership values
            u = self.__get_membership(data, data_shape)
            if spatial:     # Extracts the spatial information if indicated
                u = self.__spatial_membership(data_shape, kernel, u)
            # New centroids
            v_n = self.__update_centroids(data, data_shape, u)
            # Convergence distance
            diff_v = max(np.linalg.norm(v_n - self.v, axis = 1))
            self.v = v_n

    """
    Predict the class asociated to each data in raw_data
    """
    def predict(self, raw_data, spatial = True):
        kernel = np.ones((self.NB, self.NB))
        # Data information extraction
        data_shape = raw_data.shape
        data = raw_data.reshape(data_shape[0]*data_shape[1], -1)\
                       .astype('float32')

        # Predict the centroids
        u = self.__get_membership(data, data_shape)
        if spatial:     # Extracts the spatial information if indicated
            u = self.__spatial_membership(data_shape, kernel, u)
        # Maximum value of u is the predicted value
        return u.argmax(axis = -1).reshape((data_shape[0], data_shape[1]))

    """
    Select the layers depending of the classes fitted with sFCM
    @param cluster is the fitted cluster to the image
    @param img is the image used to fit the cluster
    @return true iff the cluster has not been corrected
    """
    def correct(self, img):
        # Distance between class centers
        center_dist = cdist(self.v, self.v)
        # If the distance is lower than the image sigma, it combine the classes
        classes2combine = []
        for [x, y] in np.argwhere(center_dist < 43):
            if x != y and not [y,x] in classes2combine:
                classes2combine.append([x,y])
    
        # Re-fit the cluster if needed
        if len(classes2combine) > 0:
            # Extract the new posible centers for the init
            new_centers = np.array([v for i,v in enumerate(self.v)\
               if i not in list(map(lambda x: x[1], classes2combine) )] )
            # Update the cluster parameters and re-fit it
            self.c = len(new_centers)
            self.init_points = new_centers
            self.fit(img)
            return False
        else:
            return True

    """
    Return true iff the cluster is correct
    """
    def is_correct(self):
        # Distance between class centers
        center_dist = cdist(self.v, self.v)
        # If the distance is lower than the image sigma, it combine the classes
        if any((np.array(center_dist) < 43).ravel()):
            return False
        else:
            return True