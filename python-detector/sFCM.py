"""
Author: Pablo Luesia Lahoz
Date: February 2020
Description: Class for performance a spatial fuzzy c-means
"""

import numpy as np
import cv2 as cv
import random as r
from scipy.spatial.distance import cdist


class sFCM(object):
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

            # First centroid
            v = data[np.random.choice(idx_data)]
            self.init_points = np.array([v])
            # Distance n x c matrix
            D = np.linalg.norm(data - v, axis=1)

            # Select the rest of the centroids
            for i in range(1, self.c):
                D_max = max(D)
                # Probability depending on the distance
                p_aux = np.power(D,2) / D_max**2   
                # Normalized probability
                p = p_aux / sum(p_aux)                  
                # Select a next centroid
                v = data[np.random.choice(idx_data, p = p)]   
                self.init_points = np.concatenate((self.init_points, [v]))
                # Compute distance
                D_aux = np.linalg.norm(data - v, axis=1)
                D = np.array([D, D_aux]).min(axis = 0)

        return self.init_points

    """
    Returns the membership value of the data in data
    """
    def __get_membership(self, data, data_shape):
        # Matrix distance from each centroid to each point
        m_dist = np.power(cdist(data, self.v), 2 / (self.m - 1))
        # Returns 1/(sum_(j=1)^(c)(v_dist_i/v_dist_j)^(2/(m-1)))
        # Nan values are converted to 1
        return np.nan_to_num(
            1 / np.divide(np.repeat(m_dist, self.c, axis = 0),
                          m_dist.reshape(-1,1) ).reshape(-1, self.c, self.c)\
                .reshape(-1, self.c, self.c).sum(axis=1),
            1)

    """
    Calculate the membership, depending on the spatial information
    """
    def __spatial_membership(self, data_shape, kernel, u):
        # Reshape u to store it in the spatial form
        sp_u = u.reshape((data_shape[0], data_shape[1], self.c))
        # Calculates h power to q, and its multiplication by u^p
        up_hq = np.power( 
            cv.filter2D(sp_u, -1, kernel)\
                .reshape((data_shape[0]*data_shape[1], -1)), self.q
            ) * np.power(u, self.p)
        return np.nan_to_num( up_hq / up_hq.sum(axis=-1).reshape(-1,1), 1)

    """
    Update the centroids v with the new values
    """
    def __update_centroids(self, data, data_shape, u):
        # v_j = (sum_(i=0)^n (u_ij*x_i) ) / sum_(i=0)^n (u_ij)
        return np.matmul(u.transpose(), data) / u.sum(axis = 0).reshape(-1,1)

    """
    Fit the cluster to the data
    @param data is a data representation. It has to be a 2d vector
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
        while diff_v > 0.05:
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

