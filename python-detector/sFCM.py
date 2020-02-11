"""
Author: Pablo Luesia Lahoz
Date: February 2020
Description: Class for performance a spacial fuzzy c-means
"""

import numpy as np
import cv2 as cv
import random as r


class sFCM(object):
    """
    Init function.
    @param c is the number of clusters
    @param m controls the fuzzyness
    @param p is the parameter to power u
    @param q is the parameter to power h
    @param NB is the window size of the spacial part
    @param init_points are the optional initial points for the clusters 
    """
    def __init__(self, c=2, m = 2, p=1, q=1, NB = 5, init_points = []):
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
    def __init_v_points(self, data, data_shape):
        # The initial points are not defined by the user
        if self.init_points == []:

            # Flat the data for the point selection
            flat_data = data.reshape((data_shape[0]*data_shape[1], -1))
            idx_data = np.arange(flat_data.shape[0])

            # First centroid
            v = flat_data[np.random.choice(idx_data)]
            self.init_points = [v]
            # Distance
            D = np.linalg.norm(flat_data - v, axis=1)

            # Select the rest of the centroids
            for i in range(1, self.c):
                D_max = max(D)
                # Probability depending on the distance
                p_aux = np.power(D,2) / D_max**2   
                # Normalized probability
                p = p_aux / sum(p_aux)                  
                # Select a next centroid
                v = flat_data[np.random.choice(idx_data, p = p)]   
                self.init_points.append(v)
                # Compute distance
                D_aux = np.linalg.norm(flat_data - v, axis=1)
                D = np.array([D, D_aux]).min(axis = 0)

        return self.init_points

    """
    Returns the membership value of the data in data
    """
    def __get_membership(self, data, data_shape):
        # Diference between the centroid and the point
        v_dist = np.linalg.norm(
            np.repeat(
                np.expand_dims(data, axis = -1), 
                self.c, axis = -1)\
                    - self.v, axis = -1)
        # Repeated values for calculations
        v_dist_r =  np.repeat(
            np.expand_dims(v_dist, axis = -1),
            self.c, axis = -1)
        # Returns 1/(sum_(j=1)^(c)(v_dist_i/v_dist_j)^(2/(m-1)))
        return 1 / np.power( 
            np.divide(v_dist_r,
                     v_dist_r.transpose((0,1,3,2))),
           2 / (self.m - 1)).sum(axis = -1)

    """
    Update the centroids v with the new values
    """
    def __update_centroids(self, data, data_shape, u):
        flat_data = data.reshape((data_shape[0]*data_shape[1], -1))
        flat_u =  u.reshape((data_shape[0]*data_shape[1], -1))
        np.multiply( np.repeat(
            np.expand_dims(data, axis = -1),
            self.c, axis = -1), u).sum(axis = -1)
        return v

    """
    Fit the cluster to the data
    @param data is a data representation. It has to be a 2d vector
    """
    def fit(self, data):
        data_shape = data.shape
        self.v = self.__init_v_points(data, data_shape)
        u = self.__get_membership(data, data_shape)


