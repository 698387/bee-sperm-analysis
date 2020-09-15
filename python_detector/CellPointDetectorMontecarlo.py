# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:57:11 2019

@author: Pablo Luesia

@description: Point into cell detection using montecarlo
"""

import math
import random
import statistics as st
from sklearn.neighbors import KDTree


"""
It returns the direction of the contour with index c_idx on the point with
index p_idx, based on the n closer points
"""
def direction(contours, c_idx, p_idx, n):
    c = contours[c_idx]                 # Extracts the contour
    dist_to_point = n // 2       # Distance to the point from index
    
    # Obtain the upper and the lower indexes, checking the range of the vector
    lower_idx = p_idx - dist_to_point
    upper_idx = p_idx + dist_to_point
    if lower_idx < 0:
        upper_idx = upper_idx - lower_idx
        lower_idx = 0
    if upper_idx >= len(c):
        upper_idx = len(c)
    
    # Obtain the angle
    delta_x = c[lower_idx][0][0] - c[upper_idx][0][0]
    delta_y = c[lower_idx][0][1] - c[upper_idx][0][1]
    return math.atan2(delta_y, delta_x)

"""
It returns a list of inner points of the cells, from a list of contours
"""
def cellInnerPoints(contours, n_points=200, max_dist = 15, direction_points=5):
    # Transform all the contours in a simple vector of contours
    all_contour_points = []
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            all_contours_points.append(contours[i][0][j],(i,j))
    total_n_points = len(all_contour_points)
    
    # Fill a kdtree for the search of neighbor points
    points_tree = KDTree(total_n_points)
    
    # Extract n_points random points from the contours
    for k in range(n_points):
        # Select a random point from a contour
        p_idx = random.randint(n_points)
        # TODO 
        points_tree.query_radius(total_n_points[p],max_dist)
        
    
            
    
