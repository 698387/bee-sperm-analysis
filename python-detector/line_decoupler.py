import cv2 as cv
import numpy as np
import math as m
from sklearn.neighbors import KDTree

PI_CUARTER = m.pi/4

""" 
Follow the line which cross the point origin, in both directions
"""
def follow_line(origin, point_tree, available_points, min_length, max_distance,
                max_angle):
    line = np.array([origin])
    point = origin
    
    # Inicializate with the nearest point
    nearest_idx = point_tree.query([point], return_distance = False)[0][0]
    point = available_points[nearest_idx]
    np.append(line, [point])
    theta = m.atan2(point[1] - origin[1], point[0] - origin[0])
    
    """
    Generate all candidate points
    Candidates are nearer than max_distance, and the direction to them from the
    last known point doesn't form more than max_angle degrees. The returned 
    array is sorted by the distance to the objective
    """
    def generate_candidates():
        #######################################################################
        # Indices of all the near points
        idx_candidates = point_tree.query_radius([point], max_distance,
                                                 return_distance = True)
        # The near points
        near_points = \
        [(available_points[idx_candidates[0][0][i]], idx_candidates[1][0][i]) \
         for i in range(0, len(idx_candidates[1][0])) ]
        # Points that are not in the line
        all_candidates = list(filter(lambda x: not x in line, near_points))
        # Generate all the directions to the points
        directions = list(map(lambda x: m.atan2(x[0][1]-point[1],
                                                x[0][0]-point[0]),
                              all_candidates) )
        # Select points that follows the direction
        return np.asarray(
                [(all_candidates[idx][0], all_candidates[idx][1], alpha)\
                for idx, alpha in list(enumerate(directions)) \
                if abs(alpha - theta) < max_angle])
        #######################################################################
    
    """
    Update the point, the direction and the line of line with the most valuable
    of the candidates candidates. The direction will be calculated with the 
    mean of num_point_theta points
    """    
    def update_line(line, candidates, num_point_theta):
        #######################################################################
        # Gives a value depending on the direction and the distance
        i = np.argmax(map(lambda x: \
                          0.3*(1 - (max_distance - x[1])/max_distance) + \
                          0.7*(1 - (max_angle - x[2])/max_angle),
                          candidates))
        point = candidates[i][0]
        # Calculates a mean for the new theta
        theta = sum(map(lambda x: \
                        m.atan2(point[1] - x[1], point[0] - x[0]),
                        line[-num_point_theta:])) / (num_point_theta - 1)
        np.append(line, [point])
        return (point, theta, line)
        #######################################################################
        
    found_end = False
    for i in range(0,3):
        candidates = generate_candidates()
        if len(candidates) == 0:
            found_end = True
            break
        point, theta, line = update_line(line, candidates, i + 2)
        
    # Keep going until it reaches the end of the line
    while(True):
        candidates = generate_candidates()
        if len(candidates) > 0:                 # There are candidates
            point, theta, line = update_line(line, candidates, 5)
        elif not found_end == 0:                # Reaches the end of the line
            # Change the direction of the search
            point = origin
            theta = sum(map(lambda x: \
                        m.atan2(point[1] - x[1], point[0] - x[0]),
                        line[:5])) / 4
            line = np.flip(line)
            found_end = True
            continue
        else:                                   # End of the other line
            break

    new_available_points = line - available_points
    
    if len(line) < min_length:      # check the minimum length
        line = None
    return line, new_available_points

""" 
Extracts all the non parametrics lines from a binary image with the lines in
white
@img is the image to extract the lines
@max_distance is the maximum distance between 2 white pixels to be consider 
                from different lines
@max_angle is the maximum angle to consider that it is to hard the curve to be
            part of the same line (in radians)
"""
def decouple_lines(img, max_distance = 10, min_length = 10, max_angle=m.pi/4):

    tmp_available_pixels = np.where(img == [255])
    # All the white pixels
    available_pixels = np.asarray([tmp_available_pixels[0],
                                   tmp_available_pixels[1]]).transpose()
    # Generates the kd-tree
    pixel_tree = KDTree(available_pixels, metric = 'euclidean')
    
    lines = []                                  # Detected lines
    while len(available_pixels) > 0:
        # Beginning point
        origin = available_pixels[0][0], available_pixels[0][1]
        
        # Find the line that contains the beginning point
        line, available_pixels = follow_line(origin, 
                                             pixel_tree,
                                             available_pixels,
                                             min_length,
                                             max_distance, 
                                             max_angle)
        if line != None:        # It has found a line
            lines.append(line)

    return lines
    
