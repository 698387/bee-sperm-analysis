import cv2 as cv
import numpy as np
import math as m
from sklearn.neighbors import KDTree

PI_CUARTER = m.pi/4

# Sets unaviable the zone near to the point
def zone_unaviable(point, point_tree, available_points, radius = 2):
    idx2unaviable = point_tree.query_radius([point], 
                                            radius,
                                            return_distance = False)[0]
    for idx in idx2unaviable:
        available_points[idx] = False

""" 
Follow the line which cross the point origin, in both directions
"""
def follow_line(origin, point_tree, all_points, available_points, max_distance,
                max_angle):
    line = np.array([origin])
    point = origin
    
    # Indices of all the near points
    idx_near_points = point_tree.query_radius([point], max_distance,
                                              return_distance = True)
    # The nearest points which are available
    idx_near_available = [(idx_near_points[0][0][i], idx_near_points[1][0][i])  
                          for i in range(0, len(idx_near_points[0][0])) 
                          if available_points[idx_near_points[0][0][i]]]
    if not idx_near_available:
        return line             # No near points
    
    # Inicializate with the nearest point
    nearest_idx, _ = min(idx_near_available, key = lambda x: x[1])
    
    available_points[nearest_idx] = False
    point = all_points[nearest_idx]
    line = np.concatenate((line, [point]))
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
        # The near points which are available
        all_candidates = \
        [(idx_candidates[0][0][i],                  # index
          all_points[idx_candidates[0][0][i]],      # point
          idx_candidates[1][0][i])                  # distance                 
         for i in range(0, len(idx_candidates[1][0])) \
         if available_points[idx_candidates[0][0][i]]]
        # Generate all the directions to the points
        directions = list(map(lambda x: m.atan2(x[1][1]-point[1],
                                                x[1][0]-point[0]),
                              all_candidates) )
        # Select points in the range (-max_angle, max_angle) of the direction
        return np.asarray(
                [(all_candidates[idx][0],   # index
                  all_candidates[idx][1],   # point
                  all_candidates[idx][2],   # distance
                  abs(alpha - theta))       # direction diff
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
                          1/(x[2]*x[3]*x[3]),
                          candidates))
        global_idx = candidates[i][0]
        available_points[global_idx] = False        # Point no longer available
        point = candidates[i][1]
        zone_unaviable(point, point_tree, available_points)
        # Calculates a mean for the new theta
        theta = sum(map(lambda x: \
                        m.atan2(point[1] - x[1], point[0] - x[0]),
                        line[-num_point_theta:-1])) / (num_point_theta - 1)
        line = np.concatenate((line, [point]))
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
        elif not found_end:                     # Reaches the end of the line
            # Change the direction of the search
            point = origin
            theta = sum(map(lambda x: \
                        m.atan2(point[1] - x[1], point[0] - x[0]),
                        line[1:5])) / 4
            line = np.flip(line, axis=0)
            found_end = True
            continue
        else:                                   # End of the other line
            break

    return line

""" 
Extracts all the non parametrics lines from a binary image with the lines in
white
@img is the image skeleton to extract the lines
@max_distance is the maximum distance between 2 white pixels to be consider 
                from different lines
@max_angle is the maximum angle to consider that it is to hard the curve to be
            part of the same line (in radians)
"""
def decouple_lines(img, max_distance = 10, min_length = 10, max_angle=m.pi/4):

    t_all_pixels = np.where(img == [255])
    # All the white pixels
    all_pixels = np.asarray([t_all_pixels[1],
                             t_all_pixels[0]]).transpose()
    available_pixels = np.ones(len(all_pixels), dtype=bool)
    # Generates the kd-tree
    pixel_tree = KDTree(all_pixels, metric = 'euclidean')
    
    lines = []                               # Detected lines
    while any(available_pixels):
        # Beginning point
        origin_idx = np.where(available_pixels)[0][0]
        available_pixels[origin_idx] = False
        origin = all_pixels[origin_idx]
        
        
        # Find the line that contains the beginning point
        line = follow_line(origin, 
                           pixel_tree,
                           all_pixels,
                           available_pixels,
                           max_distance, 
                           max_angle)
        
        if len(line) >= min_length:        # It has found a line
            lines.append(line)

    return lines
    
