import cv2 as cv
import numpy as np
import math as m
from sklearn.neighbors import KDTree
import random as r

"""
Sets unaviable the zone near to the point
"""
def set_zone_unavailable(point, point_tree, available_points, radius = 3):
    idx2unaviable = point_tree.query_radius([point], 
                                            radius,
                                            return_distance = False)[0]
    for idx in idx2unaviable:
        available_points[idx] = False
###############################################################################
    
"""
Extracts the parameters to extract the next point, depending on the previous
line
"""
def get_pred_point_params(point, line, num_point_theta):
    # Direction of the line at this point
    v_dir = point - line[-1]
    u_dir = v_dir / np.linalg.norm(v_dir)
    # Vector of the force that curves the line
    v_f = point - line[-2]
    dist = np.linalg.norm(v_f)
    f = line[-2] / dist - u_dir / dist
    return u_dir, f
    return sum(map(lambda x:
                   m.atan2(point[1] - x[1], point[0] - x[0]),
                   line[-num_point_theta:-1])) / num_point_theta
###############################################################################

"""
Returns a metric indicating how good is the point up to the direction 
parameters, for the point origin
"""
def dir_metric(point, origin, dir_params):
    # Predict a position for the point
    dist = np.linalg.norm(point - origin)
    pred_p = origin + dist *(dir_params[0] + dist*dir_params[1])
    # Distance to the prediction point as metric
    return np.linalg.norm(pred_p - point)
###############################################################################
        
"""
Initialize the line with ransac from a set of points
"""
def init_line(point, all_points, point_tree, available_points,
              max_distance, num_point_theta):
    # Select the available near candidates that 
    idx_candidates = point_tree.query_radius([point], max_distance)[0]
    candidates = [all_points[i] for i in idx_candidates 
                  if available_points[i] and (all_points[i] != point).all()]

    # Perform RANSAC to extract the direction
    hits = 0
    dir_params = get_pred_point_params(point, point, 0)
    max_tries = len(candidates)
    tries = 0
    line = [point]       # Line
    while(hits < num_point_theta and tries < max_tries):
        tries = tries + 1
        line = [point]
        # Select 2 random points
        r_p = r.choices(candidates, k=2)
        r_p.sort(key = lambda x: np.linalg.norm(x - point))
        dir_params = get_pred_point_params(point, r_p, 0)
        hits = 0
        # Check the direction with the other points
        for p in candidates:
            metric = dir_metric(p, point, dir_params)
            if metric < 3:
                hits = hits + 1
                line.append(p)
                
    if hits < num_point_theta:  # No direction has been found
        line = [point]
        
    # Set the zone around the line unavailable
    for p in line:
        set_zone_unavailable(p, point_tree, available_points)
        
    # Sort the line depending on the distance to the point
    line.sort(key = lambda x: np.linalg.norm(x - point))
    return line, dir_params
###############################################################################

"""
Give a metric depending on the direction and the distance
"""
def inv_rank(candidate, max_dist):
    return ( (candidate[2]**2)/max_dist**2 ) \
            * ( (candidate[3]**2)/(max_dist*2)**2 )
###############################################################################
    
"""
Generate all candidate points
Candidates are nearer than max_distance, and the direction to them from the
last known point doesn't form more than max_angle degrees. The returned 
array is sorted by the distance to the objective
"""
def generate_candidates(point, dir_params, all_points, point_tree, 
                        available_points, max_distance, max_angle):
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
    
    # Generate differences with the theta from to the points direction
    directions_diff = list(map(lambda x: dir_metric(x[1], point, dir_params),
                               all_candidates) )
    # Select points in the range (-max_angle, max_angle) of the direction
    return np.asarray(
            [(all_candidates[idx][0],   # index
              all_candidates[idx][1],   # point
              all_candidates[idx][2],   # distance
              alpha )                   # direction diff 
            for idx, alpha in list(enumerate(directions_diff))
            if alpha < max_angle ])
###############################################################################
    
"""
Update the point, the direction and the line of line with the most valuable
of the candidates candidates. The direction will be calculated with the 
mean of num_point_theta points
"""    
def update_line(line, candidates, point_tree, available_points,
                num_point_theta, max_distance):
    # Gives a value depending on the direction and the distance
    i = np.argmin(
            list(map(lambda x: inv_rank(x, max_distance), candidates)) )
    point = candidates[i][1]
    # The zone near to the point is now unavailable
    set_zone_unavailable(point, point_tree, available_points)
    # Calculates a mean for the new direction
    dir_params = get_pred_point_params(point, line, num_point_theta)
    line = np.concatenate((line, [point]))
    return (point, dir_params, line)
###############################################################################
    
""" 
Follow the line which cross the point in the origin index, in both directions
"""
def follow_line(origin, point_tree, all_points, available_points, 
                max_distance, max_angle):
    # Initialize the line
    line, dir_params = init_line(origin, all_points, point_tree, 
                                 available_points, max_distance, 5)
    # Has it been correctly initializied
    if len(line) == 1:
        return line
    
    # The new point is the last of the line
    point = line[-1]
    found_end = False
    # Keep going until it reaches the end of the line
    while(True):
        candidates = generate_candidates(point, dir_params, all_points, 
                                         point_tree, available_points, 
                                         max_distance, max_angle)
        if len(candidates) > 0:                 # There are candidates
            point, dir_params, line = update_line(line, candidates, point_tree, 
                                                  available_points, 
                                                  min(len(line), 4), max_distance)
        elif not found_end:                     # Reaches the first extrem
            # Change the direction of the search
            point = origin
            line = np.flip(line, axis=0)
            dir_params = get_pred_point_params(point, line, min(len(line), 4))
            found_end = True
            continue
        else:                                   # End of the other extrem
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
def decouple_lines(img, max_distance = 10, min_length = 10, max_angle=m.pi/6):
    t_all_pixels = np.where(img == [255])
    # All the white pixels
    all_pixels = np.asarray([t_all_pixels[1],
                             t_all_pixels[0]]).transpose()
    available_pixels = np.ones(len(all_pixels), dtype=bool)
    # Generates the kd-tree
    pixel_tree = KDTree(all_pixels, metric = 'euclidean')
    
    r.seed()                                # Initialize the random seed
    lines = []                              # Detected lines
    while any(available_pixels):
        # Beginning point
        origin_idx = np.where(available_pixels)[0][0]
        
        # Find the line that contains the beginning point
        line = follow_line(all_pixels[origin_idx], 
                           pixel_tree,
                           all_pixels,
                           available_pixels,
                           max_distance,
                           max_angle)
        
        if len(line) >= min_length:         # It has found a line
            lines.append(line)

    return lines
    
