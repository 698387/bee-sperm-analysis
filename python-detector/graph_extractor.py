"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: It extracts all the lines from a skeleton image. The image
             may have information of 2 layers (overlap and normal)
"""

import numpy as np
import math as m
import cv2 as cv
from graph import Graph

# Static kernel variables
eight_neighbor_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype="float32")
elliptical_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
rectangular_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))

"""
Returns the skeleton of the layers image
"""
def __extract_skeleton(layers_img):
    # The first layer is consider background
    binary_img = np.where(layers_img == 0, 0, 255).astype("ubyte")
    # Extracts the skeleton
    return cv.ximgproc.thinning(binary_img, 
                                thinningType = cv.ximgproc.THINNING_GUOHALL)\
                                    .astype("float32").clip(0,1)


"""
Performs a 8-neighbor convolution, and returns the number of neighbors of
to each pixel
"""
def __extract_num_neighbors(skeleton):
    return cv.filter2D(skeleton, -1, eight_neighbor_kernel) * skeleton

"""
Computes the local direction of the pixel, given the 8-neighbor
"""
def __extract_local_direction(skeleton):
    return np.arctan2(cv.filter2D(skeleton, -1, i_vector_kernel) * skeleton, 
                    cv.filter2D(skeleton, -1, j_vector_kernel) * skeleton)

"""
Find the extrems and the intersection points
"""
def __find_interest_points(skeleton):
    # Count the number of neighbor pixels
    num_neighbors = __extract_num_neighbors(skeleton)
    # Eliminate points not connected
    skeleton[num_neighbors == 0] = 0
    # If only one neighbor, it is an extrem
    extrem_points = num_neighbors == 1
    # If more than 2 neighbors, it is an intersection
    intersection_point = num_neighbors > 2
    return extrem_points, intersection_point


"""
Given the contour of an intersection area, it extracts all the pixels that
belong to segments that finish in the vertex
"""
def __init_segment_from_intersection(skeleton, vertex_map, vertex):
    intersection_contour = vertex.contour
    id = vertex.id
    # Rectangle to avoid using the whole image
    bound_rect = cv.boundingRect(intersection_contour)
    x_min = max(bound_rect[1] - 1, 0)
    x_max = bound_rect[1] + bound_rect[3] + 1
    y_min = max(bound_rect[0] - 1, 0)
    y_max = bound_rect[0] + bound_rect[2] + 1

    # Extracts local areas of the original images
    local_sk = skeleton[x_min:x_max, y_min:y_max]
    local_v_map = vertex_map[x_min:x_max, y_min:y_max]
    vertex_area = np.where(local_v_map == id, 1, 0).astype("ubyte")

    searching_area = cv.dilate(vertex_area, kernel = rectangular_kernel)

    sk_in_area = np.logical_and(local_sk, searching_area)
    points_in_area = np.logical_and(sk_in_area, np.logical_not(vertex_area))

    return np.argwhere(points_in_area) + np.array([x_min, y_min])

"""
Extract the vertices of the graph from the skeleton. The vertices are interest 
points. It returns the vertex map, in which each pixel has the value of the 
vertex it belongs and -1 if it belongs to none, and the list of vertices found
"""
def __extract_vertices(skeleton):
    # Detect points of interest
    extrem_p, intersec_p = __find_interest_points(skeleton)
    # Set intersection areas instead of points
    intersec_area = cv.dilate(intersec_p.astype("ubyte"), elliptical_kernel)

    # Map of the vertices id in the image
    vertex_map = np.full(skeleton.shape, -1) 
    # List with all the vertex
    vertices = np.array([], dtype = object)

    n_vertex = 0
    # Fill the map and the list with the extrems
    extrems =  np.argwhere(extrem_p)
    for single_extrem_coord in extrems:
        # Creates the vertex
        v = Graph.Vertex(n_vertex, "extrem", [single_extrem_coord])    
        vertices = np.append(vertices, v)       # Add the vertex to the list
        # Update the vertex map
        vertex_map[tuple(single_extrem_coord)] = n_vertex
        n_vertex += 1

    # Fill the map with the intersection areas
    intersec_contours, _ = cv.findContours(intersec_area, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)
    for single_intersec_contour in intersec_contours:
         # Creates the vertex
        v = Graph.Vertex(n_vertex, "intersection", single_intersec_contour)   
        vertices = np.append(vertices, v)   # Add the vertex to the list
        # Update the vertex map
        cv.fillPoly(vertex_map, pts = [single_intersec_contour], color=(n_vertex))
        n_vertex += 1

    return vertices, vertex_map


"""
Returns true, iff the point <<p>> is not a vertex, up to the vertex map 
"""
def __is_vertex(p, vertex_map):
    return vertex_map[tuple(p)] >= 0 


"""
Returns the local coordinates of all the candidates to continue the segment
given the point <<p>>
"""
def __local_candidates(skeleton, p):
    # Offset for restoring the indices to the skeleton
    offset_x = -1; offset_y = -1

    # x limits
    x_min = p[0]-1
    if 0 > x_min:
        x_min = 0
        offset_x = 0

    # y limits
    y_min = p[1]-1
    if 0 > y_min:
        y_min = 0
        offset_y = 0

    # 8 neighbor extraction from the skeleton
    eight_neighbor = skeleton[x_min:p[0]+2, y_min:p[1]+2]
    # Find the available pixels with local coordinates
    return np.argwhere(eight_neighbor) + np.array([offset_x, offset_y])


"""
Extracts the candidate points of a given point <<p>> to continue the contour
"""
def __candidate_points(skeleton, p):
    return __local_candidates(skeleton, p) + p


"""
Given a point <<p>> of the skeleton image, it extracts the next point, and sets
the previous point to 0 in the skeleton image
"""
def __next_point(skeleton, p):
    return __local_candidates(skeleton, p)[0] + p 


"""
Return the line that exists following the skeleton from line_o. Line_o can be
a subsgement.It stops in a vertex pixel
"""
def __follow_line(skeleton, vertex_map, vertices, line_o):

    line = line_o                                   # Initialize the line
    skeleton[tuple(line_o[-1])] = 0                 # Clean the point
    # Cleaning the skeleton avoids to be reconsidered
    next_p = __next_point(skeleton, line[-1])       # First next point

    skeleton[tuple(next_p)] = 0
    line = np.append(line, [next_p], axis = 0)      # Add the point to the line

    while not __is_vertex(next_p, vertex_map):      # Follows  until a vertex
        next_p = __next_point(skeleton, line[-1])   # Next point
        skeleton[tuple(next_p)] = 0
        line = np.append(line, [next_p], axis = 0)  # Add the point to the line

    # A intersection pixel can be shared with other segment
    if vertices[vertex_map[tuple(line[-1])]].type == "intersection":
        skeleton[tuple(line[-1])] = 1

    return line


"""
Extracts the lines asociated with the spermatozoon from the different layers
@param layers_img is the image with each pixel predicted to belong each class
@param overlapping indicates if there is an overlapping layer
@param debug if true, it will show debug messages and images
@return A list with all the lines found. Each line is an array of points
"""
def extractGraph(layers_img, overlapping = False, debug = False):
    # Extracts the skeleton
    skeleton = __extract_skeleton(layers_img)
    # Add a border, to prevent extracting the wrong neighbors
    skeleton = cv.copyMakeBorder(skeleton, 1, 1, 1, 1, cv.BORDER_CONSTANT, 0)
    # Extract vertices
    vertices, vertex_map = __extract_vertices(skeleton)

    extrems = np.array(list(map(lambda x: x.contour[0], filter(lambda x: x.type == "extrem", vertices)) ))
    intersec_v = list( filter(lambda x: x.type == "intersection", vertices) )
    intersec_contours = np.array( list(map(lambda x: x.contour, intersec_v) ) )
    
 
    lines = []
    for single_extrem in extrems:
        if skeleton[tuple(single_extrem)]:
            try:
                lines.append(__follow_line(skeleton, vertex_map, vertices, [single_extrem]))
            except:
                print("ups")

    other_origins = np.empty((0,2), dtype = int)
    for i in intersec_v:
        other_origins = np.append(other_origins, __init_segment_from_intersection(skeleton, vertex_map, i), axis = 0)

    g = Graph(vertices = vertices)

    if debug:
        font = cv.FONT_HERSHEY_SIMPLEX
        img2show = cv.cvtColor((skeleton*255).astype("ubyte"), cv.COLOR_GRAY2BGR)
        cv.putText(img2show, "Skeleton", (2,15), font, .5,(255,255,255),1,cv.LINE_AA)
        drawable_lines = []
        for l in lines:
            drawable_lines.append(np.array(list(map(lambda x: [x[1], x[0]], l) )) )

        cv.polylines(img2show, pts = drawable_lines, color=[255, 0, 255], isClosed = False)
        img2show[(extrems[:,0], extrems[:,1])] = [0, 255, 0]
        cv.putText(img2show, "Extrem points", (2,30), font, .5,(0, 255, 0),1,cv.LINE_AA)
        cv.fillPoly(img2show, pts = intersec_contours, color=[0, 0, 255])
        cv.putText(img2show, "Crosses", (2,45), font, .5,(0,0,255),1,cv.LINE_AA)
        
        img2show[(other_origins[:,0], other_origins[:,1])] = [0, 255, 255]
        cv.imshow("line extractor debug", img2show)




