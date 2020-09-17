"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: It extracts all the lines from a skeleton image. The image
             may have information of 2 layers (overlap and normal)
"""

import numpy as np
import math as m
import cv2 as cv
from python_detector.graph import Graph

__debug_graph = False


######################### Image treatment variables and code ##################

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

################## INTERSECTION EXTRACTION OF POINTS ######################

"""
Given the contour of an intersection area, it extracts all the pixels that
belong to segments that finish or start in the vertex
"""
def __intersection_candidates(sk_area, vertex_area):
    # Expand the area of the vertex
    searching_area = cv.dilate(vertex_area, kernel = rectangular_kernel)
    # Extract all available pixels of the skeleton in the expanded area
    sk_in_area = np.logical_and(sk_area, searching_area)
    # Eliminate pixels that belong to the intersection
    points_in_area = np.logical_and(sk_in_area, np.logical_not(vertex_area))

    return np.argwhere(points_in_area)


"""
Returns the indices of the area of interest for a contour
"""
def __vertex_area_index(vertex):
    contour = vertex.contour
    # Horizontal bounding rectangle
    bound_rect = cv.boundingRect(contour)
    x_min = max(bound_rect[1] - 1, 0)
    x_max = bound_rect[1] + bound_rect[3] + 1
    y_min = max(bound_rect[0] - 1, 0)
    y_max = bound_rect[0] + bound_rect[2] + 1

    return (x_min, x_max, y_min, y_max)


"""
Extracts the areas of interest for a intersection vertex
"""
def __intersection_area(vertex, skeleton, vertex_map, boundary_indices):
    
    # Extract parameters from the vertex
    id = vertex.id

    # Boundary coordinates for optimization
    x_min, x_max, y_min, y_max = boundary_indices 

    # Extracts local areas of the original images
    local_sk = skeleton[x_min:x_max, y_min:y_max]
    local_v_map = vertex_map[x_min:x_max, y_min:y_max]
    # Area of the intersection vertex
    vertex_area = np.where(local_v_map == id, 1, 0).astype("ubyte")

    return (local_sk, local_v_map, vertex_area)


"""
It returns a list with the current point, and the vertex pixel that is his
8-neighbor. Return an empty list if there is no vertex pixel in his
8-neighbor
"""
def __vertex_neighbor(skeleton, vertex_map, point):
    # Extract all candidates
    candidates = __candidate_points(skeleton, point)
    # Return a list of points in whose neighbor there is a vertex
    return [np.append([c_point], [point], axis = 0) for c_point in candidates \
        if __is_vertex(vertex_map, c_point)]


"""
Find the origins of all segments from an intersection vertex and return
it in a list
"""
def __origins_from_intersection(skeleton, vertex_map, vertex):
    # Sub-images of interest for the vertex
    boundary_idx = __vertex_area_index(vertex)
    local_skeleton, local_vertex_map, local_vertex_area = \
        __intersection_area(vertex, skeleton, vertex_map, boundary_idx)

    # Candidates extraction
    candidates = __intersection_candidates(local_skeleton, local_vertex_area)
    
    # Extract lines from the good candidates
    list_origins = np.empty((0,2,2), dtype = int)

    for c in candidates:
        origin = __vertex_neighbor(local_skeleton, local_vertex_map, c)
        # Only interested in one element origins. More means surrounded by
        # vertex, less means no vertex near
        if len(origin) == 1 and not __is_vertex(vertex_map, origin[0][-1]):
            list_origins = np.append(list_origins, origin, axis = 0)

    return list_origins + np.array([boundary_idx[0], boundary_idx[2]])


######################## Vertex extraction ####################################

"""
Find the extrems and the intersection points
"""
def __find_interest_points(skeleton):
    # Count the number of neighbor pixels
    num_neighbors = __extract_num_neighbors(skeleton)
    # If only one neighbor, it is an extrem
    extrem_points = num_neighbors == 1
    # If more than 2 neighbors, it is an intersection
    intersection_point = num_neighbors > 2
    return extrem_points, intersection_point


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
    extrems =  np.argwhere( np.logical_and(extrem_p,
                                           np.logical_not(intersec_area)) )
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
def __is_vertex(vertex_map, p):
    return vertex_map[tuple(p)] >= 0 


########################## Segment points extraction ##########################

"""
Returns the local coordinates of the 8neighbor of p, whose value is higher 
than v in src
"""
def __local_filter(src, p, v):
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

    # 8 neighbor extraction from the image source
    eight_neighbor = src[x_min:p[0]+2, y_min:p[1]+2]
    # Find the pixels with local coordinates higher than the value
    return np.argwhere(eight_neighbor > v) + np.array([offset_x, offset_y])


"""
Returns the local coordinates of all the candidates to continue the segment
given the point <<p>>
"""
def __local_candidates(skeleton, p):
    # Find the available pixels with local coordinates
    return __local_filter(skeleton, p, 0)


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
    candidates = __local_candidates(skeleton, p)
    # Check if it has candidates
    if len(candidates) == 0:
        return []
    else:
        # Rescale to the image size
        return candidates[0] + p 

"""
Return a non-connected neighbor that is a vertex, or an empty list
"""
def __non_connected_vertex(vertex_map, point):
    vertex_candidates = __local_filter(vertex_map, point, -1)
    if len(vertex_candidates) == 0:
        return []
    else:
        return vertex_candidates[0] + point

"""
Return the line that exists following the skeleton from line_o. Line_o can be
a subsgement.It stops in a vertex pixel
"""
def __follow_line(skeleton, vertex_map, vertices, line_o):

    line = np.array(line_o).reshape(-1, 2)          # Initialize the line
    # Clean the points to avoid being reconsidered
    skeleton[tuple(np.swapaxes(line, 0, 1))] = 0     
    next_p = __next_point(skeleton, line[-1])       # First next point
    # Check in case no points are available
    if len(next_p) == 0:
        next_p = __non_connected_vertex(vertex_map, line[-1])
        if len(next_p) == 0:
            return []                               # No vertex, no segment

    skeleton[tuple(next_p)] = 0
    line = np.append(line, [next_p], axis = 0)      # Add the point to the line

    while not __is_vertex(vertex_map, next_p):      # Follows  until a vertex
        next_p = __next_point(skeleton, line[-1])   # Next point
        # Check in case no points are available
        if len(next_p) == 0:
            next_p = __non_connected_vertex(vertex_map, line[-1])
            if len(next_p) == 0:
                return []                           # No vertex, no segment

        skeleton[tuple(next_p)] = 0
        line = np.append(line, [next_p], axis = 0)  # Add the point to the line

    # A intersection pixel can be shared with other segment
    if vertices[vertex_map[tuple(line[-1])]].type == "intersection":
        skeleton[tuple(line[-1])] = 1

    if vertices[vertex_map[tuple(line[0])]].type == "intersection":
        skeleton[tuple(line[0])] = 1

    return line


################################ Edge extractor ###############################

"""
Extract all segments found in the image, and returns it as a list
"""
def __extract_segments(skeleton, vertex_map, vertices):
    segment_v = []                                            # Segment list
    for vertex in vertices:
        # Seed for generating new segments
        segment_seeds = []                                
        # The vertex is a single extrem
        if vertex.type == "extrem":                 # EXTREM VERTEX
            extrem_origin = vertex.contour[0]
            if skeleton[tuple(extrem_origin)]:                # Unvisited pixel
                segment_seeds = [extrem_origin]               # Add the seed

        elif vertex.type == "intersection":         # INTERSECTION VERTEX
            # Extracts all posible centers
            for intersection_origin in __origins_from_intersection(skeleton,
                                                                   vertex_map,
                                                                   vertex):
                if skeleton[tuple(intersection_origin[-1])]:  # Unvisited pixel
                    segment_seeds.append(intersection_origin) # Add the seed

        # Follows and append the line
        for seed in segment_seeds:
            n_segment = __follow_line(skeleton,
                                    vertex_map,
                                    vertices,
                                    seed)
            if len(n_segment) > 0:
                segment_v.append(n_segment)

    return segment_v

"""
It returns the number of pixels of each type of the segment <<segment>>, 
deping on the layer image layers_img. Only two types as max are considered
"""
def __count_pixel_type_segment(layers_img, segment):
    layer_count = {0: 0, 1: 0, 2: 0}

    for next_p in segment:
        point_value = layers_img[tuple(next_p-1)]
        layer_count[point_value] += 1

    return (layer_count[1], layer_count[2])


"""
Add a list of lines in the graph as edges. The vertex ids are defined for each 
point in the vertex_map (no vertex equals to -1 in the vertex_map)
"""
def __insert_edges(graph, vertex_map, segment_v, 
                   vertices, overlap_info, overlap_thres,
                   layers_img):
    edge_counter = 0
    for segment in segment_v:
        # Gets the origin and end vertex ids
        vertex_o = vertex_map[tuple(segment[0])]
        vertex_e = vertex_map[tuple(segment[-1])]

        overlapping_edge = False
        # Checks if it is an overlapping edge
        if vertices[vertex_o].type == "intersection" and \
           vertices[vertex_e].type == "intersection":
            overlapping_edge = True
             
            if overlap_info:    # It exists an overlapping layer
                n_simple, n_overlap = \
                    __count_pixel_type_segment(layers_img, segment)
                overlapping_edge = (n_overlap / len(segment)) > overlap_thres
            
        # Creates the edge 
        edge = Graph.Edge(edge_counter, segment, overlapping_edge)
        edge_counter += 1
        # Add the edge to the graph
        graph.add_edge(edge, vertex_o, vertex_e)

"""
Extracts the lines asociated with the spermatozoon from the different layers
@param layers_img is the image with each pixel predicted to belong each class
@param overlapping indicates if there is an overlapping layer
@param debug if true, it will show debug messages and images
@return A list with all the lines found. Each line is an array of points
"""
def extractGraph(layers_img, overlapping_info = False, overlapping_thres = 0.0,
                 n_pixels_angle = 7, debug = False):
    # Debug options
    global __debug_graph
    __debug_graph = debug

    # Extracts the skeleton
    skeleton = __extract_skeleton(layers_img)
    # Add a border, to prevent extracting the wrong neighbors
    skeleton = cv.copyMakeBorder(skeleton, 1, 1, 1, 1, cv.BORDER_CONSTANT, 0)
    # Extract vertices
    vertices, vertex_map = __extract_vertices(skeleton)
    # Extract edges
    segments = __extract_segments(skeleton, vertex_map, vertices)

    g = Graph(vertices = vertices, n_points_angle = n_pixels_angle)
    __insert_edges(g, 
                   vertex_map, 
                   segments, 
                   vertices, 
                   overlapping_info,
                   overlapping_thres,
                   layers_img)

    if __debug_graph:
        font = cv.FONT_HERSHEY_SIMPLEX
        img2show = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype = "ubyte")
        
        cv.putText(img2show, "Edges", (2,15), font, .5,(0, 255, 0),1,cv.LINE_AA)
        cv.putText(img2show, "Overlapping Edges", (2,30), font, .5,(255, 255, 0),1,cv.LINE_AA)
        cv.putText(img2show, "Vertices", (2,45), font, .5,(0,0,255),1,cv.LINE_AA)

        img_transposed_lines = np.zeros((skeleton.shape[1], skeleton.shape[0]), dtype = "ubyte")
        ol_segments_bool = [x.overlapping for x in g.edges]
        ol_segments = [i for (i, v) in zip(segments, ol_segments_bool) if v]
        cv.polylines(img_transposed_lines, pts = segments, color=[1], isClosed = False)
        cv.polylines(img_transposed_lines, pts = ol_segments, color=[2], isClosed = False)
        img2show[np.transpose(img_transposed_lines) == 1] = [0,255,0]
        img2show[np.transpose(img_transposed_lines) == 2] = [255,255,0]

        img2show[vertex_map >= 0] = [0, 0, 255]

        cv.imshow("graph extractor debug", img2show)

    return g




