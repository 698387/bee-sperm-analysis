"""
Author: Pablo Luesia Lahoz
Date: June 2020
Description: Extracts from a single graph, all the spermatozoids it can find
"""

import numpy as np
import random as r
import math as m
import graph as G
from itertools import compress

# Given a vector of points v, it extracts the evolution per pixel of the 
# direction of the vector
def __curve_evolution(v):
    return np.mean(np.diff(v, n=2, axis = 0), axis = 0)


# Return all the evolutions of the curves given by the edges path. The key
# for each vector is the id of the edge
def __extract_all_edge_evolutions(g):
    evo_per_edge = np.empty([len(g.edges), 2])
    i = 0
    for e in g.edges:
        if len(e.path()) < 3:   # Length is not enough to extract the evolution
            # No evolution
            evo_per_edge[i] = 0 
        else:                   # Lenght is enough
            # Calculates the evolution 
            evo_per_edge[i] = __curve_evolution(e.path())
        i += 1
    return evo_per_edge


# Given two edges e1 and e2 sharing one vertex v, in the graph g
# it returns a value depending on the difference of angles, and 
# the evolution of the edges curves (evo1 and evo2). The higher, 
# the more probablity have them to be similar. Negative values
# means it is not posible
def __similarity_value(edge, vertex_type4edge, theta_edge, v, g, theta_cell, 
                       evo_cell, evo_edge, n_visited_edges, 
                       max_theta, max_evo_dist):
    # Extracts the difference between angles
    # Both are enter values of the vertex
    delta_theta = abs(abs(theta_edge - theta_cell) - m.pi)

    # Evolution difference
    # Invert the evolution if needed
    if g.edge_origin(edge) == v and vertex_type4edge == G.ORIGIN_INCIDENCE:
        evo_edge = -evo_edge
    
    evo_diff = evo_edge - evo_cell      # Difference dimension wise
    # Euclidean square distance
    evo_dist = np.dot(evo_diff, evo_diff)

    # Calculates the value
    #if delta_theta > max_theta or evo_dist > max_evo_dist:
    if delta_theta > max_theta or evo_dist > max_evo_dist:
        return -1
    
    # The value depending on how many times a vertex has been visited
    n_visited_value = 0.6*m.exp(-n_visited_edges[edge]*0.6)
    angle_value = (1-(delta_theta / max_theta))
    evo_value = (1-(evo_dist / max_evo_dist))
    edge_len = len(g.edges[edge].path())
    len_value = min(1, edge_len / g.n_incident_p)
    return (angle_value + evo_value) * n_visited_value / len_value



# Returns the input angle to the vertex from the spermatozoid
def __sperm_angle(spermatozoid_path, g, inverted):
    n_angle_pixels = min(len(spermatozoid_path), g.n_incident_p)

    if inverted:
        d_x, d_y = -np.sum(np.diff(spermatozoid_path[0:n_angle_pixels],
                                    n=1, axis = 0),
                            axis = 0)
    else:
        d_x, d_y = np.sum(np.diff(spermatozoid_path[-n_angle_pixels:],
                                    n=1, axis = 0),
                            axis = 0)

    return m.atan2(d_y, d_x)     


# Returns the most likely edge, the type of the vertex to the selected edge,
# and a boolean indicating if any vertex has been found
def __select_edge(g, current_edge, current_v_t4e, current_vertex, theta_sperm, 
                  curve_evolution,  evolutions_edge, n_visited_edges,
                  max_theta, max_evo_dist):
    err_value = 0.05
    max_value = err_value   # Very low values wouldn't be considered
    selected_edge = None
    selected_v_type4edge = None

    # Select the most similar edge
    for (candidate_edge, type_for_edge, theta_edge) \
        in g.edges_in_vertex[current_vertex]:
        # Omit same edge entrance to the vertex
        if candidate_edge != current_edge or type_for_edge != current_v_t4e:
            # Calculates the similarity value
            sim_value = __similarity_value(candidate_edge,
                                            type_for_edge,
                                            theta_edge,
                                            current_vertex,
                                            g,
                                            theta_sperm,
                                            curve_evolution,
                                            evolutions_edge[candidate_edge],
                                            n_visited_edges,
                                            max_theta,
                                            max_evo_dist)

            # If the values is the maximum, it takes it
            if sim_value > max_value:
                # Updates value, edge, and the type of vertex for the edge
                max_value = sim_value
                selected_edge = candidate_edge
                selected_v_type4edge = type_for_edge

    # Return found values
    return (selected_edge, selected_v_type4edge, max_value > err_value)


# It follows the most likely path for the edge e to the vertex v. It returns
# the path. If inverted is true, the path is appended to the beginning
def __follow_edges(e, vertex_type4edge, v, g, spermatozoid_path, 
                   n_visited_edges, evolutions_edge, max_theta, max_evo_dist,
                   max_length, inverted = False):
    current_edge = e
    current_vertex = v
    current_v_t4e = vertex_type4edge
    # The evolution of the edge
    curve_evolution = (-1*inverted)*evolutions_edge[e]

    while g.vertices[v].type != "extrem" \
        and len(spermatozoid_path) < max_length:
        # Extracts the angle of the spermatozoid to the vertex
        theta_sperm = __sperm_angle(spermatozoid_path, g, inverted)
        selected_edge, selected_v_type4edge, found_edge = \
            __select_edge(g, current_edge, current_v_t4e, current_vertex, 
                          theta_sperm, curve_evolution, evolutions_edge,
                          n_visited_edges,
                          max_theta, max_evo_dist)
        
        if not found_edge:      # There is no new edge
            break

        current_edge = selected_edge
        n_visited_edges[current_edge] += 1
        new_edge_curve_evo = evolutions_edge[e]
        #Appends the edge, and updates the vertex of the spermatozoid
        if inverted:        # The direction is inverted
            # It inverts new edge if needed
            if g.edge_origin(current_edge) == current_vertex\
              and selected_v_type4edge == G.ORIGIN_INCIDENCE:
                current_vertex = g.edge_final(current_edge)
                current_v_t4e = G.FINAL_INCIDENCE
                spermatozoid_path = np.append(
                                        np.flip(g.edges[current_edge].path(),
                                                axis = 0), 
                                             spermatozoid_path,
                                             axis = 0)
            else:
                current_vertex = g.edge_origin(current_edge)
                current_v_t4e = G.ORIGIN_INCIDENCE
                spermatozoid_path = np.append(g.edges[current_edge].path(),
                                                spermatozoid_path,
                                                axis = 0)
                new_edge_curve_evo =  -new_edge_curve_evo
                
        else:           # The direction is not inverted
            # It inverts the new edge if needed
            if g.edge_final(current_edge) == current_vertex\
              and selected_v_type4edge == G.FINAL_INCIDENCE:

                current_vertex = g.edge_origin(current_edge)
                current_v_t4e = G.ORIGIN_INCIDENCE
                spermatozoid_path = np.append(spermatozoid_path,
                                              np.flip(
                                                  g.edges[current_edge].path(),
                                                  axis = 0),
                                              axis = 0)
                new_edge_curve_evo = -new_edge_curve_evo
            else:
                current_vertex = g.edge_final(current_edge)
                current_v_t4e = G.FINAL_INCIDENCE
                spermatozoid_path = np.append(spermatozoid_path,
                                                g.edges[current_edge].path(),
                                                axis = 0)
        # Curve evolution update
        if g.edges[current_edge].overlapping:
            curve_evolution = ( curve_evolution + new_edge_curve_evo ) /2
        else:
            curve_evolution = new_edge_curve_evo
        
    return spermatozoid_path


# Extracts an spermatozoid from an edge e of the graph g.
# evolutions_edge represents the curve evolution of each edge in the graph
def __extract_spermatozoid(e, g, n_visited_edges, evolutions_edge, 
                           max_theta, max_evo_dist, max_length):
     # Origin and end vertices of the spermatozoid
    v_o = g.edge_origin(e.id); v_e = g.edge_final(e.id)
    # Path of the spermatozoid 
    spermatozoid_path = e.path()
    spermatozoid_path = __follow_edges(e.id,
                                       G.ORIGIN_INCIDENCE,
                                       v_o,
                                       g,
                                       spermatozoid_path,
                                       n_visited_edges,
                                       evolutions_edge,
                                       max_theta,
                                       max_evo_dist,
                                       max_length,
                                       inverted = True)
    spermatozoid_path = __follow_edges(e.id,
                                       G.FINAL_INCIDENCE,
                                       v_e,
                                       g,
                                       spermatozoid_path,
                                       n_visited_edges,
                                       evolutions_edge,
                                       max_theta,
                                       max_evo_dist,
                                       max_length,
                                       inverted = False)
    return spermatozoid_path


# Remove the paths that are inside a longer path in the set
def __filter_subpaths(segment_paths):
    # Dictionary to find common subpaths
    drawn_paths = {}      # Each key stores the value of the indices
    idx = 0
    # Fill the dictionary
    for s in segment_paths:
        for p in s:
            if tuple(p) in drawn_paths:
                drawn_paths[tuple(p)].append(idx)
            else:
                drawn_paths[tuple(p)] = [idx]
        idx += 1

    valid_segment_paths = np.full((len(segment_paths)), True)
    # Consult the dictionary for those paths to be removed
    # Montecarlo -> Not all points are used
    mc_samples = 7
    idx = 0
    for s in segment_paths:
        seen_indices = {idx}        # The seen indices on the drawn_paths
        shared_point_indices = []   # The indices on the selected points
        # Select mc_samples random elements
        for p in r.choices(s, k=mc_samples):
            # Append to the indices
            shared_point_indices.append(drawn_paths[tuple(p)])
            # Update the seen indices
            for indices in drawn_paths[tuple(p)]:
                seen_indices.add(indices)

        seen_indices.remove(idx)    # No need to see the current segment
        for outter_idx in seen_indices: # All the seen indices
            # If they share all the points
            if all([outter_idx in shared_point_idx \
                    for shared_point_idx in shared_point_indices]): 
                # Removes the current segment if it is shorter than the outter
                if len(segment_paths[outter_idx]) > len(segment_paths[idx]):
                    valid_segment_paths[idx] = False
                    break
        idx += 1
    # Return the valid segments
    return list(compress(segment_paths, valid_segment_paths))
        

"""
Given the morphological graph g, it extracts all the spermatozoids it can find
@param g is the morphological graph
@param n_points_cell is the number of points to codify each cell found. If 
                     negative, it returns all the points it cross
@param max_theta is the max difference between the entrance angle of two edges 
                in a vertex to be considered the same
@param max_evo is the max difference between two edges of a vertex in their 
               curve evolution to be considered the same
@param max_length is the maximum length of the cell in pixels of the original
                  image
"""
def cells_from_single_image(g, n_points4cell = 32, 
                            max_theta = m.pi/18,
                            max_evo_d = 0.01,
                            max_length = 1000):
    evolutions_edge = __extract_all_edge_evolutions(g)    # All edge evolutions
    n_visited_edges = np.full(len(g.edges), 0)        # Times visited each edge
    all_cell_path = []                                 # All spermatozoid cells

    # For each single edge, extract the spermatozoid
    shuffled_edges = g.edges.copy()
    r.shuffle(shuffled_edges)
    for e in list(filter(lambda x: not x.overlapping, shuffled_edges)):
        v_o = g.vertices[g.edge_origin(e.id)]
        v_e = g.vertices[g.edge_final(e.id)]
        if (v_o.type == "extrem" or v_e.type == "extrem") and not n_visited_edges[e.id]:
        #if not (e.overlapping or n_visited_edges[e.id]):     # It hasn't been visited
            s = __extract_spermatozoid(e, g, n_visited_edges, evolutions_edge, 
                           max_theta, max_evo_d, max_length)
            all_cell_path.append(s)

    # Extract spermatozoids from overlapping edges
    for e in shuffled_edges:
        if not n_visited_edges[e.id]:     # It hasn't been visited
            s = __extract_spermatozoid(e, g, n_visited_edges, evolutions_edge, 
                           max_theta, max_evo_d, max_length)
            all_cell_path.append(s)

    filtered_cell_paths = __filter_subpaths(all_cell_path)
    # Returns the number of given points for each cell
    if n_points4cell > 0:
        mult_indices = np.arange(0, n_points4cell-1)
        return list(map(lambda cell:
                        cell[np.append(
                            (mult_indices*(len(cell)/n_points4cell))\
                             .astype(int), [-1] )],
                        filter(lambda cell: len(cell) >= n_points4cell,
                               filtered_cell_paths)))
    else:
        return filtered_cell_paths 

