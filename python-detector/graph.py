"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: Graph implementation. The edges contain positional information
             of his path in the image (skeleton)
"""

import numpy as np
import math as m

ORIGIN_INCIDENCE = "o"
FINAL_INCIDENCE = "e"

class Graph:
    """
    Implementation of a grap class extracted from the skeleton
    """

    class Vertex:
        """
        Vertex class for the graph extracted from a skeleton
        """

        """
        The type of the point is either extreme or cross. It set the incoming edges,
        with the angle of entrance
        """
        def __init__(self, id, type, contour):
            self.type = type                            # Vertex type
            self.id = id                                # Id of the vertex
            self.contour = contour                      # Contour of the vertex

    #End of Vertex class

    class Edge:
        """
        Edge class for the graph extracted from a skeleton
        """

        """ 
        The origin and end points are type either extreme or either cross
        vertices of the graph
        """
        def __init__(self, id, path_points, overlapping):
            self._path = path_points    # List with all the points of the arc
            self.id = id                # id of the edge
            self.overlapping = overlapping  # Is it an overlapping edge?

        """
        Return the travel of the edge
        """
        def path(self):
            return self._path

        """
        Add a new point to the path of the edge
        """
        def add_path_point(self, point):
            self._path = np.append(self._path, [point], axis = 0)

        """
        Add a new points to the path of the edge
        """
        def add_path_points(self, points):
            self._path = np.append(self._path, points, axis = 0)


    # End of Edge class

    """
    Init function for the graph
    @param vertices is a list with all the vertices
    @param edges is a list with all the edges
    @param vertices_per_edge is a list of tuples in which the id of an edge
                             is equal to the index in connections. The tuple
                             value are the two vertices of the edge
    @param n_points_angle is the number of points in the edge used for 
                          extracting the incidence angle to the vertices
                          it contains it
    """
    def __init__(self, 
                 vertices = [], 
                 edges = [],
                 vertices_per_edge = np.empty((0,2),dtype = int), 
                 n_points_angle = 7):
        self.vertices = vertices
        self.edges = edges
        self.vertices_in_edge = vertices_per_edge
        self.n_incident_p = n_points_angle

        # A vertex v will have a list of tuples (id,theta), in which id is the
        # identificator of an entrance vertex, with the incident angle theta.
        # The identificator of v is used as index in edges_in_vertex
        self.edges_in_vertex = [[] for _ in range(0,len(self.vertices)) ]
        for i in range(0, len(self.edges)):
            __edge_in_vertex_extractor(self, i)

    # Extracts the entrance angle to the vertices of the edge <<id>>
    def __edge_in_vertex_extractor(self, id):
        # Path and length of the edge
        edge_path = self.edges[id].path()
        path_len = len(edge_path)
        # Id of vertices containing the edge
        vertex_o, vertex_e = self.vertices_in_edge[id]
        # Number of points to extract the vertex
        num_angle_points = min(path_len, self.n_incident_p)
        # Mean of the directions to extract the ingress angle
        do_x, do_y = -np.sum(np.diff(edge_path[0:num_angle_points],
                                  n=1, axis = 0),
                          axis = 0)
        theta_o = m.atan2(do_y, do_x)
        de_x, de_y = np.sum(np.diff(edge_path[-num_angle_points:],
                                  n=1, axis = 0),
                          axis = 0)
        theta_e = m.atan2(de_y, de_x)

        # Append to the list of edges in vertex
        self.edges_in_vertex[vertex_o].append((id, ORIGIN_INCIDENCE, theta_o))
        self.edges_in_vertex[vertex_e].append((id, FINAL_INCIDENCE, theta_e))


    # Add a vertex to the graph
    def add_vertex(self, vertex):
        np.append(self.vertices, vertex)

    # Add an edge to the graph. The two vertices are in the graph
    def add_edge(self, edge, vertex_o, vertex_e):
        self.edges = np.append(self.edges, edge)
        self.vertices_in_edge = np.append(self.vertices_in_edge,
                                         [[vertex_o, vertex_e]], 
                                         axis = 0)
        self.__edge_in_vertex_extractor(edge.id)

    # Returns the id of the origin vertex of the edge with id e_id
    def edge_origin(self, e_id):
        return self.vertices_in_edge[e_id][0]

    # Returns the id of the final vertex of the edge with id e_id
    def edge_final(self, e_id):
        return self.vertices_in_edge[e_id][1]