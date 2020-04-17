"""
Author: Pablo Luesia Lahoz
Date: March 2020
Description: Graph implementation. The edges contain positional information
             of his path in the image (skeleton)
"""

import numpy as np
import math as m

class Graph:
    """
    Implementation of a grap class extracted from the skeleton
    """

    class Vertex:
        """
        Vertex class for the graph extracted from a skeleton
        """
        vertex_counter = 0

        """
        The type of the point is either extreme or cross. It set the incoming edges,
        with the angle of entrance
        """
        def __init__(self, type):
            self.type = type                # Vertex type
            self.id = Vertex.vertex_counter # id
            Vertex.vertex_counter += 1      # Autoincrement id
        
        """
        Return the vertex id
        """
        def id(self):
            return id
    #End of Vertex class

    class Edge:
        """
        Edge class for the graph extracted from a skeleton
        """
        edge_counter = 0

        """ 
        The origin and end points are type either extreme or either cross
        vertices of the graph
        """
        def __init__(self, points_b = []):
            self._path = points_b        # List with all the points of the arc
            self.id = Edge.edge_counter   # Id of the edge
            Edge.edge_counter += 1        # Autoincrement id

        def id(self):
            return self.id

        """
        Return the travel of the edge
        """
        def path(self):
            return travel._path

        """
        Add a new point to the travel of the edge
        """
        def add_path_point(self, point):
            self._path = np.append(self._path, [point], axis = 0)
    # End of Edge class

    edges_in_vertex_type = np.dtype()

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
    def __init__(self, vertices = [], edges = [], vertices_per_edge = [], 
                 n_points_angle = 7):
        self.vertices = vertices
        self.edges = edges
        self.vertices_in_edge = vertices_per_edgetions
        self.n_incident_p = n_points_angle

        # A vertex v will have a list of tuples (id,theta), in which id is the
        # identificator of an entrance vertex, with the incident angle theta.
        # The identificator of v is used as index in edges_in_vertex
        self.edges_in_vertex = [[] for _ in range(0,len(self.vertices)) ]
        for i in range(0, len(self.vertices)):
            __edge_in_vertex_extractor(self, i)

    # Extracts the entrance angle to the vertices of the edge <<id>>
    def __edge_in_vertex_extractor(self, id):
        # Path and length of the edge
        edge_path = self.edges[id].path()
        path_len = len(edge_path)
        # Id of vertices containing the edge
        vertex_o, vertex_e = self.vertices_in_edge[id]
        theta_o = 0; theta_e = 0
        num_angle_points = min(path_len, self.n_incident_p)
        # Mean of the directions to extract the incident angle
        for p_it in range(0,num_angle_points-1):            # ORIGIN
            d_x, d_y = edge_path[p_it] - edge_path[p_it + 1]    # Difference
            theta_o += m.atan2(dy, dx)
        for p_it in range(num_angle_points+1, path_len):    # END
            d_x, d_y = edge_path[p_it-1] - edge_path[p_it]     # Difference
            theta_b += m.atan2(dy, dx)

        theta_o /= num_angle_points-1
        theta_b /= num_angle_points-1

        # Append to the list of edges in vertex
        self.edges_in_vertex[vertex_o].append((id, theta_o))
        self.edges_in_vertex[vertex_e].append((id, theta_e))


    # Add a vertex to the graph
    def add_vertex(self, vertex):
        np.append(self.vertices, vertex)

    # Add an edge to the graph. The two vertices are in the graph
    def add_edge(self, edge, vertex_o, vertex_e):
        np.append(self.edges, edge)
        np.append(self.vertices_in_edge, [[vertex_o, vertex_e]], axis = 0)
        __edge_in_vertex_extractor(self, edge.id())