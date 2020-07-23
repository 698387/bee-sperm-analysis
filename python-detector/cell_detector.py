import cv2 as cv
import sys
import statistics as st
import numpy as np
from line_decoupler import decouple_lines, follow_line
import math
import itertools as it
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
import random as r
from sFCM import sFCM, cluster_corrector
from image_preprocess import Preprocess
from graph_extractor import extractGraph
from spermatozoid_extractor import cells_from_single_image
import math as m

from matplotlib import pyplot as plt
import statistics as s


# list of colors
v = [0, 127, 255]
colors = [(a,b,c) for a in v for b in v for c in v if not(a == b and b == c)]
num_colors = len(colors)

# Reads the videofile
# Select the videofile
if (len(sys.argv) < 2):
    data_file = input('Name of the file: ')
else:
    data_file = sys.argv[1]


v = cv.VideoCapture(data_file)

# Checks if the videofile is opened
if not v.isOpened():
    print('File couldn\'t be opened')
    exit()

#original = cv.VideoWriter('original.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (576, 768))
cluster = sFCM(c=3, init_points = np.array([5, 100, 200]), NB = 3)
preproc = Preprocess()
fitted = False
# Extracts each frame
stop = -1
while stop != 27:
    raw_image_state, raw_img = v.read()
    if not raw_image_state: # It exists an image
        v.release()
        cv.destroyAllWindows()
        break

    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    if not fitted:          # First iteration. Nothing is fitted
        preproc.ext_param(gray_img)             # Preprocess fitting
        norm_img = preproc.apply(gray_img)      # Preprocess
        cluster.fit(norm_img, spatial = True)   # Cluster fitting
        # Correct the cluster
        while not cluster_corrector(cluster, norm_img):
            pass
        fitted = True
    else:
        norm_img = preproc.apply(gray_img)      # Preprocess
    
    # Clustering by layers
    pred_class = cluster.predict(norm_img, spatial = True).astype("ubyte")
    # Extract the morphological graph
    g = extractGraph(pred_class, overlapping_info = cluster.c > 2, n_pixels_angle = 7, debug = True)
    # Find the cells in a single image
    paths = cells_from_single_image(g, max_theta  = m.pi/4, max_evo_d = 4, max_length = 1000)


    printing_paths = list(map(lambda x: np.flip(x, axis = 1), paths))
    paths_img = raw_img.copy()
    for i in range(0, len(paths)):
        cv.polylines(paths_img, 
                     np.int32([printing_paths[i]]), 
                     False, 
                     colors[i % num_colors])


    vertex_img = raw_img.copy()
    for i in range(0, len(g.vertices)):
        vertex = g.vertices[i]
        if vertex.type == "intersection":
            center = np.mean(vertex.contour, axis = 0)[0]
        else:
            center = vertex.contour[0]
            center = np.flip(center)

        for (_, _, theta) in g.edges_in_vertex[vertex.id]:
            extrem = np.int32(center+7*np.array([m.sin(theta), m.cos(theta)]))
            cv.line(vertex_img,
                    tuple(np.int32(center)),
                    tuple(extrem),
                    [0,0,255])

        cv.fillPoly(vertex_img, 
                     np.int32([np.flip(vertex.contour, axis = 1)]),  
                     colors[i % num_colors])

    #skeleton_img = np.where(crosses)

    #binary_img = cv.adaptivethreshold(gray_img, 255,
    #                                 cv.adaptive_thresh_gaussian_c,
    #                                 cv.thresh_binary,11,-1)

    #skeleton, _ = find_skeleton(binary_img)

    #lines = decouple_lines(skeleton,
    #                       max_distance = 20,
    #                       min_length = 1,
    #                       max_angle = math.pi/4)
    #lines_img = raw_img.copy()
    
    #for i in range(0, len(lines)):
    #    cv.polylines(lines_img, 
    #                 np.int32([lines[i]]), 
    #                 False, 
    #                 colors[i % num_colors])
    
    #skeleton_img = raw_img.copy()
    #skeleton_img[np.where(skeleton==[255])] = [0,0,255]

    #cv.imshow('skeletons', skeleton_img)
    #cv.imshow('lines', lines_img)

    cv.imshow('gray img', gray_img)
    cv.imshow('cells img', paths_img)

    cv.imshow('Vertices', vertex_img)

    stop = cv.waitKey(0)
    # cv.waitKey(1)

cv.destroyAllWindows()
