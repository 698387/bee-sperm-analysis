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
from sFCM import sFCM
from image_preprocess import Preprocess
from graph_extractor import extractGraph


from matplotlib import pyplot as plt
import statistics as s

"""
Select the layers depending of the classes fitted with sFCM
@param cluster is the fitted cluster to the image
@param img is the image used to fit the cluster
@return true iff the cluster has not been corrected
"""
def cluster_corrector(cluster, img):
    # Distance between class centers
    center_dist = cdist(cluster.v, cluster.v)
    # If the distance is lower than the image sigma, it combine the classes
    classes2combine = []
    for [x, y] in np.argwhere(center_dist < 43):
        if x != y and not [y,x] in classes2combine:
            classes2combine.append([x,y])
    
    # Re-fit the cluster if needed
    if len(classes2combine) > 0:
        # Extract the new posible centers for the init
        new_centers = np.array([v for i,v in enumerate(cluster.v)\
           if i not in list(map(lambda x: x[1], classes2combine) )] )
        # Update the cluster parameters and re-fit it
        cluster.c = len(new_centers)
        cluster.init_points = new_centers
        cluster.fit(img)
        return False
    else:
        return True



# list of colors
v = [0, 63, 127, 191, 255]
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
    if not raw_image_state:
        v.release()
        cv.destroyAllWindows()
        break

    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    if not fitted:
        preproc.ext_param(gray_img)
        norm_img = preproc.apply(gray_img)
        cluster.fit(norm_img, spatial = True)
        while not cluster_corrector(cluster, norm_img):
            pass
        fitted = True
    else:
        norm_img = preproc.apply(gray_img)
        
    pred_class = cluster.predict(norm_img, spatial = True).astype("ubyte")
    extractGraph(pred_class, overlapping_info = cluster.c > 2, debug = True)
    
    
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

    stop = cv.waitKey(0)
    # cv.waitKey(1)

cv.destroyAllWindows()
