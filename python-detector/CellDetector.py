import cv2 as cv
import sys
from CellPointDetector import pointInCell, allContourPoints
from skeletonize import find_skeleton
import statistics as st
import numpy as np
from line_decoupler import decouple_lines, follow_line
import math
import itertools as it
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
import random as r
from sFCM import sFCM

"""
Select the importance of the class, based on the amount in the samples, 
and the distance to the other class. 
@param class_samples is the class of each sample
@param v is the value of the class
"""
def data_importance(class_samples, v):
    # All classes variables
    classes = list(range(0, len(v)))
    n_classes = len(classes)

    # Count the number samples of each class
    count_class = np.zeros(n_classes)
    total_length = len(class_samples)
    for i in classes:
        count_class[i] = np.count_nonzero(class_samples == i)
    amount_imp =  count_class / total_length
    
    # Remove the classes for the background
    max_idx = np.argmax(amount_imp)

    amount_bc = amount_imp[max_idx]
    bg = [max_idx]
    fg = classes.copy()
    fg.remove(max_idx)
    while amount_bc < 0.8:
        dist = cdist(v[bg], v[fg])
        idx_min_dist = np.argmin(dist)
        c = fg[idx_min_dist]
        bg.append(c)
        fg.remove(c)
        amount_bc = amount_bc + amount_imp[c]

    return fg

    


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
cluster = sFCM(c=3)
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
        cluster.fit(gray_img, spatial = True)
        fitted = True
        pred_class = cluster.predict(gray_img, spatial = True)
        cell_classes = data_importance(pred_class.flatten(), cluster.v )
        prediction = np.where(np.isin(pred_class, cell_classes), 255, 0).astype("ubyte")
    else:
        pred_class = cluster.predict(gray_img, spatial = True)
        prediction =  np.where(np.isin(pred_class, cell_classes), 255, 0).astype("ubyte")

    
    binary_img = cv.adaptivethreshold(gray_img, 255,
                                     cv.adaptive_thresh_gaussian_c,
                                     cv.thresh_binary,11,-1)

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
    cv.imshow('predicted img', prediction)
    cv.imshow('binary img', binary_img)

    stop = cv.waitKey(0)
    # cv.waitKey(0)

cv.destroyAllWindows()
