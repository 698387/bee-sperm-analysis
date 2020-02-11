import cv2 as cv
import sys
from CellPointDetector import pointInCell, allContourPoints
from skeletonize import find_skeleton
import statistics as st
import numpy as np
from line_decoupler import decouple_lines, follow_line
import math
from sklearn.neighbors import KDTree
import random as r
from sFCM import sFCM

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

# Extracts each frame
stop = -1
while stop == -1:
    raw_image_state, raw_img = v.read()
    if not raw_image_state:
        v.release()
        cv.destroyAllWindows()
        break

    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)

    cluster = sFCM(c=3)
    #cluster.fit(gray_img)
    cluster.fit(raw_img)

    #blur_img = cv.GaussianBlur(gray_img, (9,9), 0)      # Eliminates noise
    #binary_img = cv.adaptiveThreshold(blur_img, 255,
    #                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                 cv.THRESH_BINARY,11,-1)

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

    stop = cv.waitKey(1)
    # cv.waitKey(0)

cv.destroyAllWindows()
