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
from line_matcher import LineMatcher
import math as m

from matplotlib import pyplot as plt
import statistics as s


# list of colors
v = [0, 127, 255]
colors = [(a,b,c) for a in v for b in v for c in v if not(a == b and b == c)]
num_colors = len(colors)

# Reads the videofile
# Select the videofile
i = 1
data_file = ""
n_frames_to_use = 7
while i < len(sys.argv):
    if sys.argv[i] == "-n_frames":
       if len(sys.argv[i]) > i+2:
            n_frames_to_use = int(sys.argv[i+1])
            i += 1
       else:
            print("Usage: cell_detector [-n_frames <<n>>] [<<video_file>>]")
    else:
        data_file = sys.argv[i]
    i += 1


if data_file == "":
    data_file = input('Name of the file: ')

print("It will extract the data from \the file \""+ data_file + "\". Parameters:\n\t" + str(n_frames_to_use) + " frames to use")

v = cv.VideoCapture(data_file)
frames = []

# Checks if the videofile is opened
if not v.isOpened():
    print('File couldn\'t be opened')
    exit()

#original = cv.VideoWriter('original.avi', cv.VideoWriter_fourcc(*'DIVX'), 15, (576, 768))
cluster = sFCM(c=3, init_points = np.array([5, 100, 200]), NB = 3)
preproc = Preprocess()
cellMatcher = LineMatcher(max_distance_error = 3200, matchs_number = 1)
fitted = False
frames_used = 0
# Extracts each frame
print("Extracting information from each frame...")
stop = -1
while stop != 27:
    raw_image_state, raw_img = v.read()
    if not raw_image_state: # It exists an image
        v.release()
        cv.destroyAllWindows()
        break

    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    print("Frame " + str(frames_used+1) + " of " + str(n_frames_to_use) + ":")
    if not fitted:          # First iteration. Nothing is fitted
        print("\tAdjusting preprocessing...",end="")
        preproc.ext_param(gray_img)             # Preprocess fitting
        print("Done")
        print("\tPreprocessing...",end="")
        norm_img = preproc.apply(gray_img)      # Preprocess
        print("Done")
        print("\tFitting cluster...",end="")
        cluster.fit(norm_img, spatial = True)   # Cluster fitting
        # Correct the cluster
        while not cluster_corrector(cluster, norm_img):
            print("\n\tCluster fitting failed. Refitting...",end="")
            pass
        print("Done")
        fitted = True
    else:
        print("\tPreprocessing...",end="")
        norm_img = preproc.apply(gray_img)      # Preprocess
        print("Done")
    
    frames.append(raw_img)
    # Clustering by layers
    print("\tExtracting layers...",end="")
    pred_class = cluster.predict(norm_img, spatial = True).astype("ubyte")
    print("Done")
    # Extract the morphological graph
    print("\tExtracting graph...",end="")
    g = extractGraph(pred_class, 
                    overlapping_info = cluster.c > 2,
                    overlapping_thres = 0.8,
                    n_pixels_angle = 9,
                    debug = True)
    print("Done")
    # Find the cells in a single image
    print("\tTransforming graph into cells...",end="")
    cell_paths = cells_from_single_image(g,
                                         n_points4cell=16,
                                         max_theta  = m.pi/4,
                                         max_evo_d = 0.1,
                                         max_length = 1000)
    print("Done")

    # Add the paths to the matcher
    print("\tAdding cells to the matcher...",end="")
    cellMatcher.add_line_set(cell_paths)
    print("Done")

    frames_used += 1
    if frames_used == n_frames_to_use:
        break


    printing_cell_paths = list(map(lambda x: np.flip(x, axis = 1), cell_paths))
    cell_paths_img = raw_img.copy()
    for i in range(0, len(cell_paths)):
        cv.polylines(cell_paths_img, 
                     np.int32([printing_cell_paths[i]]), 
                     False, 
                     colors[i % num_colors])

    # Show the images
    cv.imshow('gray img', gray_img)
    cv.imshow('cells img', cell_paths_img)

    stop = cv.waitKey(1)

cv.destroyAllWindows()

# Extract the matches
print("Matching between frames...",end="")
matches = cellMatcher.matches()
print("Done")

# After n_frames_to_use first frames
cell_printing = []
for m in matches:
    cell_printing.append(np.int32(np.flip(m.position, axis = 1)))

cell_img = frames[0].copy()
cv.polylines(cell_img, 
                cell_printing, 
                False, 
                (0,0,255))

cv.imshow('Found cells', cell_img)
cv.waitKey(0)

# After n_frames_to_use first frames
print(len(matches))
for f in range(0, n_frames_to_use):
    cell_printing = []
    for m in matches:
        line = cellMatcher.match2line(m)
        idx = np.argmin(abs(np.array(line[0]) - f))
        cell_printing.append(np.int32(np.flip(line[1][idx], axis = 1)))

    cell_img = frames[f].copy()
    cv.polylines(cell_img, 
                    cell_printing, 
                    False, 
                    (0,0,255))

    cv.imshow('Found cells', cell_img)
    cv.waitKey(0)


