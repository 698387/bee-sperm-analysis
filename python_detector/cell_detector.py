import cv2 as cv
import sys
import statistics as st
import numpy as np
from python_detector.line_decoupler import decouple_lines, follow_line
import random as r
from python_detector.sFCM import sFCM, cluster_corrector
from python_detector.image_preprocess import Preprocess
from python_detector.graph_extractor import extractGraph
from python_detector.spermatozoid_extractor import cells_from_single_image
from python_detector.line_matcher import LineMatcher
import math

# list of colors for printing
color_values = [0, 127, 255]
colors = [(a,b,c) for a in color_values \
                  for b in color_values \
                  for c in color_values \
                  if not(a == b and b == c)]
num_colors = len(colors)


"""
Given the video file, it extracts the number of cells and the moving cells
@param data_file Is the video file to extracts the info from
@n_frames_to_use Is the number of frames to extract the info
"""
def sperm_movility_analysis(data_file = "", 
                            n_frames_to_use = 7,
                            view_frames = False,
                            min_length = 0.0,
                            scale = 0.0,
                            min_movement = 0.5,
                            video_fps = 0.0):
    # If there is no parameter, it returns an exception
    if data_file == "":
        raise NameError

    print("It will extract the data from the file \""+ data_file + \
          "\". Parameters:\n\t" + str(n_frames_to_use) + " frames to use")

    v = cv.VideoCapture(data_file)
    frames = []

    # Checks if the videofile is opened
    if not v.isOpened():
        print('File couldn\'t be opened')
        exit()

    # Transform all values into pixels and frame rate
    if video_fps == 0.0:
        fps = v.get(cv.CAP_PROP_FPS)
    else:
        fps = video_fps

    if scale > 0.0:
        pixel_size = scale / v.get(cv.CAP_PROP_FRAME_HEIGHT)
        pixel_min_length = min_length / pixel_size
        pixel_min_mov = min_movement / (fps * pixel_size)
    else:
        pixel_size = 0
        pixel_min_length = 0
        pixel_min_mov = 0


    # Cluster instance
    cluster = sFCM(c=3, init_points = np.array([5, 100, 200]), NB = 3)
    # Preprocess instance
    preproc = Preprocess()
    # Matcher instance
    cellMatcher = LineMatcher(max_distance_error = 3200, matchs_number = 1)
    # Iteration variables
    fitted = False
    frames_used = 0
    # Extracts each frame
    print("Extracting information from each frame...")
    stop = -1
    while frames_used < n_frames_to_use:
        raw_image_state, raw_img = v.read()
        if not raw_image_state: # It exists an image
            v.release()
            cv.destroyAllWindows()
            break
        # Image to gray
        gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
        print("Frame " + str(frames_used+1) + " of " \
              + str(n_frames_to_use) + ":")
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
                        debug = view_frames)
        print("Done")
        # Find the cells in a single image
        print("\tTransforming graph into cells...",end="")
        cell_paths = cells_from_single_image(g,
                                             n_points4cell=16,
                                             max_theta  = math.pi/4,
                                             max_evo_d = 0.1,
                                             max_length = 700)
        print("Done")

        # Add the paths to the matcher
        print("\tAdding cells to the matcher...",end="")
        cellMatcher.add_line_set(cell_paths)
        print("Done")

        frames_used += 1

        # Draws the process if indicated
        if view_frames:
            printing_cell_paths = list(map(lambda x: np.flip(x, axis = 1), 
                                           cell_paths))
            cell_paths_img = raw_img.copy()
            for i in range(0, len(cell_paths)):
                cv.polylines(cell_paths_img, 
                             np.int32([printing_cell_paths[i]]), 
                             False, 
                             colors[i % num_colors])

            # Show the images
            cv.imshow('gray img', gray_img)
            cv.imshow('cells img', cell_paths_img) 

            cv.waitKey(1)
        # While end
    cv.destroyAllWindows()

    # Extract the matches
    print("Matching between frames...",end="")
    matches = cellMatcher.matches()
    print("Done")

    # Filter by length
    matches = list(filter(lambda x:
                          np.sum( 
                              np.diff(
                                  np.mean(x.positions, axis = 0)
                                  )
                              )**2 > pixel_min_length**2,
                          matches))
    # Speed modulus 
    speed_mod = np.array(list( map(lambda x:
                                   np.mean(np.linalg.norm(x.speed, axis = 1),
                                           axis = 0),
                                   matches) ) )
    # Filter by movement
    moving_matches_idx = np.argwhere(speed_mod > pixel_min_mov)
    moving_matches = np.array(matches)[moving_matches_idx]
    # Mean speed
    mean_speed = np.mean(speed_mod[moving_matches_idx])

    # Draws the results if indicated
    stop = -1
    while stop < 0 and view_frames:
        # After n_frames_to_use first frames
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

            cv.imshow('Found cells (press any key to close)', cell_img)
            stop = cv.waitKey(400)
            if stop > 0:
                break
    cv.destroyAllWindows()

    return {"Number of detected cells": len(matches),
            "Number of moving cells": len(moving_matches),
            "Mean of movement of moving cells \u03BCm/s": mean_speed*fps*pixel_size}


     