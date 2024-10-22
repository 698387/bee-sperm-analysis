import cv2 as cv
import sys
import numpy as np
from python_detector.sFCM import sFCM
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
                            video_fps = 0.0,
                            area_filter = 0.0):
    n_points_cell = 16      # Number of points for each cell
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
        pixel_min_length = min_length / scale
        pixel_min_mov = min_movement / (fps * scale)
        max_area = area_filter/scale
    else:
        pixel_min_length = 0
        pixel_min_mov = 0
        max_area = 0


    # Cluster instance
    cluster = sFCM(c=3, init_points = np.array([5, 100, 200]), NB = 3)
    # Preprocess instance
    preproc = Preprocess(max_area = max_area)
    # Matcher instance
    cellMatcher = LineMatcher(max_distance_error = n_points_cell * 1000, matchs_number = 3,
                              init_line_sets=[])
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
        while not fitted:          # First iteration. Nothing is fitted
            print("\tAdjusting preprocessing...",end="")
            preproc.ext_param(gray_img)             # Preprocess fitting
            print("Done")
            print("\tPreprocessing...",end="")
            norm_img = preproc.apply(gray_img)      # Preprocess
            print("Done")
            print("\tFitting cluster...",end="")
            cluster.fit(norm_img, spatial = True)   # Cluster fitting
            retry = False                           
            # Check if the cluster failed
            if not cluster.is_correct():
                print("Cluster fitting failed")
                print("\tRefitting...",end="")
                # Correct the cluster
                while not cluster.correct(norm_img):
                    print("Cluster fitting failed")
                    if cluster.c == 1:
                        if not preproc.local_norm:  # Retries with local norm
                            print("\tRetrying with local ", end="")
                            print("normalization...\n", end="")
                            preproc.local_norm = True
                            cluster.c = 3
                            cluster.init_points = []
                            retry = True            # has to retry the fitting
                            break
                        else:                       # Clustering has failed
                            print("Clustering failed. Analisys will finish")
                            return {}
                    print("\tRefitting...",end="")
                if retry:           # Retry if needed
                    continue
            print("Done")
            fitted = True
        else:
            print("\tPreprocessing...",end="")
            norm_img = preproc.apply(gray_img)      # Preprocess
            print("Done")

        # Predict the class up to the cluster
        pred_class = cluster.predict(norm_img, spatial = True)\
                     .astype("ubyte")
        # View cluster and normalized image. Commented to not saturate 
        # the output
        #cluster_img = np.zeros(raw_img.shape, dtype = "ubyte")
        #cluster_img[pred_class == 1] = [255,0,0]
        #cluster_img[pred_class == 2] = [0,0,255]
        #cv.imshow("cluster", cluster_img)
        #cv.imshow("normalized", norm_img)
    
        frames.append(raw_img)
        # Clustering by layers
        print("\tExtracting layers...",end="")
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
                                             n_points4cell=n_points_cell,
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

    # Extracts each cell length
    len_matches = np.array(list( map( 
                        lambda x: 
                        math.sqrt(np.mean(
                            np.sum(
                                np.diff( x.positions, axis = 1) **2,
                                axis = (1,2)
                            )
                        )),  matches) )) 
    # Filter by length
    matches = np.array(matches)[
                    np.argwhere(len_matches > pixel_min_length )].ravel()
    len_matches = list(filter(lambda x: x > pixel_min_length, len_matches))
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
            "Moving percentage (%)": (len(moving_matches)/len(matches)) * 100,
            "Mean of movement of moving cells \u03BCm/s": mean_speed*fps*scale,
            "Minimum length": min(len_matches),
            "Maximum length": max(len_matches),
            "Mean length": np.mean(len_matches)}


     