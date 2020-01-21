import cv2 as cv
import sys
from CellPointDetector import pointInCell, allContourPoints
from skeletonize import find_skeleton
import statistics as st
import numpy as np
from line_decoupler import decouple_lines

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
    blur_img = cv.GaussianBlur(gray_img, (9,9), 0)      # Eliminates noise
    binary_img = cv.adaptiveThreshold(blur_img, 255,
                                     cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv.THRESH_BINARY,11,-1)
    skeleton, _ = find_skeleton(binary_img)
    l = decouple_lines(skeleton)
    
    """contours, _ = cv.findContours(skeleton, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contour_cod = []
    for i in range(0, len(contours)):
        circles_img = raw_img.copy()
        contour_img = cv.drawContours(np.zeros(gray_img.shape, dtype=np.uint8), contours, i, 255)
        cv.drawContours(circles_img, contours, i, (255,0,0))
        # contour_cod.append(cv.HoughCircles(contour_img, cv.HOUGH_GRADIENT, 1, 5))
        circles = cv.HoughCircles(contour_img, cv.HOUGH_GRADIENT, 1, 5)
        if circles != None:
            for c in circles:
                center = (c[0], c[1])
                r = c[2]
                cv.circle(circles_img, center, r, (0,255,0))  
        cv.imshow('circles detected', circles_img)
        cv.waitKey(0)""" 
    print(np.where(skeleton==[255], axis=1))    
        
    raw_img[np.where(skeleton==[255])] = [0,0,255]
    cv.imshow('original', raw_img)

    stop = cv.waitKey(0)

cv.destroyAllWindows()
