import cv2 as cv
import math
import numpy as np
from CellPointDetector import pointInCell, allContourPoints
from matplotlib import pyplot as plt


# Reads the videofile
data_file = input('Name of the file: ')

v = cv.VideoCapture(data_file)

# Checks if the videofile is opened
if not v.isOpened():
    print('File couldn\'t be opened')
    exit()

# Extracts each frame
stop = -1
while stop == -1:
    raw_image_state, raw_image = v.read()
    if not raw_image_state:
        v.release()
        break

    gray_image = cv.cvtColor(raw_image, cv.COLOR_BGR2GRAY)
    blur_image = cv.GaussianBlur(gray_image, (5,5), 0)  # Eliminates noise
    
    # Extracts edges with canny detector
    thrs1 = 3000
    thrs2 = 2*thrs1
    edges = cv.Canny(blur_image, thrs1, thrs2, apertureSize=7)  

    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )  

    found_contours = cv.drawContours(raw_image.copy(), contours, -1, (0,255,0)) 
    
    points = allContourPoints(contours,11)
    for p in points:
        p2 = (int(p[1][0] + 4*math.cos(p[0])), int(p[1][1]+4*math.sin(p[0])))
        found_contours = cv.line(found_contours, p[1],p2, (0,0,255))
        
    points = pointInCell(contours)
    found_points = raw_image.copy()
    for p in points:
        found_points = cv.circle(found_points, p[1],0, (0,255,255))
    
    cv.imshow('original', gray_image)
    cv.imshow('edge', edges)
    cv.imshow('Contours', found_contours)
    cv.imshow('Points', found_points)

    stop = cv.waitKey(5)

cv.destroyAllWindows()