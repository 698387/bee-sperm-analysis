import cv2 as cv
import numpy as np
from CellPointDetector import pointInCell
from matplotlib import pyplot as plt


# Reads the videofile
data_file = input('Name of the file: ')

v = cv.VideoCapture(data_file)

# Checks if the videofile is opened
if not v.isOpened():
    print('File couldn\'t be opened')
    exit()

# Extracts each frame
while True:
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

    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE )  

    to_delete = []
    for i in range(len(contours)):
        if hierarchy[0][i][0] < 0:
            to_delete.append(i)
    for j in to_delete:
        contours.pop(j)
    
    # contours3, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE ) 
    #pointos= pointInCell(contours)
    #found_pointos = raw_image.copy()
    #for p in pointos:
    #    found_pointos = cv.circle(found_pointos, p[1], 0, (0,255,255))

    found_contours = cv.drawContours(raw_image.copy(), contours, -1, (0,255,0))     
    
    cv.imshow('original', gray_image)
    cv.imshow('edge', edges)
    cv.imshow('Contours', found_contours)
    #cv.imshow('pointos', found_pointos)

    cv.waitKey(0)
