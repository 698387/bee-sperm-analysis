import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def nothing(*arg):
    pass


# Reads the videofile
data_file = input('Name of the file: ')

while True:
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
        blur_image = cv.GaussianBlur(gray_image, (3,3), 0)  # Eliminates noise 

        # Find the circles
        # (image, method, scale, distance, param2= threshold of the hough)
        circles = cv.HoughCircles(blur_image, cv.HOUGH_GRADIENT,1,20,param2=50)

        # Draw the circles in the image
        circles_img = raw_image.copy()
        for c in circles[0]:
            center = (c[0], c[1])
            r = c[2]
            cv.circle(circles_img, center, r, (0,255,0))            

        cv.imshow('original', gray_image)
        cv.imshow('Circles detected', circles_img)

        cv.waitKey(5)
