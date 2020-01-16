import cv2 as cv
import sys
from CellPointDetector import pointInCell, allContourPoints
from skeletonize import find_skeleton
import statistics as st
import numpy as np

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

# Extracts each frame
stop = -1
while stop == -1:
    raw_image_state, raw_img = v.read()
    if not raw_image_state:
        v.release()
        break

    gray_img = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
    equalized_img = cv.equalizeHist(gray_img)              # Equalized the hist
    blur_img = cv.GaussianBlur(equalized_img, (9,9), 0)         # Eliminates noise
    binary_img = cv.adaptiveThreshold(blur_img, 255,
                                         cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY,11,-3)
    contours, hierarchy = cv.findContours(binary_img, cv.RETR_LIST,
                                          cv.CHAIN_APPROX_NONE )
    
    contour_lengths = list(map(lambda x: len(x), contours))
    mean = st.mean(contour_lengths)     # Mean of the data
    sd = st.pstdev(contour_lengths)
    
    contour_length_thres = mean + sd
            
    good_contours = list(filter(lambda x: len(x) > contour_length_thres, contours))
    
    found_contours = cv.drawContours(raw_img.copy(), good_contours, -1,
                                     (0,255,0))
    
    """params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1500
    detector = cv.SimpleBlobDetector(params)
    blobs = detector.detect(binary_img)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    
    im_with_keypoints = cv.drawKeypoints(raw_img, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show blobs
    cv.imshow("Keypoints", im_with_keypoints)
    cv.waitKey(0)"""
    
    """# Extracts edges with canny detector
    thrs1 = 3000
    thrs2 = 2*thrs1
    edges = cv.Canny(equalized_img, thrs1, thrs2, apertureSize=7)  

    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE )  

    found_contours = cv.drawContours(raw_img.copy(), contours, -1, (0,255,0)) 
    
    points = allContourPoints(contours,5)
    for p in points:
        p2 = (int(p[1][0] + 4*math.cos(p[0])), int(p[1][1]+4*math.sin(p[0])))
        found_contours = cv.line(found_contours, p[1],p2, (0,0,255))
        
    points = pointInCell(contours)
    #points = []    
    found_points = raw_image.copy()
    for p in points:
        found_points = cv.circle(found_points, p[1],0, (0,255,255))

    # cv.imshow('Points', found_points) 
    cv.imshow('edge', edges)"""
    
    cv.imshow('Contours', found_contours)
    cv.imshow('original', gray_img)
    cv.imshow('Equalized', equalized_img)
    cv.imshow('blurred', blur_img)
    cv.imshow('binary', binary_img)

    stop = cv.waitKey(0)

cv.destroyAllWindows()
