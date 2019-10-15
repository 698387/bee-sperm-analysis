import cv2 as cv
import math

# Extracts the angle and the middle point from a segment of pixels
def segment2point(pixels):
    l = len(pixels)
    delta_x = pixels[0][0][0]-pixels[l-1][0][0]
    delta_y = pixels[0][0][1]-pixels[l-1][0][1]
    theta = math.atan2(delta_x, delta_y)
    point = ((pixels[0][0][0] + pixels[l-1][0][0]) / 2\
            ,(pixels[0][0][1] + pixels[l-1][0][1]) / 2)
    return theta, point

# Generate points from an array of contours
def allContourPoints(contours, length):
    segments = []
    for c in contours:      # For each contour
        while len(c) >= length:
            s = c[0:length]
            c = c[length:]
            segments.append(segment2point(s))
    return segments

# Find points that belongs to cells
def pointInCell(contours, length=11, max_distance=15, min_distance=8,\
                angle_error=0.1):
    contour_points = allContourPoints(contours, length)
    inner_points = []
    while len(contour_points) > 0:
        cp = contour_points[0]
        for a in contour_points[1:]:
            theta = min(abs(cp[0] - a[0])%(2*math.pi),\
                        abs(cp[0] - (a[0]+math.pi))%(2*math.pi))
            delta_x = cp[1][0] - a[1][0]
            delta_y = cp[1][1] - a[1][1]
            angle_points = abs(math.atan2(delta_x, delta_y))
            dist = math.sqrt(delta_x*delta_x + delta_y*delta_y)
            if dist <= max_distance and dist >= min_distance and \
              theta <= angle_error and angle_points > math.pi/4:
                inner_angle = (cp[0] + a[0]) / 2
                inner_center = ((cp[1][0] + a[1][0]) / 2,\
                                 (cp[1][1] + a[1][1]) / 2)
                inner_points.append([inner_angle, inner_center])
        contour_points = contour_points[1:]

    return inner_points
        
    
