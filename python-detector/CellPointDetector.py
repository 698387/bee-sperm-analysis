import math
import statistics

# Extracts the angle and the middle point from a segment of pixels
def segment2point(pixels):
    delta_x = pixels[0][0][0]-pixels[-1][0][0]
    delta_y = pixels[0][0][1]-pixels[-1][0][1]
    theta = math.atan2(delta_y, delta_x)
    point = (int((pixels[0][0][0] + pixels[-1][0][0]) / 2 + 0.5)\
            ,int((pixels[0][0][1] + pixels[-1][0][1]) / 2 + 0.5))
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
def pointInCell(contours, 
                length=5,
                max_distance=10, 
                min_distance=2,
                angle_error=0.1):
    contour_points = allContourPoints(contours, length)

    inner_points = []           # Vector of all the inner points
    # Vector of all the distances between the edges of the inner points    
    distances = []              
    # For all the edge points extracts the inner points    
    while len(contour_points) > 0:
        cp = contour_points[0]
        for a in contour_points[1:]:    # Compare to the rest of the points
            # Check the distance to the point (no further than max_distance)
            delta_x = cp[1][0] - a[1][0]
            delta_y = cp[1][1] - a[1][1]
            dist = math.sqrt(delta_x*delta_x + delta_y*delta_y)
            if dist > max_distance:
                continue

            parallel_diff = min(abs(cp[0] - a[0])%(2*math.pi),\
                            abs(cp[0] - (a[0]+math.pi))%(2*math.pi))
            theta = math.atan2(delta_y, delta_x)
            normal_diff = abs(cp[0]-theta)%math.pi

            # Check if the point is parallel and in a normal direction
            if dist >= min_distance  \
             and parallel_diff <= angle_error \
             and normal_diff > math.pi/3 and normal_diff < 2*math.pi/3:

                inner_angle = (cp[0] + a[0]) / 2
                inner_center = (int((cp[1][0] + a[1][0]) / 2 + 0.5),\
                                 int((cp[1][1] + a[1][1]) / 2 + 0.5))
                inner_points.append((inner_angle, inner_center))
                distances.append(dist)

        contour_points = contour_points[1:] # Remove the first element

    # Filters all points that are into the cell
    m = statistics.median(distances)
    final_inner_points = []
    for i in range(len(distances)):
        if m-1 <= distances[i] and m+1 >= distances[i]:
            final_inner_points.append(inner_points[i])

    return inner_points
        
    
