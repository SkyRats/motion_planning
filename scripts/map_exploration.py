import cv2
import numpy as np
from numpy.linalg import norm
from collections import namedtuple

THRESH = 0.06
CLOSENESS_THRESH = 3
MAP_COLOR = 114

Rectangle = namedtuple('Rectangle', 'right left top bottom')

def close(x, y):
    return True if abs(x-y) < CLOSENESS_THRESH else False

def intersecting(r1, r2):
    counter = 0
    output = None
    if  r2.bottom - CLOSENESS_THRESH < r1.top < r2.top + CLOSENESS_THRESH and r2.bottom - CLOSENESS_THRESH < r1.bottom < r2.top + CLOSENESS_THRESH:
        counter += 2
        
        if r1.left < r2.right and r1.right > r2.right: 
            output = "right"
        elif r1.left < r2.left and r1.right > r2.left:
            output = "left"
        
    if r1.left - CLOSENESS_THRESH < r2.right < r1.right + CLOSENESS_THRESH and r1.left - CLOSENESS_THRESH < r2.left < r1.right + CLOSENESS_THRESH:
        counter += 2
        
        if r1.bottom < r2.bottom and r1.top > r2.bottom:
            output = "bottom"
        elif r1.bottom < r2.top and r1.top > r2.top:
            output = "top"
    
    if counter == 4:
        output = "inside"

    return output

def clear_rectangles(rectangles):
    new_rectangles = []
    simplified = []
    removed_intersection = []
            
    for i in rectangles:
        if i in simplified:
            continue

        for j in rectangles:
            if j in simplified or i == j:
                continue
            
            intersection_i = intersecting(i, j)
            intersection_j = intersecting(j, i)
            
            if intersection_i != None and intersection_i != "inside":
                if intersection_i == "right":
                    new_rect = Rectangle( i.right, j.right, i.top, i.bottom )
                elif intersection_i == "left":
                    new_rect = Rectangle( j.left, i.left, i.top, i.bottom )
                elif intersection_i == "bottom":
                    new_rect = Rectangle( i.right, i.left, i.top, j.top )
                elif intersection_i == "top":
                    new_rect = Rectangle( i.right, i.left, j.bottom, i.bottom )
                new_rectangles.append(new_rect)
                removed_intersection.append(i)

            if intersection_i == "inside":
                new_rectangles.append(j)
                simplified.append(i)
                simplified.append(j)
            elif intersection_j == "inside":
                new_rectangles.append(i)
                simplified.append(i)
                simplified.append(j)

            elif close(i.right, j.right) and close(i.left, j.left): # Próximos em X
                new_rect = Rectangle( i.right, i.left, max(i.top, j.top), min(i.bottom, j.bottom) )
                new_rectangles.append(new_rect)
                simplified.append(i)
                simplified.append(j)
            elif close(i.top, j.top) and close(i.bottom, j.bottom): # Próximos em Y
                new_rect = Rectangle( max(i.right, j.right), min(i.left, j.left), i.top, i.bottom )
                new_rectangles.append(new_rect)
                simplified.append(i)
                simplified.append(j)

    for k in rectangles:
        if k not in new_rectangles and k not in simplified and k not in removed_intersection:
            new_rectangles.append(k)

    if len(new_rectangles) == len(rectangles):
        return new_rectangles
    else:
        return clear_rectangles(new_rectangles)

def find_rectangles():
    map = cv2.imread("map.jpeg")
    gray_map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    cv2.threshold(gray_map, 100, 255, cv2.THRESH_TOZERO, dst=gray_map)

    corners = []
    cnts, _ = cv2.findContours(gray_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        epsilon = cv2.arcLength(cnt, True)*0.01
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for point in approx.tolist():
            corners.append(point[0])
        cv2.drawContours(map, approx, -1, (0,0,255), thickness=2)

    rectangles = []
    bottom = top = right = left = 0
        
    for i in corners:
        for j in corners:
            
            if i[0] <= j[0]:
                right = j[0]
                left = i[0]
            else:
                right = i[0]
                left = j[0]

            if i[1] <= j[1]:
                top = j[1]
                bottom = i[1]
            else:
                top = i[1]
                bottom = j[1]

            area = (top-bottom)*(right-left)
            value = np.sum( gray_map[bottom:top+1, left:right+1] ) 
            if value < MAP_COLOR * THRESH * area:
                tr = (right, top)
                bl = (left, bottom)
                rect = Rectangle(right, left, top, bottom)
                rectangles.append(rect)

    rectangles = clear_rectangles(rectangles)
    print(len(rectangles))

    for nr in rectangles:
        cv2.rectangle(map, (nr.right, nr.top), (nr.left, nr.bottom), (0,255,0), thickness=1 )

    cv2.imwrite("marked_clear.png", map)
    
    return rectangles

find_rectangles()
