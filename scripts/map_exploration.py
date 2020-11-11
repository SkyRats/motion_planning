import cv2
import numpy as np
from numpy.linalg import norm
from collections import namedtuple

THRESH = 0.06
Rectangle = namedtuple('Rectangle', 'right left top bottom')

def close(x, y):
    return True if abs(x-y) < 10 else False

def inside(r1, r2): # Retorna Verdadeiro se r1 esta dentro de r2
    if r1.top <= r2.top and r1.bottom >= r2.bottom:
        if r1.right <= r2.right and r1.left >= r2.left:
            return True
    return False

def clear_rectangles(rectangles):
    new_rectangles = []
    passed = []
    
    for i in rectangles:
        if i in passed:
            continue

        for j in rectangles:
            if j in passed:
                continue
            
            if inside(i, j):
                new_rectangles.append(j)
                passed.append(i)
                passed.append(j)
            elif inside(j, i):
                new_rectangles.append(i)
                passed.append(i)
                passed.append(j)

            elif close(i.right, j.right) and close(i.left, j.left): # Próximos em X
                new_rect = Rectangle( i.right, i.left, max(i.top, j.top), min(i.bottom, j.bottom) )
                new_rectangles.append(new_rect)
                passed.append(i)
                passed.append(j)
            elif close(i.top, j.top) and close(i.bottom, j.bottom): # Próximos em Y
                new_rect = Rectangle( max(i.right, j.right), min(i.left, j.left), i.top, i.bottom )
                new_rectangles.append(new_rect)
                passed.append(i)
                passed.append(j)

    print(len(new_rectangles))
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
            if value < 114*THRESH*area:
                tr = (right, top)
                bl = (left, bottom)
                rect = Rectangle(right, left, top, bottom)
                #cv2.rectangle(map, bl, tr, (255,0,0), thickness=1)
                rectangles.append(rect)

    rectangles = clear_rectangles(rectangles)

    for nr in rectangles:
        cv2.rectangle(map, (nr.right, nr.top), (nr.left, nr.bottom), (0,255,0), thickness=1 )

    cv2.imwrite("marked1.png", map)
    
    return rectangles
    
find_rectangles()
