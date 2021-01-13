import numpy as np
import cv2

from collections import namedtuple
    
OBSTACLE_THRESH = 0.065
CLOSENESS_THRESH = 3
MAP_COLOR = 255
DRONE_RADIUS = 0.4
SAFE_DISTANCE = 1 # Se aplica aos dois lados
PERIOD = 10
NUMBER_OF_STEPS = 10

DEBUG = False

map = cv2.imread("testes/map_v4.png")
gray_map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
cv2.threshold(gray_map, 100, 255, cv2.THRESH_TOZERO, dst=gray_map)

Rectangle = namedtuple('Rectangle', 'right left top bottom')

def testSweep():
    sweep = calculate_sweep()
    for point in sweep:
        cv2.circle(map, point, 1, (0,255,0))
    cv2.imshow("map", map)
    cv2.imwrite("result.png", map)
    if cv2.waitKey(0) == 27:
        return

def testRectangles():
    rectangles = find_rectangles()
    drawRectangles(rectangles)
    cv2.imshow("map", map)
    cv2.imwrite("result.png", map)
    if cv2.waitKey(0) == 27:
        return

def drawRectangles(rectangles, color=(0,0,255)):
    global map
    for rect in rectangles:
        topright = (rect.right, rect.top)
        bottomleft = (rect.left, rect.bottom)
        cv2.rectangle(map, topright, bottomleft, color, thickness=1)

def calculate_sweep():
    sweep = []
    rectangles = find_rectangles()

    map_drone_x = 290
    map_drone_y = 190
    cv2.circle(map, (map_drone_x, map_drone_y), 1, (0,0,255), thickness=2)

    while len(rectangles) != 0:

        print(len(rectangles))

        trajectory = []
        rect = find_closest_rectangle(map_drone_x, map_drone_y, rectangles)
        rectangles.remove(rect)

        w = rect.top - rect.bottom 
        h = rect.right - rect.left

        current_x = current_y = 0
        start_x = start_y = 0

        if abs(rect.right - rect.left) > abs(rect.top - rect.bottom):   # Rectangle is horizontal
            start_y = goal_y = (rect.top + rect.bottom)/2
            
            if abs(map_drone_x - rect.right) > abs(map_drone_x - rect.left): 
                goal_x = rect.right 
                start_x = rect.left
            else: 
                goal_x = rect.left
                start_x = rect.right
            current_x = start_x

            A = abs(rect.top - rect.bottom) - SAFE_DISTANCE
            while current_x <= goal_x:
                current_y = A*np.sin(current_x * 2*np.pi/PERIOD)
                current_y = start_y + int( np.floor(current_y) )
                trajectory.append( (current_x, current_y) )

                if current_x == goal_x:
                    break
                elif current_x > goal_x:
                    current_x = goal_x
                else:
                    current_x += PERIOD//NUMBER_OF_STEPS


        else: # Rectangle is vertical
            start_x = goal_x = (rect.right + rect.left)/2

            if abs(map_drone_y - rect.top) > abs(map_drone_y - rect.bottom): 
                goal_y = rect.top
                start_y = rect.bottom
            else: 
                goal_y = rect.bottom
                start_y = rect.top
            current_y = start_y

            A = abs(rect.right - rect.left) - SAFE_DISTANCE
            while current_y <= goal_y:
                current_x = A*np.sin(current_y* 2*np.pi/PERIOD)
                current_x = start_x + int( np.floor(current_x) )
                trajectory.append( (current_x, current_y) ) 
                
                if current_y == goal_y:
                    break
                elif current_y > goal_y:
                    current_y = goal_y
                else:
                    current_y += PERIOD//NUMBER_OF_STEPS
        
        sweep.append( (start_x, start_y) )
        sweep.extend(trajectory)
        
        map_drone_x = current_x
        map_drone_y = current_y

    return sweep

def find_closest_rectangle(x, y, rectangles):
    closest_rectangle = None
    min_distance = 1000
    for rect in rectangles:
        dr, dl, dt, db = calculate_rectangle_distances(x, y, rect)

        rectangle_min_distance = min(dr, dl, dt, db)
        if rectangle_min_distance < min_distance:
            min_distance = rectangle_min_distance
            closest_rectangle = rect
    return closest_rectangle 

def calculate_rectangle_distances(x, y, rectangle):
    distance_right  = np.sqrt( (x - rectangle.right)**2 + (y - (rectangle.top + rectangle.bottom)/2)**2 )
    distance_left   = np.sqrt( (x - rectangle.left)**2  + (y - (rectangle.top + rectangle.bottom)/2)**2 )
    distance_top    = np.sqrt( (x - (rectangle.right + rectangle.left)/2)**2  + (y - rectangle.top)**2 )
    distance_bottom = np.sqrt( (x - (rectangle.right + rectangle.left)/2)**2  + (y - rectangle.bottom)**2 )
    return distance_right, distance_left, distance_top, distance_bottom

def find_rectangles():
    global map
    global gray_map

    corners = []
    cnts, _ = cv2.findContours(gray_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        epsilon = cv2.arcLength(cnt, True)*0.01
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for point in approx.tolist():
            corners.append(point[0])

            if DEBUG:
                cv2.circle(map, tuple(point[0]), 1, (0,0,255), thickness=2)

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

            value = np.sum( gray_map[bottom:top, left:right] )
            area = (top - bottom) * (right - left)
            if value < MAP_COLOR * OBSTACLE_THRESH * area:
                tr = (right, top)
                bl = (left, bottom)
                rect = Rectangle(right, left, top, bottom)
                rectangles.append(rect)

    print("# rectangles detected ", len(rectangles))

    rectangles = simplify_rectangles(rectangles)
    rectangles = remove_intersection(rectangles)
    print("# rectangles (final): ", len(rectangles))

    return rectangles

def simplify_rectangles(rectangles):
    new_rectangles = []
    simplified = []
            
    for i in rectangles:
        if i in simplified:
            continue

        for j in rectangles:
            if j in simplified or i == j:
                continue
                        
            if close(i.right, j.right) and close(i.left, j.left): # Proximos em X
                if close(i.top, j.bottom) or close(i.bottom, j.top):
                    new_rect = Rectangle( i.right, i.left, max(i.top, j.top), min(i.bottom, j.bottom) )
                    new_rectangles.append(new_rect)
                    simplified.append(i)
                    simplified.append(j)
            elif close(i.top, j.top) and close(i.bottom, j.bottom): # Proximos em Y
                if close(i.right, j.left) or close(i.left, j.right):
                    new_rect = Rectangle( max(i.right, j.right), min(i.left, j.left), i.top, i.bottom )
                    new_rectangles.append(new_rect)
                    simplified.append(i)
                    simplified.append(j)

    for k in rectangles:
        if k not in new_rectangles and k not in simplified:
            new_rectangles.append(k)

    if len(new_rectangles) == len(rectangles):
        return new_rectangles
    else:
        print("# rectangles simplified", len(new_rectangles))
        return simplify_rectangles(new_rectangles)

def remove_intersection(rectangles):
    new_rectangles = []
    removed_intersection = []
            
    for i in rectangles:
        if i in removed_intersection:
            continue

        for j in rectangles:
            if j in removed_intersection or i == j:
                continue
            
            intersection = intersecting(i, j)
            
            if intersection != None and intersection != "inside":
                if intersection == "right":
                    new_rect = Rectangle( i.right, j.right, i.top, i.bottom )
                elif intersection == "left":
                    new_rect = Rectangle( j.left, i.left, i.top, i.bottom )
                elif intersection == "top":
                    new_rect = Rectangle( i.right, i.left, i.top, j.top )
                elif intersection == "bottom":
                    new_rect = Rectangle( i.right, i.left, j.bottom, i.bottom )
                new_rectangles.append(new_rect)
                removed_intersection.append(i)

            if intersection == "inside":
                new_rectangles.append(j)
                removed_intersection.append(i)
                removed_intersection.append(j)

    for k in rectangles:
        if k not in new_rectangles and k not in removed_intersection:
            new_rectangles.append(k)

    if len(new_rectangles) == len(rectangles):
        return new_rectangles
    else:
        print("# rectangles inter", len(new_rectangles))
        return remove_intersection(new_rectangles)

def close(x, y):
    return True if abs(x-y) < CLOSENESS_THRESH else False

def intersecting(inner, outer):
    counter = 0
    output = None
    if  outer.bottom - CLOSENESS_THRESH < inner.top < outer.top + CLOSENESS_THRESH and outer.bottom - CLOSENESS_THRESH < inner.bottom < outer.top + CLOSENESS_THRESH:
        counter += 2
        
        if inner.left < outer.right and inner.right > outer.right: 
            output = "right"
        elif inner.left < outer.left and inner.right > outer.left:
            output = "left"
        
    if outer.left - CLOSENESS_THRESH < inner.right < outer.right + CLOSENESS_THRESH and outer.left - CLOSENESS_THRESH < inner.left < outer.right + CLOSENESS_THRESH:
        counter += 2
        
        if inner.bottom < outer.bottom and inner.top > outer.bottom:
            output = "bottom"
        elif inner.bottom < outer.top and inner.top > outer.top:
            output = "top"
    
    if counter == 4:
        output = "inside"

    return output    

if __name__ == "__main__":
    testSweep()