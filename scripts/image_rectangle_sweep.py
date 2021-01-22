import numpy as np
import cv2

from collections import namedtuple
    
OBSTACLE_THRESH = 0.08
CLOSENESS_THRESH = 3
MAP_COLOR = 255
DRONE_RADIUS = 0.4
SAFE_DISTANCE = 3 # Se aplica aos dois lados
PERIOD = 20
NUMBER_OF_STEPS = 20 

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

def drawRectangles(rectangles):
    global map
    colors = [  (199, 211, 141), (179, 255, 255), (218, 186, 190), 
                (114, 128, 251), (211, 177, 128), (98, 180, 253), 
                (105, 222, 179), (229, 205, 252), (217, 217, 217), 
                (189, 128, 188), (197, 235, 204), (111, 237, 255)]
    for i in range(len(rectangles)):
        color_index = i % len(colors)
        rect = rectangles[i]
        topright = (rect.right, rect.top)
        bottomleft = (rect.left, rect.bottom)
        cv2.rectangle(map, topright, bottomleft, colors[color_index])

def calculate_sweep():
    sweep = []
    rectangles = find_rectangles()

    drone_x = 290
    drone_y = 190
    cv2.circle(map, (drone_x, drone_y), 1, (0,0,255), thickness=2)

    while len(rectangles) != 0:

        print(len(rectangles))

        trajectory = []
        rect = find_closest_rectangle(drone_x, drone_y, rectangles)
        rectangles.remove(rect)

        w = rect.top - rect.bottom 
        h = rect.right - rect.left

        start_x = start_y = 0

        if abs(rect.right - rect.left) > abs(rect.top - rect.bottom):   # Rectangle is horizontal
            start_y = (rect.top + rect.bottom)//2
            start_x, goal_x, direction = calculate_start_goal_direction(drone_x, rect.right, rect.left)

            A = (rect.top - rect.bottom)//2
            A = A - SAFE_DISTANCE if A > SAFE_DISTANCE else A
            trajectory = calculate_sine_trajectory(start_x, start_y, goal_x, A, direction, True)

        else: # Rectangle is vertical
            start_x = (rect.right + rect.left)//2
            start_y, goal_y, direction = calculate_start_goal_direction(drone_y, rect.top, rect.bottom)

            A = (rect.right - rect.left)//2
            A = A - SAFE_DISTANCE if A > SAFE_DISTANCE else A
            trajectory = calculate_sine_trajectory(start_y, start_x, goal_y, A, direction, False)
        
        assert(len(trajectory) > 0), "Rectangle could not be swept"
        
        sweep.append( (start_x, start_y) )
        sweep.extend(trajectory)

        drone_x = trajectory[-1][0]
        drone_y = trajectory[-1][1]

    return sweep

def calculate_sine_trajectory(t_start, sin_t_start, t_goal, amplitude, direction, horizontal):
    trajectory = []
    t = t_start
    while True:
        sin_t = amplitude * np.sin(t * 2*np.pi/PERIOD)
        sin_t = sin_t_start + int( np.floor(sin_t) )
        
        if horizontal:
            trajectory.append( (t, sin_t) )
        else:
            trajectory.append( (sin_t, t) )
        
        if t == t_goal:
            break
        elif direction * t > direction * t_goal:
            t = t_goal
        else:
            t += direction * PERIOD//NUMBER_OF_STEPS
    return trajectory

def calculate_start_goal_direction(drone, rect_greater, rect_smaller):
    if abs(drone - rect_greater) > abs(drone - rect_smaller): 
        return rect_smaller, rect_greater, 1
    else: 
        return rect_greater, rect_smaller, -1

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

    rectangles = merge_rectangles(rectangles)
    rectangles = remove_intersection(rectangles)
    print("# rectangles (final): ", len(rectangles))

    if DEBUG:
        drawRectangles(rectangles)

    return rectangles

def merge_rectangles(rectangles):
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
        return merge_rectangles(new_rectangles)

def remove_intersection(rectangles):
    new_rectangles = []
    removed_intersection = []
            
    for j in rectangles:
        if j in removed_intersection:
            continue

        for i in rectangles:
            if i in removed_intersection or i == j:
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
    testRectangles()
