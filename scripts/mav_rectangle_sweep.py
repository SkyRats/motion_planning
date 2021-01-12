import rospy
import numpy as np
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from .motion_planning import grid_motion_planning

import cv2
from collections import namedtuple

map_data = OccupancyGrid()
pose_data = PoseStamped()
    
OBSTACLE_THRESH = 0.04
GOAL_DIST = 0.15
CLOSENESS_THRESH = 3
MAP_COLOR = 255
DEACCELERATION = 15
DRONE_RADIUS = 0.4
SAFE_DISTANCE = 0.1 # Se aplica aos dois lados
INITIAL_HEIGHT = 1.5
PERIOD = 100
NUMBER_OF_STEPS = 20

Rectangle = namedtuple('Rectangle', 'right left top bottom')

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

def find_rectangles(map):
    gray_map = np.array(map * 255, dtype=np.uint8)

    corners = []
    cnts, _ = cv2.findContours(gray_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        epsilon = cv2.arcLength(cnt, True)*0.01
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for point in approx.tolist():
            corners.append(point[0])

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

            value = np.sum( gray_map[bottom:top+1, left:right+1] )
            area = (top - bottom) * (right - left)
            if value < MAP_COLOR * OBSTACLE_THRESH * area:
                tr = (right, top)
                bl = (left, bottom)
                rect = Rectangle(right, left, top, bottom)
                rectangles.append(rect)

    rectangles = simplify_rectangles(rectangles)
    rectangles = remove_intersection(rectangles)
    print("# de retangulos: ", len(rectangles))

    return rectangles

def calculate_rectangle_distances(x, y, rectangle):
    distance_right  = (x - rectangle.right)**2 + (y - (rectangle.top + rectangle.bottom)/2)**2
    distance_left   = (x - rectangle.left)**2  + (y - (rectangle.top + rectangle.bottom)/2)**2
    distance_top    = (x - (rectangle.right + rectangle.left)/2)**2  + (y - rectangle.top)**2
    distance_bottom = (x - (rectangle.right + rectangle.left)/2)**2  + (y - rectangle.bottom)**2
    return distance_right, distance_left, distance_top, distance_bottom

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

def calculate_sweep(map, n):

    sweep = []
    rectangles = find_rectangles(map)

    map_drone_x = pose_data.pose.position.x//map_data.info.resolution  + map_data.info.width/2
    map_drone_y = pose_data.pose.position.y//map_data.info.resolution + map_data.info.height/2

    while len(rectangles) != 0:

        traejctory = [] # Tipo (lista, dicionário ou fila) depende da implementação de follow_trajectory
        rect = find_closest_rectangle(map_drone_x, map_drone_y, rectangles)
        rectangles.remove(rect)

        w = rect.top - rect.bottom 
        h = rect.right - rect.left
        if w < 2 * DRONE_RADIUS or h < 2 * DRONE_RADIUS:
            continue

        start = find_safety(map_pose(start_x, start_y))
        sweep.append( grid_motion_planning.A_star(start, n) )

        if abs(rect.right - rect.left) > abs(rect.top - rect.bottom):
            start_y = goal_y = (rect.top + rect.bottom)/2
            
            if abs(map_drone_x - rect.right) > abs(map_drone_x - rect.left): 
                goal_x = rect.right 
                start_x = rect.left
            else: 
                goal_x = rect.left
                start_x = rect.right

            vertical = False
            current_x = next_x = 0
            i = 0
            A = abs(rect.top - rect.bottom) - SAFE_DISTANCE
            while current_x < goal_x:
                next_x = current_x + PERIOD/NUMBER_OF_STEPS
                if next_x > goal_x:
                    next_x = goal_x

                current_y   = A*np.sin(current_x    * 2*np.pi/PERIOD)
                next_y      = A*np.sin(next_x       * 2*np.pi/PERIOD)
                current_point = (current_x, current_y)
                next_point = (next_x, next_y)

                trajectory[i] = current_point
                trajectory[i+1] = next_point
                i += 2

                current_x = next_x

        else:
            start_x = goal_x = (rect.right + rect.left)/2

            if abs(map_drone_y - rect.top) > abs(map_drone_y - rect.bottom): 
                goal_y = rect.top
                start_y = rect.bottom
            else: 
                goal_y = rect.bottom
                start_y = rect.top

            vertical = True
            current_y = next_y = 0
            i = 0
            A = abs(rect.right - rect.left) - SAFE_DISTANCE 
            while current_y < goal_y:
                next_y = current_y + PERIOD/NUMBER_OF_STEPS
                if next_y > goal_y:
                    next_y = goal_y

                current_x   = A*np.sin(current_y    * 2*np.pi/PERIOD)
                next_x      = A*np.sin(next_y       * 2*np.pi/PERIOD)
                current_point = (current_x, current_y)
                next_point = (next_x, next_y)

                trajectory[i] = current_point
                trajectory[i+1] = next_point
                i += 2

                current_y = next_y
        
        sweep.append(trajectory)
        # Problema: current_x e current_y podem não estar atualizados ao sairem do loop
        map_drone_x = current_x
        map_drone_y = current_y

    return sweep