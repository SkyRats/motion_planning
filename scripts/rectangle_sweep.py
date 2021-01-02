import rospy
import numpy as np
import ros_numpy
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

from .motion_planning import grid_motion_planning

import cv2
from collections import namedtuple

map_data = OccupancyGrid()
pose_data = PoseStamped()
markers = {}
obstacles = {}
sweep = {}
fixed_unknown_frontier = []
    
OBSTACLE_THRESH = 0.04
GOAL_DIST = 0.15
CANVAS_THRESH = 0.8
CLOSENESS_THRESH = 3
MAP_COLOR = 255
DEACCELERATION = 15
DRONE_RADIUS = 0.4
SAFE_DISTANCE = 0.1 # Se aplica aos dois lados

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
    # cv2.threshold(gray_map, 100, 255, cv2.THRESH_TOZERO, dst=gray_map)

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

        vertical = None

        rectangles = find_rectangles(map)
        rectangle_dists = np.array([min_dist(map_drone_x, map_drone_y, i) for i in rectangles])
        sort = np.argsort(rectangle_dists)
        rectangles = [ rectangles[i] for i in sort ]

        w = rect.top - rect.bottom 
        h = rect.right - rect.left
        area = w * h

        if np.sum( canvas[rect.bottom:rect.top+1, rect.left:rect.right+1] ) > area * CANVAS_THRESH:
            continue 
        if w < 2 * DRONE_RADIUS or h < 2 * DRONE_RADIUS:
            continue
    
        if abs(rect.right - rect.left) > abs(rect.top - rect.bottom): # Retangulo horizontal
            start_y = goal_y = (rect.top + rect.bottom)/2
            
            if abs(map_drone_x - rect.right) > abs(map_drone_x - rect.left): 
                goal_x = rect.right 
                start_x = rect.left
            else: 
                goal_x = rect.left
                start_x = rect.right

            vertical = False

        else: # Retangulo vertical
            start_x = goal_x = (rect.right + rect.left)/2

            if abs(map_drone_y - rect.top) > abs(map_drone_y - rect.bottom): 
                goal_y = rect.top
                start_y = rect.bottom
            else: 
                goal_y = rect.bottom
                start_y = rect.top

            vertical = True

        start_x = (start_x - map_data.info.width/2)*map_data.info.resolution
        start_y = (map_data.info.height/2 - start_y)*map_data.info.resolution

        start = find_safety(map_pose(start_x, start_y))
        go_to(start, n)

        goal_x = (goal_x - map_data.info.width/2)*map_data.info.resolution
        goal_y = (map_data.info.height/2 - goal_y)*map_data.info.resolution
        
        A = abs(rect.right - rect.left) - SAFE_DISTANCE if vertical else abs(rect.top - rect.bottom) - SAFE_DISTANCE
        A *= map_data.info.resolution
    
        drone_x = pose_data.pose.position.x
        drone_y = pose_data.pose.position.y

        t0 = rospy.get_time()
        print( "Vertical? {0}\n Drone_x: {1}\tGoal_x: {2}\nDrone_y: {3}\tGoal_y: {4}".format(vertical, drone_x, goal_x, drone_y, goal_y) )
        print( (( not vertical and abs(drone_x - goal_x) > GOAL_DIST ) or (vertical and abs(drone_y - goal_y) > GOAL_DIST )) )
        
        while not rospy.is_shutdown and (( not vertical and abs(drone_x - goal_x) > GOAL_DIST ) or (vertical and abs(drone_y - goal_y) > GOAL_DIST )):
            
            drone_x = pose_data.pose.position.x
            drone_y = pose_data.pose.position.y

            vel_factor_x = 1.1 - 1/(DEACCELERATION * abs(drone_x - goal_x))
            vel_factor_y = 1.1 - 1/(DEACCELERATION * abs(drone_y - goal_y))
            
            t = rospy.get_time - t0
            x_vel = -A*np.cos(t) if vertical else 1
            y_vel = 1 if vertical else -A*np.cos(t)

            x_vel *= vel_factor_x if vel_factor_x > 0 else 0.1
            y_vel *= vel_factor_y if vel_factor_y > 0 else 0.1

            mav.set_vel(x_vel, y_vel, 0)
