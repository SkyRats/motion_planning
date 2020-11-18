import cv2
import numpy as np
from numpy.linalg import norm
from collections import namedtuple

THRESH = 0.06
CLOSENESS_THRESH = 3
MAP_COLOR = 114
DEACCELERATION = 15

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
                        
            if close(i.right, j.right) and close(i.left, j.left): # Próximos em X
                if close(i.top, j.bottom) or close(i.bottom, j.top):
                    new_rect = Rectangle( i.right, i.left, max(i.top, j.top), min(i.bottom, j.bottom) )
                    new_rectangles.append(new_rect)
                    simplified.append(i)
                    simplified.append(j)
            elif close(i.top, j.top) and close(i.bottom, j.bottom): # Próximos em Y
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
        return remove_intersection(new_rectangles)

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

    rectangles = simplify_rectangles(rectangles)
    rectangles = remove_intersection(rectangles)
    print(len(rectangles))

    for nr in rectangles:
        cv2.rectangle(map, (nr.right, nr.top), (nr.left, nr.bottom), (0,255,0), thickness=1 )

    cv2.imwrite("marked_clear.png", map)
    
    return rectangles

def execute_trajectory(mav):

    drone_x = pose_data.pose.position.x
    drone_y = pose_data.pose.position.y

    POSITION_TYPEMASK = 0b0000011111111011
    SAFE_DISTANCE = 0.3 # Se aplica aos dois lados
    vertical = True

    rectangles = find_rectangles()

    for rect in rectangles:
    
        if abs(rect.right - rect.left) > abs(rect.top - rect.bottom): # Retangulo horizontal
            start_y = goal_y = (rect.top + rect.bottom)/2
            
            if abs(drone_x - rect.right) > abs(drone_x - rect.left): 
                goal_x = rect.right 
                start_x = rect.left
            else: 
                goal_x = rect.left
                start_x = rect.right

            vertical = False

        else: # Retangulo vertical
            start_x = goal_x = (rect.right + rect.left)/2

            if abs(drone_y - rect.top) > abs(drone_y - rect.bottom): 
                goal_y = rect.top
                start_y = rect.bottom
            else: 
                goal_y = rect.bottom
                start_y = rect.top
        while not mav.chegou():
            mav.set_position_target(
                type_mask=POSITION_TYPEMASK,
                x_position=start_x, y_position=start_y
            ) # Vai para ponto de inicio
        
        A = abs(rect.right - rect.left) - SAFE_DISTANCE if vertical else abs(rect.top - rect.bottom) - SAFE_DISTANCE

        t0 = rospy.get_time()
        while not rospy.is_shutdown and abs(drone_x - goal_x) > GOAL_DIST and abs(drone_y - goal_y) > GOAL_DIST :
            
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
        
        fill_marker_area([rect.right, rect.top], [rect.left, rect.bottom])
    
    mav.set_vel(0,0,0)

find_rectangles()