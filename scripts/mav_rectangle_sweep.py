import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from motion_planning.motion_planning import grid_motion_planning

import cv2

# gray_map = np.array(map * 255, dtype=np.uint8)
# cv2.threshold(gray_map, 100, 255, cv2.THRESH_TOZERO, dst=gray_map)

from collections import namedtuple

OBSTACLE_THRESH = 0.065
CLOSENESS_THRESH = 3
MAP_COLOR = 255
DRONE_RADIUS = 0.4
SAFE_DISTANCE = 3 # Se aplica aos dois lados
PERIOD = 20
NUMBER_OF_STEPS = 20 

DEBUG = False

Rectangle = namedtuple('Rectangle', 'right left top bottom')

class rectangle_sweep:
    def __init__(self, initial_mav_x, initial_mav_y, grid_2d):
        self._origin = []
        self._map = self.cut_grid(grid_2d)
        self._initial_mav_x = initial_mav_x - self._origin[0]
        self._initial_mav_y = initial_mav_y - self._origin[1]
        self._rectangles = []

    def calculate_sweep(self):
        sweep = []
        rectangles = self.find_rectangles()

        drone_x = self._initial_mav_x
        drone_y = self._initial_mav_y

        while len(rectangles) != 0:

            print(len(rectangles))

            trajectory = []
            rect = self.find_closest_rectangle(drone_x, drone_y, rectangles)
            rectangles.remove(rect)

            w = rect.top - rect.bottom 
            h = rect.right - rect.left

            start_x = start_y = 0

            if abs(rect.right - rect.left) > abs(rect.top - rect.bottom):   # Rectangle is horizontal
                start_y = (rect.top + rect.bottom)//2
                start_x, goal_x, direction = self.calculate_start_goal_direction(drone_x, rect.right, rect.left)

                A = (rect.top - rect.bottom)//2
                A = A - SAFE_DISTANCE if A > SAFE_DISTANCE else A
                trajectory = self.calculate_sine_trajectory(start_x, start_y, goal_x, A, direction, True)

            else: # Rectangle is vertical
                start_x = (rect.right + rect.left)//2
                start_y, goal_y, direction = self.calculate_start_goal_direction(drone_y, rect.top, rect.bottom)

                A = (rect.right - rect.left)//2
                A = A - SAFE_DISTANCE if A > SAFE_DISTANCE else A
                trajectory = self.calculate_sine_trajectory(start_y, start_x, goal_y, A, direction, False)
            
            assert(len(trajectory) > 0), "Rectangle could not be swept"
            
            goal = self.to_unidimensional_map_pose(start_x, start_y)
            sweep.append( grid_motion_planning.A_star(goal) )
            sweep.extend(trajectory)

            drone_x = trajectory[-1][0]
            drone_y = trajectory[-1][1]

        return sweep

    def calculate_sine_trajectory(self, t_start, sin_t_start, t_goal, amplitude, direction, horizontal):
        trajectory = []
        t = t_start
        while True:
            sin_t = amplitude * np.sin(t * 2*np.pi/PERIOD)
            sin_t = sin_t_start + int( np.floor(sin_t) )
            
            if horizontal:

                trajectory.append( self.to_unidimensional_map_pose(t, sin_t) )
            else:
                trajectory.append( self.to_unidimensional_map_pose(sin_t, t) )
            
            if t == t_goal:
                break
            elif direction * t > direction * t_goal:
                t = t_goal
            else:
                t += direction * PERIOD//NUMBER_OF_STEPS
        return trajectory

    def calculate_start_goal_direction(self, drone, rect_bigger, rect_smaller):
        if abs(drone - rect_bigger) > abs(drone - rect_smaller): 
            return rect_smaller, rect_bigger, 1
        else: 
            return rect_greater, rect_bigger, -1

    def find_closest_rectangle(self, x, y):
        closest_rectangle = None
        min_distance = 1000
        for rect in self._rectangles:
            dr, dl, dt, db = self.calculate_rectangle_distances(x, y, rect)

            rectangle_min_distance = min(dr, dl, dt, db)
            if rectangle_min_distance < min_distance:
                min_distance = rectangle_min_distance
                closest_rectangle = rect
        return closest_rectangle 

    def calculate_rectangle_distances(self, x, y):
        distance_right  = np.sqrt( (x - self._rectangles.right)**2 + (y - (self._rectangles.top + self._rectangles.bottom)/2)**2 )
        distance_left   = np.sqrt( (x - self._rectangles.left)**2  + (y - (self._rectangles.top + self._rectangles.bottom)/2)**2 )
        distance_top    = np.sqrt( (x - (self._rectangles.right + self._rectangles.left)/2)**2  + (y - self._rectangles.top)**2 )
        distance_bottom = np.sqrt( (x - (self._rectangles.right + self._rectangles.left)/2)**2  + (y - self._rectangles.bottom)**2 )
        return distance_right, distance_left, distance_top, distance_bottom

    def find_rectangles(self):
        gray_map = np.array(self._map * 255, dtype=np.uint8)
        cv2.threshold(gray_map, 100, 255, cv2.THRESH_TOZERO, dst=gray_map) 

        corners = []
        cnts, _ = cv2.findContours(gray_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            epsilon = cv2.arcLength(cnt, True)*0.01
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for point in approx.tolist():
                corners.append(point[0])

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
                    self._rectangles.append(rect)

        print("# rectangles detected ", len(self._rectangles))

        rectangles = self.simplify_rectangles()
        rectangles = self.remove_intersection()
        print("# rectangles (final): ", len(self._rectangles))

    def simplify_rectangles(self):
        new_rectangles = []
        simplified = []
        while True:
            for i in self._rectangles:
                if i in simplified:
                    continue

                for j in self._rectangles:
                    if j in simplified or i == j:
                        continue
                                
                    if self.close(i.right, j.right) and self.close(i.left, j.left): # Proximos em X
                        if self.close(i.top, j.bottom) or self.close(i.bottom, j.top):
                            new_rect = Rectangle( i.right, i.left, max(i.top, j.top), min(i.bottom, j.bottom) )
                            new_rectangles.append(new_rect)
                            simplified.append(i)
                            simplified.append(j)
                    elif self.close(i.top, j.top) and self.close(i.bottom, j.bottom): # Proximos em Y
                        if self.close(i.right, j.left) or self.close(i.left, j.right):
                            new_rect = Rectangle( max(i.right, j.right), min(i.left, j.left), i.top, i.bottom )
                            new_rectangles.append(new_rect)
                            simplified.append(i)
                            simplified.append(j)

            for k in self._rectangles:
                if k not in new_rectangles and k not in simplified:
                    new_rectangles.append(k)
            
            if len(self._rectangles) == len(new_rectangles):
                break
        
        self._rectangles = new_rectangles

    def remove_intersection(self):
        new_rectangles = []
        removed_intersection = []
        while True:
            for i in self._rectangles:
                if i in removed_intersection:
                    continue

                for j in self._rectangles:
                    if j in removed_intersection or i == j:
                        continue
                    
                    intersection = self.intersecting(i, j)
                    
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

            for k in self._rectangles:
                if k not in new_rectangles and k not in removed_intersection:
                    new_rectangles.append(k)

            if len(self._rectangles) == len(new_rectangles):
                break
        
        self._rectangles = new_rectangles

    def cut_grid(self, grid_2d):
        left, right, bottom, top = self.calculate_grid_cutpoints()
        return grid_2d[left:rigth, bottom:top]

    def calculate_cutpoints_and_origin(self):
        tr_map = grid_motion_planning.map_pose(
            grid_motion_planning.rectangle_point4[0], 
            grid_motion_planning.rectangle_point4[1])
        bl_map = grid_motion_planning.map_pose(
            grid_motion_planning.rectangle_point1[0],
            grid_motion_planning.rectangle_point1[1])
        cut_left = bl_map % grid_motion_planning.map_data.info.width
        cut_right = tr_map % grid_motion_planning.map_data.info.width
        cut_bottom = bl_map // grid_motion_planning.map_data.info.width
        cut_top = tr_map // grid_motion_planning.map_data.info.width
        self._origin = [cut_left, cut_top]
        return cut_left, cut_right, cut_bottom, cut_top

    def to_unidimensional_map_pose(x, y):
        return grid_motion_planning.map_data.info.width*(self._origin[1]+y)
                + (self._origin[0]+x)

    def close(self, x, y):
        return True if abs(x-y) < CLOSENESS_THRESH else False

    def intersecting(self, inner, outer):
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
