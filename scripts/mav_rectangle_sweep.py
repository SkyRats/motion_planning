import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

import rospy
from nav_msgs.msg import Path

from collections import namedtuple

MASK_VELOCITY = 0b0000011111000111

OBSTACLE_THRESH = 0.06
CLOSENESS_THRESH = 3
SWEEP_THRESH = 0.85
MAP_COLOR = 255 
SAFE_DISTANCE = 1 # Applies to both sides
PERIOD = 20
NUMBER_OF_STEPS = 20 

DEBUG = False

Rectangle = namedtuple('Rectangle', 'right left top bottom')

class rectangle_sweep:
    def __init__(self, initial_mav_x, initial_mav_y, motion, mav):
        self._origin = []
        self.motion = motion
        self.mav = mav
        self._map = self.cut_grid_and_get_origin(self.motion.grid_2d)
        self._map_res = self.motion.map_data.info.resolution
        self._initial_mav_x = initial_mav_x - self._origin[0]
        self._initial_mav_y = initial_mav_y - self._origin[1]
        self._rectangles = []

        self.THERMAL_CAMERA_RADIUS_IN_PIXELS = 10 / self._map_res
        
        rospy.Subscriber('/trajectory', Path, self.trajectory_cb)
        rospy.wait_for_message('/trajectory', Path)

    def calculate_sweep(self):
        drone_radius_in_pixels = self.motion.r // self._map_res
        sweep = []
        self.find_rectangles()
        astar_start = self.motion.map_pose(
            self.motion.pose_data.pose.position.x, 
            self.motion.pose_data.pose.position.y)
        
        self.dilate_path()

        drone_x = self._initial_mav_x
        drone_y = self._initial_mav_y

        while len(self._rectangles) != 0:
            if self.is_rectangle_sweep_necessary(rectangle):
                trajectory = []
                rect = self.find_closest_rectangle(drone_x, drone_y)
                self._rectangles.remove(rect)

                start_x = start_y = 0

                if abs(rect.right - rect.left) > abs(rect.top - rect.bottom):
                    # Rectangle is horizontal
                    start_y = (rect.top + rect.bottom)//2
                    start_x, goal_x, direction = self.calculate_start_goal_direction(
                        drone_x, rect.right, rect.left)
                    A = (rect.top - rect.bottom)//2
                    if (A < drone_radius_in_pixels 
                        or abs(start_x - goal_x) < 2*drone_radius_in_pixels):
                        continue

                    A = A - SAFE_DISTANCE if A > SAFE_DISTANCE else A
                    trajectory = self.calculate_sine_trajectory(
                        start_x, start_y, goal_x, A, direction, True)

                else: 
                    # Rectangle is vertical
                    start_x = (rect.right + rect.left)//2
                    start_y, goal_y, direction = self.calculate_start_goal_direction(
                        drone_y, rect.top, rect.bottom)
                    A = (rect.right - rect.left)//2
                    if (A < drone_radius_in_pixels 
                        or abs(start_y - goal_y) < 2*drone_radius_in_pixels):
                        continue

                    A = A - SAFE_DISTANCE if A > SAFE_DISTANCE else A
                    trajectory = self.calculate_sine_trajectory(
                        start_y, start_x, goal_y, A, direction, False)

                assert(len(trajectory) > 0), "Rectangle could not be swept"

                goal = self.bidimensional_to_unidimensional(start_x, start_y)
                if self.motion.inflated_grid[goal] == 100:
                    goal = self.motion.find_safety(initial = goal)
                path = self.motion.A_star(goal, start = astar_start)

                sweep.extend(path)
                for point in path:
                    print(self.motion.cartesian_pose(point))
                sweep.extend(trajectory)

                astar_start = trajectory[-1]
                drone_x = trajectory[-1] // self.motion.map_data.info.width
                drone_y = trajectory[-1] %  self.motion.map_data.info.width
                
                # Prevents drone from exiting OFFBOARD flight mode
                # while trjaectory is computed
                height = 1
                self.mav.set_position_target(
                    type_mask=MASK_VELOCITY,
                    x_velocity=0,
                    y_velocity=0,
                    z_velocity=height - self.mav.drone_pose.pose.position.z,
                    yaw_rate=-self.motion.pose_data.pose.orientation.z)

        return sweep
    
    def is_rectangle_sweep_necessary(rectangle):
        left = rectangle.left
        right = rectangle.right
        top = rectangle.top
        bottom = rectangle.bottom

        value = np.sum( self._map[left:right, bottom:top] )
        area = (top - bottom) * (right - left)
        if value < MAP_COLOR * SWEEP_THRESH * area:
            return True
        else:
            return False

    def calculate_sine_trajectory(self, t_start, sin_t_start, t_goal, amplitude, direction, horizontal):
        trajectory = []
        t = t_start
        while True:
            sin_t = amplitude * np.sin(t * 2*np.pi/PERIOD)
            sin_t = sin_t_start + int( np.floor(sin_t) )
            
            if horizontal:
                trajectory.append( self.bidimensional_to_unidimensional(t, sin_t) )
            else:
                trajectory.append( self.bidimensional_to_unidimensional(sin_t, t) )
            
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
            return rect_greater, rect_smaller, -1

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

    def calculate_rectangle_distances(self, x, y, rectangle):
        distance_right  = np.sqrt( (x - rectangle.right)**2 + (y - (rectangle.top + rectangle.bottom)/2)**2 )
        distance_left   = np.sqrt( (x - rectangle.left)**2  + (y - (rectangle.top + rectangle.bottom)/2)**2 )
        distance_top    = np.sqrt( (x - (rectangle.right + rectangle.left)/2)**2  + (y - rectangle.top)**2 )
        distance_bottom = np.sqrt( (x - (rectangle.right + rectangle.left)/2)**2  + (y - rectangle.bottom)**2 )
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
                
                if i[1] <= j[1]:
                    right = j[1]
                    left = i[1]
                else:
                    right = i[1]
                    left = j[1]

                if i[0] <= j[0]:
                    top = j[0]
                    bottom = i[0]
                else:
                    top = i[0]
                    bottom = j[0]

                value = np.sum( gray_map[left:right, bottom:top] )
                area = (top - bottom) * (right - left)
                if value < MAP_COLOR * OBSTACLE_THRESH * area:
                    tr = (right, top)
                    bl = (left, bottom)
                    rect = Rectangle(right, left, top, bottom)
                    self._rectangles.append(rect)

        self.simplify_rectangles(self._rectangles)
        self.remove_intersection(self._rectangles)

    def simplify_rectangles(self, rectangles):
        new_rectangles = []
        simplified = []
                
        for i in rectangles:
            if i in simplified:
                continue

            for j in rectangles:
                if j in simplified or i == j:
                    continue
                            
                if self.close(i.right, j.right) and self.close(i.left, j.left): # Close in X
                    if self.close(i.top, j.bottom) or self.close(i.bottom, j.top):
                        new_rect = Rectangle( i.right, i.left, max(i.top, j.top), min(i.bottom, j.bottom) )
                        new_rectangles.append(new_rect)
                        simplified.append(i)
                        simplified.append(j)
                elif self.close(i.top, j.top) and self.close(i.bottom, j.bottom): # Close in Y
                    if self.close(i.right, j.left) or self.close(i.left, j.right):
                        new_rect = Rectangle( max(i.right, j.right), min(i.left, j.left), i.top, i.bottom )
                        new_rectangles.append(new_rect)
                        simplified.append(i)
                        simplified.append(j)

        for k in rectangles:
            if k not in new_rectangles and k not in simplified:
                new_rectangles.append(k)

        if len(new_rectangles) == len(rectangles):
            self._rectangles = new_rectangles
        else:
            self.simplify_rectangles(new_rectangles)
        
    def remove_intersection(self, rectangles):
        new_rectangles = []
        removed_intersection = []
                
        for j in rectangles:
            if j in removed_intersection:
                continue

            for i in rectangles:
                if i in removed_intersection or i == j:
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

        for k in rectangles:
            if k not in new_rectangles and k not in removed_intersection:
                new_rectangles.append(k)

        if len(new_rectangles) == len(rectangles):
            self._rectangles = new_rectangles
        else:
            self.remove_intersection(new_rectangles)
        
    def dilate_path():
        for pose in self._trajectory:
            position_x = pose.pose.position.x
            position_y = pose.pose.position.y
            x, y = self.meters_to_bidimensional(position_x, position_y)
            self.dilate_around(x, y, self.THERMAL_CAMERA_RADIUS_IN_PIXELS)

    def dilate_around(x, y, radius):
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                self._map[i][j] = MAP_COLOR

    def cut_grid_and_get_origin(self, grid_2d):
        left, right, bottom, top = self.calculate_cutpoints_and_origin()
        return grid_2d[left:right, bottom:top]

    def calculate_cutpoints_and_origin(self):
        tr_map = self.motion.map_pose(
            self.motion.rectangle_point4[0], 
            self.motion.rectangle_point4[1])
        bl_map = self.motion.map_pose(
            self.motion.rectangle_point1[0],
            self.motion.rectangle_point1[1])
        cut_left = bl_map // self.motion.map_data.info.width
        cut_right = tr_map // self.motion.map_data.info.width
        cut_bottom = bl_map % self.motion.map_data.info.width
        cut_top = tr_map % self.motion.map_data.info.width
        self._origin = (cut_left, cut_bottom)
        return cut_left, cut_right, cut_bottom, cut_top

    def unidimensional_to_bidimensional(self, uni):
        x = uni // self.motion.map_data.info.width
        y = uni % self.motion.map_data.info.width
        return x, y

    def bidimensional_to_unidimensional(self, x, y):
        return ( self.motion.map_data.info.width*(self._origin[0]+x)
                + (self._origin[1]+y) )

    def meters_to_bidimensional(self, x, y):
        uni = self.motion.map_pose(x, y)
        return self.unidimensional_to_bidimensional(uni)

    def close(self, x, y):
        return True if abs(x-y) < CLOSENESS_THRESH else False

    def intersecting(self, inner, outer):
        counter = 0
        output = None
        if  (outer.bottom - CLOSENESS_THRESH < inner.top < outer.top + CLOSENESS_THRESH 
            and outer.bottom - CLOSENESS_THRESH < inner.bottom < outer.top + CLOSENESS_THRESH):
            counter += 2
            
            if inner.left < outer.right and inner.right > outer.right: 
                output = "right"
            elif inner.left < outer.left and inner.right > outer.left:
                output = "left"
            
        if (outer.left - CLOSENESS_THRESH < inner.right < outer.right + CLOSENESS_THRESH 
            and outer.left - CLOSENESS_THRESH < inner.left < outer.right + CLOSENESS_THRESH):
            counter += 2
            
            if inner.bottom < outer.bottom and inner.top > outer.bottom:
                output = "bottom"
            elif inner.bottom < outer.top and inner.top > outer.top:
                output = "top"
        
        if counter == 4:
            output = "inside"

        return output    

    def trajectory_cb(data):
        self._trajectory = data.poses