import rospy
import numpy as np
import ros_numpy
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
import math

import cv2
from collections import namedtuple

MASK_VELOCITY = 0b0000011111000111
map_data = OccupancyGrid()
pose_data = PoseStamped()
laser_data = LaserScan()
markers = {}
obstacles = {}
sweep = {}
fixed_unknown_frontier = []

#####Drone's radius#######
r = 0.4
####Safety parameter###
safety = 0.6
#####multiplicador de velocidade do drone, cuidado#####
speed_multiplier = 2

OBSTACLE_THRESH = 0.071
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

            area = (top-bottom)*(right-left)
            value = np.sum( gray_map[bottom:top+1, left:right+1] ) 
            if value < MAP_COLOR * OBSTACLE_THRESH * area:
                tr = (right, top)
                bl = (left, bottom)
                rect = Rectangle(right, left, top, bottom)
                rectangles.append(rect)

    rectangles = simplify_rectangles(rectangles)
    rectangles = remove_intersection(rectangles)
    print("# de retangulos: ", len(rectangles))

    return rectangles

def min_dist(x, y, rect):
    dr = (x - rect.right)**2 + (y - (rect.top + rect.bottom)/2)**2
    dl = (x - rect.left)**2  + (y - (rect.top + rect.bottom)/2)**2
    dt = (x - (rect.right + rect.left)/2)**2  + (y - rect.top)**2
    db = (x - (rect.right + rect.left)/2)**2  + (y - rect.bottom)**2
    return min(dr, dl, dt, db)

def execute_sweep(mav, map, canvas):

    drone_x = pose_data.pose.position.x
    drone_y = pose_data.pose.position.y

    map_drone_x = drone_x/map_data.info.resolution  + map_data.info.width/2
    map_drone_y = drone_y//map_data.info.resolution + map_data.info.height/2

    vertical = True

    rectangles = find_rectangles(map)
    rectangle_dists = [min_dist(drone_x, drone_y, i) for i in rectangles]
    rectangles = [x for _,x in sorted(zip(rectangles, rectangle_dists))]

    for rect in rectangles:

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

        start_x = (start_x - map_data.info.width/2)*map_data.info.resolution
        start_y = (map_data.info.initial_height/2 - start_y)*map_data.info.resolution

        go_to(map_pose(start_x, start_y), n)

        goal_x = (goal_x - map_data.info.width/2)*map_data.info.resolution
        goal_y = (map_data.info.initial_height/2 - goal_y)*map_data.info.resolution
        
        A = abs(rect.right - rect.left) - SAFE_DISTANCE if vertical else abs(rect.top - rect.bottom) - SAFE_DISTANCE
        A *= map_data.info.resolution

        t0 = rospy.get_time()
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

############################## This is where the exploration starts ##############################

def map_callback(data):
    global map_data
    map_data = data

def pose_callback(data):
    global pose_data
    pose_data = data

def laser_callback(data):
    global laser_data
    laser_data = data

def enqueue(queue, obj): 
    #simple function, the only use is to make WFD more readable
    queue.append(obj)
    return obj

def dequeue (queue): 
    #simple function, the only use is to make WFD more readable
    return queue.pop(0)

def mark(position,string): 
    #simple function, the only use is to make WFD more readable
    global markers
    markers[position] = string

def map_pose(x,y): 
    #given (x,y) point, returns the the map equivalent in the Occupancy Grid

    width = map_data.info.width
    res = map_data.info.resolution
    
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y

    row = int((y - origin_y)/res)
    column = (x - origin_x)/res

    map_pose = int((row * width) + 1 + column)
    return map_pose 

def cartesian_pose (position): 
    #given a map point in the Occupancy Grid, returns (x,y) point
    height = map_data.info.height
    width = map_data.info.width
    res = map_data.info.resolution
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    origin = int(width*height/2)
    y = origin_y + ((position)/width)*res
    x = origin_x + (position%width)*res
    return x,y

def is_frontier(position): 
    #detects if the map point given is a frontier between the known and the unknown
    if map_data.data[position] != 0 or map_data.data[position] == map_pose(pose_data.pose.position.x,pose_data.pose.position.y):
        return False
    for n in adj_pose(position):
        if map_data.data[n] == -1:
            return True
    return False

def is_obstacle_frontier(position):
    #detects if the m:ap point given is a frontier between an object and free space
    if map_data.data[position] != 0 or map_data.data[position] == map_pose(pose_data.pose.position.x,pose_data.pose.position.y):
        return False
    for n in adj_pose(position):
        if map_data.data[n] != -1 and map_data.data[n] != 0:
            return True
    return False

def distance(x1,y1, x2,y2):
    #calculates the distance between two points, ignoring obstacles
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

def adj_pose(position):
    #returns the Occupancy Grid position of all 8 adjacent points
    width = map_data.info.width
    adj = []
    adj.append(position + 1)
    adj.append(position - 1)
    adj.append(position + width -1)
    adj.append(position + width)
    adj.append(position + width + 1)
    if position - width > 0:
        adj.append(position - width)
        adj.append(position - width + 1)
        adj.append(position - width -1)
    return adj

def checkpoint_selection(checkpoints, camefrom, last_frontier):
    #chooses the nearest frontier, based on the size of the trajectory leading to the frontier
    distances = {}
    frontier_dic = {}
    i = 0
    if checkpoints == []:
        return []
    for frontier in checkpoints:
        if frontier in fixed_unknown_frontier:
            continue
        if frontier == last_frontier:
            fixed_unknown_frontier.append(frontier)
            continue
        #These checks are here to prevent the drone from getting stuck trying a frontier that doesn't lead to discovery (obstacles in this direction are out of lidar range)
        frontier_dic[i] = frontier
        distances[i] = len(reconstruct_path(camefrom, frontier[int(len(frontier)/2)]))
        i = i + 1
    distances = sorted(distances.items(), key=lambda x:x[1], reverse=False)
    return frontier_dic[distances[0][0]]

def WFD(n, goal=-1):
    #The heart of this code
    #WFD (Wavefront Frontier Detector) is a frontier-based exploration strategy
    #To understand the logic behind it visit: https://arxiv.org/pdf/1806.03581.pdf
    #Additions were made to the code, including the "camefrom", which makes WFD a trajectort planner aswell
    #additionaly, a "First wave" was introduced to dilate the map obstacles and prevent colisions while using the planner
    markers = {}
    Frontiers = []
    
    queuem = []
    initial = map_pose(pose_data.pose.position.x, pose_data.pose.position.y)
    if obstacles.has_key(initial):
        #prevents the drone from being stuck inside a dilated obstacle
        print("oops, inside dilated obstacle")
        initial = find_safety()
    enqueue(queuem, initial)
    camefrom = {}
    while(len(queuem) != 0):
        p = dequeue(queuem)

        if markers.has_key(p):
            if markers[p] == "First Wave":
                continue
        if obstacles.has_key(p):
            continue
        
        for adj in adj_pose(p):
            if (map_data.data[adj] == 0):
                if is_obstacle_frontier(adj):
                    queued = []
                    enqueue(queued, adj)
                    dilate_obstacle(n,adj)

                    while (len(queued) != 0):
                        p = dequeue(queued)
                        for w in adj_pose(p):
                            if is_obstacle_frontier(w):
                                if obstacles.has_key(w):
                                    if obstacles[w] == 2:
                                        continue
                                dilate_obstacle(n, w)
                                enqueue(queued, w)

                if not obstacles.has_key(adj) and not markers.has_key(adj):
                    enqueue(queuem, adj)
            markers[p] = "First Wave"
    #####Second wave starts here#####
    markers = {}
    queuem = []
    queuef = []
    NewFrontier = []
    enqueue(queuem, initial)
    mark(initial, "Map-Open-List")
    while(len(queuem) != 0):
        p = dequeue(queuem)
        if p == goal:
            #"point finder" usage of WFD, works when goal != 1 in the function parameters
            print("goal found")
            return camefrom

        if markers.has_key(p):
            if markers[p] == "Map-Close-List":
                continue
        if obstacles.has_key(p):
            continue

        if is_frontier(p):
            queuef = []
            NewFrontier = []
            enqueue(queuef, p)
            mark(p,"Frontier-Open-List")

            while len(queuef) != 0:

                q = dequeue(queuef)

                if markers.has_key(q):
                    if markers[q] == "Map-Close-List" or markers[q] == "Frontier-Close-List":
                        continue
                if obstacles.has_key(q):
                    continue

                if is_frontier(q):
                    NewFrontier.append(q)
                    for w in adj_pose(q):
                        if markers.has_key(w):
                            if markers[w] == "Frontier-Open-List" or markers[w] == "Frontier-Close-List" or markers[w] == "Map-Close-List":
                                continue
                        if is_frontier(w):
                            enqueue(queuef, w)
                            camefrom[w] = q
                            markers[w] = "Frontier-Open-List"
                markers[q] = "Frontier-Close-List"
            if NewFrontier not in Frontiers:
                Frontiers.append(NewFrontier)
            for point in NewFrontier:
                markers[point] = "Map-Close-List"
        for adj in adj_pose(p):
            if (map_data.data[adj] == 0):
                if markers.has_key(adj):
                    if markers[adj] == "Map-Open-List" or markers[adj] == "Map-Close-List":
                        continue
                if not obstacles.has_key(adj):
                    enqueue(queuem, adj)
                    camefrom[adj] = p
                    #the line above is the heart of the "camefrom" strategy, every "origin point" is stored
                    markers[adj] = "Map-Open-List"
        markers[p] = "Map-Close-List"

    return Frontiers, camefrom

def reconstruct_path(camefrom, current):
    #uses "camefrom" dictionary and a frontier point ("current") to construct a path between the drone and the frontier
    total_path = []
    total_path.append (current)
    while camefrom.has_key(current):
        current = camefrom[current]
        total_path.append(current)
    return total_path

def dilate_obstacle(n, map_pose):
    #dilates the obstacle in "map_pose" in the "obstacles" dictionary
    pose = map_pose
    for i in range(n):
        line = ((i+1) - int((n/2)+1))*map_data.info.width
        for k in range(n):
            column = ((k+1) - int((n/2)+1))
            if not obstacles.has_key(pose + line + column):
                obstacles[pose + line + column] = 1
    obstacles[pose] = 2

def paint_sweep(n):
    #"paints" the Occupancy Grid points that have been seen by the drone's camera
    pose = map_pose(pose_data.pose.position.x,pose_data.pose.position.y)
    for i in range(n):
        line = ((i+1) - int((n/2)+1))*map_data.info.width
        for k in range(n):
            column = ((k+1) - int((n/2)+1))
            if not sweep.has_key(pose + line + column) and not obstacles.has_key(pose + line + column):
                sweep[pose + line + column] = 1
    sweep[pose] = 2

def find_safety():
    #used when the drone enters an unknown area or a dilated obstacle
    #it provides the drone with a path to leave this situation
    queuek = []
    enqueue(queuek,map_pose(pose_data.pose.position.x,pose_data.pose.position.y))
    while True:
        print("finding safety")
        p = dequeue(queuek)
        if map_data.data[p] == 0 and not obstacles.has_key(p):
            return p
        for adj in adj_pose(p):
            enqueue(queuek,adj)

def run(n):

    mav = MAV("1")
    initial_height = 1
    last_frontier = []
    while not rospy.is_shutdown(): 
        mav.set_position_target(type_mask=MASK_VELOCITY,x_velocity=0,y_velocity=0,z_velocity=initial_height - mav.drone_pose.pose.position.z,yaw_rate=-pose_data.pose.orientation.z)
        #stop the drone
        frontiers = []
        checkpoints = []
        frontiers, camefrom = WFD(n)
        for frontier in frontiers:
            frontier = sorted(frontier)
            checkpoints.append(frontier)
        if checkpoints == [] and map_pose(pose_data.pose.position.x,pose_data.pose.position.y) != -1 and not obstacles.has_key(map_pose(pose_data.pose.position.x,pose_data.pose.position.y)):
            print("mapping complete :)")

            map = dict_to_np(obstacles)
            canvas = dict_to_np(sweep)
            execute_sweep(mav, map, canvas)

            mav.land()

        last_frontier = checkpoint_selection(checkpoints, camefrom,last_frontier)
        checkpoint = last_frontier[int(len(last_frontier)/2)]
        path = reconstruct_path(camefrom, checkpoint)
        path.reverse()
        print("Success, going towards goal")
        for point in path:
            print("new point detected")
            while distance(pose_data.pose.position.x, pose_data.pose.position.y, cartesian_pose(point)[0], cartesian_pose(point)[1]) > 0.2:
                if rospy.is_shutdown():
                    break

                paint_sweep(n)
                if cartesian_pose(point)[0] - pose_data.pose.position.x < 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    vel_x = speed_multiplier*vel_x
                    if vel_x < -0.6: 
                        vel_x = -0.6
                elif cartesian_pose(point)[0] - pose_data.pose.position.x > 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    vel_x = speed_multiplier*vel_x
                    if vel_x > 0.6:
                        vel_x = 0.6
                if cartesian_pose(point)[1] - pose_data.pose.position.y < 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    vel_y = speed_multiplier*vel_y
                    if vel_y < -0.6:
                        vel_y = -0.6
                elif cartesian_pose(point)[1] - pose_data.pose.position.y > 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    vel_y = speed_multiplier*vel_y
                    if vel_y > 0.6:
                        vel_y = 0.6

                mav.set_position_target(type_mask=MASK_VELOCITY,
                                    x_velocity=vel_x,
                                    y_velocity=vel_y,
                                    z_velocity=initial_height - mav.drone_pose.pose.position.z,
                                    yaw_rate=-pose_data.pose.orientation.z)
            

    mav.land()

def go_to(goal ,n):
    #find a path to goal using WFD and leads the drone to it
    camefrom = WFD(n, goal)
    path = reconstruct_path(camefrom, goal)
    path.reverse()
    for point in path:
        print("new point detected")
        while distance(pose_data.pose.position.x, pose_data.pose.position.y, cartesian_pose(point)[0], cartesian_pose(point)[1]) > 0.2:
            if rospy.is_shutdown():
                break
            paint_sweep(n)
            if cartesian_pose(point)[0] - pose_data.pose.position.x < 0:
                vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                vel_x = speed_multiplier*vel_x
                if vel_x < -0.6: 
                    vel_x = -0.6
            elif cartesian_pose(point)[0] - pose_data.pose.position.x > 0:
                vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                vel_x = speed_multiplier*vel_x
                if vel_x > 0.6:
                    vel_x = 0.6
            if cartesian_pose(point)[1] - pose_data.pose.position.y < 0:
                vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                vel_y = speed_multiplier*vel_y
                if vel_y < -0.6:
                    vel_y = -0.6
            elif cartesian_pose(point)[1] - pose_data.pose.position.y > 0:
                vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                vel_y = speed_multiplier*vel_y
                if vel_y > 0.6:
                    vel_y = 0.6
            mav.set_position_target(type_mask=MASK_VELOCITY,
                                x_velocity=vel_x,
                                y_velocity=vel_y,
                                z_velocity=initial_height - mav.drone_pose.pose.position.z,
                                yaw_rate=-pose_data.pose.orientation.z)
    print("done")

def plot(dictionary):
    #plots the dictionary as a 2d map (can be used with "obstacles" and "sweep")
    final_plot = dict_to_np(dictionary)
    print(final_plot.shape)
    plt.imshow(final_plot, cmap='hot', interpolation='nearest')
    plt.show()

def dict_to_np(dictionary):
    #transforms dictionary into a numpy array
    matrix = []
    for i in range(467):
        matrix.append([])
        for j in range(467):
            if not dictionary.has_key(467*i + j):
                dictionary[467*i + j] = 0
            if dictionary[467*i + j] == 2:
                dictionary[467*i + j] = 1 #same color 
            matrix[i].append(dictionary[467*i + j])
    return np.array(matrix)

def manual_paint(n):
    #paints manually the countours of the mission, stops WFD from going through the whole map
    for i in range(20):
        dilate_obstacle(n,map_pose(-1,-5 + 0.5*i))
    for i in range(21):
        dilate_obstacle(n,map_pose(11.5,-5 + 0.5*i))
    for i in range(25):
        dilate_obstacle(n,map_pose(-1 + i*0.5,-5))
    for i in range(25):
        dilate_obstacle(n,map_pose(-1 + i*0.5,5))

if __name__ == '__main__':
    rospy.init_node("mapping")
    mav = MAV("1")
    initial_height = 1
    mav.takeoff(initial_height)
    map_sub = rospy.Subscriber("/map", OccupancyGrid, map_callback, queue_size=1)
    pose_sub = rospy.Subscriber("/slam_out_pose", PoseStamped, pose_callback, queue_size=1)
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan, laser_callback, queue_size=1)
    rospy.wait_for_message("/map", OccupancyGrid)
    rospy.wait_for_message("/slam_out_pose", PoseStamped)
    n = int((r + safety)/map_data.info.resolution)
    manual_paint(n)
    run(n)