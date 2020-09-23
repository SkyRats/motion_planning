import rospy
import numpy as np
import ros_numpy
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import math

map_data = OccupancyGrid()
pose_data = PoseStamped()
laser_data = LaserScan()
markers = {}
obstacles = {}

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
    queue.append(obj)
    return obj

def dequeue (queue):
    return queue.pop(0)

def mark(position,string):
    global markers
    markers[position] = string

def map_pose(x,y):

    width = map_data.info.width
    res = map_data.info.resolution
    
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y

    row = int((y - origin_y)/res)
    column = (x - origin_x)/res

    map_pose = int((row * width) + 1 + column)
    return map_pose 

def cartesian_pose (position):
    height = map_data.info.height
    width = map_data.info.width
    res = map_data.info.resolution
    origin = int(height*width/2)
    y = -((origin - position)/width)*res
    if not position%width:
        var = width
    else: 
        var = position%width 
    x = -((origin % width) - var)*res
    return x,y


def is_frontier(position):
    if map_data.data[position] != 0 or map_data.data[position] == map_pose(pose_data.pose.position.x,pose_data.pose.position.y):
        return False
    for n in adj(position):
        if n == -1:
            return True
    return False

def is_obstacle_frontier(position):
    if map_data.data[position] != 0 or map_data.data[position] == map_pose(pose_data.pose.position.x,pose_data.pose.position.y):
        return False
    for n in adj(position):
        if n != -1 and n != 0:
            return True
    return False


def adj(position):
    width = map_data.info.width
    adj = []
    adj.append(map_data.data[position + 1])
    adj.append(map_data.data[position - 1])
    adj.append(map_data.data[position + width -1])
    adj.append(map_data.data[position + width])
    adj.append(map_data.data[position + width + 1])
    adj.append(map_data.data[position - width])
    adj.append(map_data.data[position - width + 1])
    adj.append(map_data.data[position - width -1])
    return adj

def adj2(position): #wtf why??????????????????
    return adj(position)

def distance(x1,y1, x2,y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def adj_pose(position):
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

def checkpoint_selection(checkpoints):
    distances = {}
    if checkpoints == []:
        return []
    for point in checkpoints:
        distances[point] = distance(pose_data.pose.position.x, pose_data.pose.position.y,cartesian_pose(point)[0],cartesian_pose(point)[1])
    distances = sorted(distances.items(), key=lambda x:x[1], reverse=False)
    return distances[0][0]

def WFD():
    markers = {}
    Frontiers = []
    r = 0.7 #####RAIO DO DRONE#######
    n = int(r/map_data.info.resolution)
    queuem = []
    initial = map_pose(pose_data.pose.position.x, pose_data.pose.position.y)
    enqueue(queuem, initial)
    camefrom = {}
    while(len(queuem) != 0):
        p = dequeue(queuem)
        ######################
        try:
            markers[p] == "First Wave"
        except KeyError:
            markers[p] = "base"
        try:
            type(obstacles[p]) == int
        except KeyError:
            obstacles[p] = "base"
        ######################
        if markers[p] == "First Wave" or type(obstacles[p]) == int:
            continue
        for adj in adj_pose(p):
            if (map_data.data[adj] == 0):
                if is_obstacle_frontier(adj):
                    queued = []
                    enqueue(queued, adj)
                    obstacles[adj] = 1

                    while (len(queued) != 0):
                        p = dequeue(queued)
                        if obstacles[p] >= n:
                            continue
                        for w in adj_pose(p):
                            ######################
                            try:
                                type(obstacles[w]) == int
                            except KeyError:
                                obstacles[w] = "base"
                            ######################
                            if type(obstacles[w]) != int and map_data.data[w] == 0:
                                obstacles[w] = obstacles[p] + 1
                                enqueue(queued, w)
                            if is_obstacle_frontier(w) and (not obstacles[w] == 1):
                                obstacles[w] = 1
                                enqueue(queued, w)
                ######################
                try:
                    markers[adj] == "First Wave"
                except KeyError:
                    markers[adj] = "base"
                try:
                    type(obstacles[adj]) == int
                except KeyError:
                    obstacles[adj] = "base"
                ######################
                if type(obstacles[adj]) != int and not markers[adj] == "First Wave":
                    enqueue(queuem, adj)
            markers[p] = "First Wave"
    #########################################################################
    queuem = []
    queuef = []
    NewFrontier = []
    enqueue(queuem, initial)
    mark(initial, "Map-Open-List")
    while(len(queuem) != 0):
        p = dequeue(queuem)
        ######################
        try:
            markers[p] == "Map-Close-List"
        except KeyError:
            markers[p] = "base"
        try:
            type(obstacles[p]) == int
        except KeyError:
            obstacles[p] = "base"
        ######################
        if markers[p] == "Map-Close-List" or type(obstacles[p]) == int:
            continue
        if is_frontier(p):
            queuef = []
            NewFrontier = []
            enqueue(queuef, p)
            mark(p,"Frontier-Open-List")

            while len(queuef) != 0:

                q = dequeue(queuef)
                ######################
                try:
                    markers[q] == "Map-Close-List" or markers[q] == "Frontier-Close-List"
                except KeyError:
                    markers[q] = "base"
                try:
                    type(obstacles[q]) == int
                except KeyError:
                    obstacles[q] = "base"
                ######################
                if markers[q] == "Map-Close-List" or markers[q] == "Frontier-Close-List" or type(obstacles[q]) == int:
                    continue
                if is_frontier(q):
                    NewFrontier.append(q)
                    for w in adj_pose(q):
                        ######################
                        try:
                            markers[w] != "Frontier-Open-List" and markers[w] != "Frontier-Close-List" and markers[w] != "Map-Close-List"
                        except KeyError:
                            markers[w] = "base"
                        ######################
                        if (markers[w] != "Frontier-Open-List" and markers[w] != "Frontier-Close-List" and markers[w] != "Map-Close-List" and is_frontier(w)):
                            enqueue(queuef, w)
                            camefrom[w] = q
                            markers[w] = "Frontier-Open-List"
                markers[q] = "Frontier-Close-List"
            if NewFrontier not in Frontiers:
                Frontiers.append(NewFrontier) #saving data
            for point in NewFrontier:
                markers[point] = "Map-Close-List"
        for adj in adj_pose(p):
            if (map_data.data[adj] == 0):
                ######################
                try:
                    markers[adj] != "Map-Open-List" and markers[adj] != "Map-Close-List"
                except KeyError:
                    markers[adj] = "base"
                try:
                    type(obstacles[adj]) != int
                except KeyError:
                    obstacles[adj] = "base"
                ######################
                if (markers[adj] != "Map-Open-List" and markers[adj] != "Map-Close-List") and type(obstacles[adj]) != int:
                    enqueue(queuem, adj)
                    camefrom[adj] = p
                    markers[adj] = "Map-Open-List"
        markers[p] = "Map-Close-List"

    return Frontiers, camefrom



def reconstruct_path(camefrom, current):
    total_path = []
    total_path.append (current)
    while camefrom.has_key(current):
        current = camefrom[current]
        total_path.append(current)
    return total_path


def run():
    rospy.init_node("mapping")
    #"""
    mav = MAV("1")
    initial_height = 1
    mav.takeoff(initial_height)
    #"""
    MASK_VELOCITY = 0b0000011111000111
    map_sub = rospy.Subscriber("/map", OccupancyGrid, map_callback, queue_size=1)
    pose_sub = rospy.Subscriber("/slam_out_pose", PoseStamped, pose_callback, queue_size=1)
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan, laser_callback, queue_size=1)
    rospy.wait_for_message("/map", OccupancyGrid)
    rospy.wait_for_message("/slam_out_pose", PoseStamped)
    

    while not rospy.is_shutdown(): 
        frontiers = []
        checkpoints = []
        trajectory = WFD()
        frontiers = trajectory[0]
        for frontier in frontiers:
            if len(frontier) >= 25:
                frontier = sorted(frontier)
                checkpoints.append(frontier[int(len(frontier)/2)])
        checkpoint = checkpoint_selection(checkpoints)
        if checkpoint == []:
            print("mapping complete :)")
            mav.land()
        #"""
        #"""
        path = reconstruct_path(trajectory[1], checkpoint)
        path.reverse()
        print("Success, going towards goal")
        for point in path:
            print("new point detected")
            while distance(pose_data.pose.position.x, pose_data.pose.position.y, cartesian_pose(point)[0], cartesian_pose(point)[1]) > 0.2 :
                if rospy.is_shutdown():
                    break
                
                if cartesian_pose(point)[0] - pose_data.pose.position.x < 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    if vel_x < -0.3: 
                        vel_x = -0.3
                elif cartesian_pose(point)[0] - pose_data.pose.position.x > 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    if vel_x > 0.3:
                        vel_x = 0.3
                
                if cartesian_pose(point)[1] - pose_data.pose.position.y < 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    if vel_y < -0.3:
                        vel_y = -0.3
                elif cartesian_pose(point)[1] - pose_data.pose.position.y > 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    if vel_y > 0.3:
                        vel_y = 0.3

                mav.set_position_target(type_mask=MASK_VELOCITY,
                                    x_velocity=vel_x,
                                    y_velocity=vel_y,
                                    z_velocity=initial_height - mav.drone_pose.pose.position.z,
                                    yaw_rate=-pose_data.pose.position.z)
    mav.land()
    #"""

    

if __name__ == '__main__':
    run()