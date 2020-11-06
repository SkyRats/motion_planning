import rospy
import numpy as np
import ros_numpy
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
import math

map_data = OccupancyGrid()
pose_data = PoseStamped()
laser_data = LaserScan()
markers = {}
obstacles = {}
fixed_unknown_frontier = []

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
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    origin = int(width*height/2)
    y = origin_y + ((position)/width)*res
    x = origin_x + (position%width)*res
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

def checkpoint_selection(checkpoints, camefrom, last_frontier):
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
        frontier_dic[i] = frontier
        distances[i] = len(reconstruct_path(camefrom, frontier[int(len(frontier)/2)]))
        i = i + 1
    distances = sorted(distances.items(), key=lambda x:x[1], reverse=False)
    return frontier_dic[distances[0][0]]

def WFD():
    markers = {}
    Frontiers = []
    r = 0.4 #####RAIO DO DRONE#######
    n = int((r + 0.4)/map_data.info.resolution)
    queuem = []
    initial = map_pose(pose_data.pose.position.x, pose_data.pose.position.y)
    try:
        type(obstacles[initial]) == int
    except KeyError:
        obstacles[initial] = "base"
        
    if type(obstacles[initial]) == int:
        print("oops, inside dilated obstacle")
        for adj in adj_pose(initial):
            enqueue(queuem,adj)
    else:
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
        if markers[p] == "First Wave" or (type(obstacles[p]) == int and obstacles[p] >= n):
            continue
        for adj in adj_pose(p):
            if (map_data.data[adj] == 0):
                if is_obstacle_frontier(adj):
                    queued = []
                    enqueue(queued, adj)
                    dilate(n,adj)

                    while (len(queued) != 0):
                        p = dequeue(queued)
                        for w in adj_pose(p):
                            ######################
                            try:
                                type(obstacles[w]) == int
                            except KeyError:
                                obstacles[w] = "base"
                            ######################
                            if is_obstacle_frontier(w) and (not obstacles[w] == 2):
                                dilate(n, w)
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
    safety = 7
    for i in range(safety):
        total_path.pop(0)
    return total_path

def dilate(n, map_pose):
    pose = map_pose
    for i in range(n):
        line = ((i+1) - int((n/2)+1))*map_data.info.width
        for k in range(n):
            column = ((k+1) - int((n/2)+1))
            try:
                type(obstacles[pose + line + column]) != int
            except KeyError:
                obstacles[pose + line + column] = "base"
            if type(obstacles[pose + line + column]) != int:
                obstacles[pose + line + column] = 1
    obstacles[pose] = 2

    
def run():
    mav = MAV("1")
    MASK_VELOCITY = 0b0000011111000111
    initial_height = 1

    r = 0.4 #####RAIO DO DRONE#######
    n = int((r + 0.4)/map_data.info.resolution)
    last_frontier = []
    while not rospy.is_shutdown(): 
        mav.set_position_target(type_mask=MASK_VELOCITY,x_velocity=0,y_velocity=0,z_velocity=initial_height - mav.drone_pose.pose.position.z,yaw_rate=-pose_data.pose.orientation.z)
        #stopping the drone
        frontiers = []
        checkpoints = []
        trajectory = WFD()
        frontiers = trajectory[0]
        for frontier in frontiers:
            frontier = sorted(frontier)
            checkpoints.append(frontier)
        if checkpoints == [] and map_pose(pose_data.pose.position.x,pose_data.pose.position.y) != -1 and type(obstacles[map_pose(pose_data.pose.position.x,pose_data.pose.position.y)]) != int:
            print("mapping complete :)")
            matrix = []
            for i in range(700):
                matrix.append([])
                for j in range(700):
                    try:
                        type(obstacles[700*i + j]) == int
                    except KeyError:
                        obstacles[700*i + j] = 0
                    if type(obstacles[700*i + j]) != int:
                        obstacles[700*i + j] = 0
                    matrix[i].append(obstacles[700*i + j])
            final_plot = np.array(matrix)
            print(final_plot.shape)
            plt.imshow(final_plot, cmap='hot', interpolation='nearest')
            plt.show()
            mav.land()
        last_frontier = checkpoint_selection(checkpoints, trajectory[1],last_frontier)
        print(last_frontier)
        checkpoint = last_frontier[int(len(last_frontier)/2)]
        path = reconstruct_path(trajectory[1], checkpoint)
        path.reverse()
        print(path)
        print("Success, going towards goal")
        for point in path:
            print("new point detected")
            while distance(pose_data.pose.position.x, pose_data.pose.position.y, cartesian_pose(point)[0], cartesian_pose(point)[1]) > 0.2:
                if rospy.is_shutdown():
                    break

                if cartesian_pose(point)[0] - pose_data.pose.position.x < 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    if vel_x < -0.5: 
                        vel_x = -0.5
                elif cartesian_pose(point)[0] - pose_data.pose.position.x > 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    if vel_x > 0.5:
                        vel_x = 0.5
                
                if cartesian_pose(point)[1] - pose_data.pose.position.y < 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    if vel_y < -0.5:
                        vel_y = -0.5
                elif cartesian_pose(point)[1] - pose_data.pose.position.y > 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    if vel_y > 0.5:
                        vel_y = 0.5

                mav.set_position_target(type_mask=MASK_VELOCITY,
                                    x_velocity=vel_x,
                                    y_velocity=vel_y,
                                    z_velocity=initial_height - mav.drone_pose.pose.position.z,
                                    yaw_rate=-pose_data.pose.orientation.z)

    mav.land()
    #"""

def manual_paint():
    r = 0.4 #####RAIO DO DRONE#######
    n = int((r + 0.4)/map_data.info.resolution)
    for i in range(20):
        dilate(n,map_pose(-1,-5 + 0.5*i))
    for i in range(21):
        dilate(n,map_pose(11.5,-5 + 0.5*i))
    for i in range(25):
        dilate(n,map_pose(-1 + i*0.5,-5))
    for i in range(25):
        dilate(n,map_pose(-1 + i*0.5,5))



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
    manual_paint()
    run()