import rospy
import numpy as np
import ros_numpy
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from sensor_msgs.msg import LaserScan
import math

MASK_VELOCITY = 0b0000011111000111
map_data = OccupancyGrid()
pose_data = PoseStamped()
laser_data = LaserScan()
markers = {}
obstacles = {}
sweep = {}





#####Drone's radius#######
r = 0.4
####Safety parameter###
safety = 0.8
#####multiplicador de velocidade do drone, cuidado#####
speed_multiplier = 5



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

def checkpoint_selection(checkpoints,n):
    #chooses the nearest checkpoint, based on the size of the trajectory leading to the point
    distances = {}
    path = {}
    i = 0
    if checkpoints == []:
        return []
    for point in checkpoints:
        path[i] = reconstruct_path(A_star(point,n), point)
        distances[i] = len(path[i])
        i = i + 1
    distances = sorted(distances.items(), key=lambda x:x[1], reverse=False)
    print("path selected")
    return path[distances[0][0]]

def update_dilated_map(n):
    inflated_grid = np.array(map_data.data)
    inflated_grid = np.reshape(inflated_grid, (map_data.info.width,map_data.info.width))
    inflated_grid = morphology.grey_dilation(inflated_grid, size=(n,n))
    inflated_grid = np.reshape(inflated_grid, (map_data.info.width*map_data.info.width))
    return inflated_grid

def WFD(n):
    #The heart of this code
    #WFD (Wavefront Frontier Detector) is a frontier-based exploration strategy
    #To understand the logic behind it visit: https://arxiv.org/pdf/1806.03581.pdf
    #additionaly, a "First wave" was introduced to dilate the map obstacles and prevent colisions while using the planner
    inflated_grid = update_dilated_map(n)
    markers = {}

    NewFrontier = []
    Frontiers = []
    
    queuem = []
    queuef = []

    initial = map_pose(pose_data.pose.position.x, pose_data.pose.position.y)
    if inflated_grid[initial] == 100:
        #prevents the drone from being stuck inside a dilated obstacle
        print("oops, inside dilated obstacle")
        initial = find_safety(n)

    enqueue(queuem, initial)
    mark(initial, "Map-Open-List")
    while(len(queuem) != 0):
        p = dequeue(queuem)

        if markers.has_key(p):
            if markers[p] == "Map-Close-List":
                continue
        if obstacles.has_key(p) or inflated_grid[p] == 100:
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
                if obstacles.has_key(q) or inflated_grid[q] == 100:
                    continue

                if is_frontier(q):
                    NewFrontier.append(q)
                    for w in adj_pose(q):
                        if markers.has_key(w):
                            if markers[w] == "Frontier-Open-List" or markers[w] == "Frontier-Close-List" or markers[w] == "Map-Close-List":
                                continue
                        if is_frontier(w):
                            enqueue(queuef, w)
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
                if not obstacles.has_key(adj) and not inflated_grid[adj] == 100:
                    enqueue(queuem, adj)
                    markers[adj] = "Map-Open-List"
        markers[p] = "Map-Close-List"

    return Frontiers

def H(node, goal):
    #heuristic function of the A* algorithm
    node = cartesian_pose(node)
    goal = cartesian_pose(goal)
    return distance(node[0], node[1], goal[0], goal[1])

def A_star(goal,n):
    inflated_grid = update_dilated_map(n)
    OPEN = []
    CLOSE = []
    h = {}
    g = {}
    camefrom = {}

    start = map_pose(pose_data.pose.position.x, pose_data.pose.position.y)

    g[start] = 0
    h[start] = H(start,goal)

    enqueue(OPEN,start)
    while(len(OPEN) != 0):


        current = OPEN[0]
        for node in OPEN:
            if g[node] + h[node] <= g[current] + h[current]:
                current = node
        OPEN.remove(current)

        if current == goal:
            print("A* sucess")
            return camefrom
        
        for sucessor in adj_pose(current):
            if sucessor == goal:
                camefrom[sucessor] = current
                print("A* sucess")
                return camefrom

            if map_data.data[sucessor] == -1 or obstacles.has_key(sucessor) or inflated_grid[sucessor] == 100:
                continue
            
            sucessor_current_cost = g[current] + map_data.info.resolution
            if sucessor in OPEN:
                if g[sucessor] <= sucessor_current_cost:
                    continue
            elif sucessor in CLOSE:
                if g[sucessor] <= sucessor_current_cost:
                    continue
                CLOSE.remove(sucessor)
                OPEN.append(sucessor)
            else:
                OPEN.append(sucessor)
                h[sucessor] = H(sucessor,goal)
            g[sucessor] = sucessor_current_cost
            camefrom[sucessor] = current
        CLOSE.append(current)
    print("A* fail")
    return 0



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

"""
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

"""
def find_safety(n):
    inflated_grid = update_dilated_map(n)
    #used when the drone enters an unknown area or a dilated obstacle
    #it provides the drone with a path to leave this situation
    queuek = []
    enqueue(queuek,map_pose(pose_data.pose.position.x,pose_data.pose.position.y))
    while True:
        print("finding safety")
        p = dequeue(queuek)
        if map_data.data[p] == 0 and not obstacles.has_key(p) and not inflated_grid[p] == 100:
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
        frontiers = WFD(n)
        for frontier in frontiers:
            frontier = sorted(frontier)
            checkpoints.append(frontier[int(len(frontier)/2)])
        if checkpoints == [] and map_pose(pose_data.pose.position.x,pose_data.pose.position.y) != -1 and not obstacles.has_key(map_pose(pose_data.pose.position.x,pose_data.pose.position.y)):
            print("mapping complete :)")



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

def go_to(goal,n):
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

def plot_dic(dictionary):
    #plots the dictionary as a 2d map (can be used with "obstacles" and "sweep")
    matrix = []
    for i in range(467):
        matrix.append([])
        for j in range(467):
            if not dictionary.has_key(467*i + j):
                dictionary[467*i + j] = 0
            if dictionary[467*i + j] == 2:
                dictionary[467*i + j] = 1 #same color 
            matrix[i].append(dictionary[467*i + j])
    final_plot = np.array(matrix)
    print(final_plot.shape)
    plt.imshow(final_plot, cmap='hot', interpolation='nearest')
    plt.show()

def plot(list):
    #plots the list as a 2d map (can be used with "obstacles" and "sweep")
    matrix = []
    for i in range(467):
        matrix.append([])
        for j in range(467): 
            matrix[i].append(list[467*i + j])
    final_plot = np.array(matrix)
    print(final_plot.shape)
    plt.imshow(final_plot, cmap='hot', interpolation='nearest')
    plt.show()

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

def test(n):
    mav = MAV("1")
    initial_height = 1
    while not rospy.is_shutdown():
        inflated_grid = update_dilated_map(n)
        mav.set_position_target(type_mask=MASK_VELOCITY,x_velocity=0,y_velocity=0,z_velocity=initial_height - mav.drone_pose.pose.position.z,yaw_rate=-pose_data.pose.orientation.z)
        frontiers = []
        checkpoints = []
        frontiers = WFD(n)
        for frontier in frontiers:
            if frontier == []:
                print(frontier)
                continue
            frontier = sorted(frontier)
            checkpoints.append(frontier[int(len(frontier)/2)])
        
        if checkpoints == [] and map_pose(pose_data.pose.position.x,pose_data.pose.position.y) != -1 and not obstacles.has_key(map_pose(pose_data.pose.position.x,pose_data.pose.position.y)) and not inflated_grid[map_pose(pose_data.pose.position.x,pose_data.pose.position.y)] == 100:
            print("mapping complete :)")
            mav.land()
        
        path = checkpoint_selection(checkpoints,n)
        path.reverse()

        for point in path:
            while distance(pose_data.pose.position.x, pose_data.pose.position.y, cartesian_pose(point)[0], cartesian_pose(point)[1]) > 0.2:
                if rospy.is_shutdown():
                    break
                """
                paint_sweep(n)
                """
                if cartesian_pose(point)[0] - pose_data.pose.position.x < 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    vel_x = speed_multiplier*vel_x
                    if vel_x < -0.8: 
                        vel_x = -0.8
                elif cartesian_pose(point)[0] - pose_data.pose.position.x > 0:
                    vel_x = cartesian_pose(point)[0] - pose_data.pose.position.x
                    vel_x = speed_multiplier*vel_x
                    if vel_x > 0.8:
                        vel_x = 0.8
                if cartesian_pose(point)[1] - pose_data.pose.position.y < 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    vel_y = speed_multiplier*vel_y
                    if vel_y < -0.8:
                        vel_y = -0.8
                elif cartesian_pose(point)[1] - pose_data.pose.position.y > 0:
                    vel_y = cartesian_pose(point)[1] - pose_data.pose.position.y
                    vel_y = speed_multiplier*vel_y
                    if vel_y > 0.8:
                        vel_y = 0.8

                mav.set_position_target(type_mask=MASK_VELOCITY,
                                    x_velocity=vel_x,
                                    y_velocity=vel_y,
                                    z_velocity=initial_height - mav.drone_pose.pose.position.z,
                                    yaw_rate=-pose_data.pose.orientation.z)
        



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
    test(n)