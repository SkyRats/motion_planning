import rospy
import numpy as np
from mavbase.MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.ndimage import morphology
import math
from random import randint
from time import time

import matplotlib.pyplot as plt

######### PARAMETERS #########

#velocity mask
MASK_VELOCITY = 0b0000011111000111 


class grid_motion_planning:

    def __init__(self):
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.pose_sub = rospy.Subscriber("/slam_out_pose", PoseStamped, self.pose_callback, queue_size=1)
        self.map_data = OccupancyGrid()
        self.pose_data = PoseStamped()
        self.inflated_grid = []
        self.grid_2d = []

        self.rectangle_point1 = [5,-5.25]
        self.rectangle_point2 = [17.5,-5.25]
        self.rectangle_point3 = [5,5.25]
        self.rectangle_point4 = [17.5,5.25]

        #drone radius
        self.r = 0.4
        #safety parameter
        self.safety = 1
        #speed parameters
        self.speed_multiplier = 5

        rospy.wait_for_message("/map", OccupancyGrid)
        self.n = int((self.r + self.safety)/self.map_data.info.resolution)

    def distance(self,x1,y1,x2,y2):
        #calculates the distance between two points, ignoring obstacles
        return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    
    ######### Callback functions #########
    def map_callback(self,data):
        self.map_data = data
    
    def pose_callback(self,data):
        self.pose_data = data

    ######### OccupancyGrid functions #########
    def map_pose(self,x,y):
        #given (x,y) point, returns the the map equivalent in the Occupancy Grid
        width = self.map_data.info.width
        res = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        row = int((y - origin_y)/res)
        column = (x - origin_x)/res
        map_pose = int((row * width) + 1 + column)
        return map_pose 
    
    def cartesian_pose (self,position): 
        #given a map point in the Occupancy Grid, returns (x,y) point
        height = self.map_data.info.height
        width = self.map_data.info.width
        res = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        origin = int(width*height/2)
        y = origin_y + ((position)/width)*res
        x = origin_x + (position%width)*res
        return x,y
    
    def is_frontier(self,position):
        #detects if the map point given is a frontier between the known and the unknown
        if self.map_data.data[position] != 0 or self.map_data.data[position] == self.map_pose(self.pose_data.pose.position.x,self.pose_data.pose.position.y):
            return False
        for a in self.adj_pose(position):
            if self.map_data.data[a] == -1:
                return True
        return False

    def adj_pose(self,position):
        #returns the Occupancy Grid position of all 8 adjacent points
        width = self.map_data.info.width
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
    
    def update_dilated_map(self):
        self.inflated_grid = np.array(self.map_data.data)

        self.inflated_grid = np.reshape(self.inflated_grid, (self.map_data.info.width,self.map_data.info.width))
        self.inflated_grid = morphology.grey_dilation(self.inflated_grid, size=(self.n,self.n))
        self.grid_2d = self.inflated_grid
        self.inflated_grid = np.reshape(self.inflated_grid, (self.map_data.info.width*self.map_data.info.width))
        self.manual_paint()
        

        plt.imshow(self.grid_2d, cmap='hot', interpolation='nearest')
        plt.show()
        return self.inflated_grid
    
    def dilate_obstacle(self, map_pose):
        #dilates the obstacle in "map_pose" in the "obstacles" dictionary
        pose = map_pose
        for i in range(self.n):
            line = ((i+1) - int((self.n/2)+1))*self.map_data.info.width
            for k in range(self.n):
                column = ((k+1) - int((self.n/2)+1))
                self.inflated_grid[pose + line + column] = 100
    
    def manual_paint(self):
        #paints manually the countours of the mission, stops WFD from going through the whole map
        for i in range(21):
            self.dilate_obstacle(self.map_pose(self.rectangle_point1[0],self.rectangle_point1[1] + 0.5*i))
        for i in range(21):
            self.dilate_obstacle(self.map_pose(self.rectangle_point2[0],self.rectangle_point2[1] + 0.5*i))
        for i in range(25):
            self.dilate_obstacle(self.map_pose(self.rectangle_point1[0] + i*0.5,self.rectangle_point1[1]))
        for i in range(25):
            self.dilate_obstacle(self.map_pose(self.rectangle_point1[0] + i*0.5,self.rectangle_point3[1]))
    
    ############################################# TRAJECTORY PLANNING FUNCTIONS #############################################
    def reconstruct_path(self,camefrom, current):
        #uses "camefrom" dictionary and a frontier point ("current") to construct a path between the drone and the frontier
        if type(camefrom) == int:
            return 0
        total_path = []
        total_path.append (current)
        while camefrom.has_key(current):
            current = camefrom[current]
            total_path.append(current)
        return total_path

    def find_safety(self):
        self.update_dilated_map()
        #used when the drone enters an unknown area or a dilated obstacle
        #it provides the drone with a path to leave this situation
        queuek = []
        queuek.append(self.map_pose(self.pose_data.pose.position.x,self.pose_data.pose.position.y))
        while True:
            p = queuek.pop()
            if self.map_data.data[p] == 0 and not self.inflated_grid[p] == 100:
                return p
            for adj in self.adj_pose(p):
                queuek.append(adj)
    
    def checkpoint_selection(self,checkpoints, camefrom):
        #chooses the nearest checkpoint, based on the size of the trajectory leading to the point
        distances = {}
        path = {}
        i = 0
        if checkpoints == []:
            return []
        for point in checkpoints:
            trajectory = self.reconstruct_path(camefrom, point)
            if type(trajectory) == int:
                continue
            path[i] = trajectory
            distances[i] = len(trajectory)
            i = i + 1
        distances = sorted(distances.items(), key=lambda x:x[1], reverse=False)
        selected_point = path[distances[0][0]][0]
        return self.A_star(selected_point)

    def follow_trajectory(self,trajectory,initial_height):
        mav = MAV("1")
        for point in trajectory:
            while self.distance(self.pose_data.pose.position.x, self.pose_data.pose.position.y, self.cartesian_pose(point)[0], self.cartesian_pose(point)[1]) > 0.2:
                if rospy.is_shutdown():
                    break
                """
                paint_sweep(n)
                """
                if self.cartesian_pose(point)[0] - self.pose_data.pose.position.x < 0:
                    vel_x = self.cartesian_pose(point)[0] - self.pose_data.pose.position.x
                    vel_x = self.speed_multiplier*vel_x
                    if vel_x < -0.8: 
                        vel_x = -0.8
                elif self.cartesian_pose(point)[0] - self.pose_data.pose.position.x > 0:
                    vel_x = self.cartesian_pose(point)[0] - self.pose_data.pose.position.x
                    vel_x = self.speed_multiplier*vel_x
                    if vel_x > 0.8:
                        vel_x = 0.8
                if self.cartesian_pose(point)[1] - self.pose_data.pose.position.y < 0:
                    vel_y = self.cartesian_pose(point)[1] - self.pose_data.pose.position.y
                    vel_y = self.speed_multiplier*vel_y
                    if vel_y < -0.8:
                        vel_y = -0.8
                elif self.cartesian_pose(point)[1] - self.pose_data.pose.position.y > 0:
                    vel_y = self.cartesian_pose(point)[1] - self.pose_data.pose.position.y
                    vel_y = self.speed_multiplier*vel_y
                    if vel_y > 0.8:
                        vel_y = 0.8

                mav.set_position_target(type_mask=MASK_VELOCITY,
                                    x_velocity=vel_x,
                                    y_velocity=vel_y,
                                    z_velocity=initial_height - mav.drone_pose.pose.position.z,
                                    yaw_rate=-self.pose_data.pose.orientation.z)
    ######### A* search #########
    def A_star(self,goal):
        self.update_dilated_map()

        t0 = time()

        OPEN = []
        CLOSE = []
        h = {}
        g = {}
        camefrom = {}

        start = self.map_pose(self.pose_data.pose.position.x, self.pose_data.pose.position.y)

        g[start] = 0
        h[start] = self.H(start,goal)

        OPEN.append(start)
        while(len(OPEN) != 0):
            print(OPEN)
            t1 = time()
            if (t1 - t0) > 1:
                return 0

            current = OPEN[0]
            for node in OPEN:
                if g[node] + h[node] <= g[current] + h[current]:
                    current = node
            OPEN.remove(current)

            if current == goal:
                return self.reconstruct_path(camefrom, goal)

            for sucessor in self.adj_pose(current):
                if sucessor == goal:
                    camefrom[sucessor] = current
                    return self.reconstruct_path(camefrom, goal)

                if self.map_data.data[sucessor] == -1 or self.inflated_grid[sucessor] == 100:
                    continue

                sucessor_current_cost = g[current] + self.map_data.info.resolution
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
                    h[sucessor] = self.H(sucessor,goal)
                g[sucessor] = sucessor_current_cost
                camefrom[sucessor] = current
            CLOSE.append(current)
        print("fudeu")
        return 0
    
    def H(self,node, goal):
        #heuristic function of the A* algorithm
        node = self.cartesian_pose(node)
        goal = self.cartesian_pose(goal)
        return self.distance(node[0], node[1], goal[0], goal[1])
    
    ######### Wavefront Frontier Detection #########
    def WFD(self):
        #WFD (Wavefront Frontier Detector) is a frontier-based exploration strategy
        #To understand the logic behind it visit: https://arxiv.org/pdf/1806.03581.pdf
        self.update_dilated_map()
        markers = {}
        camefrom = {}

        NewFrontier = []
        Frontiers = []

        queuem = []
        queuef = []

        initial = self.map_pose(self.pose_data.pose.position.x, self.pose_data.pose.position.y)
        if self.inflated_grid[initial] == 100:
            #prevents the drone from being stuck inside a dilated obstacle
            initial = self.find_safety()

        queuem.append(initial)
        markers[initial] = "Map-Open-List"
        while(len(queuem) != 0):
            p = queuem.pop(0)

            if markers.has_key(p):
                if markers[p] == "Map-Close-List":
                    continue
            if self.inflated_grid[p] == 100:
                continue


            if self.is_frontier(p):
                queuef = []
                NewFrontier = []
                queuef.append(p)
                markers[p] = "Frontier-Open-List"

                while len(queuef) != 0:

                    q = queuef.pop(0)

                    if markers.has_key(q):
                        if markers[q] == "Map-Close-List" or markers[q] == "Frontier-Close-List":
                            continue
                    if self.inflated_grid[q] == 100:
                        continue

                    if self.is_frontier(q):
                        NewFrontier.append(q)
                        for w in self.adj_pose(q):
                            if markers.has_key(w):
                                if markers[w] == "Frontier-Open-List" or markers[w] == "Frontier-Close-List" or markers[w] == "Map-Close-List":
                                    continue
                            if self.is_frontier(w):
                                queuef.append(w)
                                markers[w] = "Frontier-Open-List"
                    markers[q] = "Frontier-Close-List"
                if NewFrontier not in Frontiers:
                    Frontiers.append(NewFrontier)
                for point in NewFrontier:
                    markers[point] = "Map-Close-List"
            for adj in self.adj_pose(p):
                if (self.map_data.data[adj] == 0):
                    if markers.has_key(adj):
                        if markers[adj] == "Map-Open-List" or markers[adj] == "Map-Close-List":
                            continue
                    if not self.inflated_grid[adj] == 100:
                        queuem.append(adj)
                        camefrom[adj] = p
                        markers[adj] = "Map-Open-List"
            markers[p] = "Map-Close-List"

        return Frontiers, camefrom
    