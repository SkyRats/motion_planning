import sys, random, math, pygame, time
from pygame.locals import *
from math import sqrt, cos, sin, atan2
import numpy as np
from cylinder_det import cylinder_detector
import rospy
from sensor_msgs.msg import LaserScan
from scipy.ndimage import morphology
from searchtrees.treap import Treap
    
class dinamic_obstacle_avoidance:
    
    class node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.cost = 0.0
            self.parent = None
            self.children = []

    def __init__(self, goal_x):
        #PYGAME
        self.XDIM = 120
        self.YDIM = 180
        self.WINSIZE = [self.XDIM, self.YDIM]

        self.screen = pygame.display.set_mode(self.WINSIZE)
        black = 20, 20, 40
        self.screen.fill(black)
        pygame.display.update()


        #ROS
        self.laser_sub = rospy.Subscriber("/laser/scan", LaserScan, self.laser_callback, queue_size=1)
        rospy.wait_for_message('/laser/scan', LaserScan)
        

        #RRT PARAMS
        self.goal_x = goal_x
        self.EPSILON = 5
        self.rrt_near_radius = 15
        self.current_solution = []
        self.nodeList = []

        self.treap_x = Treap()

        #Dinamic Map
        self.cyl = cylinder_detector()
        self.update_map()

    def laser_callback(self,data):
        self.laser_data = data
    
    def distance(self,x1,y1,x2,y2): #JA ESTA NO MOTION
        return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    def nearest(self, point): #REQUER OTIMIZACAO
        tmpNode = self.node(point[0], point[1])
        nearList = self.near(tmpNode, self.rrt_near_radius)
        if len(nearList) >= 1:
            search_list = nearList
        else:
            search_list = self.nodeList
        nearestNode = search_list[0]
        for node in nearList:
            if self.distance(node.x,node.y,point[0],point[1]) < self.distance(nearestNode.x,nearestNode.y,point[0],point[1]):
                nearestNode = node
        return nearestNode

    def steer(self,origin,stretch):
        if self.distance(origin.x,origin.y,stretch[0],stretch[1]) < self.EPSILON:
            return stretch
        theta = atan2(stretch[1] - origin.y, stretch[0] - origin.x)
        return [int(origin.x + self.EPSILON * cos(theta)),int(origin.y + self.EPSILON * sin(theta))]                
    
    def collision(self,node,map):
        if map[node.x-1][node.y-1] == 100:
            return True
        else:
            return False
    
    def near(self, newNode, near_radius):
        radius = near_radius
        nearList = []

        debug_counter = 0

        while len(nearList) < 1:
            debug_counter += 1
            half_side = np.ceil(radius/np.sqrt(2))
            greater_key = newNode.x + half_side
            smaller_key = newNode.x - half_side

            is_close = lambda node: self.distance(node.x, node.y, newNode.x, newNode.y) < radius
            nearList = self.locate_close_nodes_between(self.treap_x.root, is_close, smaller_key, greater_key)

            radius += near_radius

        if debug_counter > 1:
            print('Instances: {0}\tRadius: {1}\tList size: {2}'.format(debug_counter, radius, len(nearList)))

        return nearList

    def locate_close_nodes_between(self, init_vert, is_close, smaller_key, greater_key):
        selected_nodes = []
        next_verts = []

        if init_vert != None:
            next_verts.append(init_vert)
            while len(next_verts) > 0:
                vert = next_verts.pop(0)
                if vert.key > smaller_key and vert.left != None:
                    next_verts.append(vert.left)
                if vert.key < greater_key and vert.right != None:
                    next_verts.append(vert.right)
                if smaller_key <= vert.key <= greater_key:                        
                    selected_nodes.extend(
                        self.get_close_nodes_from_vert(vert, is_close)
                    )

        return selected_nodes

    def get_close_nodes_from_vert(self, vert, is_close):
        nodes = []
        for node in vert.content:
            if is_close(node):
                nodes.append(node)
        return nodes
    
    def chooseparent(self,newNode,nearList):
        nodeMin = nearList[0]
        cmin = nodeMin.cost + self.distance(nodeMin.x,nodeMin.y,newNode.x,newNode.y)
        for node in nearList:
            if node.cost + self.distance(node.x,node.y,newNode.x,newNode.y) < cmin:
                cmin = node.cost + self.distance(node.x,node.y,newNode.x,newNode.y)
                nodeMin = node
        
        newNode.parent = nodeMin
        newNode.cost = cmin
        nodeMin.children.append(newNode)
    
    def rewire(self,nearList,newNode):
        for node in nearList:
            if newNode.cost + self.distance(newNode.x,newNode.y,node.x,node.y) < node.cost:
                node.parent.children.remove(node)
                pygame.draw.line(self.screen, (20, 20, 40), (node.x,node.y),(node.parent.x,node.parent.y))
                node.parent = newNode
                newNode.children.append(node)
                pygame.draw.line(self.screen, (0, 240, 0), (node.x,node.y),(newNode.x,newNode.y))
                node.cost = newNode.cost + self.distance(newNode.x,newNode.y,node.x,node.y)
    
    def forceRemoval(self): #requer otimizacao otimizacao (?)
        childless = []
        for node in self.nodeList:
            if node in self.current_solution:
                continue
            if node.children == []:
                childless.append(node)
        removed = childless[random.randint(0,len(childless)-1)]
        removed.parent.children.remove(removed)
        pygame.draw.line(self.screen, (20, 20, 40), (removed.x,removed.y),(removed.parent.x,removed.parent.y))
        self.nodeList.remove(removed)
        self.treap_x.remove(removed, removed.x)
    
    def generate_solution(self,node_goal):
        solution = []
        solution.append(node_goal)
        while node_goal.parent != None:
            solution.append(node_goal)
            node_goal = node_goal.parent
        solution.append(node_goal)
        solution.reverse()
        return solution

    def rrt_star_fn(self, start, i=0):
        self.start = start
        self.nodeList.append(self.start)
        self.treap_x.insert(self.start, self.start.x)
        t0 = time.time()
        found = False
        while i < 1500 or not found:
            rndPoint = [int(random.random() * 120.0), int(random.random() * 180.0)]
            
            nearestNode = self.nearest(rndPoint)

            newNode = self.steer(nearestNode,rndPoint)
            newNode = self.node(newNode[0],newNode[1])

            if not self.collision(newNode, self.DynamicMap):
                nearList = self.near(newNode, self.rrt_near_radius)

                self.chooseparent(newNode,nearList)

                self.nodeList.append(newNode)
                self.treap_x.insert(newNode, newNode.x)
                
                self.rewire(nearList,newNode)

                pygame.draw.line(self.screen, (0,240,0), (newNode.x,newNode.y),(newNode.parent.x,newNode.parent.y))
                pygame.display.update()


                if len(self.nodeList) > 2000:
                    self.forceRemoval()
            ####################################################
                found = newNode.x >= self.goal_x
                if found:
                    node_goal = newNode
            i += 1
        t1 = time.time()
        print(t1-t0)
        return self.generate_solution(node_goal)
    
    def update_map(self):
        self.DynamicMap = np.zeros((120,180))
        dilate = 5
        cylinders = self.cyl.get_cylinders_xy_and_radius(self.laser_data.ranges)
        for i in range(len(cylinders)):
            x = int(cylinders[i].cx/ 0.05)
            y = int((cylinders[i].cy + 4.5)/ 0.05) #Correct later
            obstacle = self.node(x,y)
            for collied_node in self.near(obstacle,dilate):
                if collied_node in self.nodeList:
                    self.DeleteBranch(collied_node)
            if x > 120 or y > 180:
                continue
            self.DynamicMap[x-1][y-1] = 100
        self.DynamicMap = morphology.grey_dilation(self.DynamicMap, size=(dilate*2,dilate*2))

        #import matplotlib.pyplot as plt
        #plt.imshow(self.DynamicMap, cmap='hot', interpolation='nearest')
        #plt.show()

    def CollisionInPath(self,safety):
        if safety > len(self.current_solution):
            safety = len(self.current_solution)
        for num in range(safety):
            if self.current_solution[num] not in self.nodeList:
                return True
        return False
    
    def DeleteBranch(self,node,exeption = None):
        child_list = []
        child_list.append(node)
        for child in node.children:
            if child == exeption:
                child.parent = None
                continue
            child_list.append(child)

        for child in child_list:
            for kids in child.children:
                child_list.append(kids)
        done = []
        for child in child_list:
            if child in done:
                continue
            self.treap_x.remove(child, child.x)
            self.nodeList.remove(child)
            if child.parent != None:
                pygame.draw.line(self.screen, (20,20,40), (child.x,child.y), (child.parent.x,child.parent.y))
                child.parent.children.remove(child)
            done.append(child)
        
    def nextNode(self):
        last_node = self.current_solution.pop(0)
        self.DeleteBranch(last_node, self.current_solution[0])
        self.start = self.current_solution[0]
        #drone goes to next node
        time.sleep(0.75)
        print("forward")


    def dynamic_rrt(self):
        self.current_solution = self.rrt_star_fn(self.node(0,90))
        while self.current_solution[0].x < self.goal_x and not rospy.is_shutdown():
            pygame.display.update()
            if self.collision(self.current_solution[0],self.DynamicMap):
                print("collided")
                return
            print(len(self.current_solution))
            print(self.current_solution[0].x,self.current_solution[0].y)
            self.update_map()
            if self.CollisionInPath(60):
                #StopMovement()
                self.current_solution = self.rrt_star_fn(self.current_solution[0], 1500)
                print("new path acquired")
                continue
            self.nextNode()
        print("complete!")

if __name__ == "__main__":
    rospy.init_node("dynamic_avoidance")
    doa = dinamic_obstacle_avoidance(110)
    doa.dynamic_rrt()
