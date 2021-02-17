import sys, random, math, pygame, time
from pygame.locals import *
from math import sqrt, cos, sin, atan2
import numpy as np
from cylinder_det import cylinder_detector
import rospy
from sensor_msgs.msg import LaserScan
from scipy.ndimage import morphology

MASK_VELOCITY = 0b0000011111000111 

    
class dinamic_obstacle_avoidance:
    
    class node:
        def __init__(self,x,y):
            self.x = x
            self.y = y
            self.cost = 0.0
            self.parent = None
            self.children = []

    def __init__(self, goal, mav):
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
        self.mav = mav
        

        #RRT PARAMS
        self.goal = self.node(goal[0],goal[1])
        self.EPSILON = 4
        self.rrt_near_radius = 8
        self.current_solution = []
        self.nodeList = []
        self.old_path_nodes = []

        #Dinamic Map
        self.cyl = cylinder_detector()
        self.old_obstacles = []
        self.update_map()

    def laser_callback(self,data):
        self.laser_data = data
    
    def distance(self,x1,y1,x2,y2):
        return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    def nearest(self,point): #REQUER OTIMIZACAO
        nearestNode = self.start
        for node in self.nodeList:
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
    
    def near(self,newNode,near_radius):
        nearList = []
        for node in self.nodeList:
            if self.distance(node.x,node.y,newNode.x,newNode.y) < near_radius:
                nearList.append(node)
        return nearList

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
    
    def generate_solution(self,node_goal):
        solution = []
        while True:
            solution.append(node_goal)
            node_goal = node_goal.parent
            if node_goal == self.start:
                break
        solution.append(node_goal)
        solution.reverse()
        return solution

    def rrt_star_fn(self, start, i=0):
        self.start = start
        self.nodeList.append(self.start)
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
                
                self.rewire(nearList,newNode)

                pygame.draw.line(self.screen, (0,240,0), (newNode.x,newNode.y),(newNode.parent.x,newNode.parent.y))
                pygame.display.update()


                if len(self.nodeList) > 2000:
                    self.forceRemoval()
            ####################################################
                found = self.distance(newNode.x,newNode.y,self.goal.x,self.goal.y) < 5
                if found:
                    node_goal = newNode
            i += 1
        t1 = time.time()
        print(t1-t0)
        return self.generate_solution(node_goal)
    
    def update_map(self):
        dilate = 5
        self.DynamicMap = np.zeros((120,180))
        self.screen.fill((20, 20, 40))
        if self.current_solution != []:
            for point in range(len(self.current_solution)-1):
                pygame.draw.line(self.screen, (0,240,0), (self.current_solution[point].x,self.current_solution[point].y),(self.current_solution[point+1].x,self.current_solution[point+1].y))
        for obstacle in self.old_obstacles:
            pygame.draw.circle(self.screen,(240,0,0),(obstacle.x,obstacle.y),dilate)
        self.old_obstacles = []
        pygame.display.update()
        cylinders = self.cyl.get_cylinders_xy_and_radius(self.laser_data.ranges)
        for i in range(len(cylinders)):
            x = int((cylinders[i].cx + self.mav.drone_pose.pose.position.x)/ 0.05)
            y = int((cylinders[i].cy + 4.5 + self.mav.drone_pose.pose.position.y)/ 0.05) #Correct later
            obstacle = self.node(x,y)
            self.old_obstacles.append(obstacle)
            if x > 120 or y > 180 or x < 0 or y < 0:
                continue
            self.DynamicMap[x-1][y-1] = 100
        self.DynamicMap = morphology.grey_dilation(self.DynamicMap, size=(int(dilate*4),int(dilate*4)))

        #import matplotlib.pyplot as plt
        #plt.imshow(self.DynamicMap, cmap='hot', interpolation='nearest')
        #plt.show()

    def CollisionInPath(self,safety):
        if safety > len(self.current_solution):
            safety = len(self.current_solution)
        for num in range(safety):
            if num == 0:
                continue
            if self.line_check(self.current_solution[num-1],self.current_solution[num]):
                i = 0
                current = self.current_solution[num]
                while self.collision(current,self.DynamicMap):
                    i += 1
                    current = self.current_solution[num + i]
                pygame.draw.circle(self.screen,(240,0,0),(self.current_solution[num-1].x,self.current_solution[num-1].y),2)
                pygame.draw.circle(self.screen,(240,0,0),(current.x,current.y),2)
                pygame.display.update()
                print("AAAAAAAA")
                return True
        return False

    def line_check(self,node1,node2):
        current = self.node(node1.x,node1.y)
        while current.x != node2.x or current.y != node2.y:
            if current.x < node2.x:
                x_advance = 1
            elif current.x > node2.x:
                x_advance = -1
            else:
                x_advance = 0
            if current.y < node2.y:
                y_advance = 1
            elif current.y > node2.y:
                y_advance = -1
            else:
                y_advance = 0

            current.x += x_advance
            current.y += y_advance
            if self.collision(current,self.DynamicMap):
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
            self.nodeList.remove(child)
            if child.parent != None:
                if child in child.parent.children:
                    pygame.draw.line(self.screen, (20,20,40), (child.x,child.y), (child.parent.x,child.parent.y))
                    child.parent.children.remove(child)
            done.append(child)
        
    def nextNode(self):
        last_node = self.current_solution.pop(0)
        self.old_path_nodes.append(last_node)
        self.start = self.current_solution[0]
        #drone goes to next node
        print("going for next node")
        x = self.current_solution[0].x * 0.05
        y = self.current_solution[0].y *0.05 - 4.5
        
        self.mav.set_position(x, y, 1)
        time.sleep(0.25)
        
    def dynamic_rrt(self):
        mav.hold(0.1)
        self.current_solution = self.rrt_star_fn(self.node(0,90))
        while self.distance(self.current_solution[0].x,self.current_solution[0].y,self.goal.x,self.goal.y) > 5 and not rospy.is_shutdown():
            pygame.display.update()
            self.update_map()
            if self.CollisionInPath(99):
                print("deleting old ones")
                for nodes in self.old_path_nodes:
                    print("a")
                    if nodes in self.nodeList:
                        self.DeleteBranch(nodes)
                print("recalculating...")
                #self.current_solution = self.rrt_star_fn(self.current_solution[0], 1000)
                print("new path acquired")
                continue
            #self.nextNode()
        print("complete!")

if __name__ == "__main__":
    rospy.init_node("dynamic_avoidance")
    from mavbase.MAV import MAV
    mav = MAV("1")
    #mav.takeoff(1)
    doa = dinamic_obstacle_avoidance((110,90),mav)
    doa.dynamic_rrt()
