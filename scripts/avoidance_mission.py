import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from MAV import MAV

laser_data = LaserScan()
MAX_VEL = 2
MASK_VELOCITY = 0b0000011111000111

def laser_callback(data):
    global laser_data
    laser_data = data

def saturate(vector):
    norm = np.linalg.norm(vector)
    if norm > MAX_VEL:
        vector = (vector/norm)*MAX_VEL
    return vector

def normal_dist(mean, variance, x):
    return (1/((variance*2*np.pi)**0.5))*np.exp((x-mean)/variance**0.5)


def run():
    rospy.init_node("avoidance")
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan, laser_callback, queue_size=1)
    mav = MAV("1")
    goal = np.array([8, 0])
    initial_height = 1.5
    mav.takeoff(initial_height)

    a=0.004
    b=0.9
    c=0.01
    d=-0.5

    Kr = -4 # repulsive
    Ka = 0.5 # attractive
    Kz = 0.5 # height proportional control
    Ky = -0.5 # yaw proportional control
    mean = 0
    variance = 1.2
    d = (mav.drone_pose.pose.position.x - goal[0])**2 + (mav.drone_pose.pose.position.y - goal[1])**2
    d = np.sqrt(d)
    
        
    while not rospy.is_shutdown() and d > 0.3:
        d = (mav.drone_pose.pose.position.x - goal[0])**2 + (mav.drone_pose.pose.position.y - goal[1])**2
        d = np.sqrt(d)
        
        euler_orientation = euler_from_quaternion(ros_numpy.numpify(mav.drone_pose.pose.orientation))
        ########################theta_goal global###################################
        deltaY = goal[1] - mav.drone_pose.pose.position.y
        deltaX = goal[0] - mav.drone_pose.pose.position.x
        if deltaY > 0 and deltaX >= 0:
            if deltaX == 0:
                theta_goal = 1.57079632679
            else:
                theta_goal = np.arctan(deltaY/deltaX)
        if deltaY >= 0 and deltaX < 0:
            if deltaY == 0:
                theta_goal = 3.14
            else:
                theta_goal = np.arctan(abs(deltaX/deltaY)) + 1.57079632679 #90
        if deltaY < 0 and deltaX <= 0:
            if deltaX == 0:
                theta_goal = -1.57079632679
            else:
                theta_goal = -1*np.arctan(abs(deltaX/deltaY)) - 1.57079632679 #180
        if deltaY <= 0 and deltaX > 0:
            if deltaY == 0:
                theta_goal = 0
            else:
                theta_goal = -1*np.arctan(abs(deltaY/deltaX))
        ##################################################################################
        
        theta = laser_data.angle_min
        Ft = np.array([0.0, 0.0])
        Fg = np.array([Ka*d*np.cos(theta_goal),
                                    Ka*d*np.sin(theta_goal)])
        #rospy.loginfo(laser_data)
        for laser_range in laser_data.ranges:
            theta += laser_data.angle_increment
            if laser_range != "inf" and laser_range < mav.drone_pose.pose.position.z: # not sure if that's how we verify
                Fi = Kr * ((a/((laser_range**b)*c)) + d*(laser_range-1.5) - 0.2)
                Fix = -Fi*np.cos(theta + mav.drone_pose.pose.orientation.z)
                Fiy = -Fi*np.sin(theta + mav.drone_pose.pose.orientation.z)
                Ft += np.array([Fix, Fiy])

        Fg = saturate(Fg)
        F = Ft + Fg
        rospy.loginfo("Attraction = {}; Repulsion = {}".format(Fg, Ft))
        mav.set_position_target(type_mask=MASK_VELOCITY,
                                x_velocity=F[0],
                                y_velocity=F[1],
                                z_velocity=Kz*(initial_height - mav.drone_pose.pose.position.z),
                                yaw_rate= 3)
                                
    mav.land()

run()