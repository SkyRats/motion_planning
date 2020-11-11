import rospy
import numpy as np
from numpy.linalg import norm
from MAV import MAV
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

laser_data = LaserScan()
map_data = OccupancyGrid()
pose_data = PoseStamped()

markers = 0

LINEARITY_TRESHOLD = 0.15

def map_callback(data):
    global map_data
    map_data = data

def pose_callback(data):
    global pose_data
    pose_data = data

def laser_callback(data):
    global laser_data
    laser_data = data

def laser_polar_to_cartesian():
    angles = [i for i in range(0, laser_data.angle_max+laser_data.angle_increment, laser_data.angle_increment)]
    x = np.cos(angles) * np.array(laser_data.ranges)
    y = np.sin(angles) * np.array(laser_data.ranges)
    return x, y

def cartesian_to_marker(coord):
    x, y = coord
    width = map_data.info.width
    res = map_data.info.resolution
    
    origin_x = map_data.info.origin.position.x
    origin_y = map_data.info.origin.position.y
    
    row = (x - origin_x)/res
    column = (y - origin_y)/res

    return row, column

def fill_marker_area(tr, bl):
    global markers
    tr = cartesian_to_marker(tr)
    bl = cartesian_to_marker(bl)

    for i in range(bl[0], tr[0]):
        for j in range(bl[1], tr[1]):
            markers[i][j] = -1

def free(x, y):
    global markers
    mx, my = cartesian_to_marker([x, y])
    return False if markers[mx][my] == -1 else True

def find_edges():

    edges = []

    previously_aligned = True

    x, y = laser_polar_to_cartesian()
    
    top_y = max(y)
    
    for i in range(0, len(x)-3, 3):

        v1 = [ x[i+1] - x[i], y[i+1] - y[i] ]
        v2 = [ x[i+2] - x[i+1], y[i+2] - y[i+1] ]
        angle = np.cross(v1, v2)/(norm(v1)*norm(v2))

        if abs(angle) > LINEARITY_TRESHOLD:
            if previously_aligned:
                edges.extend( [x[i+1], y[i+1]], [x[i+2], y[i+2]] )
                top_y = min( [top_y, y[i+1], y[i+2]] )
            elif free(x[i+2], y[i+2]):
                top_y = min( [top_y, y[i+2]] )
            previously_aligned = False
        elif not previously_aligned:
            previously_aligned = True
    
    rec_x = [ x[i] for i in range(len(x)) if y[i] == top_y ]
    rec_x.sort(reverse=True)
    bottom_y = max( [y[i] for i in range(len(y)) if x[i] in rec_x] )
        
    
    return rec_x, [top_y, bottom_y]

def execute_trajectory(mav):

    drone_x = pose_data.pose.position.x
    drone_y = pose_data.pose.position.y

    POSITION_TYPEMASK = 0b0000011111111011
    SAFE_DISTANCE = 0.3 # Se aplica aos dois lados
    vertical = True

    t_x, b_x, t_y, b_y = find_edges()
    
    if abs(t_x - b_x) > abs(t_y - b_y): # Retangulo horizontal
        if abs(drone_x - t_x) > abs(drone_x - b_x): 
            goal_x = t_x 
            start_x = b_x
        else: 
            goal_x = b_x
            start_x = t_x

        start_y = goal_y = (t_y + b_y)/2

        vertical = False

    else: # Retangulo vertical
        start_x = goal_x = (t_x + b_x)/2

        if abs(drone_y - t_y) > abs(drone_y - b_y): 
            goal_y = t_y
            start_y = b_y
        else: 
            goal_y = b_y
            start_y = t_y
    
    mav.set_position_target(
        type_mask=POSITION_TYPEMASK,
        x_position=start_x, y_position=start_y
    ) # Vai para ponto de inicio
    
    A = abs(t_x - b_x) - SAFE_DISTANCE if vertical else abs(t_y - b_y) - SAFE_DISTANCE

    t0 = rospy.get_time()
    while not rospy.is_shutdown and abs(drone_x - goal_x) > GOAL_DIST and abs(drone_y - goal_y) > GOAL_DIST :
        t = rospy.get_time - t0
        x_vel = -A*np.cos(t) if vertical else 1
        y_vel = 1 if vertical else -A*np.cos(t)

        mav.set_vel(x_vel, y_vel, 0)
    
    fill_marker_area([t_x, t_y], [b_x, b_y])
    
    mav.set_vel(0,0,0)

def run():
    global markers
    markers = np.zeros([map_data.info.width, map_data.info.height], dtype=np.int8)
    
    HEIGHT = 1

    rospy.init_node('exploration')
    mav = MAV('1')
    mav.takeoff(HEIGHT)
    map_sub = rospy.Subscriber("/map", OccupancyGrid, map_callback, queue_size=1)
    pose_sub = rospy.Subscriber("/slam_out_pose", PoseStamped, pose_callback, queue_size=1)
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan, laser_callback)
    rospy.wait_for_message("/map", OccupancyGrid)

    while not rospy.is_shutdown:
        execute_trajectory(mav)

if __name__ == '__main__':
    run()
