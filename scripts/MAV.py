#!/usr/bin/env python3

import rospy
import smach
from progress.bar import ChargingBar
import smach_ros
import mavros_msgs
from mavros_msgs import srv
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState, PositionTarget
from sensor_msgs.msg import BatteryState
from mavros_msgs.msg import Mavlink


TOL = 0.2
MAX_TIME_DISARM = 15
CONFIG = {"mavros_local_position_pub" : "/mavros/setpoint_position/local",
                "mavros_velocity_pub" : "/mavros/setpoint_velocity/cmd_vel",
                "mavros_target_pub" :   "/mavros/setpoint_raw/local",
                "mavros_local_atual" :  "/mavros/local_position/pose",
                "mavros_state_sub" :    "/mavros/state",
                "mavros_arm" :          "/mavros/cmd/arming",
                "mavros_set_mode" :     "/mavros/set_mode",
                "mavros_battery_sub" :  "/mavros/battery"}
class MAV:
    
               

    def __init__(self, mav_name, mav_type="mavros"):
        self.rate = rospy.Rate(20.0)
        self.desired_state = ""

        self.drone_pose = PoseStamped()
        self.goal_pose = PoseStamped()
        self.goal_vel = TwistStamped()
        self.pose_target = PositionTarget()
        self.drone_state = State()
        self.battery = BatteryState()
        # ############## Publishers ##############
        self.local_position_pub = rospy.Publisher(CONFIG[mav_type + "_local_position_pub"], PoseStamped, queue_size = 20)
        self.velocity_pub = rospy.Publisher(CONFIG[mav_type + "_velocity_pub"],  TwistStamped, queue_size=5)
        self.target_pub = rospy.Publisher(CONFIG[mav_type + "_target_pub"], PositionTarget, queue_size=5)

        ########## Subscribers ##################
        self.local_atual = rospy.Subscriber(CONFIG[mav_type + "_local_atual"], PoseStamped, self.local_callback)
        self.state_sub = rospy.Subscriber(CONFIG[mav_type + "_set_mode"], State, self.state_callback)
        self.battery_sub = rospy.Subscriber(CONFIG[mav_type + "_battery_sub"], BatteryState, self.battery_callback)
        self.extended_state_sub = rospy.Subscriber("/mavros/extended_status", ExtendedState, self.extended_state_callback, queue_size=2)
        ############# Services ##################
        self.arm = rospy.ServiceProxy(CONFIG[mav_type + "_arm"], CommandBool)
        self.set_mode_srv = rospy.ServiceProxy(CONFIG[mav_type + "_set_mode"], SetMode)

        self.LAND_STATE = ExtendedState.LANDED_STATE_UNDEFINED # landing state
        '''
        0: Undefined
        1: On ground
        2: In air
        '''



    ###### Callback Functions ##########
    def state_callback(self, state_data):
        self.drone_state = state_data
        if self.drone_state.mode != self.desired_state:
            #rospy.loginfo("Setting {} flight mode".format(self.desired_state))
            self.set_mode_srv(0, self.desired_state)

    def battery_callback(self, bat_data):
        self.battery = bat_data

    def local_callback(self, local):
        self.drone_pose.pose.position.x = local.pose.position.x
        self.drone_pose.pose.position.y = local.pose.position.y
        self.drone_pose.pose.position.z = local.pose.position.z
        
        self.drone_pose.pose.orientation.x = local.pose.orientation.x
        self.drone_pose.pose.orientation.y = local.pose.orientation.y
        self.drone_pose.pose.orientation.z = local.pose.orientation.z
        self.drone_pose.pose.orientation.w = local.pose.orientation.w

    def extended_state_callback(self, es_data):
        self.LAND_STATE = es_data.landed_state

    ####### Set Position and Velocity ################
    def set_position(self, x, y, z):
        self.goal_pose.pose.position.x = x
        self.goal_pose.pose.position.y = y
        self.goal_pose.pose.position.z = z
        self.local_position_pub.publish(self.goal_pose)
        self.rate.sleep()

    def set_vel(self, x, y, z, roll=0, pitch=0, yaw=0):
        self.goal_vel.twist.linear.x = x
        self.goal_vel.twist.linear.y = y
        self.goal_vel.twist.linear.z = z

        self.goal_vel.twist.angular.x = roll
        self.goal_vel.twist.angular.y = pitch
        self.goal_vel.twist.angular.z = yaw
        self.stamp = rospy.Time.now()
        self.velocity_pub.publish(self.goal_vel)
    
    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        self.desired_state = mode
        old_mode = self.drone_state.mode
        loop_freq = 1  # Hz
        loop_rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in range(timeout * loop_freq):
            if self.drone_state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    result = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not result.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                loop_rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr(e)


    def set_position_target(self, type_mask, x_position=0, y_position=0, z_position=0, x_velocity=0, y_velocity=0, z_velocity=0, x_aceleration=0, y_aceleration=0, z_aceleration=0, yaw=0, yaw_rate=0):
        self.pose_target.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        self.pose_target.type_mask = type_mask
        #https://mavlink.io/en/messages/common.html#POSITION_TARGET_TYPEMASK
        #4095 ignores every dimension (subtract the usefull ones from it)

        self.pose_target.position.x = x_position
        self.pose_target.position.y = y_position
        self.pose_target.position.z = z_position

        self.pose_target.velocity.x = x_velocity
        self.pose_target.velocity.y = y_velocity
        self.pose_target.velocity.z = z_velocity

        self.pose_target.acceleration_or_force.x = x_aceleration
        self.pose_target.acceleration_or_force.y = y_aceleration
        self.pose_target.acceleration_or_force.z = z_aceleration

        self.pose_target.yaw = yaw
        self.pose_target.yaw_rate = yaw_rate

        self.target_pub.publish(self.pose_target)

    def chegou(self):
        if (abs(self.goal_pose.pose.position.x - self.drone_pose.pose.position.x) < TOL) and (abs(self.goal_pose.pose.position.y - self.drone_pose.pose.position.y) < TOL) and (abs(self.goal_pose.pose.position.z - self.drone_pose.pose.position.z) < TOL):
            return True
        else:
            return False

    def takeoff(self, height):
        velocity = 1.0
        part = velocity/60.0
        inicial_height = (0.8 * 60 * part) / velocity
        for i in range(100):
            self.set_position(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, 0)
            self.rate.sleep()
        self.set_mode("OFFBOARD", 1)

        rospy.logwarn("ARMING DRONE")
        fb = self.arm(True)
        while not fb.success:
            rospy.logwarn("ARMING DRONE {}".format(fb))
            fb = self.arm(True)
            self.rate.sleep()
        rospy.loginfo("DRONE ARMED")
        self.rate.sleep()

        t=0
        t += inicial_height
        while abs(self.drone_pose.pose.position.z - height) >= TOL and not rospy.is_shutdown():
            if t < height:
                rospy.logwarn('TAKING OFF AT ' + str(velocity) + ' m/s')
                self.set_position(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, t)
                t += part
            else:
                self.set_position(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, height)

            rospy.loginfo('Position: (' + str(self.drone_pose.pose.position.x) + ', ' + str(self.drone_pose.pose.position.y) + ', '+ str(self.drone_pose.pose.position.z) + ')')
            self.rate.sleep()

        self.set_position(self.drone_pose.pose.position.x, self.drone_pose.pose.position.y, height)

        return "done"

    def RTL(self):
        velocity = 0.3
        ds = velocity/60.0

        self.rate.sleep()
        height = self.drone_pose.pose.position.z
        rospy.loginfo('Position: (' + str(self.drone_pose.pose.position.x) + ', ' + str(self.drone_pose.pose.position.y) + ', ' + str(self.drone_pose.pose.position.z) + ')')

        self.set_position(0,0,height)
        self.rate.sleep()
        rospy.loginfo('Position: (' + str(self.drone_pose.pose.position.x) + ', ' + str(self.drone_pose.pose.position.y) + ', ' + str(self.drone_pose.pose.position.z) + ')')
        rospy.loginfo('Goal Position: (' + str(self.goal_pose.pose.position.x) + ', ' + str(self.goal_pose.pose.position.y) + ', ' + str(self.goal_pose.pose.position.z) + ')')


        while not self.chegou():
            rospy.loginfo('Executing State RTL')
            rospy.loginfo("STARING HOME")
            self.set_position(0,0,height)
            self.rate.sleep()

        t=0
        self.set_position(0,0,height-ds)
        self.rate.sleep()

        init_time = rospy.get_rostime().secs
        #while not (self.drone_pose.pose.position.z < -0.1) and rospy.get_rostime().secs - init_time < (height/velocity)*1.3: #30% tolerance in time
        while not self.LAND_STATE == ExtendedState.LANDED_STATE_ON_GROUND:
            rospy.loginfo('Executing State RTL')

            rospy.loginfo('Height: ' + str(abs(self.drone_pose.pose.position.z)))
            ################# Velocity Control #################
            self.set_vel(0, 0, -velocity, 0, 0, 0)
            rospy.loginfo('LANDING AT ' + str(velocity) + 'm/s')
            self.rate.sleep()
        rospy.logwarn("LANDED_STATE: ON GROUND\nDISARMING")
        self.arm(False)
        return "succeeded"

    def hold(self, time):
        now = rospy.Time.now()
        bar = ChargingBar("Holding Position", max=time*60)
        while not rospy.Time.now() - now > rospy.Duration(secs=time):
            bar.next()
            self.local_position_pub.publish(self.drone_pose)
            self.rate.sleep()
        bar.finish()
    

    def land(self):
        velocity = 0.3
        height = self.drone_pose.pose.position.z
        init_time = rospy.get_rostime().secs
        #while not (self.drone_pose.pose.position.z < 0.4 and not rospy.is_shutdown()) or  self.LAND_STATE == ExtendedState.LANDED_STATE_ON_GROUND:
        while not self.LAND_STATE == ExtendedState.LANDED_STATE_ON_GROUND:
            rospy.logwarn('Landing')
            rospy.loginfo('Height: ' + str(abs(self.drone_pose.pose.position.z)))
            ################# Velocity Control #################
            self.set_vel(0, 0, -velocity, 0, 0, 0)
            self.rate.sleep()
        rospy.logwarn("LANDED_STATE: ON GROUND\nDISARMING")
        self.arm(False)
        return "succeeded"

    def _disarm(self):
        rospy.logwarn("DISARM MAV")
        if self.drone_pose.pose.position.z < TOL:
            for i in range(3):
                rospy.loginfo('Drone height' + str(self.drone_pose.pose.position.z))
                self.arm(False)
        else:
            rospy.logwarn("Altitude too high for disarming!")
            self.land()
            self.arm(False)

if __name__ == '__main__':
    mav = MAV(1) #MAV name
    mav.takeoff(3)
    mav.RTL()
