#!/usr/bin/env python
import rospy
import numpy as np
import time

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import Joy
from visual_servoing.msg import ConeLocation

class PurePursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.odom_topic       = rospy.get_param("~odom_topic","/pf/pose/odom")
        self.speed            = 2 #filled in for testing purposes, please update
        self.wheelbase_length = 0.32 #flilled in for testing purposes, please update

        self.drive_pub = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.drive_pub2 = rospy.Publisher("/drive", AckermannDriveStamped, queue_size=1)

        self.cterr_pub = rospy.Publisher("/crosstrackerror", Float64,queue_size=1)

        self.bufferr_pub = rospy.Publisher("/buffererror",Float64,queue_size=1)

        self.cone_sub = rospy.Subscriber("/relative_cone", ConeLocation, self.cone_callback, queue_size = 20)

        self.buffer_count = 0
        
        self.buffer_y = 0
        
        self.imgBufferLim = 5

        self.avg_pt = np.array([0,0])

        self.brake = False # Boolean condition to determine whether to stop
        # self.thresh = self.speed # distance from final path point at which car will stop
        self.teleop_subscriber = rospy.Subscriber("/vesc/joy", Joy, self.tcb, queue_size=1)        
        self.pressed = False	

    def tcb(self, msg):
        buttons = msg.buttons
        if(buttons[5] == 1):
            self.pressed = True
        else:
            self.pressed = False

    def cone_callback(self, msg):
	#x = msg.x_pos
	#y = msg.y_pos
        #d = np.sqrt(x*x + y*y)
	#radius = 3
	#scale = radius / d
        x = msg.x_pos
        y = msg.y_pos 
        #rospy.logerr("x: {} y: {}".format(x, y))
        def cte(x,y): # Currently publishes relative cross track error
            return y
        bfe = lambda y: abs(self.buffer_y - y)

        cterr = Float64()
        bferr = Float64()
        # Averaging Buffer, TODO listener for state of driving around turn vs not.

        if self.buffer_count == self.imgBufferLim:
            self.avg_pt /= self.imgBufferLim
            self.drive_command(self.avg_pt[0], self.avg_pt[1])
            self.buffer_y = self.avg_pt[1]
            self.buffer_count = 0
            cterr.data = cte(self.avg_pt[0],self.avg_pt[1])
            bferr.data = bfe(self.avg_pt[1])
            self.cterr_pub.publish(cterr)
            self.bufferr_pub.publish(bferr)
            self.avg_pt = np.array([0,0])
            return
        self.avg_pt += np.array([x,y])

        cterr.data = cte(x,y)
        bferr.data = bfe(y)
        self.cterr_pub.publish(cterr)
        self.bufferr_pub.publish(bferr)

        

    def drive_command(self, goalx, goaly):
        #rospy.logerr("x: {}, y: {}".format(self.x, self.y))
        #rospy.logerr("goalx: {}, goaly: {}".format(goalx, goaly))
        #ospy.logerr("a")
        
        print("target point(m):",  (goalx, goaly))
        #!!! (x,y) as output by homography transform are reversed to those expected by pure pursuit
        eta = np.arctan2(goaly, goalx)
        # R = self.lookahead / (2 * np.sin(eta))
        AckermannDrive = AckermannDriveStamped()
        AckermannDrive.header.stamp = rospy.Time.now()
        AckermannDrive.header.frame_id = "base_link"
        if (self.brake):
            rospy.logerr("STOPPING)")
            AckermannDrive.drive.speed = 0
            AckermannDrive.drive.steering_angle = 0
        else:
            AckermannDrive.drive.speed = self.speed
            angle = np.arctan2(2 * self.wheelbase_length * np.sin(eta), np.sqrt(goalx**2 + goaly**2))
            if abs(angle) >= 0.34:
                angle = 0.34 * np.sign(angle)
            AckermannDrive.drive.steering_angle = angle
            rospy.logerr(angle)

        #generalized sttering law by having a point ahead lecture slides
        # lfw = 0.05 #randomly defined based on lecture slides
        # AckermannDrive.drive.steering_angle = -1 * np.arctan(self.wheelbase_length * np.sin(eta) / (self.lookahead / 2  + lfw/np.cos(eta)))
        if (self.pressed):
            self.drive_pub2.publish(AckermannDrive)
        self.drive_pub.publish(AckermannDrive)

if __name__=="__main__":
    rospy.init_node("pure_pursuit")
    pf = PurePursuit()
    rospy.spin()
