#!/usr/bin/env python

import rospy
import numpy as np

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    history_dist = 0
    history_para = 0
    ierr_para = 0
    turn_angle = 0.33/np.tan(0.34) #0.33/np.tan(0.34)
    meters_from_feet = 0.3048 

    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)

        self.parking_distance = .75 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0
        self.buffer = 0.0

    def angle_correct(self, error):
        p_gain = 0.4#.04
        d_gain = 0.3#.03
        derr_para = error-self.history_para
        self.history_para = error
        P = p_gain*error
        D = -d_gain*derr_para
        steering_angle = P+D
        if steering_angle > .34:
            steering_angle = .34
        elif steering_angle <-.34:
            steering_angle = -.34
    
        return steering_angle

    def relative_cone_callback(self, msg):
        # print("GOT HERE")
        # print("msg.x_pos :", msg.x_pos)
        # print("msg.y_pos :", msg.y_pos)

        self.relative_x = msg.x_pos 
        self.relative_y = msg.y_pos 
        drive_cmd = AckermannDriveStamped()

        #################################

        # YOUR CODE HERE
        # Use relative position and your control law to set drive_cmd

        self.angle = np.arctan([self.relative_y, self.relative_x])[0]
        self.distance = np.sqrt(self.relative_x**2+self.relative_y**2)

        if self.distance <(1.5 + self.buffer)*self.meters_from_feet: #drive backwards if too close 
            drive_cmd.drive.steering_angle = -self.angle_correct(self.angle)
            drive_cmd.drive.speed = -1
            self.buffer = 0.5
        else:
            self.buffer = 0.0
            if (abs(self.angle) > self.turn_angle) and (self.distance <3): #turn if too far past cone
                drive_cmd.drive.steering_angle = -0.34 * np.sign(self.angle)
                drive_cmd.drive.speed = -1
            elif abs(self.angle) > 0.1: #adjust angle 
                drive_cmd.drive.steering_angle =  self.angle_correct(self.angle)
                drive_cmd.drive.speed = 1
            elif self.distance > (2.0)*self.meters_from_feet: #move forward part
                drive_cmd.drive.steering_angle = 0 
                drive_cmd.drive.speed = 1
            else:
                drive_cmd.drive.speed = 0

        #################################

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        #################################

        # YOUR CODE HERE
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x**2+self.relative_y**2)

        #################################
        
        self.error_pub.publish(error_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
