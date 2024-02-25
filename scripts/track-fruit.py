#!/usr/bin/env python

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import numpy as np

class trackFruit:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        image_sub = message_filters.Subscriber('rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('depth/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=5, slop=0.2)
        ts.registerCallback(self.callback)
        
        self.cv_bridge = CvBridge()

    def callback(self, image_msg, depth_msg):
        img = self.cv_bridge.imgmsg_to_cv2(image_msg)
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        # PERCEPTION LOGIC GOES HERE

if __name__ == "__main__":
    track_fruit = trackFruit()
    rospy.spin()