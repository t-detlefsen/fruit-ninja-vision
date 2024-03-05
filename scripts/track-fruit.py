#!/usr/bin/env python

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/thomasdetlefsen/FruitNinja/src/fruit-ninja-vision/model.pth'

class trackFruit:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        image_sub = message_filters.Subscriber('rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('depth/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=5, slop=0.2)
        ts.registerCallback(self.callback)
        
        self.cv_bridge = CvBridge()

        self.model = YOLO(MODEL_PATH) 

    def callback(self, image_msg, depth_msg):
        img = self.cv_bridge.imgmsg_to_cv2(image_msg)
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        # Run inference on image
        result = self.model(img)[0]
        box = result.boxes[0].xyxy.squeeze()

        # TO-DO: Get location of ball with respect to camera

if __name__ == "__main__":
    track_fruit = trackFruit()
    rospy.spin()