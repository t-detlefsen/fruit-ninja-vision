#!/usr/bin/env python

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/thomasdetlefsen/FruitNinja/src/fruit-ninja-vision/best.pt'
VERBOSE = True
VISUALIZE = False

class trackFruit:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        image_sub = message_filters.Subscriber('rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('depth/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=5, slop=0.2)
        ts.registerCallback(self.callback)
        
        self.cv_bridge = CvBridge()

        if VERBOSE: rospy.loginfo("Loading model...")
        self.model = YOLO(MODEL_PATH)
        if VERBOSE: rospy.loginfo("Done!")

    def callback(self, image_msg, depth_msg):
        img = self.cv_bridge.imgmsg_to_cv2(image_msg)[:, :, :3]
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        # Run inference on image
        if VERBOSE: rospy.loginfo("Running Inference...")
        result = self.model.predict(source=img, conf=0.1)[0]
        if VERBOSE: rospy.loginfo("Done!")

        out_img = img.copy()
        if len(result.boxes) > 0:
            box = result.boxes[0].xyxy.squeeze()
            out_img = cv2.rectangle(out_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))

        if VISUALIZE:
            cv2.imshow("", cv2.resize(out_img, (0, 0), fx = 0.5, fy = 0.5))
            cv2.waitKey(1)

        # TO-DO: Get location of ball with respect to camera
        # TO-DO: depth/image_raw is 512x512 check other depth topics to see if there's one that matches camera

if __name__ == "__main__":
    track_fruit = trackFruit()
    rospy.spin()