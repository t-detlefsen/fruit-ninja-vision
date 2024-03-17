#!/usr/bin/env python

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/thomasdetlefsen/FruitNinja/src/fruit-ninja-vision/best.pt'
VERBOSE = True
VISUALIZE = False
SUB_BOX_SCALE = 2

class trackFruit:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        image_sub = message_filters.Subscriber('rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=5, slop=0.2)
        ts.registerCallback(self.callback)
        
        self.cv_bridge = CvBridge()

        if VERBOSE: rospy.loginfo("Loading model...")
        self.model = YOLO(MODEL_PATH)

        if VERBOSE: rospy.loginfo("Waiting for camera_info...")
        camera_info = rospy.wait_for_message('rgb/camera_info', CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape((3, 3))

        if VERBOSE: rospy.loginfo("Done!")

    def callback(self, image_msg, depth_msg):
        img = self.cv_bridge.imgmsg_to_cv2(image_msg)[:, :, :3]
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        rospy.loginfo(img.shape)
        rospy.loginfo(depth.shape)

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

            # # TEMPORARY
            # size = 20
            # box = [1024 - size / 2, 768 - size/ 2, 1024 + size / 2, 768 + size/ 2,]

            # Get offset value for scaled box
            lx = box[2] - box[0]
            ly = box[3] - box[1]
            x_offset = (lx - (lx / SUB_BOX_SCALE) ) / 2
            y_offset = (ly - (ly / SUB_BOX_SCALE) ) / 2

            # Get coordinates to sample depth image
            x1 = int(box[0] + x_offset)
            y1 = int(box[1] + y_offset)
            x2 = int(box[2] - x_offset)
            y2 = int(box[3] - y_offset)

            depth = np.median(depth[y1:y2, x1:x2])

            if VERBOSE: rospy.loginfo("Average Depth = {}".format(depth))

            # Transform to camera coordinates
            px_x = box[0] + lx / 2
            px_y = box[1] + ly / 2

            x = (px_x - self.intrinsic[0, 2]) / self.intrinsic[0, 0] * depth
            y = (px_y - self.intrinsic[1, 2]) / self.intrinsic[1, 1] * depth
            z = depth

            if VERBOSE: rospy.loginfo("Ball detected at = {}, {}, {}".format(x, y, z))

if __name__ == "__main__":
    track_fruit = trackFruit()
    rospy.spin()