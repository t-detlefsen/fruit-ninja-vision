#!/usr/bin/env python

import rospy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/student/16662_RobotAutonomy/src/devel_packages/fruit-ninja-vision/best_new.pt'
VERBOSE = False
VISUALIZE = True
SUB_BOX_SCALE = 2

class trackFruit:
    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        
        # Load model
        if VERBOSE: rospy.loginfo("Loading model...")
        self.model = YOLO(MODEL_PATH)

        # Get camera info for 3D conversion
        if VERBOSE: rospy.loginfo("Waiting for camera_info...")
        camera_info = rospy.wait_for_message('rgb/camera_info', CameraInfo)
        self.intrinsic = np.array(camera_info.K).reshape((3, 3))

        # Initialize publishers and subscribers
        image_sub = message_filters.Subscriber('rgb/image_raw', Image)
        depth_sub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
        if VISUALIZE:
            self.yolo_pub = rospy.Publisher('yolo_detections',Image,queue_size=10)
            self.ball_pub = rospy.Publisher("ball_marker", Marker, queue_size=10)
            
            # Define ball marker
            self.marker = Marker()
            self.marker.header.frame_id = "camera_base"
            self.marker.header.stamp = rospy.Time.now()
            self.marker.type = 2
            self.marker.id = 0
            self.marker.scale.x = 0.075
            self.marker.scale.y = 0.075
            self.marker.scale.z = 0.075
            self.marker.color.r = 1.0
            self.marker.color.g = 0.65
            self.marker.color.b = 0.0
            self.marker.color.a = 1.0
            self.marker.pose.orientation.w = 1

        # Create time synchronizer for image and depth
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=5, slop=0.2)
        ts.registerCallback(self.callback)

        self.cv_bridge = CvBridge()

        if VERBOSE: rospy.loginfo("Done!")

    def callback(self, image_msg, depth_msg):
        # Convert images to arrays
        img = self.cv_bridge.imgmsg_to_cv2(image_msg)[:, :, :3]
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg)

        _, r, _ = img.shape

        # Run inference
        result = self.model.predict(source=img, conf=0.1, verbose=VERBOSE)[0]

        # Determine bounding box
        out_img = cv2.rotate(img.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
        if len(result.boxes) > 0:
            box = result.boxes[0].xyxy.squeeze()
            out_img = cv2.rectangle(out_img, (int(box[3]), int(r - box[2])), (int(box[1]), int(r - box[0])), (255, 0, 0))

            # Crop to center of box
            lx = box[3] - box[1]
            ly = box[2] - box[0]
            x_offset = (lx - (lx / SUB_BOX_SCALE) ) / 2
            y_offset = (ly - (ly / SUB_BOX_SCALE) ) / 2

            y1 = int(box[1] + x_offset)
            x1 = int(box[0] + y_offset)
            y2 = int(box[3] - x_offset)
            x2 = int(box[2] - y_offset)

            # Get median of depth
            depth_sample = depth[y1:y2, x1:x2]
            med_depth = np.median(depth_sample[depth_sample > 0])

            if np.isnan(med_depth): return

            # Determine center of box in pixel coords
            px_x = box[0] + lx / 2
            px_y = box[1] + ly / 2

            # Determine ball location in 3D
            x = med_depth
            y = -(px_x - self.intrinsic[0, 2]) / self.intrinsic[0, 0] * med_depth
            z = -(px_y - self.intrinsic[1, 2]) / self.intrinsic[1, 1] * med_depth

            if VISUALIZE:
                self.marker.pose.position.x = x / 1000
                self.marker.pose.position.y = y / 1000
                self.marker.pose.position.z = z / 1000
                self.ball_pub.publish(self.marker)
            
        if VISUALIZE:
            self.yolo_pub.publish(self.cv_bridge.cv2_to_imgmsg(out_img))

if __name__ == "__main__":
    track_fruit = trackFruit()
    rospy.spin()