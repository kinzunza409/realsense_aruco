#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv

def main():
    rospy.init_node('realsense_viewer')
    rospy.loginfo("RealSense Viewer Node Started")
    
    cvbridge = CvBridge()

    def image_callback(msg):
        frame = cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv.imshow("RealSense Feed", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            rospy.signal_shutdown("User quit")

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()