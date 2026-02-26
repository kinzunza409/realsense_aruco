#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv

def main():
    rospy.init_node('realsense_viewer')
    rospy.loginfo("RealSense Viewer Node Started")
    cvbridge = CvBridge()

    # aruco detection initializations
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    aruco_params = cv.aruco.DetectorParameters_create()

    def image_callback(msg):
        frame = cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # aruco detection
        # TODO: try this with the OpenCV 4.8 params
        corners, ids, rejected = cv.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if ids is not None:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
        else:
            cv.putText(frame, 'NO MARKERS DETECTED!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv.imshow("RealSense Feed", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            rospy.signal_shutdown("User quit")

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()