#!/usr/bin/env python3
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2 as cv

def main():
    rospy.init_node('realsense_viewer_aruco')
    rospy.loginfo("RealSense Aruco Viewer Node Started")
    cvbridge = CvBridge()

    # get camera intrinsics
    camera_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
    camera_k = np.array(camera_info.K).reshape(3, 3)  # 3x3 intrinsic matrix as flat list of 9 values
    camera_d = np.array(camera_info.D)  # distortion coefficients

    # aruco detection initializations
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    aruco_params = cv.aruco.DetectorParameters_create()
    marker_length = 0.04 # m

    def image_callback(msg):
        frame = cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # aruco detection
        # TODO: try this with the OpenCV 4.8 params to see if it performs better
        corners, ids, rejected = cv.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        rvecs, tvecs, _objPoints = cv.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_k, camera_d)

        if ids is not None:
            #cv.aruco.drawDetectedMarkers(frame, corners, ids)
            for r,t in zip(rvecs, tvecs):
                frame = cv.aruco.drawAxis(frame, camera_k, camera_d, r, t, marker_length)
                x, y, z = t[0]
                dist = np.sqrt((x*1000)**2 + (y*1000)**2 + (z*1000)**2)
                cv.putText(frame, f"({x:.3f}, {y:.3f}, {z:.3f})", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.putText(frame, f"(DIST: {dist:.2f}mm)", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
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