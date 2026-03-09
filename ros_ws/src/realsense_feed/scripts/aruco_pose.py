#!/usr/bin/env python3
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
import tf2_msgs.msg
import tf.transformations
import geometry_msgs.msg
import visualization_msgs.msg
import std_msgs.msg

import cv2 as cv

IDENTITY_TRANSFORM = geometry_msgs.msg.TransformStamped()
IDENTITY_TRANSFORM.transform.rotation.w = 1.0

RED = std_msgs.msg.ColorRGBA(r=1, g=0, b=0, a=1)

class ArucoTag:
    _aruco_lengths = { # in meters
        "default": 0.04,
        1 : 0.1,
        2 : 0.1
    }

    def __init__(self, id, corners, pose = None, frame_id = "camera_color_optical_frame"):
        self.id = id
        self.corners = corners
        self.pose = pose if pose is not None else geometry_msgs.msg.Pose()
        self.frame_id = frame_id
        self.length = self._aruco_lengths.get(id, self._aruco_lengths["default"])

    def pose_from_opencv(self, rvec, tvec):

        self.rvec = rvec
        self.tvec = tvec

        t = tvec.flatten()
        r = rvec.flatten()
        angle = np.linalg.norm(r)
        
        if angle == 0:
            q = tf.transformations.quaternion_about_axis(0, [1, 0, 0])  # identity rotation
        else:
            axis = r / angle
            q = tf.transformations.quaternion_about_axis(angle, axis)
        
        self.pose.position.x = t[0]
        self.pose.position.y = t[1]
        self.pose.position.z = t[2]
        self.pose.orientation.x = q[0]
        self.pose.orientation.y = q[1]
        self.pose.orientation.z = q[2]
        self.pose.orientation.w = q[3]

        return self.pose
    
    def get_marker(self) -> visualization_msgs.msg.Marker:
        marker = visualization_msgs.msg.Marker()

        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id

        marker.pose = self.pose
        marker.id = self.id

        # need this so that it displays in rviz
        marker.color = RED
        marker.type = visualization_msgs.msg.Marker.CUBE
        marker.scale.x = self.length
        marker.scale.y = self.length
        marker.scale.z = self.length

        return marker


def main():

    rospy.init_node('realsense_aruco_pose')
    rospy.loginfo("RealSense Aruco Pose Node Started")
    marker_pub = rospy.Publisher(
        '/aruco/marker_poses',
        visualization_msgs.msg.MarkerArray,
        queue_size=10)
    cvbridge = CvBridge()
    origin_id = rospy.get_param('~origin_id', default=1)

    # get camera intrinsics
    camera_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
    camera_k = np.array(camera_info.K).reshape(3, 3)  # 3x3 intrinsic matrix as flat list of 9 values
    camera_d = np.array(camera_info.D)  # distortion coefficients

    # aruco detection initializations
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
    aruco_params = cv.aruco.DetectorParameters_create()

    # improved defaults from newer OpenCV
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.adaptiveThreshConstant = 7

    aruco_params.minMarkerPerimeterRate = 0.03
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.03
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minDistanceToBorder = 3
    aruco_params.minMarkerDistanceRate = 0.05

    aruco_params.cornerRefinementMethod = 1  # CORNER_REFINE_SUBPIX — not default in 4.2 but is in newer
    aruco_params.cornerRefinementWinSize = 5
    aruco_params.cornerRefinementMaxIterations = 30
    aruco_params.cornerRefinementMinAccuracy = 0.1

    aruco_params.markerBorderBits = 1
    aruco_params.perspectiveRemovePixelPerCell = 4
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    aruco_params.maxErroneousBitsInBorderRate = 0.35
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.errorCorrectionRate = 0.6
    
    
    marker_length = 0.04 # side length of aruco marker in meters
    origin_marker_length = 0.10
    checkersquare_length = 0.0667 # meters

    def image_callback(msg):
        
        frame = cvbridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # TODO: try this with the OpenCV 4.8 params to see if it performs better
        start = rospy.Time.now()

        # preproccesing
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # find Arucos positions wrt image frame
        corners, ids, rejected = cv.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        #corners, ids, rejected = cv.aruco.refineDetectedMarkers(gray, aruco_dict, corners, ids, rejected, camera_k, camera_d)

        elapsed_ms = (rospy.Time.now() - start).to_sec() * 1000
        rospy.loginfo_throttle(30, f"Detection time: {elapsed_ms:.1f}ms")

        
        if ids is not None:

            # convert into a single list for easier sorting/searching
            aruco_tags = [ArucoTag(id, c) for c,id in zip(corners, ids.flatten())]

            # find poses of each tag
            for a in aruco_tags:
                rvecs, tvecs, _objPoints = cv.aruco.estimatePoseSingleMarkers(a.corners, a.length, camera_k, camera_d) # this needs to be done tag by tag because of different lengths
                a.pose_from_opencv(rvecs, tvecs)

            # convert vectors to ROS poses
            #tag_poses = [cv_to_pose(r,t) for r,t in zip(rvecs, tvecs)]

            #rospy.loginfo_throttle(30, "\n".join([f"Found {len(ids)} markers (coordinate wrt camera frame)"] + [f"Marker {id[0]}: ({pose.position.x*1000:.1f}mm, {pose.position.y*1000:.1f}mm, {pose.position.z*1000:.1f}mm)" for id, pose in zip(ids, tag_poses)]))


            # draw marker axes on frame
            for a in aruco_tags:
                frame = cv.aruco.drawAxis(frame, camera_k, camera_d, a.rvec, a.tvec, a.length)

            # publish all found tags as a MarkerArray
            marker_pub.publish([a.get_marker() for a in aruco_tags])


        else:
            cv.putText(frame, 'NO MARKERS DETECTED!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # TODO: make this a publisher instead so it can be seen in RVIZ
        cv.imshow("RealSense Feed", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            rospy.signal_shutdown("User quit")

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()