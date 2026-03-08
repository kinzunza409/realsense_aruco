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


def pose_distance(pose1, pose2):
    p1 = np.array([pose1.position.x, pose1.position.y, pose1.position.z])
    p2 = np.array([pose2.position.x, pose2.position.y, pose2.position.z])
    return np.linalg.norm(p2 - p1)

# converts OpenCV translatation and rotation vectors to a ROS Pose
def cv_to_pose(r, t):

    t = t.flatten()
    r = r.flatten()
    
    angle = np.linalg.norm(r)
    axis = r / angle
    q = tf.transformations.quaternion_about_axis(angle, axis)
    
    p = geometry_msgs.msg.Pose()
    p.position.x = t[0]
    p.position.y = t[1]
    p.position.z = t[2]
    p.orientation.x = q[0]
    p.orientation.y = q[1]
    p.orientation.z = q[2]
    p.orientation.w = q[3]

    return p

# returns translation distance between two transforms in mm
def transform_translation_distance(t1, t2):
    
    # extract the transform from the transfromstamped
    t1 = t1.transform
    t2 = t2.transform
    
    p1 = np.array([t1.translation.x, t1.translation.y, t1.translation.z])
    p2 = np.array([t2.translation.x, t2.translation.y, t2.translation.z])
    return np.linalg.norm(p2 - p1) * 1000

def transform_pose(pose, transform):
    # build 4x4 matrix from transform
    T = tf.transformations.quaternion_matrix([
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w
    ])
    T[0][3] = transform.translation.x
    T[1][3] = transform.translation.y
    T[2][3] = transform.translation.z

    # build 4x4 matrix from pose
    P = tf.transformations.quaternion_matrix([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])
    P[0][3] = pose.position.x
    P[1][3] = pose.position.y
    P[2][3] = pose.position.z

    # multiply
    result_matrix = np.dot(T, P)

    # extract result back into a Pose
    result = geometry_msgs.msg.Pose()
    result.position.x = result_matrix[0][3]
    result.position.y = result_matrix[1][3]
    result.position.z = result_matrix[2][3]
    q = tf.transformations.quaternion_from_matrix(result_matrix)
    result.orientation.x = q[0]
    result.orientation.y = q[1]
    result.orientation.z = q[2]
    result.orientation.w = q[3]
    return result

def multiply_transforms(T1, T2):
    # build 4x4 matrices
    M1 = tf.transformations.quaternion_matrix([
        T1.transform.rotation.x,
        T1.transform.rotation.y,
        T1.transform.rotation.z,
        T1.transform.rotation.w
    ])
    M1[0][3] = T1.transform.translation.x
    M1[1][3] = T1.transform.translation.y
    M1[2][3] = T1.transform.translation.z

    M2 = tf.transformations.quaternion_matrix([
        T2.transform.rotation.x,
        T2.transform.rotation.y,
        T2.transform.rotation.z,
        T2.transform.rotation.w
    ])
    M2[0][3] = T2.transform.translation.x
    M2[1][3] = T2.transform.translation.y
    M2[2][3] = T2.transform.translation.z

    M = np.dot(M1, M2)

    T = geometry_msgs.msg.TransformStamped()
    T.header.stamp = rospy.Time.now()
    q = tf.transformations.quaternion_from_matrix(M)
    T.transform.translation.x = M[0][3]
    T.transform.translation.y = M[1][3]
    T.transform.translation.z = M[2][3]
    T.transform.rotation.x = q[0]
    T.transform.rotation.y = q[1]
    T.transform.rotation.z = q[2]
    T.transform.rotation.w = q[3]
    return T

def apply_transforms(p, transforms):
    result = p
    for transform in transforms:
        t = transform.transform if isinstance(transform, geometry_msgs.msg.TransformStamped) else transform
        result = transform_pose(result, t)
    return result

# find transform from o1 to o2
def find_transform(p, q, id = None):
    # assume that p and q are in o1 frame of reference
    
    T = geometry_msgs.msg.TransformStamped()
    
    T.header.frame_id = id
    T.header.stamp = rospy.Time.now()

    T.transform.translation = p
    T.transform.rotation = q

    return T

# returns Marker object from Pose and ID
def get_marker(pose, id):
    m = visualization_msgs.msg.Marker()
    m.pose = pose
    m.id = id

    return m

def main():
    rospy.init_node('realsense_aruco_pose')
    rospy.loginfo("RealSense Aruco Pose Node Started")
    marker_pub = rospy.Publisher(
        '/aruco/marker_poses',
        visualization_msgs.msg.MarkerArray,
        queue_size=10)
    cvbridge = CvBridge()

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
    origin_id = 1 # aruco id for marking robot base

    # frame transforms
    T_ca = IDENTITY_TRANSFORM # tranform between camera and aruco
    T_ar_R = geometry_msgs.msg.TransformStamped()
    T_ar_T = geometry_msgs.msg.TransformStamped()
    #T_ar = geometry_msgs.msg.TransformStamped() # transform between aruco and robot base

    

    # rotation: 45 deg around y, then 90 deg around z to align tag x with robot y
    R = tf.transformations.euler_matrix(0,0,0)
    q = tf.transformations.quaternion_from_matrix(R)
    T_ar_R.transform.rotation.x = q[0]
    T_ar_R.transform.rotation.y = q[1]
    T_ar_R.transform.rotation.z = q[2]
    T_ar_R.transform.rotation.w = q[3]

    # 40mm above robot in z
    T_ar_T.transform.translation.x = 0.34925/2 + 0.01 + origin_marker_length/2
    T_ar_T.transform.translation.y = 0
    T_ar_T.transform.translation.z = 0.012

    #T_ar = multiply_transforms(T_ar_R, T_ar_T)
    T_ar = IDENTITY_TRANSFORM

    #flags
    extrisnics_found = False

    def image_callback(msg):
        nonlocal T_ca, T_ar, extrisnics_found
        
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



        # get aruco poses wrt camera pose
        #rvecs, tvecs, _objPoints = cv.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_k, camera_d)
        #num_rejected = len(rejected)
        #rospy.loginfo_throttle(5.0, f"Rejected candidates: {num_rejected}")
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
            
            '''
            # check if origin ID found
            if origin_id in ids:
                idx = list(ids.flatten()).index(origin_id)
                r_origin, t_origin, _objPoints = cv.aruco.estimatePoseSingleMarkers(corners[idx], origin_marker_length, camera_k, camera_d)
                origin_pose = cv_to_pose(r_origin, t_origin) # pose of tag wrt camera
                T_temp = find_transform(origin_pose.position, origin_pose.orientation)

                # check if origin aruco has moved
                if transform_translation_distance(T_temp, T_ca) > 10:
                    # if this is the first time the transform was found
                    if not extrisnics_found:
                        extrisnics_found = True
                        rospy.loginfo(f"Origin Marker ID {origin_id} has been found")
                    else:
                        rospy.logwarn_throttle(10, f"Origin Marker ID {origin_id} or CAMERA has moved")

                    # update transform
                    T_ca = T_temp
                    
            else:
                rospy.logwarn_throttle(3*60,f"Origin Marker ID {origin_id} is obscured")

            

            #transform poses to robot base frame of reference
            tag_poses_transformed = [apply_transforms(p, [T_ca, T_ar]) for p in tag_poses]
            # publish poses if the camera posiotion is known (otherwise the values will be innacurate)
            if extrisnics_found: 
                marker_pub.publish([get_marker(t, id) for t,id in zip(tag_poses_transformed, ids.flatten())])
            else:
                rospy.logerr_throttle(30, "Not publishing marker poses...origin marker may not be visible")

            #cv.aruco.drawDetectedMarkers(frame, corners, ids)
            for r,t in zip(rvecs, tvecs):
                frame = cv.aruco.drawAxis(frame, camera_k, camera_d, r, t, marker_length)

                #cv.putText(frame, f"({x:.3f}, {y:.3f}, {z:.3f})", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                #cv.putText(frame, f"(DIST: {dist:.2f}mm)", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            '''

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