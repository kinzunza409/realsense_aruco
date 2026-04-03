#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
import rospy
import cv2 as cv
import mediapipe as mp
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
import tf2_geometry_msgs
from mediapipe.framework.formats import landmark_pb2

PL = mp.solutions.pose.PoseLandmark

_DEFAULT_POSE_DETECTION_THRESHOLD = 0.5
_DEFAULT_ALPHA = 0.8


@dataclass
class Landmark2D:
    name: str
    x_px: int
    y_px: int
    visibility: float

@dataclass
class WristState:
    pos:      np.ndarray
    prev_pos: np.ndarray
    vel:      np.ndarray
    prev_vel: np.ndarray

    def __post_init__(self):
        assert self.pos.shape      == (3,), f"expected (3,), got {self.pos.shape}"
        assert self.prev_pos.shape == (3,), f"expected (3,), got {self.prev_pos.shape}"
        assert self.vel.shape      == (3,), f"expected (3,), got {self.vel.shape}"
        assert self.prev_vel.shape == (3,), f"expected (3,), got {self.prev_vel.shape}"

    @classmethod
    def nan(cls) -> 'WristState':
        return cls(
            pos      = np.full(3, np.nan),
            prev_pos = np.full(3, np.nan),
            vel      = np.full(3, np.nan),
            prev_vel = np.full(3, np.nan),
        )


@dataclass
class UpperPoseResult:
    left_wrist:  np.ndarray
    right_wrist: np.ndarray

    def __post_init__(self):
        assert self.left_wrist.shape  == (3,), f"expected (3,), got {self.left_wrist.shape}"
        assert self.right_wrist.shape == (3,), f"expected (3,), got {self.right_wrist.shape}"


class UpperPoseFinder:
    def __init__(self, camera_k: np.ndarray, pose_threshold : float):
        
        self.camera_k = camera_k
        self.pose_th = pose_threshold
        
        self._tracker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_pose_3d(self, frame: np.ndarray, frame_depth: np.ndarray) -> UpperPoseResult | None:
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        raw = self._tracker.process(frame_rgb)

        if raw.pose_landmarks is None:
            return None

        h, w = frame.shape[:2]
        landmarks = self._extract_landmarks(raw.pose_landmarks, w, h)

        def deproject_landmark(name : str):
            lm = landmarks[name]
            # if mediapipe confidence is below threshold
            if lm.visibility < self.pose_th:
                return np.full(3, np.nan)
            return self.deproject((lm.x_px,lm.y_px), frame_depth)
        
        result = UpperPoseResult(
            left_wrist  = deproject_landmark("left_wrist"),
            right_wrist = deproject_landmark("right_wrist"),
        )
        
        return result 

        

    # determines the x,y,z location of a point in a 2d image using a synced depth image
    def deproject(self, px: tuple[int, int], depth_frame: np.ndarray) -> np.ndarray:
        fx, fy = self.camera_k[0, 0], self.camera_k[1, 1] # focal length
        cx, cy = self.camera_k[0, 2], self.camera_k[1, 2] # principle point
        u, v = px
        
        z = depth_frame[v, u] / 1000.0 # convert to meters

        # RealSense sets depth to zero if invalid measurement
        if z == 0:
            return np.full(3, np.nan)

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        return np.array([x, y, z])


    @staticmethod
    def _extract_landmarks(raw_landmarks, frame_w: int, frame_h: int) -> dict[str, Landmark2D]:
        lms = raw_landmarks.landmark
        upper_body = [
            PL.NOSE,
            PL.LEFT_EYE_INNER,  PL.LEFT_EYE,  PL.LEFT_EYE_OUTER,
            PL.RIGHT_EYE_INNER, PL.RIGHT_EYE, PL.RIGHT_EYE_OUTER,
            PL.LEFT_EAR,        PL.RIGHT_EAR,
            PL.MOUTH_LEFT,      PL.MOUTH_RIGHT,
            PL.LEFT_SHOULDER,   PL.RIGHT_SHOULDER,
            PL.LEFT_ELBOW,      PL.RIGHT_ELBOW,
            PL.LEFT_WRIST,      PL.RIGHT_WRIST,
            PL.LEFT_PINKY,      PL.RIGHT_PINKY,
            PL.LEFT_INDEX,      PL.RIGHT_INDEX,
            PL.LEFT_THUMB,      PL.RIGHT_THUMB,
        ]
        return {
            pl.name.lower(): Landmark2D(
                pl.name.lower(),
                int(lms[pl.value].x * frame_w),
                int(lms[pl.value].y * frame_h),
                lms[pl.value].visibility,
            )
            for pl in upper_body
        }
    
def velocity_filtered(p1 : np.ndarray, p2 : np.ndarray, v1 : np.ndarray, dt : float, alpha : float) -> np.ndarray :
    v = (p2 - p1)/dt # caclulate raw velocity
    v2 = alpha * v + (1 - alpha) * v1 # apply EMA low pass filter with last velocity
    return v2

def main():
    rospy.init_node('human_intent_node')
    rospy.loginfo("Human Intent Node Started")

    cvbridge = CvBridge()

    pose_detection_th = rospy.get_param('~pose_detection_th', _DEFAULT_POSE_DETECTION_THRESHOLD)
    alpha = rospy.get_param('alpha', _DEFAULT_ALPHA)

    camera_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
    camera_k = np.array(camera_info.K).reshape(3, 3)

    pose_finder = UpperPoseFinder(camera_k, pose_detection_th)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    wrist_vel_pub = rospy.Publisher('/human_intent/wrist_velocity', PointStamped, queue_size=10)

    prev_time:   rospy.Time | None = None
    left_state:  WristState | None = None
    right_state: WristState | None = None

    def callback(rgb_msg: Image, depth_msg: Image) -> None:
        nonlocal prev_time, left_state, right_state

        frame       = cvbridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        frame_depth = cvbridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        current_time = rgb_msg.header.stamp

        # account for first call of function
        if prev_time is None:
            prev_time = current_time
            return

        dt = (current_time - prev_time).to_sec()
        prev_time = current_time

        # no person detected — reset time so dt is clean on next valid frame
        pose = pose_finder.get_pose_3d(frame, frame_depth)
        if pose is None:
            prev_time = None
            return

        # right wrist velocity
        if right_state is not None and not np.isnan(pose.right_wrist).any():
            raw_vel     = (pose.right_wrist - right_state.pos) / dt
            filtered    = alpha * raw_vel + (1 - alpha) * right_state.vel
            right_state = WristState(pos=pose.right_wrist, prev_pos=right_state.pos, vel=filtered, prev_vel=right_state.vel)
        else:
            right_state = WristState.nan()

        # left wrist velocity
        if left_state is not None and not np.isnan(pose.left_wrist).any():
            raw_vel    = (pose.left_wrist - left_state.pos) / dt
            filtered   = alpha * raw_vel + (1 - alpha) * left_state.vel
            left_state = WristState(pos=pose.left_wrist, prev_pos=left_state.pos, vel=filtered, prev_vel=left_state.vel)
        else:
            left_state = WristState.nan()

    
    # TODO: Make visualizer to check if velocity is correct
    
    rgb_sub = Subscriber('/camera/color/image_raw', Image)
    depth_sub = Subscriber('/camera/aligned_depth_to_color/depth/image_raw', Image)
    sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.05)
    sync.registerCallback(callback)

    rospy.spin()


if __name__ == '__main__':
    main()