#!/usr/bin/env python3

# this is needed otherwise you need to install tkinter
import matplotlib
matplotlib.use("Agg")

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
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
    left_wrist:     np.ndarray
    right_wrist:    np.ndarray
    left_wrist_px:  Tuple[int, int]
    right_wrist_px: Tuple[int, int]

    def __post_init__(self):
        assert self.left_wrist.shape  == (3,), f"expected (3,), got {self.left_wrist.shape}"
        assert self.right_wrist.shape == (3,), f"expected (3,), got {self.right_wrist.shape}"


class UpperPoseFinder:
    def __init__(self, camera_k: np.ndarray, pose_threshold: float):
        self.camera_k = camera_k
        self.pose_th  = pose_threshold
        self._last_landmarks = None

        self._tracker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_pose_3d(self, frame: np.ndarray, frame_depth: np.ndarray) -> Optional[UpperPoseResult]:
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        raw = self._tracker.process(frame_rgb)

        if raw.pose_landmarks is None:
            return None

        h, w = frame.shape[:2]
        landmarks = self._extract_landmarks(raw.pose_landmarks, w, h)

        def deproject_landmark(name: str) -> np.ndarray:
            lm = landmarks[name]
            # if mediapipe confidence is below threshold
            if lm.visibility < self.pose_th:
                return np.full(3, np.nan)
            return self.deproject((lm.x_px, lm.y_px), frame_depth)

        self._last_landmarks = raw.pose_landmarks

        result = UpperPoseResult(
            left_wrist     = deproject_landmark("left_wrist"),
            right_wrist    = deproject_landmark("right_wrist"),
            left_wrist_px  = (landmarks["left_wrist"].x_px,  landmarks["left_wrist"].y_px),
            right_wrist_px = (landmarks["right_wrist"].x_px, landmarks["right_wrist"].y_px),
        )

        return result

    # determines the x,y,z location of a point in a 2d image using a synced depth image
    def deproject(self, px: Tuple[int, int], depth_frame: np.ndarray) -> np.ndarray:
        fx, fy = self.camera_k[0, 0], self.camera_k[1, 1]
        cx, cy = self.camera_k[0, 2], self.camera_k[1, 2]
        u, v = px

        h, w = depth_frame.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            return np.full(3, np.nan)

        z = depth_frame[v, u] / 1000.0  # convert to meters

        # RealSense sets depth to zero if invalid measurement
        if z == 0:
            return np.full(3, np.nan)

        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        return np.array([x, y, z])

    def draw_pose(self, frame: np.ndarray, pose: UpperPoseResult,
                  right_state: Optional['WristState'] = None, show_values: bool = False) -> None:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, self._last_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )
        for px in [pose.left_wrist_px, pose.right_wrist_px]:
            cv.circle(frame, px, 8, (0, 255, 0), -1)

        if show_values and right_state is not None and not np.isnan(pose.right_wrist).any():
            p = pose.right_wrist
            v = right_state.vel
            mag = np.linalg.norm(v) if not np.isnan(v).any() else float('nan')
            lines = [
                f"R wrist pos  x:{p[0]:.3f} y:{p[1]:.3f} z:{p[2]:.3f} m",
                f"R wrist vel  x:{v[0]:.3f} y:{v[1]:.3f} z:{v[2]:.3f} m/s",
                f"R wrist |v|  {mag:.3f} m/s",
            ]
            y = 30
            for line in lines:
                cv.putText(frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
                y += 28

    @staticmethod
    def _extract_landmarks(raw_landmarks, frame_w: int, frame_h: int) -> Dict[str, Landmark2D]:
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


def is_valid(state: WristState) -> bool:
    return not np.isnan(state.pos).any()


def velocity_filtered(p1: np.ndarray, p2: np.ndarray, v1: np.ndarray, dt: float, alpha: float) -> np.ndarray:
    v  = (p2 - p1) / dt  # caclulate raw velocity
    v2 = alpha * v + (1 - alpha) * v1  # apply EMA low pass filter with last velocity
    return v2


def main():
    rospy.init_node('human_intent_node')
    rospy.loginfo("Human Intent Node Started")

    cvbridge = CvBridge()

    pose_detection_th = rospy.get_param('~pose_detection_th', _DEFAULT_POSE_DETECTION_THRESHOLD)
    alpha             = rospy.get_param('~alpha', _DEFAULT_ALPHA)
    viewer_enabled    = rospy.get_param('~viewer_enabled', False)
    viz_show_values   = rospy.get_param('~viz_show_values', False)

    camera_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
    camera_k    = np.array(camera_info.K).reshape(3, 3)

    pose_finder = UpperPoseFinder(camera_k, pose_detection_th)

    tf_buffer   = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    wrist_pos_pub = rospy.Publisher('/human_intent/right_wrist/position', PointStamped, queue_size=10)
    wrist_vel_pub = rospy.Publisher('/human_intent/right_wrist/velocity', PointStamped, queue_size=10)
    viz_pub       = rospy.Publisher('/human_intent/visualization', Image, queue_size=1)

    prev_time:   Optional[rospy.Time] = None
    left_state:  Optional[WristState] = None
    right_state: Optional[WristState] = None

    def callback(rgb_msg: Image, depth_msg: Image) -> None:
        nonlocal prev_time, left_state, right_state

        #rospy.loginfo_throttle(1, "Callback firing")

        frame       = cvbridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        frame_depth = cvbridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        current_time = rgb_msg.header.stamp

        # account for first call of function
        if prev_time is None:
            prev_time = current_time
            return

        dt        = (current_time - prev_time).to_sec()
        prev_time = current_time

        # no person detected — reset time so dt is clean on next valid frame
        pose = pose_finder.get_pose_3d(frame, frame_depth)
        if pose is None:
            prev_time = None
            return

        # right wrist velocity
        if not np.isnan(pose.right_wrist).any():
            if right_state is not None and is_valid(right_state):
                right_state = WristState(
                    pos      = pose.right_wrist,
                    prev_pos = right_state.pos,
                    vel      = velocity_filtered(right_state.pos, pose.right_wrist, right_state.vel, dt, alpha),
                    prev_vel = right_state.vel,
                )
            else:
                right_state = WristState(pos=pose.right_wrist, prev_pos=pose.right_wrist, vel=np.zeros(3), prev_vel=np.zeros(3))

            if is_valid(right_state):
                pos_msg = PointStamped()
                pos_msg.header.stamp    = current_time
                pos_msg.header.frame_id = 'camera_color_optical_frame'
                pos_msg.point.x = right_state.pos[0]
                pos_msg.point.y = right_state.pos[1]
                pos_msg.point.z = right_state.pos[2]
                wrist_pos_pub.publish(pos_msg)

                vel_msg = PointStamped()
                vel_msg.header.stamp    = current_time
                vel_msg.header.frame_id = 'camera_color_optical_frame'
                vel_msg.point.x = right_state.vel[0]
                vel_msg.point.y = right_state.vel[1]
                vel_msg.point.z = right_state.vel[2]
                wrist_vel_pub.publish(vel_msg)

        # left wrist velocity
        if not np.isnan(pose.left_wrist).any():
            if left_state is not None and is_valid(left_state):
                left_state = WristState(
                    pos      = pose.left_wrist,
                    prev_pos = left_state.pos,
                    vel      = velocity_filtered(left_state.pos, pose.left_wrist, left_state.vel, dt, alpha),
                    prev_vel = left_state.vel,
                )
            else:
                left_state = WristState(pos=pose.left_wrist, prev_pos=pose.left_wrist, vel=np.zeros(3), prev_vel=np.zeros(3))

        # TODO: Make visualizer to check if velocity is correct
        if viewer_enabled:
            pose_finder.draw_pose(frame, pose, right_state, viz_show_values)
            viz_pub.publish(cvbridge.cv2_to_imgmsg(frame, encoding='bgr8'))

    rgb_sub   = Subscriber('/camera/color/image_raw', Image)
    depth_sub = Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
    sync      = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.05)
    sync.registerCallback(callback)
    rospy.loginfo("Waiting for camera frames...")

    rospy.spin()


if __name__ == '__main__':
    main()