#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray, Marker

class Aruco:
    def __init__(self, marker : Marker):
        self.update(marker)

    def update(self, marker: Marker):
        self.stamp : rospy.Time = marker.header.stamp
        p = marker.pose.position
        self.pos : np.ndarray = np.array([p.x, p.y, p.z])


def main():
    rospy.init_node('intent_estimation_node')
    rospy.loginfo("Intent Estimation Node Started")

    vel_threshold = rospy.get_param('~vel_threshold', 0.05)

    arucos : Dict[int, Aruco] = {} # aruco tag objects
    scores : Dict[int, float] = {} # score that human intends to interact with aruco

    def wrist_vel_callback(msg: PointStamped) -> None:
        point = msg.point
        v = np.array([point.x, point.y, point.z]) # wrist velocity

        norm_v = np.linalg.norm(v)

        # if wrist is stationary don't update scores
        if norm_v < vel_threshold:
            return

        for id, a in arucos.items():
            s = scores[id]
            p = a.pos
            norm_p = np.linalg.norm(p)

            # if aruco is at origin
            if norm_p < 1e-6:
                rospy.logerr(f"Aruco id: {id} is too close to camera x: {p[0]:.3f}, y: {p[1]:.3f}, z: {p[2]:.3f}")
                continue

            # need to clip to prevent floating point erros pusing past arccos input limits
            theta = np.arccos(np.clip(np.dot(v,p)/(norm_v*norm_p) ,-1.0, 1.0))
            s = theta
            rospy.loginfo_throttle(1, f"Aruco id: {id}, angle: {np.degrees(theta):.1f} deg")


        # softmax
                
    
    def aruco_callback(msg: MarkerArray) -> None:

        for m in msg.markers:
            if m.id not in arucos:
                arucos[m.id] = Aruco(m)
                scores[m.id] = np.nan
                rospy.loginfo(f"New aruco found! Id: {m.id}, x: {m.pose.position.x:.3f}, y: {m.pose.position.y:.3f}, z: {m.pose.position.z:.3f}")
            else:
                arucos[m.id].update(m)


        

    rospy.Subscriber('/human_intent/right_wrist/velocity', PointStamped, wrist_vel_callback)
    rospy.Subscriber('/aruco/marker_poses', MarkerArray, aruco_callback)

    rospy.spin()


if __name__ == '__main__':
    main()