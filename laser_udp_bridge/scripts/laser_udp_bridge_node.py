#!/usr/bin/env python3

import socket
import struct
import threading

import rospy
import tf2_ros

from nn_laser_spot_tracking.msg import KeypointImage


def main():
    rospy.init_node("laser_udp_bridge")

    target_host = rospy.get_param("~target_host", "127.0.0.1")
    target_port = int(rospy.get_param("~target_port", 5005))

    reference_frame = rospy.get_param("~reference_frame", "k4a_rgb_camera_link")
    target_frame = rospy.get_param("~target_frame", "laser_spot_frame")
    keypoint_topic = rospy.get_param(
        "~keypoint_topic",
        "/nn_laser_spot_tracking/detection_output_keypoint",
    )
    rate_hz = float(rospy.get_param("~rate", 60.0))

    seq = 0

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    keypoint_lock = threading.Lock()
    latest_keypoint = {"msg": None}

    def keypoint_cb(msg):
        with keypoint_lock:
            latest_keypoint["msg"] = msg

    rospy.Subscriber(keypoint_topic, KeypointImage, keypoint_cb, queue_size=1)

    last_sent_stamp = None

    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        with keypoint_lock:
            kp = latest_keypoint["msg"]
        if kp is None:
            rate.sleep()
            continue
        if kp.confidence <= 0.0:
            rospy.loginfo_throttle(
                5.0,
                "Skipping keypoint with non-positive confidence (%.3f) at stamp %s",
                kp.confidence,
                str(kp.header.stamp),
            )
            rate.sleep()
            continue
        if last_sent_stamp is not None and kp.header.stamp == last_sent_stamp:
            rospy.loginfo_throttle(
                2.0,
                "Skipping duplicate keypoint stamp %s (already sent)",
                str(kp.header.stamp),
            )
            rate.sleep()
            continue
        try:
            transform = tf_buffer.lookup_transform(
                reference_frame,
                target_frame,
                kp.header.stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.loginfo_throttle(
                2.0,
                "TF lookup failed for %s->%s at stamp %s: %s",
                reference_frame,
                target_frame,
                str(kp.header.stamp),
                str(e),
            )
            rate.sleep()
            continue

        t_ros_ns = kp.header.stamp.to_nsec()

        translation = transform.transform.translation
        payload = struct.pack(
            "<IQffff",
            seq,
            t_ros_ns,
            float(translation.x),
            float(translation.y),
            float(translation.z),
            float(kp.confidence),
        )
        udp_socket.sendto(payload, (target_host, target_port))
        last_sent_stamp = kp.header.stamp
        seq = (seq + 1) % (2 ** 32)

        rate.sleep()


if __name__ == "__main__":
    main()
