#!/usr/bin/env python3

import socket
import struct

import rospy
import tf2_ros


def main():
    rospy.init_node("laser_udp_bridge")

    target_host = "127.0.0.1"
    target_port = 5005

    reference_frame = "k4a_rgb_camera_link"
    target_frame = "laser_spot_frame"

    seq = 0

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        try:
            transform = tf_buffer.lookup_transform(
                reference_frame,
                target_frame,
                rospy.Time(0),
                rospy.Duration(0.0),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        now = rospy.Time.now()
        t_ros_ns = now.to_nsec()

        translation = transform.transform.translation
        payload = struct.pack(
            "<IQffff",
            seq,
            t_ros_ns,
            float(translation.x),
            float(translation.y),
            float(translation.z),
            1.0,
        )
        udp_socket.sendto(payload, (target_host, target_port))
        seq = (seq + 1) % (2 ** 32)

        rate.sleep()


if __name__ == "__main__":
    main()
