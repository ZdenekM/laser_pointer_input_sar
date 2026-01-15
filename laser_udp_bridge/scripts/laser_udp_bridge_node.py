#!/usr/bin/env python3

import json
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import rospy
import tf2_ros

from nn_laser_spot_tracking.msg import KeypointImage
from std_srvs.srv import Trigger


class CalibrationHttpHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == self.server.pose_path:
            payload = self.server.pose_payload_fn()
        elif path == self.server.point_path:
            payload = self.server.point_payload_fn()
        else:
            self.send_error(404, "Not Found")
            return
        body = json.dumps(payload).encode("utf-8")
        status = 200 if payload.get("ok", False) else 503
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?")[0] != self.server.calibrate_path:
            self.send_error(404, "Not Found")
            return
        payload = self.server.calibrate_fn()
        body = json.dumps(payload).encode("utf-8")
        status = 200 if payload.get("ok", False) else 503
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main():
    rospy.init_node("laser_udp_bridge")

    target_host = rospy.get_param("~target_host", "127.0.0.1")
    target_port = int(rospy.get_param("~target_port", 5005))

    reference_frame = rospy.get_param("~reference_frame", "table_frame")
    target_frame = rospy.get_param("~target_frame", "laser_spot_frame")
    camera_frame = rospy.get_param("~camera_frame", "k4a_rgb_camera_link")
    keypoint_topic = rospy.get_param(
        "~keypoint_topic",
        "/nn_laser_spot_tracking/detection_output_keypoint",
    )
    rate_hz = float(rospy.get_param("~rate", 60.0))
    calibration_param_ns = rospy.get_param("~calibration_param_ns", "/table_calibration")

    http_enable = bool(rospy.get_param("~http_enable", True))
    http_bind = rospy.get_param("~http_bind", "0.0.0.0")
    http_port = int(rospy.get_param("~http_port", 8000))
    http_pose_path = rospy.get_param("~http_pose_path", "/camera_pose")
    http_point_path = "/laser_point"
    http_calibrate_path = rospy.get_param("~http_calibrate_path", "/calibrate")
    calibration_service = rospy.get_param(
        "~calibration_service",
        "/aruco_table_calibration/calibrate",
    )
    calibration_service_timeout = float(rospy.get_param("~calibration_service_timeout", 2.0))

    seq = 0

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    def get_calibration_payload():
        payload = {
            "ok": False,
            "reference_frame": reference_frame,
            "camera_frame": camera_frame,
        }
        try:
            transform = tf_buffer.lookup_transform(
                reference_frame,
                camera_frame,
                rospy.Time(0),
                rospy.Duration(0.2),
            )
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            payload["ok"] = True
            payload["pose"] = {
                "frame_id": transform.header.frame_id,
                "child_frame_id": transform.child_frame_id,
                "stamp": {
                    "secs": int(transform.header.stamp.secs),
                    "nsecs": int(transform.header.stamp.nsecs),
                },
                "translation": {
                    "x": float(translation.x),
                    "y": float(translation.y),
                    "z": float(translation.z),
                },
                "rotation": {
                    "x": float(rotation.x),
                    "y": float(rotation.y),
                    "z": float(rotation.z),
                    "w": float(rotation.w),
                },
            }
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            payload["pose_error"] = str(exc)

        ns = calibration_param_ns.rstrip("/")
        payload["table"] = {
            "width_m": rospy.get_param(ns + "/width_m", None),
            "height_m": rospy.get_param(ns + "/height_m", None),
            "marker_size_m": rospy.get_param(ns + "/marker_size_m", None),
            "origin_corner": rospy.get_param(ns + "/origin_corner", None),
            "x_axis_corner": rospy.get_param(ns + "/x_axis_corner", None),
            "y_axis_corner": rospy.get_param(ns + "/y_axis_corner", None),
            "origin_id": rospy.get_param(ns + "/origin_id", None),
            "x_axis_id": rospy.get_param(ns + "/x_axis_id", None),
            "y_axis_id": rospy.get_param(ns + "/y_axis_id", None),
            "stamp": {
                "secs": rospy.get_param(ns + "/stamp_secs", None),
                "nsecs": rospy.get_param(ns + "/stamp_nsecs", None),
            },
        }
        return payload

    def trigger_calibration():
        payload = {"ok": False}
        try:
            rospy.wait_for_service(calibration_service, timeout=calibration_service_timeout)
        except (rospy.ROSException, rospy.ROSInterruptException) as exc:
            payload["error"] = f"Calibration service unavailable: {exc}"
            return payload

        try:
            calibrate = rospy.ServiceProxy(calibration_service, Trigger)
            response = calibrate()
        except rospy.ServiceException as exc:
            payload["error"] = f"Calibration service call failed: {exc}"
            return payload

        payload["ok"] = bool(response.success)
        payload["message"] = response.message
        return payload

    point_lock = threading.Lock()
    latest_point = {"payload": None}

    def get_point_payload():
        with point_lock:
            payload = latest_point["payload"]
        if payload is None:
            return {"ok": False, "error": "No laser point data available yet"}
        return payload

    if http_enable:
        def http_thread_fn():
            server = HTTPServer((http_bind, http_port), CalibrationHttpHandler)
            server.pose_path = http_pose_path
            server.point_path = http_point_path
            server.calibrate_path = http_calibrate_path
            server.pose_payload_fn = get_calibration_payload
            server.point_payload_fn = get_point_payload
            server.calibrate_fn = trigger_calibration
            rospy.loginfo(
                "HTTP endpoints: pose=http://%s:%d%s point=http://%s:%d%s calibrate=http://%s:%d%s",
                http_bind,
                http_port,
                http_pose_path,
                http_bind,
                http_port,
                http_point_path,
                http_bind,
                http_port,
                http_calibrate_path,
            )
            server.serve_forever()

        thread = threading.Thread(target=http_thread_fn, daemon=True)
        thread.start()

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
        flags = 1 if getattr(kp, "predicted", False) else 0
        payload = struct.pack(
            "<IQffffI",
            seq,
            t_ros_ns,
            float(translation.x),
            float(translation.y),
            float(translation.z),
            float(kp.confidence),
            flags,
        )
        point_payload = {
            "ok": True,
            "seq": int(seq),
            "stamp": {
                "secs": int(kp.header.stamp.secs),
                "nsecs": int(kp.header.stamp.nsecs),
            },
            "frame_id": reference_frame,
            "target_frame": target_frame,
            "position": {
                "x": float(translation.x),
                "y": float(translation.y),
                "z": float(translation.z),
            },
            "confidence": float(kp.confidence),
            "predicted": bool(getattr(kp, "predicted", False)),
        }
        with point_lock:
            latest_point["payload"] = point_payload
        udp_socket.sendto(payload, (target_host, target_port))
        last_sent_stamp = kp.header.stamp
        seq = (seq + 1) % (2 ** 32)

        rate.sleep()


if __name__ == "__main__":
    main()
