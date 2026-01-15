#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aruco-based table calibration with on-demand capture and TF publication.
"""

import threading

import cv2
import numpy as np

import rospy
import tf2_ros
import tf.transformations as tft

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
from std_srvs.srv import Trigger, TriggerResponse


class ArucoTableCalibrator:
    def __init__(self):
        if not hasattr(cv2, "aruco"):
            rospy.logfatal("cv2.aruco not available; install opencv-contrib-python")
            raise RuntimeError("cv2.aruco missing")
        if not hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rospy.logfatal("cv2.aruco.estimatePoseSingleMarkers missing; OpenCV contrib required")
            raise RuntimeError("cv2.aruco.estimatePoseSingleMarkers missing")

        self.bridge = CvBridge()

        self.marker_size = float(rospy.get_param("~marker_size", 0.1))
        self.origin_id = int(rospy.get_param("~origin_id", 0))
        self.x_axis_id = int(rospy.get_param("~x_axis_id", 49))
        self.y_axis_id = int(rospy.get_param("~y_axis_id", 30))
        self.target_detections = int(rospy.get_param("~target_detections_per_marker", 15))
        self.capture_timeout = float(rospy.get_param("~capture_timeout", 15.0))
        self.origin_corner = rospy.get_param("~origin_corner", "lower_left")
        self.x_axis_corner = rospy.get_param("~x_axis_corner", "lower_right")
        self.y_axis_corner = rospy.get_param("~y_axis_corner", "upper_left")
        self.calibration_param_ns = rospy.get_param("~calibration_param_ns", "/table_calibration")

        self.table_frame = rospy.get_param("~table_frame", "table_frame")
        self.camera_frame_override = rospy.get_param("~camera_frame", "")

        self.camera_image_topic = rospy.get_param("~camera_image_topic", "/k4a/rgb/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/k4a/rgb/camera_info")
        transport_param = rospy.get_param("~transport", "raw")
        if transport_param != "raw":
            raise ValueError("Only raw image transport is supported (transport=raw)")

        dictionary_name = rospy.get_param("~aruco_dictionary", "DICT_4X4_100")
        if not hasattr(cv2.aruco, dictionary_name):
            rospy.logfatal("Unknown aruco dictionary: %s", dictionary_name)
            raise RuntimeError("Unknown aruco dictionary")
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        try:
            self.detector_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None

        self.samples_lock = threading.Lock()
        self.samples = {}
        self.capture_active = False
        self.capture_end_time = rospy.Time(0)
        self.last_calibration = None
        self.seen_marker_ids = set()

        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self._setup_subscribers()
        self._setup_services()
        rospy.loginfo(
            "Aruco calibrator listening on image=%s info=%s",
            self.camera_image_topic,
            self.camera_info_topic,
        )

    def _setup_subscribers(self):
        self.image_sub = rospy.Subscriber(
            self.camera_image_topic,
            ROSImage,
            self._image_callback,
            queue_size=1,
        )
        self.info_sub = rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self._info_callback,
            queue_size=1,
        )

    def _setup_services(self):
        self.calib_service = rospy.Service(
            "~calibrate",
            Trigger,
            self._handle_calibrate,
        )

    def _info_callback(self, info_msg):
        rospy.loginfo_throttle(5.0, "Received camera_info (frame_id=%s)", info_msg.header.frame_id)
        if self.camera_matrix is None:
            self._update_camera_info(info_msg)
        elif info_msg.header.frame_id and not self.camera_frame_override:
            self.camera_frame = info_msg.header.frame_id

    def _image_callback(self, image_msg):
        if not self.camera_frame and image_msg.header.frame_id:
            self.camera_frame = image_msg.header.frame_id

        if not self.capture_active:
            return
        if rospy.Time.now() > self.capture_end_time:
            return
        rospy.loginfo_throttle(
            2.0,
            "Capture active: image stamp=%s, intrinsics_ready=%s",
            str(image_msg.header.stamp),
            "yes" if self.camera_matrix is not None else "no",
        )

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        detections, detected_ids = self._detect_markers(cv_image)
        if detected_ids:
            self.seen_marker_ids.update(detected_ids)
        if not detections:
            return

        with self.samples_lock:
            for marker_id, tvec in detections.items():
                self.samples.setdefault(marker_id, []).append(tvec)

    def _update_camera_info(self, info_msg):
        if info_msg.K and len(info_msg.K) == 9:
            self.camera_matrix = np.array(info_msg.K, dtype=np.float64).reshape((3, 3))
        else:
            self.camera_matrix = None
        if info_msg.D and len(info_msg.D) > 0:
            self.dist_coeffs = np.array(info_msg.D, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros((5,), dtype=np.float64)
        if self.camera_frame_override:
            self.camera_frame = self.camera_frame_override
        else:
            self.camera_frame = info_msg.header.frame_id

    def _detect_markers(self, cv_image):
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.detector_params,
            )
        if ids is None or len(ids) == 0:
            return {}, []
        detected_ids = [int(marker_id) for marker_id in ids.flatten().tolist()]
        if self.camera_matrix is None:
            rospy.logwarn_throttle(
                5.0,
                "Detected ArUco ids %s but camera intrinsics are missing",
                ", ".join(map(str, detected_ids)),
            )
            return {}, detected_ids
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_size,
            self.camera_matrix,
            self.dist_coeffs,
        )
        detections = {}
        for idx, marker_id in enumerate(ids.flatten().tolist()):
            if marker_id in (self.origin_id, self.x_axis_id, self.y_axis_id):
                tvec = tvecs[idx][0]
                detections[int(marker_id)] = np.array(tvec, dtype=np.float64)
        return detections, detected_ids

    def _handle_calibrate(self, _req):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return TriggerResponse(False, "Camera intrinsics not received yet")
        if self.capture_active:
            return TriggerResponse(False, "Calibration already in progress")
        if self.target_detections <= 0:
            return TriggerResponse(False, "target_detections_per_marker must be > 0")
        if self.capture_timeout <= 0.0:
            return TriggerResponse(False, "capture_timeout must be > 0")

        with self.samples_lock:
            self.samples = {}
        self.seen_marker_ids = set()
        self.capture_active = True
        self.capture_end_time = rospy.Time.now() + rospy.Duration.from_sec(self.capture_timeout)
        rospy.loginfo(
            "Starting aruco capture (target=%d per marker, timeout=%.2f s, ids: %d, %d, %d)",
            self.target_detections,
            self.capture_timeout,
            self.origin_id,
            self.y_axis_id,
            self.x_axis_id,
        )

        rate = rospy.Rate(50)
        while not rospy.is_shutdown() and rospy.Time.now() < self.capture_end_time:
            if self._has_enough_detections():
                break
            rate.sleep()

        self.capture_active = False
        if not self._has_enough_detections():
            counts = self._detection_counts()
            return TriggerResponse(
                False,
                "Timeout before reaching %d detections per marker: %s" % (
                    self.target_detections,
                    counts,
                ),
            )
        success, message = self._compute_and_publish()
        return TriggerResponse(success, message)

    def _compute_and_publish(self):
        with self.samples_lock:
            samples_copy = {mid: list(vals) for mid, vals in self.samples.items()}

        missing = [mid for mid in (self.origin_id, self.x_axis_id, self.y_axis_id) if mid not in samples_copy]
        if missing:
            seen = sorted(self.seen_marker_ids)
            seen_msg = ""
            if seen:
                seen_msg = " (seen ids: %s)" % ", ".join(map(str, seen))
            return False, "Missing detections for marker ids: %s%s" % (
                ", ".join(map(str, missing)),
                seen_msg,
            )

        p_origin_center = np.mean(samples_copy[self.origin_id], axis=0)
        p_x_center = np.mean(samples_copy[self.x_axis_id], axis=0)
        p_y_center = np.mean(samples_copy[self.y_axis_id], axis=0)

        x_axis = p_x_center - p_origin_center
        y_axis = p_y_center - p_origin_center

        x_norm = np.linalg.norm(x_axis)
        y_norm = np.linalg.norm(y_axis)
        if x_norm < 1e-6 or y_norm < 1e-6:
            return False, "Marker positions too close; cannot define axes"

        x_axis = x_axis / x_norm
        y_axis = y_axis / y_norm
        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-6:
            return False, "Markers are nearly colinear; cannot define table plane"
        z_axis = z_axis / z_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        corner_signs = {
            "lower_left": (-1.0, -1.0),
            "lower_right": (1.0, -1.0),
            "upper_left": (-1.0, 1.0),
            "upper_right": (1.0, 1.0),
            "center": (0.0, 0.0),
        }
        if self.origin_corner not in corner_signs:
            return False, "Invalid origin_corner: %s" % self.origin_corner
        if self.x_axis_corner not in corner_signs:
            return False, "Invalid x_axis_corner: %s" % self.x_axis_corner
        if self.y_axis_corner not in corner_signs:
            return False, "Invalid y_axis_corner: %s" % self.y_axis_corner

        def corner_position(center, corner_name):
            sign_x, sign_y = corner_signs[corner_name]
            offset = (sign_x * 0.5 * self.marker_size * x_axis) + (sign_y * 0.5 * self.marker_size * y_axis)
            return center + offset

        origin = corner_position(p_origin_center, self.origin_corner)
        corner_x = corner_position(p_x_center, self.x_axis_corner)
        corner_y = corner_position(p_y_center, self.y_axis_corner)

        width = abs(float(np.dot(corner_x - origin, x_axis)))
        height = abs(float(np.dot(corner_y - origin, y_axis)))

        rotation = np.column_stack((x_axis, y_axis, z_axis))
        rotation_4x4 = np.eye(4, dtype=np.float64)
        rotation_4x4[:3, :3] = rotation
        qx, qy, qz, qw = tft.quaternion_from_matrix(rotation_4x4)

        if not self.camera_frame:
            return False, "Camera frame is not known"

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.camera_frame
        transform.child_frame_id = self.table_frame
        transform.transform.translation.x = float(origin[0])
        transform.transform.translation.y = float(origin[1])
        transform.transform.translation.z = float(origin[2])
        transform.transform.rotation.x = float(qx)
        transform.transform.rotation.y = float(qy)
        transform.transform.rotation.z = float(qz)
        transform.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(transform)

        self.last_calibration = transform
        self._store_calibration_params(width, height, transform.header.stamp)
        rospy.loginfo(
            "Published table TF %s -> %s (origin %.3f, %.3f, %.3f)",
            self.camera_frame,
            self.table_frame,
            origin[0],
            origin[1],
            origin[2],
        )
        return True, "Table calibration published"

    def _detection_counts(self):
        with self.samples_lock:
            counts = {
                self.origin_id: len(self.samples.get(self.origin_id, [])),
                self.x_axis_id: len(self.samples.get(self.x_axis_id, [])),
                self.y_axis_id: len(self.samples.get(self.y_axis_id, [])),
            }
        return counts

    def _has_enough_detections(self):
        counts = self._detection_counts()
        return all(count >= self.target_detections for count in counts.values())

    def _store_calibration_params(self, width, height, stamp):
        ns = self.calibration_param_ns.rstrip("/")
        rospy.set_param(ns + "/width_m", width)
        rospy.set_param(ns + "/height_m", height)
        rospy.set_param(ns + "/marker_size_m", float(self.marker_size))
        rospy.set_param(ns + "/origin_corner", self.origin_corner)
        rospy.set_param(ns + "/x_axis_corner", self.x_axis_corner)
        rospy.set_param(ns + "/y_axis_corner", self.y_axis_corner)
        rospy.set_param(ns + "/origin_id", int(self.origin_id))
        rospy.set_param(ns + "/x_axis_id", int(self.x_axis_id))
        rospy.set_param(ns + "/y_axis_id", int(self.y_axis_id))
        rospy.set_param(ns + "/stamp_secs", int(stamp.secs))
        rospy.set_param(ns + "/stamp_nsecs", int(stamp.nsecs))


def main():
    rospy.init_node("aruco_table_calibration")
    rospy.loginfo("Starting aruco table calibration node")
    try:
        ArucoTableCalibrator()
    except RuntimeError as exc:
        rospy.logfatal("Failed to start aruco calibration: %s", exc)
        return
    rospy.spin()


if __name__ == "__main__":
    main()
