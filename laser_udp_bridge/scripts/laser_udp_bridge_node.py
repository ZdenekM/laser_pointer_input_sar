#!/usr/bin/env python3

import json
import math
import os
import socket
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import rospy
import tf2_ros

from nn_laser_spot_tracking.msg import KeypointImage
from std_srvs.srv import Trigger


class CalibrationHttpHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == self.server.pose_path:
            payload = self.server.pose_payload_fn()
            status = 200 if payload.get("ok", False) else 503
            self._send_json(payload, status)
            return
        if path == self.server.point_path:
            payload = self.server.point_payload_fn()
            status = 200 if payload.get("ok", False) else 503
            self._send_json(payload, status)
            return
        if path == self.server.projector_calibration_path:
            payload, status = self.server.projector_calibration_get_fn()
            self._send_json(payload, status)
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path.split("?")[0] != self.server.calibrate_path:
            self.send_error(404, "Not Found")
            return
        payload = self.server.calibrate_fn()
        status = 200 if payload.get("ok", False) else 503
        self._send_json(payload, status)

    def do_PUT(self):
        if self.path.split("?")[0] != self.server.projector_calibration_path:
            self.send_error(404, "Not Found")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            payload = {"ok": False, "error": "Empty request body"}
            self._send_json(payload, 400)
            return
        body = self.rfile.read(content_length)
        payload, status = self.server.projector_calibration_put_fn(body)
        self._send_json(payload, status)

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
    http_projector_calibration_path = rospy.get_param(
        "~http_projector_calibration_path",
        "/projector_calibration",
    )
    calibration_service = rospy.get_param(
        "~calibration_service",
        "/aruco_table_calibration/calibrate",
    )
    calibration_service_timeout = float(rospy.get_param("~calibration_service_timeout", 2.0))
    projector_calibration_path = rospy.get_param(
        "~projector_calibration_path",
        "/data/projector_calibration.json",
    )
    projector_calibration_plane_rms_tolerance_m = float(
        rospy.get_param("~projector_calibration_plane_rms_tolerance_m", 0.01)
    )
    projector_calibration_full3d_min_depth_m = float(
        rospy.get_param("~projector_calibration_full3d_min_depth_m", 0.01)
    )
    projector_calibration_frame_id = "table_frame"

    seq = 0

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    def _validate_finite(name, value):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value for {name}: {value}")

    def _normalize_points_2d(points):
        mean = np.mean(points, axis=0)
        diffs = points - mean
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        mean_dist = float(np.mean(dists))
        if mean_dist <= 1e-12:
            raise ValueError("Degenerate 2D points: mean distance too small")
        scale = math.sqrt(2.0) / mean_dist
        transform = np.array(
            [
                [scale, 0.0, -scale * mean[0]],
                [0.0, scale, -scale * mean[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points_h = np.column_stack([points, np.ones((points.shape[0], 1))])
        normalized = (transform @ points_h.T).T
        return normalized[:, :2], transform

    def _normalize_points_3d(points):
        mean = np.mean(points, axis=0)
        diffs = points - mean
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        mean_dist = float(np.mean(dists))
        if mean_dist <= 1e-12:
            raise ValueError("Degenerate 3D points: mean distance too small")
        scale = math.sqrt(3.0) / mean_dist
        transform = np.array(
            [
                [scale, 0.0, 0.0, -scale * mean[0]],
                [0.0, scale, 0.0, -scale * mean[1]],
                [0.0, 0.0, scale, -scale * mean[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        points_h = np.column_stack([points, np.ones((points.shape[0], 1))])
        normalized = (transform @ points_h.T).T
        return normalized[:, :3], transform

    def _compute_plane_basis(points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, singular_values, v_t = np.linalg.svd(centered, full_matrices=False)
        normal = v_t[2, :]
        plane_rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
        basis_u = v_t[0, :]
        basis_v = v_t[1, :]
        return centroid, basis_u, basis_v, normal, singular_values, plane_rms

    def _compute_homography(src_points, dst_points):
        src_norm, t_src = _normalize_points_2d(src_points)
        dst_norm, t_dst = _normalize_points_2d(dst_points)

        num_points = src_points.shape[0]
        a = np.zeros((num_points * 2, 9), dtype=np.float64)
        for i in range(num_points):
            x, y = src_norm[i]
            u, v = dst_norm[i]
            a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
            a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]
        _, _, v_t = np.linalg.svd(a)
        h = v_t[-1].reshape(3, 3)
        h = np.linalg.inv(t_dst) @ h @ t_src
        if abs(h[2, 2]) > 1e-12:
            h = h / h[2, 2]
        return h

    def _reprojection_error_homography(h, src_points, dst_points):
        src_h = np.column_stack([src_points, np.ones((src_points.shape[0], 1))])
        proj = (h @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        errors = np.sqrt(np.sum((proj - dst_points) ** 2, axis=1))
        return float(np.sqrt(np.mean(errors ** 2)))

    def _compute_projection_matrix(world_points, image_points):
        world_norm, t_world = _normalize_points_3d(world_points)
        image_norm, t_img = _normalize_points_2d(image_points)
        num_points = world_points.shape[0]
        a = np.zeros((num_points * 2, 12), dtype=np.float64)
        for i in range(num_points):
            x, y, z = world_norm[i]
            u, v = image_norm[i]
            a[2 * i] = [0.0, 0.0, 0.0, 0.0, -x, -y, -z, -1.0, v * x, v * y, v * z, v]
            a[2 * i + 1] = [x, y, z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u * z, -u]
        _, _, v_t = np.linalg.svd(a)
        p = v_t[-1].reshape(3, 4)
        p = np.linalg.inv(t_img) @ p @ t_world
        norm = np.linalg.norm(p)
        if norm > 1e-12:
            p = p / norm
        return p

    def _reprojection_error_projection(p, world_points, image_points):
        world_h = np.column_stack([world_points, np.ones((world_points.shape[0], 1))])
        proj = (p @ world_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        errors = np.sqrt(np.sum((proj - image_points) ** 2, axis=1))
        return float(np.sqrt(np.mean(errors ** 2)))

    def _ensure_directory(path):
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_saved_projector_calibration():
        if not os.path.exists(projector_calibration_path):
            return None
        try:
            with open(projector_calibration_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("Saved calibration JSON is not an object")
            return data
        except Exception as exc:
            rospy.logerr("Failed to load projector calibration from %s: %s", projector_calibration_path, exc)
            return None

    def _save_projector_calibration(payload):
        _ensure_directory(projector_calibration_path)
        with open(projector_calibration_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

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

    projector_calibration_lock = threading.Lock()
    latest_projector_calibration = {"payload": _load_saved_projector_calibration()}

    def _validate_projector_calibration_payload(payload):
        if not isinstance(payload, dict):
            raise ValueError("Calibration payload must be a JSON object")
        mode = payload.get("mode", "planar")
        if mode not in ("planar", "full3d"):
            raise ValueError("mode must be 'planar' or 'full3d'")

        if "frame_id" in payload and payload["frame_id"] != projector_calibration_frame_id:
            raise ValueError(f"frame_id must be '{projector_calibration_frame_id}'")

        projector = payload.get("projector")
        if not isinstance(projector, dict):
            raise ValueError("projector must be an object with width_px and height_px")
        width_px = projector.get("width_px")
        height_px = projector.get("height_px")
        if not isinstance(width_px, (int, float)) or not isinstance(height_px, (int, float)):
            raise ValueError("projector.width_px and projector.height_px must be numbers")
        _validate_finite("projector.width_px", float(width_px))
        _validate_finite("projector.height_px", float(height_px))
        if width_px <= 0 or height_px <= 0:
            raise ValueError("projector.width_px and projector.height_px must be positive")

        points = payload.get("points")
        if not isinstance(points, list) or len(points) == 0:
            raise ValueError("points must be a non-empty array")

        min_points = 4 if mode == "planar" else 6
        if len(points) < min_points:
            raise ValueError(f"mode '{mode}' requires at least {min_points} points")

        world_points = []
        image_points = []
        for idx, item in enumerate(points):
            if not isinstance(item, dict):
                raise ValueError(f"points[{idx}] must be an object")
            proj = item.get("projector")
            world = item.get("world")
            if not isinstance(proj, dict) or not isinstance(world, dict):
                raise ValueError(f"points[{idx}] must include projector and world objects")
            for key in ("u", "v"):
                if key not in proj:
                    raise ValueError(f"points[{idx}].projector.{key} is required")
            for key in ("x", "y", "z"):
                if key not in world:
                    raise ValueError(f"points[{idx}].world.{key} is required")
            u = float(proj["u"])
            v = float(proj["v"])
            x = float(world["x"])
            y = float(world["y"])
            z = float(world["z"])
            _validate_finite(f"points[{idx}].projector.u", u)
            _validate_finite(f"points[{idx}].projector.v", v)
            _validate_finite(f"points[{idx}].world.x", x)
            _validate_finite(f"points[{idx}].world.y", y)
            _validate_finite(f"points[{idx}].world.z", z)
            image_points.append([u, v])
            world_points.append([x, y, z])

        return mode, float(width_px), float(height_px), np.array(world_points), np.array(image_points)

    def _projector_calibration_get():
        with projector_calibration_lock:
            payload = latest_projector_calibration["payload"]
        if payload is None:
            return {"ok": False, "error": "No projector calibration available yet"}, 404
        return payload, 200

    def _projector_calibration_put(body):
        try:
            payload_in = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"ok": False, "error": f"Invalid JSON body: {exc}"}, 400
        try:
            mode, width_px, height_px, world_points, image_points = _validate_projector_calibration_payload(
                payload_in
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}, 400

        try:
            centroid, basis_u, basis_v, _normal, singular_values, plane_rms = _compute_plane_basis(world_points)
            if singular_values[0] <= 1e-12 or (singular_values[1] / singular_values[0]) < 1e-3:
                return {
                    "ok": False,
                    "error": "Point set is nearly colinear; cannot compute calibration",
                }, 400
            if mode == "planar":
                if plane_rms > projector_calibration_plane_rms_tolerance_m:
                    return {
                        "ok": False,
                        "error": (
                            "planar mode requires near-planar points; "
                            f"RMS distance to plane {plane_rms:.6f} m exceeds "
                            f"tolerance {projector_calibration_plane_rms_tolerance_m:.6f} m"
                        ),
                    }, 400

                plane_coords = np.column_stack(
                    [
                        (world_points - centroid) @ basis_u,
                        (world_points - centroid) @ basis_v,
                    ]
                )
                h = _compute_homography(plane_coords, image_points)
                rms = _reprojection_error_homography(h, plane_coords, image_points)
                payload_out = {
                    "ok": True,
                    "mode": mode,
                    "frame_id": projector_calibration_frame_id,
                    "projector": {"width_px": width_px, "height_px": height_px},
                    "points_used": int(world_points.shape[0]),
                    "plane_rms_m": plane_rms,
                    "reprojection_rms_px": rms,
                    "H_3x3": h.tolist(),
                }
            else:
                if plane_rms < projector_calibration_full3d_min_depth_m:
                    return {
                        "ok": False,
                        "error": (
                            "full3d mode requires non-coplanar points; "
                            f"RMS distance to plane {plane_rms:.6f} m is below "
                            f"minimum depth {projector_calibration_full3d_min_depth_m:.6f} m"
                        ),
                    }, 400
                p = _compute_projection_matrix(world_points, image_points)
                rms = _reprojection_error_projection(p, world_points, image_points)
                payload_out = {
                    "ok": True,
                    "mode": mode,
                    "frame_id": projector_calibration_frame_id,
                    "projector": {"width_px": width_px, "height_px": height_px},
                    "points_used": int(world_points.shape[0]),
                    "plane_rms_m": plane_rms,
                    "reprojection_rms_px": rms,
                    "P_3x4": p.tolist(),
                }
        except Exception as exc:
            rospy.logerr("Projector calibration failed: %s", exc)
            return {"ok": False, "error": f"Projector calibration failed: {exc}"}, 500

        try:
            _save_projector_calibration(payload_out)
        except Exception as exc:
            rospy.logerr("Failed to save projector calibration: %s", exc)
            return {"ok": False, "error": f"Failed to save projector calibration: {exc}"}, 500

        with projector_calibration_lock:
            latest_projector_calibration["payload"] = payload_out
        return payload_out, 200

    if http_enable:
        def http_thread_fn():
            server = HTTPServer((http_bind, http_port), CalibrationHttpHandler)
            server.pose_path = http_pose_path
            server.point_path = http_point_path
            server.calibrate_path = http_calibrate_path
            server.projector_calibration_path = http_projector_calibration_path
            server.pose_payload_fn = get_calibration_payload
            server.point_payload_fn = get_point_payload
            server.calibrate_fn = trigger_calibration
            server.projector_calibration_get_fn = _projector_calibration_get
            server.projector_calibration_put_fn = _projector_calibration_put
            rospy.loginfo(
                "HTTP endpoints: pose=http://%s:%d%s point=http://%s:%d%s calibrate=http://%s:%d%s projector_calibration=http://%s:%d%s",
                http_bind,
                http_port,
                http_pose_path,
                http_bind,
                http_port,
                http_point_path,
                http_bind,
                http_port,
                http_calibrate_path,
                http_bind,
                http_port,
                http_projector_calibration_path,
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
