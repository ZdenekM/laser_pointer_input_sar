#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:48:52 2022

@author: tori
with ideas taken from https://github.com/vvasilo/yolov3_pytorch_ros/blob/master/src/yolov3_pytorch_ros/detector.py
"""
import os
from collections import deque
import threading
import numpy as np


# Pytorch stuff
import torch
import torchvision.transforms

#Opencv stuff
import cv2
from cv_bridge import CvBridge, CvBridgeError

# ROS imports
import rospy
import message_filters
import tf2_ros
import tf.transformations
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo

from nn_laser_spot_tracking.msg import KeypointImage
import image_geometry

class AlphaBetaTracker2D:
    """Alpha-beta tracker for 2D pixel positions with gating and short prediction windows."""

    def __init__(self, alpha, beta, gate_px, max_prediction_frames, reset_on_jump=True, min_dt=1e-3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("tracking_alpha must be in (0, 1]")
        if beta < 0.0:
            raise ValueError("tracking_beta must be >= 0")
        if gate_px <= 0.0:
            raise ValueError("tracking_gate_px must be > 0")
        if max_prediction_frames < 0:
            raise ValueError("tracking_max_prediction_frames must be >= 0")
        if min_dt <= 0.0:
            raise ValueError("tracking_min_dt must be > 0")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gate_px = float(gate_px)
        self.max_prediction_frames = int(max_prediction_frames)
        self.reset_on_jump = bool(reset_on_jump)
        self.min_dt = float(min_dt)

        self.initialized = False
        self.pos = None
        self.vel = np.zeros(2, dtype=np.float64)
        self.last_stamp = None
        self.missed_frames = 0

    def reset(self):
        self.initialized = False
        self.pos = None
        self.vel = np.zeros(2, dtype=np.float64)
        self.last_stamp = None
        self.missed_frames = 0

    def update(self, measurement, stamp):
        if measurement is None:
            if not self.initialized:
                return None
            if self.missed_frames >= self.max_prediction_frames:
                return None
            dt = self._dt(stamp)
            self.pos = self.pos + self.vel * dt
            self.last_stamp = stamp
            self.missed_frames += 1
            return {"pos": self.pos.copy(), "predicted": True, "reset": False}

        measurement = np.array(measurement, dtype=np.float64)
        if not self.initialized:
            self.pos = measurement
            self.vel = np.zeros(2, dtype=np.float64)
            self.last_stamp = stamp
            self.missed_frames = 0
            self.initialized = True
            return {"pos": self.pos.copy(), "predicted": False, "reset": True}

        dt = self._dt(stamp)
        pred_pos = self.pos + self.vel * dt
        residual = measurement - pred_pos

        if self.reset_on_jump and np.linalg.norm(residual) > self.gate_px:
            self.pos = measurement
            self.vel = np.zeros(2, dtype=np.float64)
            self.last_stamp = stamp
            self.missed_frames = 0
            return {"pos": self.pos.copy(), "predicted": False, "reset": True}

        self.pos = pred_pos + self.alpha * residual
        self.vel = self.vel + (self.beta / dt) * residual
        self.last_stamp = stamp
        self.missed_frames = 0
        return {"pos": self.pos.copy(), "predicted": False, "reset": False}

    def _dt(self, stamp):
        if self.last_stamp is None:
            return self.min_dt
        dt = (stamp - self.last_stamp).to_sec()
        if dt <= 0.0:
            dt = self.min_dt
        return dt

class AlphaBetaTracker1D:
    """Alpha-beta tracker for scalar values with gating and short prediction windows."""

    def __init__(self, alpha, beta, gate, max_prediction_frames, reset_on_jump=True, min_dt=1e-3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("depth_tracking_alpha must be in (0, 1]")
        if beta < 0.0:
            raise ValueError("depth_tracking_beta must be >= 0")
        if gate <= 0.0:
            raise ValueError("depth_tracking_gate_m must be > 0")
        if max_prediction_frames < 0:
            raise ValueError("depth_tracking_max_prediction_frames must be >= 0")
        if min_dt <= 0.0:
            raise ValueError("depth_tracking_min_dt must be > 0")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gate = float(gate)
        self.max_prediction_frames = int(max_prediction_frames)
        self.reset_on_jump = bool(reset_on_jump)
        self.min_dt = float(min_dt)

        self.initialized = False
        self.value = None
        self.vel = 0.0
        self.last_stamp = None
        self.missed_frames = 0

    def reset(self):
        self.initialized = False
        self.value = None
        self.vel = 0.0
        self.last_stamp = None
        self.missed_frames = 0

    def update(self, measurement, stamp):
        if measurement is None:
            if not self.initialized:
                return None
            if self.missed_frames >= self.max_prediction_frames:
                return None
            dt = self._dt(stamp)
            self.value = self.value + self.vel * dt
            self.last_stamp = stamp
            self.missed_frames += 1
            return {"value": float(self.value), "predicted": True, "reset": False}

        measurement = float(measurement)
        if not self.initialized:
            self.value = measurement
            self.vel = 0.0
            self.last_stamp = stamp
            self.missed_frames = 0
            self.initialized = True
            return {"value": float(self.value), "predicted": False, "reset": True}

        dt = self._dt(stamp)
        pred_val = self.value + self.vel * dt
        residual = measurement - pred_val

        if self.reset_on_jump and abs(residual) > self.gate:
            self.value = measurement
            self.vel = 0.0
            self.last_stamp = stamp
            self.missed_frames = 0
            return {"value": float(self.value), "predicted": False, "reset": True}

        self.value = pred_val + self.alpha * residual
        self.vel = self.vel + (self.beta / dt) * residual
        self.last_stamp = stamp
        self.missed_frames = 0
        return {"value": float(self.value), "predicted": False, "reset": False}

    def _dt(self, stamp):
        if self.last_stamp is None:
            return self.min_dt
        dt = (stamp - self.last_stamp).to_sec()
        if dt <= 0.0:
            dt = self.min_dt
        return dt

class getCameraInfo:
    
    cam_info = {}
    
    def __init__(self, image_info_topic):
        self.sub = rospy.Subscriber(image_info_topic, CameraInfo, self.__callback)
        rospy.loginfo("waiting for camerainfo...")
        rospy.wait_for_message(image_info_topic, CameraInfo, timeout=10)
        rospy.loginfo("... camerainfo arrived")

    def __callback(self, msg):
        self.cam_info["width"] = msg.width
        self.cam_info["height"] = msg.height
        self.sub.unregister()

class GenericModel : 
    model = None
    device = None
    _transform_chain = None
    tensor_images = []
    
    def __init__(self):
        pass
        
    def initialize(self, model_path_name, yolo_path="", device='gpu'):
        pass
        
    def infer(self, cv_image_input):
        pass
    
class NoYoloModel(GenericModel) : 
    
    def __init__(self):
        super().__init__()
        
    def __process_image(self, cv_image_input):
        
        #pil_image_input = PILImage.fromarray(self.cv_image_input) #img as opencv
        #pil_image_input.show()
        
        self.tensor_images = [(self._transform_chain(cv_image_input).to(self.device))]        

        #beh = torchvision.transforms.functional.to_pil_image(self.tensor_images[0], "RGB")
       # beh.show()    
        
    def initialize(self, model_path_name, yolo_path="", device='gpu'):
        
        if device == 'cpu' :
            self.device = torch.device('cpu')
            self.model = torch.load(model_path_name, map_location=torch.device('cpu'))
     
        elif device == 'gpu' :
            self.device = torch.device('cuda')
            self.model = torch.load(model_path_name)
       
        else:
            raise Exception("Invalid device")   
            
        # wants a tensor
        self._transform_chain = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            ])
        
        self.model.eval()
        self.model.to(self.device)
        
    def infer(self, cv_image_input):
        
        self.__process_image(cv_image_input)
        out = self.model(self.tensor_images)[0]
        
        return out
    
    
class YoloModel(GenericModel) : 
    
    def __init__(self):
        super().__init__()
        
    def initialize(self, model_path, yolo_path="", device='gpu'):

        if device == 'cpu' :
            self.device = torch.device('cpu')
            self.model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True, device='cpu')

        elif device == 'gpu' :
            self.device = torch.device('cuda:0')
            self.model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True, device='cuda:0')
       
        else:
            raise Exception("Invalid device " + device)   
            
        # wants a tensor?
        self._transform_chain = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            ])
        
        self.model.eval()
        self.model.to(self.device)
        
    def infer(self, cv_image_input):
        
        self.tensor_images = [(self._transform_chain(cv_image_input).to(self.device))]        
        
        out_yolo = self.model(cv_image_input)
        
        #out_yolo.print()
        #print(out_yolo.xyxy)
        
        self.out = {
            'boxes': torch.tensor(torch.zeros(len(out_yolo.xyxy[0]), 4), device=self.device),
            'labels': torch.tensor(torch.zeros(len(out_yolo.xyxy[0])), dtype=torch.int32, device=self.device),
            'scores': torch.tensor(torch.zeros(len(out_yolo.xyxy[0])), device=self.device)
        }

        # xyxy has is array with elements of format : 
        # xmin    ymin    xmax   ymax  confidence  class
        for i in range(0, len(out_yolo.xyxy[0])) :
            self.out['boxes'][i] = out_yolo.xyxy[0][i][0:4]
            self.out['scores'][i] = out_yolo.xyxy[0][i][4]
            self.out['labels'][i] = out_yolo.xyxy[0][i][5].int()
        
        #print(self.out)

        return self.out

class DetectorManager():
    
    ros_image_input = ros_image_output = None
    cv_image_input = cv_image_output = np.zeros((100,100,3), np.uint8)
    new_image = False
    model_helper = None
    out = {'scores' : []}
    best_index = -1
    inference_stamp = None
    
    def __init__(self):
        
        self.inference_stamp = rospy.Time.now()

        ### Input Params
        model_path = rospy.get_param('~model_path')
        model_name = rospy.get_param('~model_name')
        yolo_path = rospy.get_param('~yolo_path', "ultralytics/yolov5")

        camera_image_topic = rospy.get_param('~camera_image_topic')
        transport_param = rospy.get_param('~transport', 'raw')
        if transport_param != "raw":
            raise ValueError("Only raw image transport is supported (transport=raw)")
        ros_image_input_topic = camera_image_topic

        ## Detection Params
        self.detection_confidence_threshold = rospy.get_param('~detection_confidence_threshold', 0.55)

        ## Tracking Params
        self.tracking_enable = rospy.get_param('~tracking_enable', True)
        if not isinstance(self.tracking_enable, bool):
            raise ValueError("tracking_enable must be a boolean")
        tracking_alpha = rospy.get_param('~tracking_alpha', 0.85)
        tracking_beta = rospy.get_param('~tracking_beta', 0.005)
        tracking_gate_px = rospy.get_param('~tracking_gate_px', 30.0)
        tracking_max_prediction_frames = rospy.get_param('~tracking_max_prediction_frames', 6)
        tracking_reset_on_jump = rospy.get_param('~tracking_reset_on_jump', True)
        self.tracking_predicted_confidence_scale = rospy.get_param('~tracking_predicted_confidence_scale', 1.0)

        ## Depth filter Params
        if self.tracking_predicted_confidence_scale <= 0.0:
            raise ValueError("tracking_predicted_confidence_scale must be > 0")
        self.depth_history_max_age = float(rospy.get_param("~depth_history_max_age", 1.0))
        if self.depth_history_max_age <= 0.0:
            raise ValueError("depth_history_max_age must be > 0")
        self.depth_frame_history_size = int(rospy.get_param("~depth_frame_history_size", 7))
        if self.depth_frame_history_size < 1:
            raise ValueError("depth_frame_history_size must be >= 1")
        self.depth_tracking_enable = rospy.get_param("~depth_tracking_enable", True)
        if not isinstance(self.depth_tracking_enable, bool):
            raise ValueError("depth_tracking_enable must be a boolean")
        depth_tracking_alpha = rospy.get_param("~depth_tracking_alpha", 0.7)
        depth_tracking_beta = rospy.get_param("~depth_tracking_beta", 0.02)
        depth_tracking_gate_m = rospy.get_param("~depth_tracking_gate_m", 0.2)
        depth_tracking_max_prediction_frames = rospy.get_param("~depth_tracking_max_prediction_frames", 2)
        depth_tracking_reset_on_jump = rospy.get_param("~depth_tracking_reset_on_jump", True)
        self.depth_fallback_plane_enable = rospy.get_param("~depth_fallback_plane_enable", True)
        if not isinstance(self.depth_fallback_plane_enable, bool):
            raise ValueError("depth_fallback_plane_enable must be a boolean")
        self.depth_fallback_plane_frame = rospy.get_param("~depth_fallback_plane_frame", "table_frame")
        self.depth_fallback_plane_timeout = float(rospy.get_param("~depth_fallback_plane_timeout", 0.05))
        if self.depth_fallback_plane_timeout <= 0.0:
            raise ValueError("depth_fallback_plane_timeout must be > 0")
        self.require_table_calibration = rospy.get_param("~require_table_calibration", True)
        if not isinstance(self.require_table_calibration, bool):
            raise ValueError("require_table_calibration must be a boolean")
        self.table_frame = rospy.get_param("~table_frame", "table_frame")
        self.calibration_param_ns = rospy.get_param("~calibration_param_ns", "/table_calibration")
        
        ### Output Params
        pub_out_keypoint_topic = rospy.get_param('~pub_out_keypoint_topic', "/detection_output_keypoint")
        self.pub_out_images = rospy.get_param('~pub_out_images', True)
        self.pub_out_all_keypoints = rospy.get_param('~pub_out_images_all_keypoints', False)
        pub_out_images_topic = rospy.get_param('~pub_out_images_topic', "/detection_output_img")
        self.laser_spot_frame = rospy.get_param('~laser_spot_frame', 'laser_spot_frame')
        
        #camera_info_topic = rospy.get_param('~camera_info_topic', '/D435_head_camera/color/camera_info')
        #getCameraInfo(camera_info_topic)
        #self.cam_info = getCameraInfo.cam_info
        
        if (model_name.startswith('yolo')) :
            self.model_helper = YoloModel()
        
        else:
            self.model_helper = NoYoloModel()
        
        ############ PYTHORCH STUFF
        model_path_name = os.path.join(model_path, model_name)
        
        rospy.loginfo(f"Using model {model_path_name}")
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            rospy.loginfo("CUDA available, use GPU")
            self.model_helper.initialize(model_path_name, yolo_path, 'gpu')

        else:
            self.device = torch.device('cpu')
            rospy.loginfo("CUDA not available, use CPU") 
            self.model_helper.initialize(model_path_name, yolo_path, 'cpu')
        
        ############ ROS STUFF
        
        self.bridge = CvBridge()
        depth_image_topic = rospy.get_param('~depth_image_topic')
        camera_info_topic = rospy.get_param('~camera_info_topic')

        self.cam_model = image_geometry.PinholeCameraModel()

        image_sub = message_filters.Subscriber(ros_image_input_topic, ROSImage, queue_size=1)
        depth_sub = message_filters.Subscriber(depth_image_topic, ROSImage, queue_size=1)

        try:
            self.camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=10.0)
        except rospy.ROSException as exc:
            raise RuntimeError(
                "CameraInfo not received on %s: %s" % (camera_info_topic, exc)
            )
        self.cam_model.fromCameraInfo(self.camera_info_msg)

        sync = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 5, 0.1)
        sync.registerCallback(self.__sync_clbk)

        self.keypoint_pub = rospy.Publisher(pub_out_keypoint_topic, KeypointImage, queue_size=10)
        if self.pub_out_images:
            self.image_pub = rospy.Publisher(pub_out_images_topic, ROSImage, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.depth_header = None
        self.frame_queue = deque(maxlen=1)
        self.queue_lock = threading.Lock()
        self.depth_frame_history = deque(maxlen=self.depth_frame_history_size)
        self.last_label = 0
        self.last_confidence = 0.0

        self.tracker = None
        if self.tracking_enable:
            self.tracker = AlphaBetaTracker2D(
                tracking_alpha,
                tracking_beta,
                tracking_gate_px,
                tracking_max_prediction_frames,
                tracking_reset_on_jump,
            )
        self.depth_tracker = None
        if self.depth_tracking_enable:
            self.depth_tracker = AlphaBetaTracker1D(
                depth_tracking_alpha,
                depth_tracking_beta,
                depth_tracking_gate_m,
                depth_tracking_max_prediction_frames,
                depth_tracking_reset_on_jump,
            )


    def __sync_clbk(self, rgb_msg, depth_msg):
        rospy.logdebug(
            "Synced RGB/Depth stamps rgb=%s depth=%s",
            str(rgb_msg.header.stamp),
            str(depth_msg.header.stamp),
        )
        with self.queue_lock:
            self.frame_queue.append((rgb_msg, depth_msg))
       
    def infer(self):
        
        with self.queue_lock:
            if len(self.frame_queue) == 0:
                if self.ros_image_input is None:
                    rospy.loginfo_throttle(5.0, "Waiting for synced RGB/Depth messages...")
                    return False
                rospy.loginfo_throttle(2.0, "No new synced frame yet; skipping publish")
                return False
            ros_image_input, depth_msg = self.frame_queue.pop()

        if ros_image_input.encoding not in ("bgra8", "rgb8"):
            raise ValueError(
                f"Unsupported color image encoding '{ros_image_input.encoding}'. Expected bgra8 or rgb8."
            )

        try:
            cv_image_input = self.bridge.imgmsg_to_cv2(ros_image_input, desired_encoding="rgb8")
        except CvBridgeError as e:
            rospy.logerror(e)
            return False

        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerror(e)
            return False

        self.cv_image_input = cv_image_input
        self.ros_image_input = ros_image_input
        self.depth_image = depth_cv
        self.depth_header = depth_msg.header
        self.__append_depth_frame(depth_cv, depth_msg.header.stamp)

        if self.ros_image_input is None:
            rospy.loginfo_throttle(5.0, "Waiting for synced RGB/Depth messages...")
            return False

        if self.require_table_calibration:
            ready, error = self.__table_calibration_ready(self.ros_image_input.header.stamp)
            if not ready:
                rospy.logwarn_throttle(
                    5.0,
                    "Table calibration required but missing: %s",
                    error,
                )
                if self.tracker is not None:
                    self.tracker.reset()
                if self.depth_tracker is not None:
                    self.depth_tracker.reset()
                self.__reset_depth_history()
                return False
        
        with torch.no_grad():
            
            #tic = rospy.Time().now()
            #tic_py = time.time()
            self.out = self.model_helper.infer(self.cv_image_input)
            #self.out = non_max_suppression(out, 80, self.confidence_th, self.nms_th)
        
            #toc = rospy.Time().now()
            #toc_py = time.time()
            #rospy.loginfo ('Inference time: %s s', (toc-tic).to_sec())
            #rospy.loginfo ('Inference time py: %s s', toc_py-tic_py )
            #rospy.loginfo ('%s', toc_py-tic_py )

            #images[0] = images[0].detach().cpu()
        
        if (len(self.out['scores']) == 0):
            rospy.loginfo("No detections in this frame (scores empty)")
            self.inference_stamp = self.ros_image_input.header.stamp
            self.__publish_tracking(self.inference_stamp, None)
            return False
        
        #IDK if the best box is always the first one, so lets the argmax
        self.best_index = int(torch.argmax(self.out['scores']))
        best_score = float(self.out['scores'][self.best_index].item())
        rospy.loginfo("Detections=%d, best_score=%.3f, threshold=%.3f",
                      len(self.out['scores']), best_score, self.detection_confidence_threshold)
        
        #show_image_with_boxes(img, self.out['boxes'][self.best_index], self.out['labels'][self.best_index])
        
        # Keep ROS timestamps aligned with the source image
        self.inference_stamp = self.ros_image_input.header.stamp
        if best_score < self.detection_confidence_threshold:
            rospy.loginfo("Best score %.3f below threshold %.3f; skipping measurement",
                          best_score, self.detection_confidence_threshold)
            self.__publish_tracking(self.inference_stamp, None)
            return True

        measurement = self.__box_center(self.out['boxes'][self.best_index])
        self.__publish_tracking(
            self.inference_stamp,
            measurement,
            self.out['labels'][self.best_index],
            self.out['scores'][self.best_index],
            self.out['boxes'][self.best_index],
            self.out['boxes'],
            self.out['labels'],
        )
        return True

    def __publish_tracking(self, stamp, measurement, label=None, score=None, box=None, boxes=None, labels=None):
        tracking = self.__update_tracking(stamp, measurement, label, score)
        if tracking is None:
            self.__reset_depth_history()
            if self.pub_out_images:
                self.__pubImageWithRectangle()
            return
        xyz, depth_assumed_plane, depth_predicted = self.__compute_xyz(tracking["pixel"], stamp)
        predicted = bool(tracking["predicted"] or depth_predicted)
        kp_msg = self.__pubKeypoint(
            stamp,
            tracking["pixel"],
            tracking["confidence"],
            tracking["label"],
            predicted,
            depth_assumed_plane,
        )
        if kp_msg is None:
            if self.tracker is not None:
                self.tracker.reset()
            self.__reset_depth_history()
            return
        self.__publish_tf(xyz, stamp)

        if self.pub_out_images:
            if measurement is None:
                self.__pubImageWithRectangle(
                    tracking_pixel=tracking["pixel"],
                    tracking_predicted=tracking["predicted"],
                )
            elif self.pub_out_all_keypoints and boxes is not None and labels is not None:
                self.__pubImageWithAllRectangles(
                    boxes,
                    labels,
                    tracking_pixel=tracking["pixel"],
                    tracking_predicted=tracking["predicted"],
                )
            else:
                self.__pubImageWithRectangle(
                    box,
                    score,
                    label,
                    tracking_pixel=tracking["pixel"],
                    tracking_predicted=tracking["predicted"],
                )

    def __update_tracking(self, stamp, measurement, label, score):
        if measurement is None:
            if self.tracker is None:
                return None
            was_initialized = self.tracker.initialized
            result = self.tracker.update(None, stamp)
            if result is None:
                if was_initialized:
                    rospy.loginfo_throttle(
                        2.0,
                        "Tracking lost: exceeded max_prediction_frames=%d",
                        self.tracker.max_prediction_frames,
                    )
                return None
            if result["reset"]:
                self.__reset_depth_history()
            confidence = self.last_confidence * self.tracking_predicted_confidence_scale
            return {
                "pixel": result["pos"],
                "predicted": True,
                "confidence": confidence,
                "label": self.last_label,
            }

        if score is None or label is None:
            raise ValueError("Measurement provided without label/score")
        confidence = float(score.item()) if torch.is_tensor(score) else float(score)
        self.last_label = label
        self.last_confidence = confidence

        if self.tracker is None:
            return {
                "pixel": measurement,
                "predicted": False,
                "confidence": confidence,
                "label": label,
            }

        was_initialized = self.tracker.initialized
        result = self.tracker.update(measurement, stamp)
        if result is None:
            return None
        if result["reset"]:
            if was_initialized:
                rospy.loginfo_throttle(
                    2.0,
                    "Tracking reset on jump: gate_px=%.1f",
                    self.tracker.gate_px,
                )
            self.__reset_depth_history()
        return {
            "pixel": result["pos"],
            "predicted": result["predicted"],
            "confidence": confidence,
            "label": label,
        }

    def __reset_depth_history(self):
        self.depth_frame_history.clear()

    def __append_depth_frame(self, depth_image, stamp):
        self.depth_frame_history.append((stamp.to_sec(), depth_image))
        self.__purge_depth_frame_history(stamp)

    def __purge_depth_frame_history(self, stamp):
        now_sec = stamp.to_sec()
        while self.depth_frame_history and (now_sec - self.depth_frame_history[0][0]) > self.depth_history_max_age:
            self.depth_frame_history.popleft()

    def __collect_depth_samples(self, depth_image, min_u, max_u, min_v, max_v):
        samples = []
        for vv in range(min_v, max_v + 1):
            for uu in range(min_u, max_u + 1):
                if depth_image.dtype == np.uint16:
                    d = float(depth_image[vv, uu]) * 0.001
                    if d <= 0:
                        continue
                else:
                    d = float(depth_image[vv, uu])
                    if not np.isfinite(d) or d <= 0:
                        continue
                samples.append(d)
        return samples

    def __table_calibration_ready(self, stamp):
        ns = self.calibration_param_ns.rstrip("/")
        width = rospy.get_param(ns + "/width_m", None)
        height = rospy.get_param(ns + "/height_m", None)
        if width is None or height is None:
            return False, "Table calibration parameters missing"
        try:
            self.tf_buffer.lookup_transform(
                self.table_frame,
                self.ros_image_input.header.frame_id,
                stamp,
                rospy.Duration(0.05),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            return False, f"Table calibration TF missing: {exc}"
        return True, None

    def __fallback_to_table_plane(self, pixel, stamp):
        if not self.depth_fallback_plane_enable:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.depth_fallback_plane_frame,
                self.ros_image_input.header.frame_id,
                stamp,
                rospy.Duration(self.depth_fallback_plane_timeout),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.loginfo_throttle(
                5.0,
                "TF lookup failed for %s->%s at stamp %s: %s",
                self.depth_fallback_plane_frame,
                self.ros_image_input.header.frame_id,
                str(stamp),
                str(exc),
            )
            return None

        ray = np.array(self.cam_model.projectPixelTo3dRay(pixel), dtype=np.float64)
        ray_norm = np.linalg.norm(ray)
        if ray_norm <= 1e-9:
            rospy.loginfo_throttle(5.0, "Invalid camera ray norm for fallback; skipping")
            return None
        ray = ray / ray_norm

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot_matrix = tf.transformations.quaternion_matrix(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        )[:3, :3]
        origin_plane = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
        dir_plane = rot_matrix.dot(ray)
        if abs(dir_plane[2]) < 1e-6:
            rospy.loginfo_throttle(5.0, "Ray parallel to table plane; skipping fallback")
            return None
        t_param = -origin_plane[2] / dir_plane[2]
        if t_param <= 0.0:
            rospy.loginfo_throttle(5.0, "Table plane intersection behind camera; skipping fallback")
            return None
        point_plane = origin_plane + t_param * dir_plane
        point_cam = rot_matrix.T.dot(point_plane - origin_plane)
        return (float(point_cam[0]), float(point_cam[1]), float(point_cam[2]))

    def __box_center(self, box):
        if box is None:
            return None
        u = round(box[0].item() + (box[2].item() - box[0].item()) / 2)
        v = round(box[1].item() + (box[3].item() - box[1].item()) / 2)
        return (u, v)

    def __draw_tracking_marker(self, image, pixel, predicted):
        if pixel is None:
            return
        height, width = image.shape[:2]
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        color = (255, 255, 0) if predicted else (0, 255, 0)
        size = 6
        thickness = 2
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)

    def __pubImageWithRectangle(self, box=None, score=None, label=None, tracking_pixel=None, tracking_predicted=False):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.model_helper.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = np.array(self.cv_image_output, copy=True, order="C")
        
        if (not box == None) and (score is not None):
            score_val = float(score.item()) if torch.is_tensor(score) else float(score)
        else:
            score_val = None

        if (not box == None) and (score_val is not None) and (score_val > self.detection_confidence_threshold):
            cv2.rectangle(self.cv_image_output, 
                          (round(box[0].item()), round(box[1].item())),
                          (round(box[2].item()), round(box[3].item())),
                          (255,0,0), 3)

        self.__draw_tracking_marker(self.cv_image_output, tracking_pixel, tracking_predicted)
        
        #if label:
            #cv2.putText(self.cv_image_output, str(label.item()), (round(box[0].item()), round(box[3].item()+10)), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_imgmsg(self.cv_image_output, encoding="rgb8")
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)
        
    def __pubImageWithAllRectangles(self, box=None, label=None, tracking_pixel=None, tracking_predicted=False):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.model_helper.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = np.array(self.cv_image_output, copy=True, order="C")
        
        if not box == None:
            i = 0
            for b in box:
                cv2.rectangle(self.cv_image_output, 
                              (round(b[0].item()), round(b[1].item())),
                              (round(b[2].item()), round(b[3].item())),
                              (255,0,0), 2)
        
                if not label == None:
                        cv2.putText(self.cv_image_output, str(label[i].item()), (round(b[0].item()), round(b[3].item()+10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                i = i+1

        self.__draw_tracking_marker(self.cv_image_output, tracking_pixel, tracking_predicted)
        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_imgmsg(self.cv_image_output, encoding="rgb8")
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)

            
    """
    pixel coordinates are rounded and validated against image bounds.
    """
    def __pubKeypoint(self, stamp, pixel=None, confidence=0.0, label=0, predicted=False, depth_assumed_plane=False):

        msg = KeypointImage()
        if self.ros_image_input is None:
            return None
        msg.header.frame_id = self.ros_image_input.header.frame_id
        msg.header.seq = self.ros_image_input.header.seq
        msg.header.stamp = stamp
        msg.image_width = self.cv_image_input.shape[1]
        msg.image_height = self.cv_image_input.shape[0]

        if pixel is None:
            msg.x_pixel = 0
            msg.y_pixel = 0
            msg.label = 0
            msg.confidence = 0.0
            msg.predicted = False
            msg.depth_assumed_plane = False
            self.keypoint_pub.publish(msg)
            return msg

        x, y = int(round(pixel[0])), int(round(pixel[1]))
        if x < 0 or y < 0 or x >= msg.image_width or y >= msg.image_height:
            rospy.logwarn(
                "Keypoint pixel out of bounds (x=%d, y=%d, w=%d, h=%d); dropping output",
                x, y, msg.image_width, msg.image_height,
            )
            return None

        msg.x_pixel = x
        msg.y_pixel = y
        msg.label = int(label.item()) if torch.is_tensor(label) else int(label)
        msg.confidence = float(confidence)
        msg.predicted = bool(predicted)
        msg.depth_assumed_plane = bool(depth_assumed_plane)

        self.keypoint_pub.publish(msg)
        return msg

    def __compute_xyz(self, pixel, stamp):
        if not hasattr(self, "depth_image") or self.depth_image is None:
            rospy.loginfo_throttle(5.0, "No depth image available; skipping xyz/TF publish")
            return None, False, False
        if pixel is None:
            return None, False, False
        u = int(round(pixel[0]))
        v = int(round(pixel[1]))
        window_radius = 2
        img_h, img_w = self.depth_image.shape[:2]
        if u < 0 or v < 0 or u >= img_w or v >= img_h:
            rospy.loginfo_throttle(
                5.0,
                "Keypoint out of depth bounds (u=%d, v=%d, w=%d, h=%d); skipping xyz/TF",
                u, v, img_w, img_h,
            )
            return None, False, False

        min_u = max(0, u - window_radius)
        max_u = min(img_w - 1, u + window_radius)
        min_v = max(0, v - window_radius)
        max_v = min(img_h - 1, v + window_radius)

        self.__purge_depth_frame_history(stamp)
        depths = []
        for _, frame in self.depth_frame_history:
            depths.extend(self.__collect_depth_samples(frame, min_u, max_u, min_v, max_v))

        depth_meas = None
        if len(depths) >= 3:
            depth_meas = float(np.median(depths))

        depth_value = None
        depth_predicted = False
        if self.depth_tracker is not None:
            result = self.depth_tracker.update(depth_meas, stamp)
            if result is not None:
                depth_value = float(result["value"])
                depth_predicted = bool(result["predicted"])
        else:
            depth_value = depth_meas

        if depth_value is None:
            xyz = self.__fallback_to_table_plane((u, v), stamp)
            if xyz is not None:
                return xyz, True, False
            rospy.loginfo_throttle(
                5.0,
                "Insufficient valid depth samples (%d) around (u=%d, v=%d); skipping xyz/TF",
                len(depths), u, v,
            )
            return None, False, False

        ray = self.cam_model.projectPixelTo3dRay((u, v))
        scale = depth_value / float(ray[2])
        x = float(ray[0] * scale)
        y = float(ray[1] * scale)
        z = depth_value
        return (x, y, z), False, depth_predicted

    def __publish_tf(self, xyz, stamp):
        if xyz is None:
            return
        frames = [self.laser_spot_frame + "_raw", self.laser_spot_frame]
        for child in frames:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.ros_image_input.header.frame_id
            t.child_frame_id = child
            t.transform.translation.x = xyz[0]
            t.transform.translation.y = xyz[1]
            t.transform.translation.z = xyz[2]
            t.transform.rotation.w = 1.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            self.tf_broadcaster.sendTransform(t)
        

if __name__=="__main__":
    # Initialize node
    rospy.init_node("tracking_2D")

    rospy.loginfo("Starting node...")
    
    rate_param = rospy.get_param('~rate', 5)

    # Define detector object
    dm = DetectorManager()

    rate = rospy.Rate(rate_param) # ROS Rate
    
    while not rospy.is_shutdown():
        new_infer = dm.infer()
        rate.sleep()
    

    
