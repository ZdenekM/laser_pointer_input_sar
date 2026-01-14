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
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage as ROSCompressedImage
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import CameraInfo

from nn_laser_spot_tracking.msg import KeypointImage
import image_geometry

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
            self.device = torch.device('cuda')
            self.model = torch.hub.load(yolo_path, 'custom', source='local', path=model_path, force_reload=True, device='cuda')
       
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
        self.camera_image_transport = rospy.get_param('~transport', 'compressed')
        if self.camera_image_transport == "raw":
            ros_image_input_topic = camera_image_topic
        else:
            ros_image_input_topic = camera_image_topic + '/' + self.camera_image_transport

        ## Detection Params
        self.detection_confidence_threshold = rospy.get_param('~detection_confidence_threshold', 0.55)
        
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

        if self.camera_image_transport == "compressed":
            image_sub = message_filters.Subscriber(ros_image_input_topic, ROSCompressedImage, queue_size=1)
        else:
            image_sub = message_filters.Subscriber(ros_image_input_topic, ROSImage, queue_size=1)

        depth_sub = message_filters.Subscriber(depth_image_topic, ROSImage, queue_size=1)
        info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo, queue_size=1)

        sync = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, info_sub], 5, 0.1)
        sync.registerCallback(self.__sync_clbk)

        self.keypoint_pub = rospy.Publisher(pub_out_keypoint_topic, KeypointImage, queue_size=10)
        if self.pub_out_images:
            self.image_pub = rospy.Publisher(pub_out_images_topic+"/compressed", ROSCompressedImage, queue_size=10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.cam_model = image_geometry.PinholeCameraModel()
        self.depth_header = None
        self.frame_queue = deque(maxlen=1)
        self.queue_lock = threading.Lock()


    def __sync_clbk(self, rgb_msg, depth_msg, info_msg):
        rospy.loginfo("Synced RGB/Depth/Info stamps rgb=%s depth=%s", str(rgb_msg.header.stamp), str(depth_msg.header.stamp))
        try:
            if self.camera_image_transport == "compressed":
                cv_image_input = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            else:
                cv_image_input = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        except CvBridgeError as e:
            rospy.logerror(e)
            return

        try:
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerror(e)
            return

        with self.queue_lock:
            self.frame_queue.append((cv_image_input, rgb_msg, depth_cv, depth_msg.header, info_msg))
        rospy.logdebug("Buffered rgb/depth at %s, rgb shape=%s depth shape=%s", str(rgb_msg.header.stamp), cv_image_input.shape, depth_cv.shape)
       
    def infer(self):
        
        with self.queue_lock:
            if len(self.frame_queue) == 0:
                if self.ros_image_input is None:
                    rospy.loginfo_throttle(5.0, "Waiting for synced RGB/Depth/Info messages...")
                    return False
                rospy.loginfo_throttle(2.0, "No new synced frame yet; skipping publish")
                return False
            cv_image_input, ros_image_input, depth_cv, depth_header, info_msg = self.frame_queue.pop()

        self.cv_image_input = cv_image_input
        self.ros_image_input = ros_image_input
        self.depth_image = depth_cv
        self.depth_header = depth_header
        self.cam_model.fromCameraInfo(info_msg)

        if self.ros_image_input is None:
            rospy.loginfo_throttle(5.0, "Waiting for synced RGB/Depth/Info messages...")
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
            self.__pubROS(self.inference_stamp)
            return False
        
        #IDK if the best box is always the first one, so lets the argmax
        self.best_index = int(torch.argmax(self.out['scores']))
        best_score = float(self.out['scores'][self.best_index].item())
        rospy.loginfo("Detections=%d, best_score=%.3f, threshold=%.3f",
                      len(self.out['scores']), best_score, self.detection_confidence_threshold)
        
        #show_image_with_boxes(img, self.out['boxes'][self.best_index], self.out['labels'][self.best_index])
        
        # Keep ROS timestamps aligned with the source image
        self.inference_stamp = self.ros_image_input.header.stamp
        self.__pubROS(self.inference_stamp, self.best_index, self.out['boxes'], self.out['scores'], self.out['labels'])
        
        return True
    
    def __pubROS(self, stamp, best_index=-1, box=None, score=None, label=None):
        
        if (best_index == -1):
            kp_msg = self.__pubKeypoint(stamp)
            if kp_msg is None:
                return
            
            if self.pub_out_images:
                self.__pubImageWithRectangle()

        else:
            best_score = float(score[best_index].item())
            if best_score < self.detection_confidence_threshold:
                rospy.loginfo("Best score %.3f below threshold %.3f; publishing empty keypoint",
                              best_score, self.detection_confidence_threshold)
                kp_msg = self.__pubKeypoint(stamp)
                if kp_msg is None:
                    return
                if self.pub_out_images:
                    self.__pubImageWithRectangle()
                return

            kp_msg = self.__pubKeypoint(stamp, box[best_index], score[best_index], label[best_index])
            if kp_msg is None:
                return
            xyz = self.__compute_xyz(box[best_index])
            self.__publish_tf(xyz, stamp)
            
            if self.pub_out_images:
                if self.pub_out_all_keypoints:
                    self.__pubImageWithAllRectangles(box, label)
                else:
                    self.__pubImageWithRectangle(box[best_index], score[best_index], label[best_index])
                
            

    def __pubImageWithRectangle(self, box=None, score=None, label=None):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.model_helper.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = cv2.cvtColor(self.cv_image_output, cv2.COLOR_BGR2RGB)
        
        if (not box == None) and (score is not None):
            score_val = float(score.item()) if torch.is_tensor(score) else float(score)
        else:
            score_val = None

        if (not box == None) and (score_val is not None) and (score_val > self.detection_confidence_threshold):
            cv2.rectangle(self.cv_image_output, 
                          (round(box[0].item()), round(box[1].item())),
                          (round(box[2].item()), round(box[3].item())),
                          (255,0,0), 3)
        
        #if label:
            #cv2.putText(self.cv_image_output, str(label.item()), (round(box[0].item()), round(box[3].item()+10)), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_compressed_imgmsg(self.cv_image_output)
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)
        
    def __pubImageWithAllRectangles(self, box=None, label=None):
        
        #first convert back to unit8
        self.cv_image_output = torchvision.transforms.functional.convert_image_dtype(
            self.model_helper.tensor_images[0].cpu(), torch.uint8).numpy().transpose([1,2,0])
        self.cv_image_output = cv2.cvtColor(self.cv_image_output, cv2.COLOR_BGR2RGB)
        
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
        
        #cv2.imshow("test_boxes", self.cv_image_output)
        #cv2.waitKey()
        
        self.ros_image_output = self.bridge.cv2_to_compressed_imgmsg(self.cv_image_output)
        self.ros_image_output.header.seq = self.ros_image_input.header.seq
        self.ros_image_output.header.frame_id = self.ros_image_input.header.frame_id
        self.ros_image_output.header.stamp = rospy.Time.now()
        
        self.image_pub.publish(self.ros_image_output)

            
    """
    box is tensor and may be still float, we round befor filling the msg
    """        
    def __pubKeypoint(self, stamp, box=None, score=None, label=None):
        
        msg = KeypointImage()
        if self.ros_image_input is None:
            return None
        msg.header.frame_id = self.ros_image_input.header.frame_id
        msg.header.seq = self.ros_image_input.header.seq
        msg.header.stamp = stamp
        msg.image_width = self.cv_image_input.shape[1]
        msg.image_height = self.cv_image_input.shape[0]
        
        if (not box == None) and (not score == None) and (not label == None):
        
            #box from model has format: [x_0, y_0, x_1, y_1]
            msg.x_pixel = round(box[0].item() + (box[2].item() - box[0].item())/2)
            msg.y_pixel = round(box[1].item() + (box[3].item()  - box[1].item())/2)
            msg.label = int(label.item()) if torch.is_tensor(label) else int(label)
            msg.confidence = float(score.item()) if torch.is_tensor(score) else float(score)
            
        else:
            msg.x_pixel = 0
            msg.y_pixel = 0
            msg.label = 0
            msg.confidence = 0
        
        self.keypoint_pub.publish(msg)
        return msg

    def __compute_xyz(self, box):
        if not hasattr(self, "depth_image") or self.depth_image is None:
            rospy.loginfo_throttle(5.0, "No depth image available; skipping xyz/TF publish")
            return None
        u = round(box[0].item() + (box[2].item() - box[0].item())/2)
        v = round(box[1].item() + (box[3].item() - box[1].item())/2)
        window_radius = 2
        img_h, img_w = self.depth_image.shape[:2]
        if u < 0 or v < 0 or u >= img_w or v >= img_h:
            rospy.loginfo_throttle(
                5.0,
                "Keypoint out of depth bounds (u=%d, v=%d, w=%d, h=%d); skipping xyz/TF",
                u, v, img_w, img_h,
            )
            return None

        min_u = max(0, u - window_radius)
        max_u = min(img_w - 1, u + window_radius)
        min_v = max(0, v - window_radius)
        max_v = min(img_h - 1, v + window_radius)

        depths = []
        for vv in range(min_v, max_v + 1):
            for uu in range(min_u, max_u + 1):
                if self.depth_image.dtype == np.uint16:
                    d = float(self.depth_image[vv, uu]) * 0.001
                    if d <= 0:
                        continue
                else:
                    d = float(self.depth_image[vv, uu])
                    if not np.isfinite(d) or d <= 0:
                        continue
                depths.append(d)

        if len(depths) < 5:
            rospy.loginfo_throttle(
                5.0,
                "Insufficient valid depth samples (%d) around (u=%d, v=%d); skipping xyz/TF",
                len(depths), u, v,
            )
            return None

        depth_m = float(np.median(depths))
        ray = self.cam_model.projectPixelTo3dRay((u, v))
        scale = depth_m / float(ray[2])
        x = float(ray[0] * scale)
        y = float(ray[1] * scale)
        z = depth_m
        return (x, y, z)

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
    

    
