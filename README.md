# nn_laser_spot_tracking

![logic scheme](./scheme/scheme.png)

Detect (and track) in ROS a laser spot emitted from a common laser pointer.


## Requirements
- ROS 1 Noetic (host) or Docker (see below).
- Python 3.8+ with PyTorch (CUDA optional for GPU inference).
- If using YOLOv5, a local clone is recommended (see setup).

## Overview
This repo provides:
- `tracking_2D` node (`scripts/inferNodeROS.py`): runs the NN, consumes RGB + depth + CameraInfo, publishes `KeypointImage` (with tracking) and TF (`laser_spot_frame`).
- `laser_udp_bridge` (optional): publishes the `laser_spot_frame` TF as UDP packets and serves HTTP endpoints for camera pose, laser point, and table size.
- `aruco_table_calibration` node (`scripts/aruco_table_calib_node.py`): on-demand ArUco-based table calibration, publishes a `table_frame` TF and stores table size on the parameter server.


## Setup and running (ROS)
In brief:
- `tracking_2D` subscribes to `/$(arg camera)/$(arg image)` (raw), aligned depth, and matching `camera_info`.
- It publishes a TF from the camera frame to `$(arg laser_spot_frame)` and a `KeypointImage`.

- Indentify the model/weights you want. Some are provided at [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835). In any case they should have been trained on laser spots, obviously. Put the model you want to use in a folder, `models` folder of this repo is the default.

- [Optional, but suggested] If Yolov5 is used, better to clone their [repo](https://github.com/ultralytics/yolov5/), and provide its path to the `yolo_path` argument. Otherwise, pytorch will download it every time (since the default is "ultralytics/yolov5"). If cloning, go in the folder and install the requirments: `pip install -r requirements.txt`.

- Run the launch file: 
  `roslaunch nn_laser_spot_tracking laser_tracking.launch model_name:=<> camera:=<> depth_image:=<> camera_info:=<>`

## Table calibration (ArUco)
This is optional and runs only on demand via a service call.

Marker layout (top view, right-handed):
- Marker 0: lower-left table corner (origin).
- Marker 30: upper-left.
- Marker 49: lower-right.
- X axis: from marker 0 to 49. Y axis: from marker 0 to 30. Z axis: `X x Y`.
- Origin is the lower-left corner of marker 0 (set `origin_corner` to change).
- Marker corner offsets used for size: `x_axis_corner=lower_right`, `y_axis_corner=upper_left` by default.

Run calibration (default service name):
```
rosservice call /aruco_table_calibration/calibrate
```

Query camera pose (table -> camera) and table size over HTTP:
```
curl http://<host>:8000/camera_pose
```
Query latest laser point over HTTP:
```
curl http://<host>:8000/laser_point
```
Trigger calibration over HTTP:
```
curl -X POST http://<host>:8000/calibrate
```

Notes:
- Requires OpenCV ArUco module (opencv-contrib). The Docker image installs `opencv-contrib-python`.
- Detection runs until `target_detections_per_marker` is reached or `capture_timeout` expires.
- Table size is stored on the ROS parameter server (namespace `/table_calibration` by default).
- The HTTP endpoints are hosted by `laser_udp_bridge`; keep that node running to query pose/point/size or trigger calibration.
- If you are not using table calibration, set `laser_udp_bridge` `reference_frame` back to your camera frame.

Parameters (rosparams for `aruco_table_calibration`):
- `target_detections_per_marker` (default: 15)
- `capture_timeout` (default: 15.0 s)

## Tracking (alpha-beta)
- Tracking runs in 2D image space and outputs filtered pixels in `KeypointImage`.
- `KeypointImage.predicted` is true when no measurement was available in the current frame.
- Depth is computed from the filtered pixel and smoothed with a short median window.

### Launch file arguments (`laser_tracking.launch`)
#### Required
- **`model_name`**: Name of the model `.pt` file in `model_path`.
- **`camera`**: Camera ROS name (root of the RGB topic).
- **`depth_image`**: Depth image aligned to RGB (e.g., `/k4a/depth_to_rgb/image_raw`).
- **`camera_info`**: CameraInfo matching the depth image (aligned intrinsics).

#### Common optional
- **`model_path`** (default: "$(find nn_laser_spot_tracking)/models/"): Path to the model directory.
- **`yolo_path`** (default: "ultralytics/yolov5"): Local YOLOv5 repo path.
- **`image`** (default: "color/image_raw"): RGB image topic name.
- **`dl_rate`** (default: 30): Main loop rate (inference is blocking, so actual rate may be lower).
- **`detection_confidence_threshold`** (default: 0.55): Confidence threshold for detections.
- **`keypoint_topic`** (default: "/nn_laser_spot_tracking/detection_output_keypoint"): `KeypointImage` output.
- **`laser_spot_frame`** (default: "laser" in `laser_tracking.launch`, "laser_spot_frame" in the docker launch).
- **`pub_out_images`** (default: true): Publish debug images with rectangle.
- **`pub_out_images_all_keypoints`** (default: false): Publish all detections.
- **`pub_out_images_topic`** (default: "/detection_output_img"): Debug image topic base.
- **`log_level`** (default: INFO, docker launch only): Logger level for `tracking_2D`.
- **`tracking_enable`** (default: true): Enable 2D alpha-beta tracking.
- **`tracking_alpha`** (default: 0.85): Alpha parameter for tracking (higher = less lag).
- **`tracking_beta`** (default: 0.005): Beta parameter for tracking velocity update.
- **`tracking_gate_px`** (default: 30.0): Gating threshold in pixels before reset.
- **`tracking_max_prediction_frames`** (default: 2): Max predicted frames without measurement.
- **`tracking_reset_on_jump`** (default: true): Reset tracker when jump exceeds gate.
- **`tracking_predicted_confidence_scale`** (default: 1.0): Confidence scale for predicted frames.
- **`depth_median_window`** (default: 3): Temporal median window for depth (odd >= 1).

#### Legacy (present in launch file but unused by `tracking_2D`)
- **`tracking_rate`**, **`cloud_detection_max_sec_diff`**, **`position_filter_enable`**, **`position_filter_bw`**, **`position_filter_damping`**

## Docker prototype (Azure Kinect + UDP)
- Prerequisites: Docker, docker compose, and an Azure Kinect device connected to the host.
- Download the model `yolov5l6_e400_b8_tvt302010_laser_v4.pt` from [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835) and place it in `models/` before building.
- Run everything in Docker with: `docker compose up --build`
- Toggle tracking with `TRACKING_ENABLE=true|false` (compose environment variable).
- Set Kinect FPS with `K4A_FPS=5|15|30` (also used for `dl_rate`).
- The full ROS stack runs inside the container; no host-side ROS installation is required.
- The container runs privileged with host networking for USB access to the Kinect and publishes UDP port 5005/udp to the host.
- The docker stack launches the Azure Kinect driver, `tracking_2D`, and `laser_udp_bridge`.
- UDP packet layout (little-endian, 32 bytes): `uint32 seq`, `uint64 t_ros_ns`, `float32 x_m`, `float32 y_m`, `float32 z_m`, `float32 confidence`, `uint32 flags` (bit0 = predicted).
- Packets are sent only when a detection exists and a valid depth/TF is available.
- Coordinates are expressed in the `table_frame` after calibration (or in `reference_frame` when configured otherwise).
- The HTTP endpoints for pose/size and laser point are served by `laser_udp_bridge` (default `:8000/camera_pose` and `:8000/laser_point`).

### GPU build (Docker)
This image is CPU by default. To build/run with CUDA wheels:
```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```
Optional overrides:
```
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 \
TORCH_SUFFIX=+cu118 \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Debug images in Docker (no host ROS)
If you want to view `/detection_output_img` from the container:
```
xhost +local:root
docker exec -it -e DISPLAY=$DISPLAY <container_id> bash
rosrun rqt_image_view rqt_image_view
```
Select `/detection_output_img` in the UI.

## Training new models
- See [hhcm_yolo_training](https://github.com/ADVRHumanoids/hhcm_yolo_training) repo

## Testing/comparing models
- See [benchmark](benchmark) folder

## Image dataset: 
Available at [https://zenodo.org/records/15230870](https://zenodo.org/records/15230870)
Two formats are given:
- COCO format (for non-yolo models) as:
  - a folder containing images and annotation folders.    
    - in images, all the images (not divided by train, val, test, this is done by the code)
    - in annotations, an instances_default.json file 

- YOLOv5 pytorch format for YOLOV5 model
  - a folder containing data.yaml file which points to two folders in the same location:
    - train
    - valid
    Both have images and labels folders

## Trained models: 
Available at [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835)

## Troubleshoot
If a too old version of `setuptools` is found on the system, Ultralytics Yolo will upgrade it. Recently, when upgrading to >71, this errors occurs:
`AttributeError: module 'importlib_metadata' has no attribute 'EntryPoints'`
You should solve downgrading a bit setuptools: `pip3 install setuptools==70.3.0`. See [here](https://github.com/pypa/setuptools/issues/4478)

## Papers

[https://www.sciencedirect.com/science/article/pii/S092188902500140X](https://www.sciencedirect.com/science/article/pii/S092188902500140X)
@article{LaserJournal,
  title = {An intuitive tele-collaboration interface exploring laser-based interaction and behavior trees},
  author = {Torielli, Davide and Muratore, Luca and Tsagarakis, Nikos},
  journal = {Robotics and Autonomous Systems},
  volume = {193},
  pages = {105054},
  year = {2025},
  issn = {0921-8890},
  doi = {https://doi.org/10.1016/j.robot.2025.105054},
  url = {https://www.sciencedirect.com/science/article/pii/S092188902500140X},
  keywords = {Human-robot interface, Human-centered robotics, Visual servoing, Motion planning},
  dimensions = {true},
}

[https://ieeexplore.ieee.org/document/10602529](https://ieeexplore.ieee.org/document/10602529)
```
@ARTICLE{10602529,
  author={Torielli, Davide and Bertoni, Liana and Muratore, Luca and Tsagarakis, Nikos},
  journal={IEEE Robotics and Automation Letters}, 
  title={A Laser-Guided Interaction Interface for Providing Effective Robot Assistance to People With Upper Limbs Impairments}, 
  year={2024},
  volume={9},
  number={9},
  pages={7653-7660},
  keywords={Robots;Lasers;Task analysis;Keyboards;Magnetic heads;Surface emitting lasers;Grippers;Human-robot collaboration;physically assistive devices;visual servoing},
  doi={10.1109/LRA.2024.3430709}}
```
