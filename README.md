# nn_laser_spot_tracking

![logic scheme](./scheme/scheme.png)

Detect (and track) in ROS a laser spot emitted from a common laser pointer.


## Requirements
- ROS 1 Noetic (host) or Docker (see below).
- Python 3.8+ with PyTorch (CUDA optional for GPU inference).
- If using YOLOv5, a local clone is recommended (see setup).

## Overview
This repo provides:
- `tracking_2D` node (`scripts/inferNodeROS.py`): runs the NN, consumes RGB + depth + CameraInfo, publishes `KeypointImage` and TF (`laser_spot_frame`).
- `laser_udp_bridge` (optional): publishes the `laser_spot_frame` TF as UDP packets.


## Setup and running (ROS)
In brief:
- `tracking_2D` subscribes to `/$(arg camera)/$(arg image)` (with `transport`), aligned depth, and matching `camera_info`.
- It publishes a TF from the camera frame to `$(arg laser_spot_frame)` and a `KeypointImage`.

- Indentify the model/weights you want. Some are provided at [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835). In any case they should have been trained on laser spots, obviously. Put the model you want to use in a folder, `models` folder of this repo is the default.

- [Optional, but suggested] If Yolov5 is used, better to clone their [repo](https://github.com/ultralytics/yolov5/), and provide its path to the `yolo_path` argument. Otherwise, pytorch will download it every time (since the default is "ultralytics/yolov5"). If cloning, go in the folder and install the requirments: `pip install -r requirements.txt`.

- Run the launch file: 
  `roslaunch nn_laser_spot_tracking laser_tracking.launch model_name:=<> camera:=<> depth_image:=<> camera_info:=<>`

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
- **`transport`** (default: "compressed" in `laser_tracking.launch`, "raw" in `docker_k4a_laser_tracking.launch`).
- **`dl_rate`** (default: 30): Main loop rate (inference is blocking, so actual rate may be lower).
- **`detection_confidence_threshold`** (default: 0.55): Confidence threshold for detections.
- **`keypoint_topic`** (default: "/nn_laser_spot_tracking/detection_output_keypoint"): `KeypointImage` output.
- **`laser_spot_frame`** (default: "laser" in `laser_tracking.launch`, "laser_spot_frame" in the docker launch).
- **`pub_out_images`** (default: true): Publish debug images with rectangle.
- **`pub_out_images_all_keypoints`** (default: false): Publish all detections.
- **`pub_out_images_topic`** (default: "/detection_output_img"): Debug image topic base.
- **`log_level`** (default: INFO, docker launch only): Logger level for `tracking_2D`.

#### Legacy (present in launch file but unused by `tracking_2D`)
- **`tracking_rate`**, **`cloud_detection_max_sec_diff`**, **`position_filter_enable`**, **`position_filter_bw`**, **`position_filter_damping`**

## Docker prototype (Azure Kinect + UDP)
- Prerequisites: Docker, docker compose, and an Azure Kinect device connected to the host.
- Download the model `yolov5l6_e400_b8_tvt302010_laser_v4.pt` from [https://zenodo.org/records/10471835](https://zenodo.org/records/10471835) and place it in `models/` before building.
- Run everything in Docker with: `docker compose up --build`
- The full ROS stack runs inside the container; no host-side ROS installation is required.
- The container runs privileged with host networking for USB access to the Kinect and publishes UDP port 5005/udp to the host.
- The docker stack launches the Azure Kinect driver, `tracking_2D`, and `laser_udp_bridge`.
- UDP packet layout (little-endian, 28 bytes): `uint32 seq`, `uint64 t_ros_ns`, `float32 x_m`, `float32 y_m`, `float32 z_m`, `float32 confidence`.
- Packets are sent only when a detection exists and a valid depth/TF is available.
- Coordinates are expressed in the `k4a_rgb_camera_link` frame; no world/table calibration is included.

### GPU build (Docker)
This image is CPU by default. To build with CUDA wheels:
```
export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
export TORCH_SUFFIX=+cu118
docker compose build
```
Then enable GPU runtime by uncommenting `gpus: all` in `docker-compose.yml`.

### Debug images in Docker (no host ROS)
If you want to view `/detection_output_img/compressed` from the container:
```
xhost +local:root
docker exec -it -e DISPLAY=$DISPLAY <container_id> bash
rosrun rqt_image_view rqt_image_view
```
Select `/detection_output_img/compressed` in the UI.

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
