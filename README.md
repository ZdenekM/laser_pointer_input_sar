# Laser Pointer Input for Spatial Augmented Reality

![logic scheme](./scheme/scheme.png)

Interactive laser-pointer tracking for projection-based interfaces. Runs a ROS 1 stack in Docker, outputs 3D laser points over UDP and calibration data over HTTP for Unity or similar clients.

This project is based on https://github.com/ADVRHumanoids/nn_laser_spot_tracking and adapted for interactive use (low-latency tracking, Unity-friendly IO, projector calibration) with a Docker-first workflow.

## What this provides
- `tracking_2D` node (`scripts/inferNodeROS.py`): runs the NN, consumes RGB + depth + CameraInfo, publishes `KeypointImage` (with tracking) and TF (`laser_spot_frame`).
- `laser_udp_bridge`: publishes the `laser_spot_frame` TF as UDP packets and serves HTTP endpoints for camera pose, laser point, table size, and projector calibration.
- `aruco_table_calibration` node (`scripts/aruco_table_calib_node.py`): on-demand ArUco-based table calibration, publishes a `table_frame` TF and stores table size on the parameter server.

## Requirements (Docker)
- Docker and docker compose.
- Azure Kinect device (default docker stack).
- Optional NVIDIA GPU for faster inference (see GPU build below).

## Quick start (Docker, primary)
- Initialize submodules:
  `git submodule update --init`
- Download the model `yolov5l6_e400_b8_tvt302010_laser_v4.pt` from https://zenodo.org/records/10471835 and place it in `models/`.
- The Docker build now fails fast if the model file is missing (override with build arg `MODEL_NAME` for custom models).
- Run everything in Docker:
  `docker compose up --build`
- Toggle tracking with `TRACKING_ENABLE=true|false` in `docker-compose.yml`.
- Set Kinect FPS with `K4A_FPS=5|15|30` (also used for `dl_rate`).
- Set color resolution with `COLOR_RESOLUTION=720P|1080P|1440P|1536P|2160P|3072P`.
- Control whether roslaunch exits when any node dies with `NODES_REQUIRED=true|false`.
- Table calibration is saved to `TABLE_CALIBRATION_PATH` (default `/data/table_calibration.json`).
- The full ROS stack runs inside the container; no host-side ROS installation is required.

### GPU build (Docker)
This image is CPU by default. To build/run with CUDA wheels:
```
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

### Debug images over SSH X11
If you are connected via `ssh -X` and want to use X11 from Docker:
```
XAUTH=/tmp/.docker.xauth
touch "$XAUTH"
xauth nlist "$DISPLAY" | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge -
chmod 644 "$XAUTH"
```
Then run:
```
docker compose -f docker-compose.yml -f docker-compose.x11-ssh.yml up --build
```

## Outputs
### UDP
- Port: `5005/udp` by default (see `docker-compose.yml`).
- Packet layout (little-endian, 32 bytes): `uint32 seq`, `uint64 t_ros_ns`, `float32 x_m`, `float32 y_m`, `float32 z_m`, `float32 confidence`, `uint32 flags` (bit0 = predicted, bit1 = depth_assumed_plane).
- Table calibration is required; UDP/HTTP laser point outputs are disabled until it is available.
- Packets are sent only when a detection exists and a valid depth/TF is available.
- Coordinates are expressed in the `table_frame` after calibration (or in `reference_frame` when configured otherwise).

### HTTP
Default host port: `8000`.
- `GET /camera_pose`
- `GET /laser_point`
- `POST /calibrate`
- `PUT /projector_calibration`
- `GET /projector_calibration`

Endpoint details (served by `laser_udp_bridge`):

#### GET /camera_pose
Returns the latest camera pose (TF) and table calibration metadata.
- Status: `200` when `ok=true`, `503` when `ok=false`.
- Response body:
  - `ok` (bool): `true` when the TF lookup succeeded.
  - `reference_frame` (string): the configured reference frame.
  - `camera_frame` (string): the configured camera frame.
  - `pose` (object, present when `ok=true`): `frame_id`, `child_frame_id`, `stamp{secs,nsecs}`, `translation{x,y,z}`, `rotation{x,y,z,w}`.
  - `pose_error` (string, present when `ok=false`): TF error message.
  - `table` (object, always present): `width_m`, `height_m`, `marker_size_m`, `origin_corner`, `x_axis_corner`, `y_axis_corner`, `origin_id`, `x_axis_id`, `y_axis_id`, `stamp{secs,nsecs}`. Values are `null` if calibration is missing.

#### GET /laser_point
Returns the latest laser point in the reference frame.
- Status: `200` when `ok=true`, `503` when `ok=false`.
- Response body:
  - `ok` (bool): `true` when a point is available.
  - `error` (string, when `ok=false`): reason (e.g., missing table calibration or no data yet).
  - `seq` (uint32), `stamp{secs,nsecs}`, `frame_id` (reference frame), `target_frame` (laser TF frame).
  - `position{x,y,z}` in meters, `confidence` (float), `predicted` (bool), `depth_assumed_plane` (bool).

#### POST /calibrate
Triggers ArUco table calibration via the ROS service.
- Status: `200` when `ok=true`, `503` when `ok=false`.
- Response body: `ok` (bool), `message` (string).

#### PUT /projector_calibration
Stores a projector calibration computed from point correspondences.
- Status: `200` on success, `400` on validation errors, `503` if table calibration is required but missing, `500` on compute/save errors.
- Request body:
  - `mode` (`planar` or `full3d`, default `planar`).
  - `projector{width_px,height_px}`.
  - `points[]`: array of `{projector{u,v}, world{x,y,z}}` (min 4 points for `planar`, 6 for `full3d`).
  - Optional `frame_id` must be `table_frame` if provided.
- Response body (success):
  - `ok` (true), `mode`, `frame_id`, `projector`, `points_used`, `plane_rms_m`, `reprojection_rms_px`.
  - `H_3x3` for `planar`, or `P_3x4` for `full3d`.

#### GET /projector_calibration
Returns the last saved projector calibration.
- Status: `200` on success, `404` if none is available.
- Response body: same as successful `PUT /projector_calibration` (or `{ok:false,error}` when missing).

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
HTTP trigger:
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
- `calibration_store_path` (default: `/data/table_calibration.json`)

## Projector calibration (HTTP)
Planar (homography) and full 3D (projection matrix) calibration based on `projector_pixel <-> world_point` correspondences.

Example request (`table_frame`):
```
curl -X PUT http://<host>:8000/projector_calibration \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "planar",
    "projector": { "width_px": 1920, "height_px": 1080 },
    "points": [
      { "projector": { "u": 100, "v": 120 }, "world": { "x": 0.12, "y": 0.34, "z": 0.0 } }
    ]
  }'
```
Retrieve last projector calibration:
```
curl http://<host>:8000/projector_calibration
```

Notes:
- `mode=planar` expects near-planar points and returns `H_3x3`.
- `mode=full3d` expects non-coplanar points and returns `P_3x4`.
- Projector calibration is saved to `projector_calibration_path` (default `/data/projector_calibration.json`).
- In Docker, the compose file mounts `./data:/data` for persistence.

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
- **`image`** (default: "color/image_raw"): Color image topic name (Kinect publishes `bgra8`; the node accepts `bgra8` or `rgb8` and converts to `rgb8` internally).
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
- **`depth_history_max_age`** (default: 1.0): Max age (s) for cached depth frames.
- **`depth_frame_history_size`** (default: 7): Number of recent depth frames used to collect samples around the pixel.
- **`depth_tracking_enable`** (default: true): Enable alpha-beta tracking on depth values.
- **`depth_tracking_alpha`** (default: 0.7): Alpha for depth tracking.
- **`depth_tracking_beta`** (default: 0.02): Beta for depth tracking.
- **`depth_tracking_gate_m`** (default: 0.2): Gating threshold (m) for depth tracking resets.
- **`depth_tracking_max_prediction_frames`** (default: 2): Max predicted frames without depth measurement.
- **`depth_tracking_reset_on_jump`** (default: true): Reset depth tracker when jump exceeds gate.
- **`depth_fallback_plane_enable`** (default: true): If depth is missing, project the ray onto the table plane.
- **`depth_fallback_plane_frame`** (default: "table_frame"): Frame defining the table plane (z=0).
- **`depth_fallback_plane_timeout`** (default: 0.05): TF lookup timeout for plane fallback (s).
- **`require_table_calibration`** (default: true): Skip inference until table calibration is available.
- **`table_frame`** (default: "table_frame"): Table frame used for calibration checks.
- **`calibration_param_ns`** (default: "/table_calibration"): Namespace used to verify calibration params.

## Models and training
- Place model weights in `models/`.
- The original authors provide trained models and datasets at https://zenodo.org/records/10471835 and https://zenodo.org/records/15230870.
- Training workflows and dataset details are documented in the upstream repo: https://github.com/ADVRHumanoids/nn_laser_spot_tracking.
- For YOLOv5 training pipelines, see https://github.com/ADVRHumanoids/hhcm_yolo_training.
