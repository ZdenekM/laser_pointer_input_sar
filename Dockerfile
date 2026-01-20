FROM ros:noetic-ros-base

ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
ARG TORCH_VERSION="2.0.1"
ARG TORCHVISION_VERSION="0.15.2"
ARG TORCH_SUFFIX="+cpu"
ARG MODEL_NAME="yolov5l6_e400_b8_tvt302010_laser_v4.pt"

ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV YOLOv5_AUTOINSTALL=False

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    apt-transport-https \
    && curl -sSL https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -o /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && rm /tmp/packages-microsoft-prod.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-empy \
    python3-catkin-pkg-modules \
    python3-rospkg-modules \
    ros-noetic-catkin \
    ros-noetic-roscpp \
    ros-noetic-rospy \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-camera-info-manager \
    ros-noetic-image-geometry \
    ros-noetic-cv-bridge \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-nodelet \
    ros-noetic-xacro \
    ros-noetic-angles \
    ros-noetic-rgbd-launch \
    ros-noetic-joint-state-publisher \
    ros-noetic-robot-state-publisher \
    ros-noetic-rqt-image-view \
    libopencv-dev \
    libopencv-contrib-dev \
    libsoundio1 \
    libgl1-mesa-dri \
    libgl1 \
    mesa-utils \
    x11-utils \
    xserver-xorg-core \
    xserver-xorg-video-dummy \
    xserver-xorg \
    ocl-icd-opencl-dev \
    pocl-opencl-icd \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN set -e; \
    echo "libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76" | debconf-set-selections; \
    curl -sSLo /tmp/libk4a1.4_1.4.2_amd64.deb https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.2_amd64.deb; \
    curl -sSLo /tmp/libk4a1.4-dev_1.4.2_amd64.deb https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.2_amd64.deb; \
    curl -sSLo /tmp/k4a-tools_1.4.2_amd64.deb https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools/k4a-tools_1.4.2_amd64.deb; \
    ACCEPT_EULA=Y dpkg -i /tmp/libk4a1.4_1.4.2_amd64.deb /tmp/libk4a1.4-dev_1.4.2_amd64.deb /tmp/k4a-tools_1.4.2_amd64.deb || true; \
    apt-get -f install -y --no-install-recommends; \
    rm /tmp/libk4a1.4_1.4.2_amd64.deb /tmp/libk4a1.4-dev_1.4.2_amd64.deb /tmp/k4a-tools_1.4.2_amd64.deb; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --extra-index-url ${PYTORCH_INDEX_URL} \
    typing_extensions==4.5.0 \
    numpy==1.24.4 \
    torch==${TORCH_VERSION}${TORCH_SUFFIX} torchvision==${TORCHVISION_VERSION}${TORCH_SUFFIX}

WORKDIR /catkin_ws/src
COPY third_party/yolov5/requirements.txt /tmp/yolov5_requirements.txt

RUN pip3 install --no-cache-dir --extra-index-url ${PYTORCH_INDEX_URL} \
    -r /tmp/yolov5_requirements.txt \
    "python-dateutil>=2.8.2"

RUN pip3 install --no-cache-dir --force-reinstall \
    opencv-contrib-python==4.8.1.78

RUN python3 - <<'PY'
import cv2
if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
    raise SystemExit("cv2.aruco.estimatePoseSingleMarkers missing after install")
print("OpenCV", cv2.__version__, "aruco OK")
PY

WORKDIR /catkin_ws/src
COPY . /catkin_ws/src/nn_laser_spot_tracking

COPY config/xorg.conf /etc/X11/xorg.conf

RUN test -f "/catkin_ws/src/nn_laser_spot_tracking/models/${MODEL_NAME}" || \
    (echo "Missing model file models/${MODEL_NAME}. Download it or set build arg MODEL_NAME to your .pt file." >&2; exit 1)

RUN mv /catkin_ws/src/nn_laser_spot_tracking/third_party/Azure_Kinect_ROS_Driver /catkin_ws/src/Azure_Kinect_ROS_Driver && \
    mv /catkin_ws/src/nn_laser_spot_tracking/laser_udp_bridge /catkin_ws/src/laser_udp_bridge

WORKDIR /catkin_ws
RUN source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5005/udp

ENTRYPOINT ["/entrypoint.sh"]
CMD ["roslaunch", "nn_laser_spot_tracking", "docker_stack.launch"]
