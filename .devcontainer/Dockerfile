FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    ros-noetic-tf2-tools \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-realsense2-camera \
    ros-noetic-realsense2-description \
    python3-opencv \
    python3-catkin-tools \
    python3-pip \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo 'if [ -f /workspace/ros_ws/devel/setup.bash ]; then source /workspace/ros_ws/devel/setup.bash; fi' >> /root/.bashrc
RUN echo 'export ROSCONSOLE_COLORIZE_OUTPUT=1' >> /root/.bashrc

# ROS networking
RUN echo 'export ROS_MASTER_URI=http://localhost:11311' >> /root/.bashrc
RUN echo 'export ROS_HOSTNAME=localhost' >> /root/.bashrc

# NVIDIA Optimus — force rendering on discrete GPU, not Intel integrated
RUN echo 'export __NV_PRIME_RENDER_OFFLOAD=1' >> /root/.bashrc
RUN echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> /root/.bashrc
RUN echo 'export NVIDIA_DRIVER_CAPABILITIES=all' >> /root/.bashrc

# Fix XDG_RUNTIME_DIR warning from Qt/RViz
RUN echo 'export XDG_RUNTIME_DIR=/tmp/runtime-root' >> /root/.bashrc

RUN echo 'cd /workspace/ros_ws' >> /root/.bashrc

WORKDIR /workspace