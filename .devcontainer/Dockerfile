FROM osrf/ros:noetic-desktop-full
#FROM ros:noetic

RUN apt-get update && apt-get install -y \
    python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-realsense2-camera \
    ros-noetic-realsense2-description \
    python3-catkin-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo 'if [ -f /workspace/ros_ws/devel/setup.bash ]; then source /workspace/ros_ws/devel/setup.bash; fi' >> /root/.bashrc
RUN echo 'export ROSCONSOLE_COLORIZE_OUTPUT=1' >> /root/.bashrc

# this is here for quick tests
#RUN python3 -m pip install opencv-python 

WORKDIR /workspace