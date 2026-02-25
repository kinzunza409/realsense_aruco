FROM osrf/ros:noetic-desktop-full
#FROM ros:noetic

RUN apt-get update && apt-get install -y \
    #python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# this is here for quick tests
RUN python3 -m pip install opencv-python 

WORKDIR /workspace