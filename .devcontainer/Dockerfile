FROM ros:noetic

RUN apt-get update && apt-get install -y \
    python3-opencv \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

WORKDIR /workspace