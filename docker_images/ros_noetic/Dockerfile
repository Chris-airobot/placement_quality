# Use the official ROS Noetic base image (built on Ubuntu 20.04)
FROM ros:noetic-ros-base

# Set the DEBIAN_FRONTEND to noninteractive to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install additional packages.
RUN apt-get update && apt-get install -y \
    nano \
    git \
    curl \
    gnupg2 \
    lsb-release \
    cmake \
    build-essential \
    libpcl-dev \
    libeigen3-dev \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install opencv-python==4.2.0.32

# Create the ROS workspace directory
RUN mkdir -p /home/ros_ws/src

# Source ROS environment automatically for interactive sessions.
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# --------------------------
# Install gpd package
# --------------------------
RUN cd /home && \
    git clone https://github.com/atenpas/gpd && \
    cd gpd && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4 && \
    sudo make install


RUN bash -c "source /opt/ros/noetic/setup.bash"
# --------------------------
# Install gpd_ros package
# --------------------------
RUN cd /home/ros_ws/src && \
    git clone https://github.com/atenpas/gpd_ros && \
    apt-get update && \
    apt-get install -y ros-noetic-catkin \
    ros-noetic-eigen-conversions \
    ros-noetic-pcl-conversions 

RUN pip3 install pyquaternion

# Set the default command to run a bash shell.
CMD ["/bin/bash"]




