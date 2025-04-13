# Use the NVIDIA Isaac Sim image as the base.
# IMPORTANT: Ensure that this base image is Ubuntu 22.04 or has been modified accordingly.
FROM nvcr.io/nvidia/isaac-sim:4.2.0

# Set DEBIAN_FRONTEND to noninteractive to prevent interactive tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set up locales (if not already set) for ROS2
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Install utilities required for installing ROS2 and cloning your repo
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg2 \
    lsb-release

# Add ROS2 Humble apt repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2-latest.list

# Update apt and install ROS2 Humble (desktop version)
RUN apt-get update && apt-get install -y ros-humble-desktop

# Install colcon build tools for building your ROS2 workspace
RUN apt-get install -y python3-colcon-common-extensions

# (Optional) Add ROS2 sourcing to the bashrc for interactive sessions
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Specify the build argument for the GitHub token
ARG GITHUB_TOKEN

# Create a ROS2 workspace directory structure
RUN mkdir -p /home/chris/Chris/placement_ws/src
WORKDIR /home/chris/Chris/placement_ws/src

# Use the token to clone the repository
RUN git clone https://${GITHUB_TOKEN}@github.com/Chris-airobot/placement_quality.git

# Install pip3 (so pip3 is available for the requirements step)
RUN apt-get update && apt-get install -y python3-pip

# If your repo contains a requirements.txt, install Python dependencies.
# Adjust the path if necessary.
RUN if [ -f "placement_quality/requirements.txt" ]; then pip3 install -r placement_quality/requirements.txt; fi

# Build your ROS2 workspace (using colcon)
WORKDIR /home/chris/Chris/placement_ws/
RUN /bin/bash -c "source /opt/ros/humble/setup.bash"


RUN apt install -y python3-rosdep && rosdep init && rosdep update && apt update && apt dist-upgrade -y && \
apt install -y python3-colcon-common-extensions && apt install -y python3-colcon-mixin && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && colcon mixin update default
RUN apt install python3-vcstool

RUN mkdir -p ~/ws_moveit/src && \
cd ~/ws_moveit/src && \
git clone -b humble https://github.com/moveit/moveit2_tutorials

RUN vcs import --recursive < moveit2_tutorials/moveit2_tutorials.repos

RUN apt remove ros-humble-moveit*

RUN apt update && rosdep install -r --from-paths . --ignore-src --rosdistro humble -y

RUN cd ~/ws_moveit && \
colcon build --mixin release


RUN source ~/ws_moveit/install/setup.bash
# Create the bash file with your desired content
# RUN mkdir -p /home/chris/Chris/placement_ws/src && \
#     echo '#!/usr/bin/env bash\nwhile true; do\n    echo "Running ycb_collection.py..."\n    cd /home/chris/Chris/placement_ws/src/placement_quality\n    /isaac-sim/python.sh -m ycb_simulation.ycb_collection --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error\n    echo "Script finished. Resting for 1 minute..."\n    sleep 60\ndone' > /home/chris/Chris/placement_ws/src/start_collection.sh

# # Make the script executable
# RUN chmod +x /home/chris/Chris/placement_ws/src/start_collection.sh


# Set the system's default python to be Isaac Sim's python by creating symbolic links.
RUN sed -i 's|^python_exe=${PYTHONEXE:-"${SCRIPT_DIR}/kit/python/bin/python3"}|python_exe=/usr/bin/python3|' /isaac-sim/python.sh


# The default command: open a bash shell.
CMD ["/bin/bash"]
