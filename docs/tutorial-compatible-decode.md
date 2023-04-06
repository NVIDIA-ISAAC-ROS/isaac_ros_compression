# Decoding Jetson H.264 images on non-NVIDIA powered systems

<div align="center"><img alt="Decoding Jetson H.264 images on non-NVIDIA powered systems sample expected output" src="../resources/image_view_h264_decoded.png" width="400px"/></div>

## Overview

Using hardware-accelerated Isaac ROS compression on Jetson to H.264 encode data for playback through Isaac ROS H.264 decoder on NVIDIA-powered x86_64 is fast and efficient. However,
you may have need to decode recorded data on x86_64 systems that are not NVIDIA-powered. Fortunately, the output of the `isaac_ros_h264_encoder` package can easily be reformatted to work with other available ROS 2 decoder packages.

This tutorial shows you how to replay a rosbag recorded on a Jetson using a third-party H.264 decoder [image_transport plugin](https://github.com/KDharmarajanDev/h264_image_transport.git) used in [FogROS2](https://github.com/berkeleyAutomation/FogROS2).

The launch file here replays a target rosbag recorded with `isaac_ros_h264_encoder` through a node to reformat the compressed messages for compatibility before being decoded by the third-party decoder to be displayed in an image view window.

## Tutorial Walkthrough

1. Complete the [Quickstart section](../README.md#quickstart) in the main README.
2. Clone the following third-party repo into your workspace:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    git clone https://github.com/KDharmarajanDev/h264_image_transport.git

    # Install dependencies for the third-party package
    sudo apt install libavdevice-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
    ```

3. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

4. Inside the container, build the third-party `h264_image_transport` package:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install --packages-up-to \
        h264_image_transport isaac_ros_to_h264_msgs_packet && \
      source install/setup.bash
    ```

5. Launch the graph which brings up an image viewer to show the decoded output.

    ```bash
    ros2 launch isaac_ros_to_h264_msgs_packet h264_compatible_decoder.launch.py rosbag_path:=/workspaces/isaac_ros-dev/src/isaac_ros_compression/resources/rosbags/h264_compressed_sample.bag
    ```

    > Note: You may have to click on the "/image" to trigger window resizing.
