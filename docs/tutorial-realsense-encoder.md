# Tutorial For RealSense-based Encoding

In this tutorial, we'll demonstrate how you can perform H.264 encoding using a [RealSense](https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html) camera and [isaac_ros_h264_encoder](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_compression/blob/main/isaac_ros_h264_encoder/src/encoder_node.cpp), and save the compressed images into a rosbag.  

> **Note**: `isaac_ros_h264_encoder` needs to run on a Jetson platform.
<!-- Split blockquote -->
> **Note**: This tutorial requires a compatible RealSense camera from the list available [here](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md#camera-compatibility).

1. Complete the [RealSense setup tutorial](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md).
2. Complete steps 1-2 described in the [quickstart guide](../README.md#quickstart).
3. Open a new terminal and launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

4. Build and source the workspace:

    ```bash
    cd /workspaces/isaac_ros-dev && \
      colcon build --symlink-install && \
      source install/setup.bash
    ```

5. Run the launch file. This launch file will launch the example and save `CompressedImage` and `CameraInfo` topic data into a rosbag to your current folder:

    ```bash
    ros2 launch isaac_ros_h264_encoder isaac_ros_h264_encoder_realsense.launch.py
    ```

6. (Optional) If you want to decode and visualize the images from the rosbag, you can place the recorded rosbag into an x86 machine equipped with NVIDIA GPU, then follow steps 7 & 8 in the [Quickstart section](../README.md#quickstart). (Change the rosbag path and input dimension accordingly in step 7):

    ```bash
    ros2 launch isaac_ros_h264_decoder isaac_ros_h264_decoder_rosbag.launch.py rosbag_path:=<"path to your rosbag folder"> input_width:=640 input_height:=480
    ```

Here is a screenshot of the result example:
<div align="center"><img src="../resources/realsense_example.png"/></div>
