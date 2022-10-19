# Isaac ROS Compression

<div align="center"><img alt="Isaac ROS Compression Sample Output" src="resources/compression_pipeline.png" width="800px"/></div>

## Overview

This repository provides an H.264 image encoder and decoder that leverages the specialized hardware in NVIDIA GPUs and the [Jetson](https://developer.nvidia.com/embedded-computing) platform. The `isaac_ros_h264_encoder` package can compress an image into H.264 data using the NVENC on the Jetson platform. The `isaac_ros_h264_decoder` package can decode the H.264 data into original images using the NVDEC on x86 platform with NVIDIA GPUs.

Image compression reduces the data footprint of images when written to storage or transmitted between computers.  A 1080p camera at 30fps produces 177MB/s of data; image compression reduces this by approximately 10 times to 17MB/s of data reducing the throughput needed to send this to another computer or write out to storage; a one minute 1080p camera recording is reduced from ~10GB to ~1GB. This compression is provided by dedicated hardware acceleration (NvEnc) separate from other hardware engines such as the GPU.

A common use case for image compression during the development of robots is to capture camera images to storage.  This captured data is processed offline from the robot to produce training datasets for AI models, test datasets for perception functions, and test data for open-loop re-simulation of software in development with real data. The compression parameters used are tuned to minimize visual quality reduction from lossy compression for AI model and perception function development. Compression reduces the amount of data written to storage, the time required to offload the recording, and footprint of the data at rest in a data lake.

Compression can be used with event data recorders that capture camera images to storage when an events of interest occurs, often due to failures on the robot. This provides visual information to assist in the debugging of the event, or to improve perception and robot functions.

[H.264](https://en.wikipedia.org/wiki/Advanced_Video_Coding) is an efficient and popular compression algorithm with broad support across many platforms. The output of the `isaac_ros_h264_encoder` package run on Jetson can then be decoded with hardware acceleration using the `isaac_ros_h264_decoder` on x86_64 systems or by third-party H.264 decoder packages on non-NVIDIA powered platforms.

> Check your requirements against package [input limitations](#input-restrictions).

### Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

The following are the benchmark performance results of the prepared pipelines in this package, by supported platform:

| Pipeline           | AGX Orin            | Orin Nano | x86_64 w/ RTX 3060 Ti |
| ------------------ | ------------------- | --------- | --------------------- |
| H.264 encoder node | 170 fps <br> 17.4ms | N/A       | N/A                   |
| H.264 decoder node | N/A                 | N/A       | 400 fps <br> 2.3ms    |

These data have been collected per the methodology described [here](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/performance-summary.md#methodology).

## Table of Contents

- [Isaac ROS Compression](#isaac-ros-compression)
  - [Overview](#overview)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try More Examples](#try-more-examples)
    - [Customize your Dev Environment](#customize-your-dev-environment)
  - [Package Reference](#package-reference)
    - [`isaac_ros_h264_encoder`](#isaac_ros_h264_encoder)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
      - [Input Restrictions](#input-restrictions)
      - [Output Interpretations](#output-interpretations)
    - [`isaac_ros_h264_decoder`](#isaac_ros_h264_decoder)
      - [Usage](#usage-1)
      - [ROS Parameters](#ros-parameters-1)
      - [ROS Topics Subscribed](#ros-topics-subscribed-1)
      - [ROS Topics Published](#ros-topics-published-1)
      - [Input Restrictions](#input-restrictions-1)
      - [Output Interpretations](#output-interpretations-1)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2022-10-19: Initial release of `isaac_ros_h264_encoder` and `isaac_ros_h264_decoder`

## Supported Platforms

`isaac_ros_h264_encoder` is designed and tested to be compatible with ROS2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing).  
`isaac_ros_h264_decoder` is designed and tested to be compatible with ROS2 Humble running on x86_64 system with an NVIDIA GPU.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                             | Notes                                                                                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack)                                                       | For best performance, ensure that the [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.6.1+](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note:** All Isaac ROS quick start guides, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

This quickstart shows an example of how to use the `isaac_ros_h264_decoder` with a pre-recorded rosbag, which contains compressed H.264 images generated from `isaac_ros_h264_encoder` with two argus cameras as the input source. You will be able to visualize the decoded images after the last step.
> **Warning**: step 7 & 8 must be performed on an `x86_64` platform with an NVIDIA GPU.

1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).  
2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

    ```bash
    cd ~/workspaces/isaac_ros-dev/src
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline
    ```

    ```bash
    git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_compression
    ```

3. Pull down a rosbag of sample data:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_compression && \ 
    git lfs pull -X "" -I "resources/rosbags/h264_compressed_sample.bag"
    ```

4. Launch the Docker container using the `run_dev.sh` script:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```

5. Inside the container, build and source the workspace:  

    ```bash
    cd /workspaces/isaac_ros-dev && \
    colcon build --symlink-install && \
    source install/setup.bash
    ```

6. (Optional) Run tests to verify complete and correct installation:  

    ```bash
    colcon test --executor sequential
    ```

7. Run the following launch files to run the demo of this `isaac_ros_h264_decoder`:

    ```bash
    ros2 launch isaac_ros_h264_decoder isaac_ros_h264_decoder_rosbag.launch.py rosbag_path:=/workspaces/isaac_ros-dev/src/isaac_ros_compression/resources/rosbags/h264_compressed_sample.bag
    ```

8. Open a second terminal and attach to the container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
    ./scripts/run_dev.sh
    ```

9. Visualize and validate the output of the package:

    ```bash
    ros2 run image_view image_view --ros-args -r image:=/left/image_uncompressed
    ```

    ```bash
    ros2 run image_view image_view --ros-args -r image:=/right/image_uncompressed
    ```

## Next Steps

### Try More Examples

To continue exploring the Compression packages, check out the following suggested examples:

- [Tutorial with RealSense and H.264 data recording](./docs/tutorial-realsense-encoder.md)
- [Tutorial with Argus-compatible camera and H.264 data recording](./docs/tutorial-nitros-graph.md)
- [Tutorial with rosbag playback and software-based decoder](./docs/tutorial-compatible-decode.md)

### Customize your Dev Environment

To customize your development environment, reference [this guide](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/modify-dockerfile.md).

## Package Reference

### `isaac_ros_h264_encoder`

#### Usage

```bash
# Must be run on a Jetson platform.
ros2 launch isaac_ros_h264_encoder isaac_ros_h264_encoder.launch.py input_width:=<"your input image width"> input_height:=<"your input image height">
```

#### ROS Parameters

| ROS Parameter  | Type          | Default | Description                                                                                                                                                                                                                                    |
| -------------- | ------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_width`  | `uint32_t`    | `1920`  | The width of the input image.                                                                                                                                                                                                                  |
| `input_height` | `uint32_t`    | `1200`  | The height of the input image.                                                                                                                                                                                                                 |
| `qp`           | `uint32_t`    | `20`    | The  encoder constant QP value.                                                                                                                                                                                                                |
| `hw_preset`    | `uint32_t`    | `3`     | The encoder hardware preset type. The value can be an integer from `0` to `3`, representing `Ultrafast`, `Fast`, `Medium` and, `Slow`, respectviely.                                                                                           |
| `profile`      | `uint32_t`    | `0`     | The profile to be used for encoding. The value can be an integer from `0` to `2`,  representing `Main`, `Baseline`, and `High`, respectviely.                                                                                                  |
| `config`       | `std::string` | `lossy` | A preset combination of `qp`, `hw_preset`, and `profile`. The value can be `lossy` or `custom`. When `lossy`, the default value of these three parameters will be used.  Only when this field is `custom` will the custom values will be used. |

#### ROS Topics Subscribed

| ROS Topic   | Type                                                                                                 | Description      |
| ----------- | ---------------------------------------------------------------------------------------------------- | ---------------- |
| `image_raw` | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | Raw input image. |

> **Limitation:** All input images are required to have height and width that are both an even number of pixels.

#### ROS Topics Published

| ROS Topic          | Interface                                                                                                                | Description             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| `image_compressed` | [sensor_msgs/CompressedImage](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CompressedImage.msg) | H.264 compressed image. |

#### Input Restrictions

1. The input image resolution must be the same as the dimension you provided, and the resolution must be  **no larger than `1920x1200`**.
2. The input image should be in `rgb8` or `bgr8` format, and it will be converted to `nv12` format before being sent to the encoder.

#### Output Interpretations

1. The encoder will perform All-I frame encoding and output the H.264 compressed data.

### `isaac_ros_h264_decoder`

#### Usage

```bash
# Must be run on a x86_64 platform.
ros2 launch isaac_ros_h264_decoder isaac_ros_h264_decoder.launch.py input_width:=<"your original image width"> input_height:=<"your original image height">
```

#### ROS Parameters

| ROS Parameter  | Type       | Default | Description                      |
| -------------- | ---------- | ------- | -------------------------------- |
| `input_width`  | `uint32_t` | `1920`  | The width of the original image  |
| `input_height` | `uint32_t` | `1200`  | The height of the original image |

#### ROS Topics Subscribed

| ROS Topic          | Interface                                                                                                                | Description                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `image_compressed` | [sensor_msgs/CompressedImage](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CompressedImage.msg) | The H.264 compressed image |

#### ROS Topics Published

| ROS Topic            | Type                                                                                                 | Description                                |
| -------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `image_uncompressed` | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | The uncompressed image with RGB8 encoding. |

#### Input Restrictions

1. The input resolution must be **no larger than `1920x1200`** and must be the same as the original image resolution.

#### Output Interpretations

1. The `isaas_ros_h264_decoder` package outputs a uncompressed image with the same resolution as the original image. The output image will be in `RGB8` format.

## Troubleshooting

### Isaac ROS Troubleshooting

For solutions to problems with Isaac ROS, check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

## Updates

| Date       | Changes         |
| ---------- | --------------- |
| 2022-10-19 | Initial release |
