# Isaac ROS Compression

<div align="center"><img alt="graph of nodes using Isaac ROS Compression" src="resources/isaac_ros_compression_nodegraph.png" width="800px"/></div>

## Overview

This repository provides an H.264 image encoder and decoder that leverages the specialized hardware in NVIDIA GPUs and the [Jetson](https://developer.nvidia.com/embedded-computing) platform. The `isaac_ros_h264_encoder` package can compress an image into H.264 data using the NVENC on the Jetson platform. The `isaac_ros_h264_decoder` package can decode the H.264 data into original images using the NVDEC on the Jetson and x86 platform with NVIDIA GPUs.

Image compression reduces the data footprint of images when written to storage or transmitted between computers. A 1080p camera at 30fps produces 177MB/s of data; image compression reduces this by approximately 10 times to 17MB/s of data, reducing the throughput needed to send this to another computer or write out to storage; a one minute 1080p camera recording is reduced from ~10GB to ~1GB. This compression is provided by dedicated hardware acceleration (NvEnc) separate from other hardware engines such as the GPU.

A common use case for image compression during the development of robots is to capture camera images to storage. This captured data is processed offline from the robot to produce training datasets for AI models, test datasets for perception functions, and test data for open-loop re-simulation of software in development with real data. The compression parameters are tuned to minimize visual quality reduction from lossy compression for AI model and perception function development. Compression reduces the amount of data written to storage, the time required to offload the recording, and footprint of the data at rest in a data lake.

Compression can be used with event data recorders to capture camera images to storage when an event of interest occurs, often due to failures on the robot. This provides visual information to assist in the debugging of the event or to improve perception and robot functions.

[H.264](https://en.wikipedia.org/wiki/Advanced_Video_Coding) is an efficient and popular compression algorithm with broad support across many platforms. The output of the `isaac_ros_h264_encoder` package on Jetson can then be decoded with hardware acceleration using the `isaac_ros_h264_decoder` on Jetson and x86_64 systems, or by third-party H.264 decoder packages on non-NVIDIA platforms.

### H.264 compared to JPEG image_transport_plugins

ROS [image_transport_plugins](https://index.ros.org/p/image_transport_plugins/) use the JPEG standard for image compression on the CPU, where each frame is compressed spatially within the 2D image. `isaac_ros_h264_encoder` uses the H.264 standard for video compression, which allows it to compress more efficiently than JPEG by using more advanced spatial and temporal compression. ‘isaac_ros_h264_encoder’ uses the GPU for higher performance, while offloading compute work from the CPU.

H264 video compression of individual images, or I-frame (inter-frame) are smaller in size than equivalent quality JPEG images; H.264 uses improved compression functions for smooth areas and less artifacts in high frequencies areas of the image. H.264 adds temporal compression on a sequence of images by using a P-frame (predicted frame) from a previous I or P frame.  P-frames benefit on image similarity from frame to frame as occurs from a camera stream. P-frames are ~25% the size of an I-Frame. A sequence of images is compressed to one I-Frame followed by one or more P-Frames.  P-Frames require less computation improving frame rate of compression.

As a result, `isaac_ros_h264_encoder` can perform compression to smaller sizes with the same quality as JPEG `image_transport_plugins`, at higher frame rates (throughput), while simultaneously offloading the CPU from this compute for compression.

> Check your requirements against package [input limitations](#input-restrictions).

### Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

> **Note**:
ROS 2 relies on `image_transport_plugins` for CPU based compression.  We recommend using `isaac_ros_h264_encoder` as part of the graph of nodes when capturing to a rosbag for performance benefits of NITROS;  ROS 2 type adaptation used by NITROS is not supported by `image_transport_plugins`, resulting in more CPU load, and less encode performance.

## Performance

The following table summarizes the per-platform performance statistics of sample graphs that use this package, with links included to the full benchmark output. These benchmark configurations are taken from the [Isaac ROS Benchmark](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark#list-of-isaac-ros-benchmarks) collection, based on the [`ros2_benchmark`](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark) framework.

| Sample Graph                                                                                                                                                      | Input Size | AGX Orin                                                                                                                                                | x86_64 w/ RTX 3060 Ti                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [H.264 Decoder Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_h264_decoder_node.py)                           | 1080p      | [144 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_h264_decoder_node-agx_orin.json)<br>15 ms         | [290 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_h264_decoder_node-x86_64_rtx_3060Ti.json)<br>3.0 ms |
| [H.264 Encoder Node<br>I-frame Support](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_h264_encoder_iframe_node.py) | 1080p      | [305 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_h264_encoder_iframe_node-agx_orin.json)<br>11 ms  | --                                                                                                                                                        |
| [H.264 Encoder Node<br>P-frame Support](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_h264_encoder_pframe_node.py) | 1080p      | [293 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_h264_encoder_pframe_node-agx_orin.json)<br>9.7 ms | --                                                                                                                                                        |

## Table of Contents

- [Isaac ROS Compression](#isaac-ros-compression)
  - [Overview](#overview)
    - [H.264 compared to JPEG image\_transport\_plugins](#h264-compared-to-jpeg-image_transport_plugins)
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
    - [Symptom](#symptom)
    - [Solution](#solution)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2023-04-05: P-frame encoder and Tegra decoder support

## Supported Platforms

| Node                     | Jetson aarch64 | GPU x86_64 |
| ------------------------ | -------------- | ---------- |
| `isaac_ros_h264_encoder` | ✓              | N/A        |
| `isaac_ros_h264_decoder` | ✓              | ✓          |

On the hardware platforms listed in the table above `isaac_ros_h264_encoder` and `isaac_ros_h264_decoder` are designed and tested to be compatible with ROS 2 Humble.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                           | Notes                                                                                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack)                                                     | For best performance, ensure that the [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.8+](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note**: All Isaac ROS quick start guides, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

This quickstart shows an example of how to use the `isaac_ros_h264_decoder` with a pre-recorded rosbag, which contains compressed H.264 images generated from `isaac_ros_h264_encoder` with two argus cameras as the input source. You will be able to visualize the decoded images after the last step.

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

| ROS Parameter     | Type          | Default  | Description                                                                                                                                                                                                                                                                                        |
| ----------------- | ------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_width`     | `uint32_t`    | `1920`   | The width of the input image.                                                                                                                                                                                                                                                                      |
| `input_height`    | `uint32_t`    | `1200`   | The height of the input image.                                                                                                                                                                                                                                                                     |
| `qp`              | `uint32_t`    | `20`     | The  encoder constant QP value.                                                                                                                                                                                                                                                                    |
| `hw_preset`       | `uint32_t`    | `0`      | The encoder hardware preset type. The value can be an integer from `0` to `3`, representing `Ultrafast`, `Fast`, `Medium` and, `Slow`, respectviely.                                                                                                                                               |
| `profile`         | `uint32_t`    | `0`      | The profile to be used for encoding. The value can be an integer from `0` to `2`,  representing `Main`, `Baseline`, and `High`, respectviely.                                                                                                                                                      |
| `iframe_interval` | `int32_t`     | `5`      | Interval between two I frames, in number of frames. E.g., iframe_interval=5 represents IPPPPI..., iframe_interval=1 represents I frame only                                                                                                                                                        |
| `config`          | `std::string` | `pframe` | A preset combination of `qp`, `hw_preset`, `profile` and `iframe_interval`. The value can be `iframe`, `pframe` or `custom`. When `iframe` or `pframe` is used, the default value of these four parameters will be applied.  Only when this field is `custom` will the custom values will be used. |

#### ROS Topics Subscribed

| ROS Topic   | Type                                                                                                 | Description      |
| ----------- | ---------------------------------------------------------------------------------------------------- | ---------------- |
| `image_raw` | [sensor_msgs/Image](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Image.msg) | Raw input image. |

> **Limitation**: All input images are required to have height and width that are both an even number of pixels.

#### ROS Topics Published

| ROS Topic          | Interface                                                                                                                | Description             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ | ----------------------- |
| `image_compressed` | [sensor_msgs/CompressedImage](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/CompressedImage.msg) | H.264 compressed image. |

#### Input Restrictions

1. The input image resolution must be the same as the dimension you provided, and the resolution must be  **no larger than `1920x1200`**.
2. The input image should be in `rgb8` or `bgr8` format, and it will be converted to `nv12` format before being sent to the encoder.

#### Output Interpretations

1. The encoder could perform All-I frame or P-frame encoding and output the H.264 compressed data. The input and output are one-to-one mapped.

### `isaac_ros_h264_decoder`

#### Usage

```bash
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

### Symptom

Launching the decoder node using `ros2 launch isaac_ros_h264_decoder isaac_ros_h264_decoder_rosbag.launch.py`, produces the below error message:

x86_64 Platform:

```log
[component_container_mt-2] 2023-04-04 22:19:02.609 WARN  /workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_compression/isaac_ros_h264_decoder/gxf/codec/components/decoder.cpp@310: Decoder output is empty, current frame will be dropped
```

Jetson Platform:

```log
[component_container_mt-2] 2023-04-04 06:39:57.766 WARN  gxf/std/greedy_scheduler.cpp@235: Error while executing entity 20 named 'KIWPUYFVJT_decoder_response': GXF_FAILURE
[component_container_mt-2] [ERROR] (/usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2ElementPlane.cpp:178) <dec0> Capture Plane:Error while DQing buffer: Invalid argument
[component_container_mt-2] 2023-04-04 06:39:57.766 ERROR /workspaces/isaac_ros-dev/ros_ws/src/isaac_ros_compression/isaac_ros_h264_decoder/gxf/codec/components/decoder_request.cpp@530: Failed to dequeue buffer from capture plane
[component_container_mt-2] 2023-04-04 06:39:57.766 ERROR gxf/std/entity_executor.cpp@509: Failed to tick codelet  in entity: KIWPUYFVJT_decoder_response code: GXF_FAILURE
[component_container_mt-2] 2023-04-04 06:39:57.766 ERROR gxf/std/entity_executor.cpp@203: Entity with eid 35 not found!
```

### Solution

You may see these errors when decoder receives P-frames first. Decoder can not decode P-frames until it has received the first I-frame. You can start the decoder before playing the rosbag to avoid these errors.

### Isaac ROS Troubleshooting

For solutions to problems with Isaac ROS, check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md).

## Updates

| Date       | Changes                                   |
| ---------- | ----------------------------------------- |
| 2023-04-05 | P-frame encoder and Tegra decoder support |
| 2022-10-19 | Initial release                           |
