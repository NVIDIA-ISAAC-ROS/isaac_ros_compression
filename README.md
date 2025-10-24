# Isaac ROS Compression

NVIDIA-accelerated data compression.

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_compression/isaac_ros_compression_nodegraph.png/"><img alt="image" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_compression/isaac_ros_compression_nodegraph.png/" width="800px"/></a></div>

## Overview

Isaac ROS Compression provides H.264 image encoder
and decoder that leverages the specialized hardware in NVIDIA GPUs and the
[Jetson](https://developer.nvidia.com/embedded-computing) platform.
The `isaac_ros_h264_encoder` package can compress an image into H.264
data using the NVENC. The
`isaac_ros_h264_decoder` package can decode the H.264 data into
original images using the NVDEC.

Image compression reduces the data footprint of images when written to
storage or transmitted between computers. A 1080p camera at 30fps
produces 177MB/s of data; image compression reduces this by
approximately 10 times to 17MB/s of data, reducing the throughput needed
to send this to another computer or write out to storage; a one minute
1080p camera recording is reduced from ~10GB to ~1GB. This compression
is provided by dedicated NVIDIA acceleration (NvEnc) separate from
other hardware engines such as the GPU.

A common use case for image compression during the development of robots
is to capture camera images to storage. This captured data is processed
offline from the robot to produce training datasets for AI models, test
datasets for perception functions, and test data for open-loop
re-simulation of software in development with real data. The compression
parameters are tuned to minimize visual quality reduction from lossy
compression for AI model and perception function development.
Compression reduces the amount of data written to storage, the time
required to offload the recording, and footprint of the data at rest in
a data lake.

Compression can be used with event data recorders to capture camera
images to storage when an event of interest occurs, often due to
failures on the robot. This provides visual information to assist in the
debugging of the event or to improve perception and robot functions.

[H.264](https://en.wikipedia.org/wiki/Advanced_Video_Coding) is an
efficient and popular compression algorithm with broad support across
many platforms. The output of the `isaac_ros_h264_encoder` package can then
be decoded with NVIDIA acceleration using the
`isaac_ros_h264_decoder` on Jetson and x86_64 systems, or by
third-party H.264 decoder packages on non-NVIDIA platforms.

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/),
which leverages type adaptation and negotiation to optimize message
formats and dramatically accelerate communication between participating nodes.

> [!Note]
> ROS 2 relies on `image_transport_plugins` for CPU based compression.
> We recommend using `isaac_ros_h264_encoder` as part of the graph of
> nodes when capturing to a rosbag for performance benefits of NITROS;
> ROS 2 type adaptation used by NITROS is not supported by `image_transport_plugins`,
> resulting in more CPU load, and less encode performance.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                                           | Input Size<br/><br/>   | AGX Thor<br/><br/>                                                                                                                                                                 | x86_64 w/ RTX 5090<br/><br/>                                                                                                                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [H.264 Decoder Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_h264_decoder_benchmark/scripts/isaac_ros_h264_decoder_node.py)<br/><br/>                                      | 1080p<br/><br/>        | [596 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_h264_decoder_node-agx_thor.json)<br/><br/><br/>10 ms @ 30Hz<br/><br/>         | [596 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_h264_decoder_node-x86-5090.json)<br/><br/><br/>4.0 ms @ 30Hz<br/><br/>        |
| [H.264 Encoder Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_h264_encoder_benchmark/scripts/isaac_ros_h264_encoder_iframe_node.py)<br/><br/><br/>I-frame Support<br/><br/> | 1080p<br/><br/>        | [390 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_h264_encoder_iframe_node-agx_thor.json)<br/><br/><br/>11 ms @ 30Hz<br/><br/>  | [596 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_h264_encoder_iframe_node-x86-5090.json)<br/><br/><br/>3.9 ms @ 30Hz<br/><br/> |
| [H.264 Encoder Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/benchmarks/isaac_ros_h264_encoder_benchmark/scripts/isaac_ros_h264_encoder_pframe_node.py)<br/><br/><br/>P-frame Support<br/><br/> | 1080p<br/><br/>        | [429 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_h264_encoder_pframe_node-agx_thor.json)<br/><br/><br/>8.6 ms @ 30Hz<br/><br/> | [596 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-4.0/results/isaac_ros_h264_encoder_pframe_node-x86-5090.json)<br/><br/><br/>4.4 ms @ 30Hz<br/><br/> |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_h264_decoder`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_decoder/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_decoder/index.html#quickstart)
  * [Troubleshooting](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_decoder/index.html#troubleshooting)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_decoder/index.html#api)
* [`isaac_ros_h264_encoder`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_encoder/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_encoder/index.html#quickstart)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_compression/isaac_ros_h264_encoder/index.html#api)

## Latest

Update 2025-10-24: Support for ROS 2 Jazzy
