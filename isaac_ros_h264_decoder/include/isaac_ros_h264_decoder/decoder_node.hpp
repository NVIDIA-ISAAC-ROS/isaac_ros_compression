// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_H264_DECODER__DECODER_NODE_HPP_
#define ISAAC_ROS_H264_DECODER__DECODER_NODE_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_h264_decoder/decoder_v4l2_impl.hpp"
#include "vpi_format_converter.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_decoder
{

class DecoderNode : public rclcpp::Node
{
public:
  explicit DecoderNode(const rclcpp::NodeOptions & options);
  ~DecoderNode();

  DecoderNode(const DecoderNode &) = delete;
  DecoderNode & operator=(const DecoderNode &) = delete;

private:
  // Callback for incoming H264 compressed image messages
  void compressed_callback(nitros::NitrosCompressedImage::SharedPtr msg);

  // Callback invoked when decoder produces uncompressed frame
  void on_decoded_frame(DecodedFrame && frame);

  // Subscription to input H264 compressed image messages
  rclcpp::Subscription<nitros::NitrosCompressedImage>::SharedPtr compressed_sub_;

  // Publisher for output RGB8 image messages
  rclcpp::Publisher<nitros::NitrosImage>::SharedPtr image_pub_;

  // V4L2 decoder implementation
  std::unique_ptr<V4L2Decoder> decoder_;

  // Decoder parameters
  int32_t max_bitstream_size_;
  bool low_latency_;
  int32_t output_width_;
  int32_t output_height_;

  // CUDA stream for async memory operations
  cudaStream_t cuda_stream_{nullptr};

  // VPI NV12 -> RGB8 color conversion
  VPIStream vpi_stream_{nullptr};
  codec::VPIFormatConverter vpi_converter_;

  // Memory pool for RGB8 output images
  nitros::CUDAMemoryPool output_pool_;
  static constexpr size_t kOutputPoolBlockCount = 40;

  // Track frame drops between publishes
  uint64_t last_frames_dropped_{0};
};

}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_DECODER__DECODER_NODE_HPP_
