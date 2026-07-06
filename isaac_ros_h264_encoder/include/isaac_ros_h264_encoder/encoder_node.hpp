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

#ifndef ISAAC_ROS_H264_ENCODER__ENCODER_NODE_HPP_
#define ISAAC_ROS_H264_ENCODER__ENCODER_NODE_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"
#include "isaac_ros_nitros/types/cuda_memory_pool.hpp"
#include "isaac_ros_h264_encoder/encoder_v4l2_impl.hpp"
#include "vpi_format_converter.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
{

class EncoderNode : public rclcpp::Node
{
public:
  explicit EncoderNode(const rclcpp::NodeOptions & options);
  ~EncoderNode();

  EncoderNode(const EncoderNode &) = delete;
  EncoderNode & operator=(const EncoderNode &) = delete;

private:
  // Initializes V4L2 encoder and output pool using actual frame dimensions.
  // Called once on first received frame.
  bool initialize_encoder(uint32_t width, uint32_t height);

  void image_callback(nitros::NitrosImage::SharedPtr msg);
  void encode_nv12(const nitros::NitrosImage & msg);
  void convert_and_encode(const nitros::NitrosImage & msg);

  // Callback invoked when encoder produces compressed frame
  void on_encoded_frame(EncodedFrame && frame);

  // Subscription to input NV12 image messages
  rclcpp::Subscription<nitros::NitrosImage>::SharedPtr image_sub_;

  // Publisher for output H264 compressed image messages
  rclcpp::Publisher<nitros::NitrosCompressedImage>::SharedPtr compressed_pub_;

  // V4L2 encoder implementation
  std::unique_ptr<V4L2Encoder> encoder_;

  // Encoder parameters
  int32_t input_width_;
  int32_t input_height_;
  bool encoder_initialized_{false};
  int32_t qp_;
  int32_t hw_preset_type_;
  int32_t profile_;
  int32_t iframe_interval_;
  int32_t idr_interval_;
  int32_t num_bframes_;
  int32_t entropy_;
  int32_t bitrate_;
  int32_t framerate_;
  int32_t level_;
  std::string config_;

  // CUDA stream for input path (image_callback -> encode_frame)
  cudaStream_t input_stream_{nullptr};

  // CUDA stream for output path (on_encoded_frame, called from encoder_thread)
  cudaStream_t output_stream_{nullptr};

  // VPI RGB8 -> NV12 conversion
  VPIStream vpi_stream_{nullptr};
  codec::VPIFormatConverter vpi_converter_;
  uint8_t * nv12_staging_ptr_{nullptr};

  // Memory pool for output compressed H.264 data
  nitros::CUDAMemoryPool output_pool_;

  // Track frame drops between publishes
  uint64_t last_frames_dropped_{0};
};

}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_ENCODER__ENCODER_NODE_HPP_
