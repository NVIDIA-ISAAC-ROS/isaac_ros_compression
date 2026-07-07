// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_h264_decoder/decoder_node.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_decoder
{

DecoderNode::DecoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("h264_decoder", options),
  max_bitstream_size_(declare_parameter<int32_t>("max_bitstream_size", 2097152)),
  low_latency_(declare_parameter<bool>("low_latency", false)),
  output_width_(declare_parameter<int32_t>("output_width", 1920)),
  output_height_(declare_parameter<int32_t>("output_height", 1200))
{
  RCLCPP_INFO(get_logger(), "[DecoderNode] Initializing H264 Decoder Node");

  cudaError_t cuda_err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[DecoderNode] Failed to create CUDA stream: %s",
      cudaGetErrorString(cuda_err));
    return;
  }

  VPIStatus vpi_s = vpiStreamCreateWrapperCUDA(cuda_stream_, VPI_BACKEND_CUDA, &vpi_stream_);
  if (vpi_s != VPI_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "[DecoderNode] vpiStreamCreateWrapperCUDA failed: %s",
      vpiStatusGetName(vpi_s));
    return;
  }

  DecoderConfig dec_config;
  dec_config.max_bitstream_size = max_bitstream_size_;
  dec_config.low_latency = low_latency_;

  decoder_ = std::make_unique<V4L2Decoder>();
  decoder_->set_output_callback(
    [this](DecodedFrame && frame) {
      on_decoded_frame(std::move(frame));
    });

  if (!decoder_->initialize(dec_config)) {
    RCLCPP_ERROR(get_logger(), "[DecoderNode] Failed to initialize V4L2 decoder");
    return;
  }

  // Enable intra-process communication for zero-copy when nodes are in same process
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_pub_ = create_publisher<nitros::NitrosImage>(
    "image_uncompressed", rclcpp::QoS(1), pub_options);

  compressed_sub_ = create_subscription<nitros::NitrosCompressedImage>(
    "image_compressed", rclcpp::QoS(1),
    [this](nitros::NitrosCompressedImage::SharedPtr msg) {
      compressed_callback(msg);
    },
    sub_options);

  RCLCPP_INFO(get_logger(), "[DecoderNode] H264 Decoder Node initialized");
}

DecoderNode::~DecoderNode()
{
  if (decoder_) {
    decoder_->shutdown();
  }

  if (vpi_stream_) {
    vpiStreamDestroy(vpi_stream_);
  }

  output_pool_.destroy();

  if (cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
    cuda_stream_ = nullptr;
  }
}

void DecoderNode::compressed_callback(nitros::NitrosCompressedImage::SharedPtr msg)
{
  size_t size = msg->size();
  auto read_handle = msg->get_read_handle(cuda_stream_);
  const uint8_t * data_ptr = read_handle.get_ptr();

  if (!data_ptr) {
    RCLCPP_ERROR(get_logger(), "[DecoderNode] Input compressed image pointer is null");
    return;
  }

  uint64_t timestamp_ns = static_cast<uint64_t>(msg->timestamp_sec) * 1000000000ULL +
    msg->timestamp_nsec;

  if (!decoder_->decode_frame(data_ptr, size, timestamp_ns, msg->frame_id, cuda_stream_)) {
    RCLCPP_ERROR(get_logger(), "[DecoderNode] Failed to decode frame");
  }
}

void DecoderNode::on_decoded_frame(DecodedFrame && frame)
{
  uint32_t width = frame.width;
  uint32_t height = frame.height;
  size_t rgb_size = static_cast<size_t>(width) * height * 3;

  if (!output_pool_.initialized()) {
    cudaError_t err = output_pool_.create(
      rgb_size, kOutputPoolBlockCount,
      nitros::CUDAMemoryPool::MemoryType::Device);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(
        get_logger(), "[DecoderNode] Failed to create output memory pool: %s",
        cudaGetErrorString(err));
      decoder_->return_buffer(frame.buffer_index);
      return;
    }
  }

  nitros::NitrosImage output;
  uint32_t rgb_step = width * 3;
  auto write_handle = output.from_pool(
    output_pool_, width, height, rgb_step, "rgb8", cuda_stream_);

  // DEPRECATED: NV12->RGB8 conversion for backward compatibility.
  // The V4L2 decoder outputs NV12 natively. This conversion maintains the
  // RGB8 output contract for downstream consumers. Will be removed in the
  // next major release; consumers should migrate to accept NV12 directly.
  VPIImageData nv12_data;
  codec::FillNV12ImageData(
    nv12_data,
    const_cast<uint8_t *>(frame.device_ptr), frame.y_pitch,
    const_cast<uint8_t *>(frame.device_ptr + frame.uv_offset), frame.uv_pitch,
    width, height);

  VPIImageData rgb_data;
  codec::FillRGB8ImageData(rgb_data, write_handle.get_ptr(), rgb_step, width, height);

  VPIStatus vpi_err = vpi_converter_.convert(vpi_stream_, nv12_data, rgb_data);
  if (vpi_err != VPI_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "[DecoderNode] VPI convert failed: %s",
      vpiStatusGetName(vpi_err));
    decoder_->return_buffer(frame.buffer_index);
    return;
  }

  // vpiStreamSync inside convert() guarantees the NV12 read from frame.device_ptr
  // is complete, so the V4L2 capture buffer can be safely returned for reuse.
  decoder_->return_buffer(frame.buffer_index);

  output.timestamp_sec = static_cast<uint32_t>(frame.timestamp_ns / 1000000000ULL);
  output.timestamp_nsec = static_cast<uint32_t>(frame.timestamp_ns % 1000000000ULL);
  output.frame_id = frame.frame_id;

  auto stats = decoder_->get_frame_stats();
  if (stats.frames_dropped > last_frames_dropped_) {
    uint64_t new_drops = stats.frames_dropped - last_frames_dropped_;
    RCLCPP_ERROR(
      get_logger(),
      "[DecoderNode] FRAME DROP DETECTED: %lu new drop(s), total=%lu "
      "(in=%lu, out=%lu, pending=%lu)",
      new_drops, stats.frames_dropped,
      stats.frames_in, stats.frames_out, stats.pending);
    last_frames_dropped_ = stats.frames_dropped;
  }

  image_pub_->publish(output);
}

}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::h264_decoder::DecoderNode)
