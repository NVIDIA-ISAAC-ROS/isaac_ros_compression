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
#include <optional>
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
  output_height_(declare_parameter<int32_t>("output_height", 1200)),
  output_encoding_(declare_parameter<std::string>("output_encoding", "rgb8"))
{
  RCLCPP_INFO(get_logger(), "[DecoderNode] Initializing H264 Decoder Node");

  if (output_encoding_ != "rgb8" && output_encoding_ != "nv12" &&
    output_encoding_ != "mono8")
  {
    RCLCPP_WARN(get_logger(),
      "[DecoderNode] Unsupported output_encoding '%s'; supported values are "
      "'rgb8', 'nv12', 'mono8'. Falling back to 'rgb8'.",
      output_encoding_.c_str());
    output_encoding_ = "rgb8";
  }
  RCLCPP_INFO(get_logger(), "[DecoderNode] Output encoding: %s", output_encoding_.c_str());

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

  // Output geometry depends on the configured encoding. For nv12/mono8 the row
  // stride equals the width (1 byte per luma sample); rgb8 is 3 bytes/pixel.
  uint32_t out_step;
  size_t block_bytes;
  if (output_encoding_ == "mono8") {
    out_step = width;
    block_bytes = static_cast<size_t>(out_step) * height;
  } else if (output_encoding_ == "nv12") {
    out_step = width;
    block_bytes = static_cast<size_t>(out_step) * height * 3 / 2;
  } else {  // "rgb8"
    out_step = width * 3;
    block_bytes = static_cast<size_t>(out_step) * height;
  }

  // The pool block is sized from the first frame. A smaller later frame fits
  // the oversized block, but a larger one (resolution increase) would not, so
  // recreate the pool to grow the block. destroy() blocks until in-flight
  // output frames are recycled, so the new block size always takes effect.
  if (output_pool_.initialized() && block_bytes > output_pool_.block_size()) {
    output_pool_.destroy();
  }
  if (!output_pool_.initialized()) {
    cudaError_t err = output_pool_.create(
      block_bytes, kOutputPoolBlockCount,
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
  // from_pool throws if the pool is exhausted (or, defensively, if the request
  // still exceeds the block size). Drop the frame instead of letting an
  // uncaught exception crash this callback. std::optional keeps the WriteHandle
  // (not default-constructible) alive until the end of the function so its
  // completion event is recorded after the copies, as before.
  std::optional<nitros::WriteHandle> write_handle_opt;
  try {
    write_handle_opt.emplace(
      output.from_pool(
        output_pool_, width, height, out_step, output_encoding_, cuda_stream_));
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      get_logger(),
      "[DecoderNode] Failed to acquire output image from pool (%ux%u %s): %s",
      width, height, output_encoding_.c_str(), e.what());
    decoder_->return_buffer(frame.buffer_index);
    return;
  }
  nitros::WriteHandle & write_handle = *write_handle_opt;

  // On cuvid (dGPU), Y and UV live in one contiguous buffer, so the UV plane
  // pointer is device_ptr + uv_offset. On Tegra (NvBufSurfaceMap + cudaHostRegister),
  // the two planes come from independent mappings, so uv_device_ptr is set.
  uint8_t * uv_ptr = frame.uv_device_ptr != nullptr ?
    frame.uv_device_ptr :
    frame.device_ptr + frame.uv_offset;

  if (output_encoding_ == "rgb8") {
    // DEPRECATED: NV12->RGB8 conversion for backward compatibility.
    // The V4L2 decoder outputs NV12 natively. This conversion maintains the
    // RGB8 output contract for downstream consumers. Will be removed in the
    // next major release; consumers should migrate to accept NV12 directly.
    VPIImageData nv12_data;
    codec::FillNV12ImageData(
      nv12_data,
      const_cast<uint8_t *>(frame.device_ptr), frame.y_pitch,
      const_cast<uint8_t *>(uv_ptr), frame.uv_pitch,
      width, height);

    VPIImageData rgb_data;
    codec::FillRGB8ImageData(rgb_data, write_handle.get_ptr(), out_step, width, height);

    VPIStatus vpi_err = vpi_converter_.convert(vpi_stream_, nv12_data, rgb_data);
    if (vpi_err != VPI_SUCCESS) {
      RCLCPP_ERROR(get_logger(), "[DecoderNode] VPI convert failed: %s",
        vpiStatusGetName(vpi_err));
      decoder_->return_buffer(frame.buffer_index);
      return;
    }
    // vpiStreamSync inside convert() guarantees the NV12 read from
    // frame.device_ptr is complete before the capture buffer is returned below.
  } else {
    // nv12/mono8: copy the decoded planes directly, no color conversion.
    // The Y (luma) plane alone is a valid grayscale image, so mono8 copies only
    // it; nv12 additionally copies the interleaved UV plane.
    uint8_t * dst = write_handle.get_ptr();
    cudaError_t err = cudaMemcpy2DAsync(
      dst, out_step,
      frame.device_ptr, frame.y_pitch,
      width, height, cudaMemcpyDeviceToDevice, cuda_stream_);
    if (err == cudaSuccess && output_encoding_ == "nv12") {
      // UV plane is interleaved: width bytes per row (w/2 samples x 2), h/2 rows.
      // It is laid out contiguously after the Y plane in the NitrosImage buffer.
      uint8_t * dst_uv = dst + static_cast<size_t>(out_step) * height;
      err = cudaMemcpy2DAsync(
        dst_uv, out_step,
        uv_ptr, frame.uv_pitch,
        width, height / 2, cudaMemcpyDeviceToDevice, cuda_stream_);
    }
    // Always drain the stream before returning the capture buffer: even if a
    // copy failed to enqueue, an earlier async copy (e.g. the Y plane when the
    // UV copy fails) may still be reading frame.device_ptr, so the buffer must
    // not be returned for reuse until the stream is idle.
    cudaError_t sync_err = cudaStreamSynchronize(cuda_stream_);
    if (err == cudaSuccess) {
      err = sync_err;
    }
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "[DecoderNode] Failed to copy decoded %s frame: %s",
        output_encoding_.c_str(), cudaGetErrorString(err));
      decoder_->return_buffer(frame.buffer_index);
      return;
    }
  }

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
