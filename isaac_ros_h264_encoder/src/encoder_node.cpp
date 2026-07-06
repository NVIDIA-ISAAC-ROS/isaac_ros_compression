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

#include "isaac_ros_h264_encoder/encoder_node.hpp"

#include <memory>
#include <string>
#include <utility>

#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
{

namespace
{
// Memory pool configuration
constexpr size_t kOutputPoolBlockCount = 40;
}  // namespace

EncoderNode::EncoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("h264_encoder", options),
  input_width_(declare_parameter<int32_t>("input_width", 1920)),
  input_height_(declare_parameter<int32_t>("input_height", 1200)),
  qp_(declare_parameter<int32_t>("qp", 20)),
  hw_preset_type_(declare_parameter<int32_t>("hw_preset_type", 0)),
  profile_(declare_parameter<int32_t>("profile", 0)),
  iframe_interval_(declare_parameter<int32_t>("iframe_interval", 5)),
  idr_interval_(declare_parameter<int32_t>("idr_interval", 5)),
  num_bframes_(declare_parameter<int32_t>("num_bframes", 0)),
  entropy_(declare_parameter<int32_t>("entropy", 1)),
  bitrate_(declare_parameter<int32_t>("bitrate", 20000000)),
  framerate_(declare_parameter<int32_t>("framerate", 30)),
  level_(declare_parameter<int32_t>("level", 14)),
  config_(declare_parameter<std::string>("config", "pframe_cqp"))
{
  RCLCPP_INFO(get_logger(), "[EncoderNode] Initializing H264 Encoder Node");

  // Create separate streams for input and output paths to avoid thread contention:
  // - input_stream_: used by image_callback (ROS executor thread)
  // - output_stream_: used by on_encoded_frame (encoder_thread)
  cudaError_t cuda_err = cudaStreamCreateWithFlags(&input_stream_, cudaStreamNonBlocking);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[EncoderNode] Failed to create input CUDA stream: %s",
      cudaGetErrorString(cuda_err));
    return;
  }

  cuda_err = cudaStreamCreateWithFlags(&output_stream_, cudaStreamNonBlocking);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[EncoderNode] Failed to create output CUDA stream: %s",
      cudaGetErrorString(cuda_err));
    cudaStreamDestroy(input_stream_);
    input_stream_ = nullptr;
    return;
  }

  VPIStatus vpi_s = vpiStreamCreateWrapperCUDA(input_stream_, VPI_BACKEND_CUDA, &vpi_stream_);
  if (vpi_s != VPI_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] vpiStreamCreateWrapperCUDA failed: %s",
      vpiStatusGetName(vpi_s));
    return;
  }

  // Enable intra-process communication for zero-copy when nodes are in same process
  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  compressed_pub_ = create_publisher<nitros::NitrosCompressedImage>(
    "image_compressed", rclcpp::QoS(1), pub_options);

  image_sub_ = create_subscription<nitros::NitrosImage>(
    "image_raw", rclcpp::QoS(1),
    [this](nitros::NitrosImage::SharedPtr msg) {
      image_callback(msg);
    },
    sub_options);

  RCLCPP_INFO(get_logger(),
    "[EncoderNode] H264 Encoder Node ready (encoder deferred until first frame)");
}

bool EncoderNode::initialize_encoder(uint32_t width, uint32_t height)
{
  input_width_ = width;
  input_height_ = height;

  EncoderConfig enc_config;
  enc_config.width = width;
  enc_config.height = height;
  enc_config.qp = qp_;
  enc_config.profile = profile_;
  enc_config.hw_preset_type = hw_preset_type_;
  enc_config.iframe_interval = iframe_interval_;
  enc_config.idr_interval = idr_interval_;
  enc_config.num_bframes = num_bframes_;
  enc_config.entropy = entropy_;
  enc_config.bitrate = bitrate_;
  enc_config.framerate = framerate_;
  enc_config.level = level_;

  // Apply preset configs (matching GXF implementation)
  if (config_ == "iframe_cqp") {
    // I-frame only mode: every frame is an IDR frame
    enc_config.rate_control_mode = 0;  // CQP mode
    enc_config.qp = 20;
    enc_config.iframe_interval = 1;
    enc_config.idr_interval = 1;
    enc_config.profile = 1;  // Main profile
    enc_config.hw_preset_type = 0;  // ULTRAFAST
    enc_config.entropy = 0;  // CAVLC
    RCLCPP_INFO(get_logger(),
      "[EncoderNode] Using iframe_cqp preset: qp=20, iframe_interval=1");
  } else if (config_ == "pframe_cqp") {
    // P-frame mode with periodic I-frames
    enc_config.rate_control_mode = 0;  // CQP mode
    enc_config.qp = 20;
    enc_config.iframe_interval = 5;
    enc_config.idr_interval = 5;
    enc_config.profile = 1;  // Main profile
    enc_config.hw_preset_type = 0;  // ULTRAFAST
    enc_config.entropy = 0;  // CAVLC
    RCLCPP_INFO(get_logger(),
      "[EncoderNode] Using pframe_cqp preset: qp=20, iframe_interval=5");
  } else {
    // Custom mode: use user-provided parameters
    enc_config.rate_control_mode = 1;  // CBR mode
    RCLCPP_INFO(get_logger(), "[EncoderNode] Using custom config with user parameters");
  }

  encoder_ = std::make_unique<V4L2Encoder>();
  encoder_->set_output_callback(
    [this](EncodedFrame && frame) {
      on_encoded_frame(std::move(frame));
    });

  if (!encoder_->initialize(enc_config)) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Failed to initialize V4L2 encoder");
    return false;
  }

  // Allocate conservative size for H.264 bitstream (raw NV12 size * 2 as upper bound)
  size_t output_pool_block_size = static_cast<size_t>(width) * height * 3;
  cudaError_t cuda_err = output_pool_.create(
    output_pool_block_size, kOutputPoolBlockCount,
    nitros::CUDAMemoryPool::MemoryType::Device);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[EncoderNode] Failed to create output memory pool: %s",
      cudaGetErrorString(cuda_err));
    encoder_->shutdown();
    encoder_.reset();
    return false;
  }

  RCLCPP_INFO(
    get_logger(), "[EncoderNode] H264 Encoder initialized: %ux%u (accepts RGB8 or NV12)",
    width, height);
  return true;
}

EncoderNode::~EncoderNode()
{
  if (encoder_) {
    encoder_->shutdown();
  }

  if (vpi_stream_) {
    vpiStreamDestroy(vpi_stream_);
  }
  if (nv12_staging_ptr_) {
    cudaFree(nv12_staging_ptr_);
  }

  output_pool_.destroy();

  if (input_stream_) {
    cudaStreamDestroy(input_stream_);
    input_stream_ = nullptr;
  }

  if (output_stream_) {
    cudaStreamDestroy(output_stream_);
    output_stream_ = nullptr;
  }
}

void EncoderNode::image_callback(nitros::NitrosImage::SharedPtr msg)
{
  if (!encoder_initialized_) {
    uint32_t w = msg->width;
    uint32_t h = msg->height;
    if (w != static_cast<uint32_t>(input_width_) ||
      h != static_cast<uint32_t>(input_height_))
    {
      RCLCPP_WARN(get_logger(),
        "[EncoderNode] Configured resolution %dx%d does not match incoming frame %ux%u. "
        "Using actual frame dimensions.",
        input_width_, input_height_, w, h);
    }
    if (!initialize_encoder(w, h)) {
      RCLCPP_ERROR(get_logger(),
        "[EncoderNode] Failed to initialize encoder for %ux%u. "
        "No frames will be encoded.", w, h);
      return;
    }
    encoder_initialized_ = true;
  }

  if (msg->encoding == "nv12") {
    encode_nv12(*msg);
  } else {
    convert_and_encode(*msg);
  }
}

void EncoderNode::encode_nv12(const nitros::NitrosImage & msg)
{
  if (msg.num_planes() != 2) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] NV12 requires 2 planes, got %zu", msg.num_planes());
    return;
  }

  auto read_handle = msg.get_read_handle(input_stream_);
  const uint8_t * base_ptr = read_handle.get_ptr();
  if (!base_ptr) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Input image buffer pointer is null");
    return;
  }

  const uint8_t * y_ptr = base_ptr + msg.get_plane(0).offset;
  const uint8_t * uv_ptr = base_ptr + msg.get_plane(1).offset;
  uint32_t y_stride = msg.get_plane(0).stride;
  uint32_t uv_stride = msg.get_plane(1).stride;

  uint64_t timestamp_ns = static_cast<uint64_t>(msg.timestamp_sec) * 1000000000ULL +
    msg.timestamp_nsec;

  if (!encoder_->encode_frame(
      y_ptr, uv_ptr, y_stride, uv_stride,
      timestamp_ns, msg.frame_id, input_stream_))
  {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Failed to encode frame");
  }
}

// DEPRECATED: RGB8->NV12 conversion for backward compatibility.
// The V4L2 encoder requires NV12 input natively. This conversion maintains the
// RGB8 input contract for upstream producers. Will be removed in the next major
// release; producers should migrate to supply NV12 directly.
void EncoderNode::convert_and_encode(const nitros::NitrosImage & msg)
{
  uint32_t width = msg.width;
  uint32_t height = msg.height;
  size_t nv12_size = static_cast<size_t>(width) * height * 3 / 2;

  if (!nv12_staging_ptr_) {
    cudaError_t err = cudaMalloc(&nv12_staging_ptr_, nv12_size);
    if (err != cudaSuccess) {
      RCLCPP_ERROR(get_logger(),
        "[EncoderNode] cudaMalloc for NV12 staging failed: %s", cudaGetErrorString(err));
      return;
    }
  }

  auto read_handle = msg.get_read_handle(input_stream_);
  const uint8_t * input_ptr = read_handle.get_ptr();
  if (!input_ptr) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Input image buffer pointer is null");
    return;
  }

  VPIImageData rgb_data;
  if (msg.encoding == "bgr8" || msg.encoding == "8UC3") {
    codec::FillBGR8ImageData(
      rgb_data, const_cast<uint8_t *>(input_ptr), msg.step, width, height);
  } else {
    codec::FillRGB8ImageData(
      rgb_data, const_cast<uint8_t *>(input_ptr), msg.step, width, height);
  }

  uint8_t * uv_ptr = nv12_staging_ptr_ + static_cast<size_t>(width) * height;
  VPIImageData nv12_data;
  codec::FillNV12ImageData(nv12_data, nv12_staging_ptr_, width, uv_ptr, width, width, height);

  VPIStatus vpi_err = vpi_converter_.convert(vpi_stream_, rgb_data, nv12_data);
  if (vpi_err != VPI_SUCCESS) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] VPI convert failed: %s",
      vpiStatusGetName(vpi_err));
    return;
  }

  uint64_t timestamp_ns = static_cast<uint64_t>(msg.timestamp_sec) * 1000000000ULL +
    msg.timestamp_nsec;

  if (!encoder_->encode_frame(
      nv12_staging_ptr_, uv_ptr, width, width,
      timestamp_ns, msg.frame_id, input_stream_))
  {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Failed to encode frame");
  }
}

void EncoderNode::on_encoded_frame(EncodedFrame && frame)
{
  // Get size from appropriate source (device_ptr path vs host data path)
  size_t size = frame.device_ptr ? frame.size : frame.data.size();

  // Validate encoded frame size against static pool capacity
  if (size > output_pool_.block_size()) {
    RCLCPP_ERROR(
      get_logger(),
      "[EncoderNode] Encoded frame size (%zu bytes) exceeds pool capacity (%zu bytes). "
      "Increase encoder parameters or reduce bitrate/quality.",
      size, output_pool_.block_size());
    return;
  }

  nitros::NitrosCompressedImage output;

  // Use output_stream_ here since on_encoded_frame is called from encoder_thread,
  // not the ROS executor thread that calls image_callback
  auto write_handle = output.from_pool(output_pool_, size, "h264", output_stream_);

  cudaError_t err;
  if (frame.device_ptr) {
    // Both dGPU and Tegra: D2D copy from V4L2 capture buffer (device) to output (device)
    // - dGPU: dataPtr is CUDA device memory
    // - Tegra: NvBufSurfaceMapCudaBuffer provides CUDA-accessible pointer
    err = cudaMemcpyAsync(
      write_handle.get_ptr(), frame.device_ptr, size,
      cudaMemcpyDeviceToDevice, output_stream_);
  } else {
    // Fallback: H2D copy from host vector to output (device)
    // This path should not be hit in normal operation.
    RCLCPP_WARN_ONCE(get_logger(), "[EncoderNode] Using fallback H2D copy path");
    err = cudaMemcpyAsync(
      write_handle.get_ptr(), frame.data.data(), size,
      cudaMemcpyHostToDevice, output_stream_);
  }
  if (err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[EncoderNode] cudaMemcpyAsync failed: %s", cudaGetErrorString(err));
    return;
  }

  // CRITICAL: Wait for async copy to complete before returning to encoder_thread.
  // After this callback returns, the V4L2 capture buffer is re-queued and may be
  // overwritten by the next encoded frame. Without this sync, the encoder can
  // overwrite the buffer while copy is in progress, causing data corruption.
  err = cudaStreamSynchronize(output_stream_);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(
      get_logger(), "[EncoderNode] cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
  }

  output.timestamp_sec = static_cast<uint32_t>(frame.timestamp_ns / 1000000000ULL);
  output.timestamp_nsec = static_cast<uint32_t>(frame.timestamp_ns % 1000000000ULL);
  output.frame_id = frame.frame_id;

  // Check for frame drops since last publish
  auto stats = encoder_->get_frame_stats();
  if (stats.frames_dropped > last_frames_dropped_) {
    uint64_t new_drops = stats.frames_dropped - last_frames_dropped_;
    RCLCPP_ERROR(
      get_logger(),
      "[EncoderNode] FRAME DROP DETECTED: %lu new drop(s), total=%lu "
      "(in=%lu, out=%lu, pending=%lu)",
      new_drops, stats.frames_dropped,
      stats.frames_in, stats.frames_out, stats.pending);
    last_frames_dropped_ = stats.frames_dropped;
  }

  compressed_pub_->publish(output);
}

}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::h264_encoder::EncoderNode)
