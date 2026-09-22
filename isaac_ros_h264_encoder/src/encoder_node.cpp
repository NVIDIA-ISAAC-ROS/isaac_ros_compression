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

#include "isaac_ros_h264_encoder/encoder_node.hpp"

#include <memory>
#include <string>
#include <utility>

#include "cuda_buffer/cuda_buffer_api.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
{

EncoderNode::EncoderNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("h264_encoder", options),
  input_width_(declare_parameter<int32_t>("input_width", 1920)),
  input_height_(declare_parameter<int32_t>("input_height", 1200)),
  qp_(declare_parameter<int32_t>("qp", 20)),
  hw_preset_type_(declare_parameter<int32_t>("hw_preset_type", 1)),
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
  // Accept GPU-backed image buffers; from_input_buffer promotes CPU buffers as needed.
  sub_options.acceptable_buffer_backends = "any";

  compressed_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
    "image_compressed", rclcpp::QoS(1), pub_options);

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image_raw", rclcpp::QoS(1),
    [this](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
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
    enc_config.hw_preset_type = 1;
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
    enc_config.hw_preset_type = 1;
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

  RCLCPP_INFO(
    get_logger(), "[EncoderNode] H264 Encoder initialized: %ux%u (accepts RGB8, NV12, or Mono8)",
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
  if (mono8_uv_staging_ptr_) {
    cudaFree(mono8_uv_staging_ptr_);
  }

  if (input_stream_) {
    cudaStreamDestroy(input_stream_);
    input_stream_ = nullptr;
  }

  if (output_stream_) {
    cudaStreamDestroy(output_stream_);
    output_stream_ = nullptr;
  }
}

void EncoderNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
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
  } else if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
    encode_mono8(*msg);
  } else {
    convert_and_encode(*msg);
  }
}

void EncoderNode::encode_nv12(const sensor_msgs::msg::Image & msg)
{
  auto read_handle = cuda_buffer_backend::from_input_buffer(msg.data, input_stream_);
  const uint8_t * base_ptr = read_handle.get_ptr();
  if (!base_ptr) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Input image buffer pointer is null");
    return;
  }

  // Packed NV12 in a sensor_msgs::Image: the Y plane occupies `height` rows of
  // `step` bytes, immediately followed by the interleaved UV plane. Both planes
  // share the row stride `step`.
  const uint8_t * y_ptr = base_ptr;
  const uint8_t * uv_ptr = base_ptr + static_cast<size_t>(msg.step) * msg.height;
  uint32_t y_stride = msg.step;
  uint32_t uv_stride = msg.step;

  uint64_t timestamp_ns = static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL +
    msg.header.stamp.nanosec;

  if (!encoder_->encode_frame(
      y_ptr, uv_ptr, y_stride, uv_stride,
      timestamp_ns, msg.header.frame_id, input_stream_))
  {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Failed to encode frame");
  }
}

// mono8 (grayscale, e.g. RealSense infrared streams) maps directly onto the
// NV12 luma plane, so no color-space conversion is needed: the input buffer is
// fed straight in as Y and a constant neutral-chroma (UV = 128) plane is
// synthesized once and reused. This is the inverse of the NV12->MONO8 Y-plane
// copy used by isaac_ros_image_proc's ImageFormatConverterNode.
void EncoderNode::encode_mono8(const sensor_msgs::msg::Image & msg)
{
  auto read_handle = cuda_buffer_backend::from_input_buffer(msg.data, input_stream_);
  const uint8_t * input_ptr = read_handle.get_ptr();
  if (!input_ptr) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Input image buffer pointer is null");
    return;
  }

  if (!encoder_) {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Encoder not initialized");
    return;
  }

  // encode_frame reads the UV plane sized to the encoder's CONFIGURED
  // resolution (cudaMemcpy2DAsync of config.width x config.height/2), which may
  // differ from this message if the encoder was initialized on an earlier frame
  // of a different size. Size the staging buffer to the encoder's resolution to
  // avoid an over-read, and reallocate if that resolution changes.
  const EncoderConfig & enc_config = encoder_->config();
  size_t uv_size = static_cast<size_t>(enc_config.width) * (enc_config.height / 2);
  if (!ensure_mono8_uv_staging(uv_size)) {
    return;
  }

  uint64_t timestamp_ns = static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL +
    msg.header.stamp.nanosec;

  // UV staging is a tightly packed neutral-chroma plane at the encoder's width,
  // so its row stride equals the configured width.
  if (!encoder_->encode_frame(
      input_ptr, mono8_uv_staging_ptr_, msg.step, enc_config.width,
      timestamp_ns, msg.header.frame_id, input_stream_))
  {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Failed to encode frame");
  }
}

bool EncoderNode::ensure_mono8_uv_staging(size_t uv_size)
{
  if (mono8_uv_staging_ptr_ && uv_size == mono8_uv_staging_size_) {
    return true;
  }
  if (mono8_uv_staging_ptr_) {
    cudaFree(mono8_uv_staging_ptr_);
    mono8_uv_staging_ptr_ = nullptr;
    mono8_uv_staging_size_ = 0;
  }
  cudaError_t err = cudaMalloc(&mono8_uv_staging_ptr_, uv_size);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(),
      "[EncoderNode] cudaMalloc for mono8 UV staging failed: %s", cudaGetErrorString(err));
    mono8_uv_staging_ptr_ = nullptr;
    return false;
  }
  // 0x80 = neutral chroma; renders the luma-only stream as true grayscale.
  err = cudaMemset(mono8_uv_staging_ptr_, 0x80, uv_size);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(get_logger(),
      "[EncoderNode] cudaMemset for mono8 UV staging failed: %s", cudaGetErrorString(err));
    cudaFree(mono8_uv_staging_ptr_);
    mono8_uv_staging_ptr_ = nullptr;
    return false;
  }
  mono8_uv_staging_size_ = uv_size;
  return true;
}

// DEPRECATED: RGB8->NV12 conversion for backward compatibility.
// The V4L2 encoder requires NV12 input natively. This conversion maintains the
// RGB8 input contract for upstream producers. Will be removed in the next major
// release; producers should migrate to supply NV12 directly.
void EncoderNode::convert_and_encode(const sensor_msgs::msg::Image & msg)
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

  auto read_handle = cuda_buffer_backend::from_input_buffer(msg.data, input_stream_);
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

  uint64_t timestamp_ns = static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL +
    msg.header.stamp.nanosec;

  if (!encoder_->encode_frame(
      nv12_staging_ptr_, uv_ptr, width, width,
      timestamp_ns, msg.header.frame_id, input_stream_))
  {
    RCLCPP_ERROR(get_logger(), "[EncoderNode] Failed to encode frame");
  }
}

void EncoderNode::on_encoded_frame(EncodedFrame && frame)
{
  // Pick the active source of bytes (dGPU device pointer, Tegra host pointer,
  // or legacy host-vector fallback).
  size_t size = (frame.device_ptr != nullptr || frame.host_ptr != nullptr) ?
    frame.size :
    frame.data.size();

  auto output = std::make_unique<sensor_msgs::msg::CompressedImage>();
  output->format = "h264";
  output->data = cuda_buffer_backend::allocate_buffer(size);

  // Scope the write handle so its CUDA completion event is recorded on
  // output_stream_ before the message is published. on_encoded_frame runs on the
  // encoder_thread, so output_stream_ (not the executor's input_stream_) is used.
  {
    auto write_handle = cuda_buffer_backend::from_output_buffer(output->data, output_stream_);
    uint8_t * dst = write_handle.get_ptr();

    cudaError_t err;
    if (frame.device_ptr) {
      // dGPU path: dataPtr on the V4L2 capture buffer is already CUDA device
      // memory, so a single D2D copy moves the bitstream to the output buffer.
      err = cudaMemcpyAsync(
        dst, frame.device_ptr, size, cudaMemcpyDeviceToDevice, output_stream_);
    } else if (frame.host_ptr) {
      // Tegra (nvgpu) path: the NVENC bitstream buffer is CPU-mapped. Issue
      // a synchronous cudaMemcpy to the output buffer. This blocks until the copy is done,
      // so the V4L2 capture buffer is safe to re-enqueue when this callback returns.
      err = cudaMemcpy(dst, frame.host_ptr, size, cudaMemcpyHostToDevice);
    } else {
      // Fallback: H2D copy from host vector to output (device)
      // This path should not be hit in normal operation.
      RCLCPP_WARN_ONCE(get_logger(), "[EncoderNode] Using fallback H2D copy path");
      err = cudaMemcpyAsync(
        dst, frame.data.data(), size, cudaMemcpyHostToDevice, output_stream_);
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
  }

  output->header.stamp.sec = static_cast<int32_t>(frame.timestamp_ns / 1000000000ULL);
  output->header.stamp.nanosec = static_cast<uint32_t>(frame.timestamp_ns % 1000000000ULL);
  output->header.frame_id = frame.frame_id;

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

  compressed_pub_->publish(std::move(output));
}

}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::h264_encoder::EncoderNode)
