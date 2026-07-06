// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_ROS_H264_ENCODER__ENCODER_V4L2_IMPL_HPP_
#define ISAAC_ROS_H264_ENCODER__ENCODER_V4L2_IMPL_HPP_

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
{

/// Configuration parameters for V4L2 H264 encoder
struct EncoderConfig
{
  uint32_t width{1920};             ///< Input frame width in pixels
  uint32_t height{1200};            ///< Input frame height in pixels
  uint32_t qp{20};                  ///< Quantization parameter (0-51)
  int32_t profile{0};               ///< H264 profile: 0=Baseline, 1=Main, 2=High
  int32_t hw_preset_type{0};        ///< HW preset: 0-3 for Tegra, 0-7 for CUVID
  int32_t iframe_interval{5};       ///< Interval between I-frames
  int32_t idr_interval{256};        ///< Interval between IDR frames
  int32_t num_bframes{0};           ///< Number of B-frames between P-frames
  int32_t entropy{1};               ///< Entropy coding: 0=CAVLC, 1=CABAC
  int32_t rate_control_mode{0};     ///< Rate control: 0=CQP, 1=CBR, 2=VBR
  int32_t bitrate{20000000};        ///< Target bitrate in bits per second
  int32_t framerate{30};            ///< Target frame rate
  int32_t level{14};                ///< H264 level (0-14)
};

/// Output structure containing encoded H264 frame data
struct EncodedFrame
{
  std::vector<uint8_t> data;  ///< H264 bitstream data (host memory, for Tegra)
  /// Device pointer to H264 data (for dGPU, valid until callback returns)
  const void * device_ptr{nullptr};
  size_t size{0};              ///< Size in bytes (used with device_ptr)
  uint64_t timestamp_ns{0};    ///< Timestamp in nanoseconds
  std::string frame_id;        ///< Frame ID from input message
  bool is_keyframe{false};     ///< True if this is an I-frame
};

/// Callback type for receiving encoded frames
using EncodedFrameCallback = std::function<void(EncodedFrame &&)>;

/// Frame statistics for monitoring encoder health
struct EncoderFrameStats
{
  uint64_t frames_in{0};       ///< Number of frames submitted to encoder
  uint64_t frames_out{0};      ///< Number of frames output from encoder
  uint64_t frames_dropped{0};  ///< Frames dropped (metadata queue empty)
  uint64_t pending{0};         ///< Frames currently in V4L2 pipeline (metadata queue size)

  /// Check if any frames were dropped (indicates a bug)
  bool has_drops() const {return frames_dropped > 0;}
};

/// V4L2-based H264 encoder implementation
/// Supports both CUVID (dGPU) and Tegra (iGPU) platforms
class V4L2Encoder
{
public:
  V4L2Encoder();
  ~V4L2Encoder();

  V4L2Encoder(const V4L2Encoder &) = delete;
  V4L2Encoder & operator=(const V4L2Encoder &) = delete;

  /// Initialize encoder with given configuration
  /// @param config Encoder configuration parameters
  /// @return true on success, false on failure
  bool initialize(const EncoderConfig & config);

  /// Shutdown encoder and release resources
  void shutdown();

  /// Encode NV12 frame from device memory with multi-planar support
  /// @param y_device_ptr Pointer to Y plane in device memory
  /// @param uv_device_ptr Pointer to UV plane in device memory
  /// @param y_stride Stride of Y plane in bytes
  /// @param uv_stride Stride of UV plane in bytes
  /// @param timestamp_ns Timestamp to associate with this frame
  /// @param frame_id Frame ID from input message
  /// @param stream CUDA stream for async copy (synchronized before V4L2 QBUF)
  /// @return true on success, false on failure
  bool encode_frame(
    const uint8_t * y_device_ptr,
    const uint8_t * uv_device_ptr,
    uint32_t y_stride,
    uint32_t uv_stride,
    uint64_t timestamp_ns,
    const std::string & frame_id,
    cudaStream_t stream);

  /// Set callback for receiving encoded frames
  /// @param callback Function to call when encoded frame is ready
  void set_output_callback(EncodedFrameCallback callback);

  /// Check if encoder is initialized
  bool is_initialized() const {return initialized_;}

  /// Get current encoder configuration
  const EncoderConfig & config() const {return config_;}

  /// Get frame statistics for monitoring
  /// @return Current frame counts (in, out, dropped)
  EncoderFrameStats get_frame_stats() const;

  /// Invoke the output callback with encoded frame (internal use)
  void invoke_callback(EncodedFrame && frame);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  EncoderConfig config_;
  bool initialized_{false};

  EncodedFrameCallback output_callback_;
  std::mutex callback_mutex_;
};

}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_ENCODER__ENCODER_V4L2_IMPL_HPP_
