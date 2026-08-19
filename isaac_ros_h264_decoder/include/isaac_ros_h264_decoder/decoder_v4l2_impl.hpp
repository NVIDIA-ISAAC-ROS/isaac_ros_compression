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

#ifndef ISAAC_ROS_H264_DECODER__DECODER_V4L2_IMPL_HPP_
#define ISAAC_ROS_H264_DECODER__DECODER_V4L2_IMPL_HPP_

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
namespace h264_decoder
{

/// Configuration parameters for V4L2 H264 decoder
struct DecoderConfig
{
  uint32_t max_bitstream_size{2097152};  ///< Maximum bitstream buffer size (2MB default)
  bool low_latency{false};               ///< Enable low latency decode mode
  bool output_nv12{true};                ///< Output format: true=NV12, false=RGB
};

/// Output structure containing decoded frame data
struct DecodedFrame
{
  uint8_t * device_ptr{nullptr};     ///< Decoded pixel data on GPU (NV12 Y plane)
  uint8_t * uv_device_ptr{nullptr};  ///< UV plane device pointer. When nullptr, UV is at
                                     ///< device_ptr + uv_offset (cuvid single-buffer layout).
                                     ///< When set (Tegra (nvgpu): cudaHostRegister path), Y and UV
                                     ///< come from separate NvBufSurfaceMap plane mappings
                                     ///< and are not necessarily contiguous.
  uint32_t width{0};                ///< Frame width in pixels
  uint32_t height{0};               ///< Frame height in pixels
  uint32_t y_pitch{0};              ///< Y plane pitch (stride) in bytes
  uint32_t uv_pitch{0};             ///< UV plane pitch (stride) in bytes
  uint32_t uv_offset{0};            ///< UV plane offset from device_ptr in bytes
  uint64_t timestamp_ns{0};         ///< Timestamp in nanoseconds
  std::string frame_id;             ///< Frame ID from input message
  int buffer_index{-1};             ///< V4L2 capture buffer index (for returning buffer)
};

/// Callback type for receiving decoded frames
using DecodedFrameCallback = std::function<void(DecodedFrame &&)>;

/// Frame statistics for monitoring decoder health
struct DecoderFrameStats
{
  uint64_t frames_in{0};       ///< Number of frames submitted to decoder
  uint64_t frames_out{0};      ///< Number of frames output from decoder
  uint64_t frames_dropped{0};  ///< Frames dropped (metadata queue empty or error)
  uint64_t pending{0};         ///< Frames currently in V4L2 pipeline (metadata queue size)

  /// Check if any frames were dropped (indicates a bug)
  bool has_drops() const {return frames_dropped > 0;}
};

/// V4L2-based H264 decoder implementation
/// Supports both CUVID (dGPU) and Tegra (iGPU) platforms
class V4L2Decoder
{
public:
  V4L2Decoder();
  ~V4L2Decoder();

  V4L2Decoder(const V4L2Decoder &) = delete;
  V4L2Decoder & operator=(const V4L2Decoder &) = delete;

  /// Initialize decoder with given configuration
  /// @param config Decoder configuration parameters
  /// @return true on success, false on failure
  bool initialize(const DecoderConfig & config);

  /// Shutdown decoder and release resources
  void shutdown();

  /// Decode H264 frame from device memory
  /// @param data Pointer to H264 bitstream data in device memory
  /// @param size Size of bitstream data in bytes
  /// @param timestamp_ns Timestamp to associate with this frame
  /// @param frame_id Frame ID to associate with this frame
  /// @param stream CUDA stream for synchronization (data must be ready on this stream)
  /// @return true on success, false on failure
  bool decode_frame(
    const uint8_t * data,
    size_t size,
    uint64_t timestamp_ns,
    const std::string & frame_id,
    cudaStream_t stream);

  /// Set callback for receiving decoded frames
  /// @param callback Function to call when decoded frame is ready
  void set_output_callback(DecodedFrameCallback callback);

  /// Return a capture buffer back to the decoder after processing
  /// @param buffer_index The buffer index from DecodedFrame::buffer_index
  void return_buffer(int buffer_index);

  /// Check if decoder is initialized
  bool is_initialized() const {return initialized_;}

  /// Get current decoder configuration
  const DecoderConfig & config() const {return config_;}

  /// Get decoded video width (available after first frame decoded)
  uint32_t video_width() const;

  /// Get decoded video height (available after first frame decoded)
  uint32_t video_height() const;

  /// Get frame statistics for monitoring
  /// @return Current frame counts (in, out, dropped)
  DecoderFrameStats get_frame_stats() const;

  /// Invoke the output callback with decoded frame (internal use)
  void invoke_callback(DecodedFrame && frame);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  DecoderConfig config_;
  bool initialized_{false};

  DecodedFrameCallback output_callback_;
  std::mutex callback_mutex_;
};

}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_DECODER__DECODER_V4L2_IMPL_HPP_
