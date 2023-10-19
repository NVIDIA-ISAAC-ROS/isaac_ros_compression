// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_CONTEXT_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_CONTEXT_HPP_

#include <queue>
#include "gxf/core/component.hpp"
#include "gxf/std/scheduling_terms.hpp"

#define MAX_BUFFERS 32
#define MAX_PLANES 4

namespace nvidia {
namespace gxf {

// Struct to hold buffer info
typedef struct buffer_info_rec {
  void *buf_surface;
  int buf_fd;
  int enqueued;
  int length;
} buffer_info;

// Struct to store the context/meta data of the encoder
struct nvmpictx {
  // File descriptor for the device
  int dev_fd;
  // Index of the output plane buffer
  uint32_t index;
  // Input image width
  uint32_t width;
  // Input image height
  uint32_t height;
  // Profile to be used for encoding
  uint32_t profile;
  // Bitrate of the encoded stream, in bits per second
  uint32_t bitrate;
  // Max number of buffers used to hold the output plane data
  uint32_t num_buffers;
  // Encoded data size
  uint32_t bitstream_size;
  // Number of buffers on output and capture planes
  uint32_t output_buffer_count;
  uint32_t capture_buffer_count;
  uint32_t output_buffer_idx;
  uint32_t capture_buffer_idx;
  // Buffer info for capture and output planes
  buffer_info output_buffers[MAX_BUFFERS];
  buffer_info capture_buffers[MAX_BUFFERS];
  // Size of the output planes
  uint32_t outbuf_bytesused[MAX_PLANES];
  // Input yuv format
  uint32_t input_format;
  // Async scheduling term for response codelet
  nvidia::gxf::Handle<nvidia::gxf::AsynchronousSchedulingTerm> scheduling_term;
  std::queue<gxf::Entity> output_entity_queue;
  gxf_context_t gxf_context;
  // Thread funciton to issue dequeue on capture plane
  pthread_t  ctx_thread;
  volatile uint32_t bistreamBuf_queued;
  volatile uint32_t eos;
  volatile uint32_t dqbuf_index;
  volatile uint64_t request_count;
  volatile uint64_t response_count;
  uint32_t is_cuvid;
  uint32_t qp;
  // Input iamge format, support nv24 and nv12
  uint32_t raw_pixfmt;
  // Output format, only support V4L2_PIX_FMT_H264 yet
  uint32_t encoder_pixfmt;
  // Maximum data rate and resolution, select from 0 to 14
  uint32_t level;
  // Encode hw preset type, select from 0 to 3
  uint32_t hw_preset_type;
  // Entropy encoding, 0 represents CAVLC, 1 represents CABAC
  uint32_t entropy;
  // Interval between two I frames, in number of frames
  uint32_t iframe_interval;
  // Interval between two IDR frames, in number of frames
  uint32_t idr_interval;
  // Number of B frames in a GOP
  uint32_t num_of_bframes;
  int32_t rate_control_mode;
};


/// @brief Video encodercontext
class VideoEncoderContext : public gxf::Component {
 public:
  VideoEncoderContext();
  ~VideoEncoderContext();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  nvmpictx* ctx_;

 private:
  // Fill the encoder context wil default values
  gxf_result_t initalizeContext();
  // Async Scheduling Term required to get/set event state.
  gxf::Parameter<gxf::Handle<gxf::AsynchronousSchedulingTerm>> scheduling_term_;
};


}  // namespace gxf
}  // namespace nvidia

#endif
