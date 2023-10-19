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
#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEODECODER_CONTEXT_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEODECODER_CONTEXT_HPP_

#include <queue>
#include "gxf/core/component.hpp"
#include "gxf/std/scheduling_terms.hpp"

#define MAX_V4L2_BUFFERS 32

namespace nvidia {
namespace gxf {

typedef struct buffer_info_rec {
  void *buf_surface;
  int32_t buf_fd;
  int32_t enqueued; /*Set to 1 if enqueued. */
  int32_t length;
} buffer_info;

typedef struct timestamp_info_rec {
  uint64_t tv_usec;
  uint64_t tv_sec;
} timestamp_info;

// Struct to store the meta data passed to the encoder
struct nvmpictx {
  int32_t dev_fd;
  uint32_t output_buffer_count;
  uint32_t output_buffer_idx;
  uint32_t capture_buffer_count;
  uint32_t colorspace;
  uint32_t quantization;

  buffer_info output_buffers[MAX_V4L2_BUFFERS];
  buffer_info capture_buffers[MAX_V4L2_BUFFERS];

  pthread_t ctx_thread;
  int32_t capture_format_set;
  int32_t resolution_change_event;
  int32_t video_width;
  int32_t video_height;
  uint32_t eos;
  uint32_t got_eos;
  int32_t dst_dma_fd;
  uint32_t is_cuvid;
  uint32_t planar;
  uint32_t error_in_decode_thread;
  timestamp_info input_timestamp;
  timestamp_info output_timestamp;

  nvidia::gxf::Handle<nvidia::gxf::AsynchronousSchedulingTerm> response_scheduling_term;
  std::queue<gxf::Entity> output_entity_queue;
  gxf_context_t gxf_context;
  volatile uint32_t cp_dqbuf_available;
  volatile uint32_t cp_dqbuf_index;
};

/// @brief Video decodercontext
class VideoDecoderContext : public gxf::Component {
 public:
  VideoDecoderContext();
  ~VideoDecoderContext();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  nvmpictx* ctx_;
 private:
  // Async Scheduling Term required to get/set event state.
  gxf::Parameter<gxf::Handle<gxf::AsynchronousSchedulingTerm>> response_scheduling_term_;
};

}  // namespace gxf
}  // namespace nvidia

#endif
