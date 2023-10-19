// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_DECODE_VIDEODECODER_UTILS_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_DECODE_VIDEODECODER_UTILS_HPP_

#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <pthread.h>

#include "libv4l2.h"
#include "linux/v4l2_nv_extensions.h"
#include "nvbufsurface.h"
#include "videodecoder_context.hpp"

constexpr char kTimestampName[] = "timestamp";
constexpr uint32_t max_bitstream_size = (2 * 1024 * 1024);

namespace nvidia {
namespace gxf {

int32_t get_fmt_capture_plane(nvmpictx* ctx, struct v4l2_format* fmt);
int32_t set_capture_plane_format(nvmpictx* ctx);
int32_t set_output_plane_format(nvmpictx* ctx);
int32_t reqbufs_output_plane(nvmpictx* ctx);
int32_t get_num_capture_buffers(nvmpictx* ctx);
int32_t reqbufs_capture_plane(nvmpictx* ctx);
int32_t enqueue_all_capture_plane_buffers(nvmpictx* ctx);
int32_t enqueue_plane_buffer(nvmpictx* ctx, int32_t q_index, uint32_t bytes_used,
                             uint32_t buftype);
int32_t streamon_plane(nvmpictx* ctx, uint32_t stream_type);
int32_t streamoff_plane(nvmpictx* ctx, uint32_t stream_type);
int32_t set_disable_complete_frame_input(nvmpictx* ctx);
int32_t enable_low_latency_deocde(nvmpictx* ctx);
int32_t subscribe_events(nvmpictx* ctx);
int32_t dqbuf_plane(nvmpictx* ctx, int32_t* dqed_index, uint32_t buftype);
int32_t set_cuda_gpu_id(nvmpictx* ctx);

}  // namespace gxf
}  // namespace nvidia
#endif  // NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_DECODE_VIDEODECODER_UTILS_HPP_
