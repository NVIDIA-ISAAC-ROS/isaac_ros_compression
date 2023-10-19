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
#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_UTILS_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_UTILS_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <linux/videodev2.h>
#include <semaphore.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <atomic>
#include <queue>
#include <thread>

#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#include "libv4l2.h"
#include "linux/v4l2_nv_extensions.h"
#include "nvbufsurface.h"
#include "videoencoder_context.hpp"

namespace nvidia {
namespace gxf {

// Different utility functions for v4l2 encoder implementaion
int set_capture_plane_format(nvmpictx* ctx);
int set_output_plane_format(nvmpictx* ctx);
int reqbufs_output_plane(nvmpictx* ctx, int32_t framerate);
int reqbufs_capture_plane(nvmpictx* ctx);
int enqueue_output_plane_buffer(nvmpictx* ctx, int q_index, int bytes_used);
int enqueue_capture_plane_buffer(nvmpictx* ctx, int q_index);
int streamon_plane(nvmpictx* ctx, unsigned int stream_type);
int streamoff_plane(nvmpictx* ctx, unsigned int stream_type);
int dqbuf_on_output_plane(nvmpictx* ctx, int* dqed_index);
int dqbuf_on_capture_plane(nvmpictx* ctx, int* dqed_index, int* bytes_copied);
int enqueue_all_capture_plane_buffers(nvmpictx* ctx);

// Utility functions for setting Encoder parameters
int setProfile(nvmpictx* ctx, uint32_t profile);
int setBitrate(nvmpictx* ctx, uint32_t bitrate);
int setInitQP(nvmpictx* ctx, uint32_t IinitQP, uint32_t PinitQP,
               uint32_t BinitQP);
int setCABAC(nvmpictx* ctx, bool enabled);
int setInsertSpsPpsAtIdrEnabled(nvmpictx* ctx, bool enabled);
int setNumBFrames(nvmpictx* ctx, uint32_t num);
int setHWPresetType(nvmpictx* ctx, uint32_t type);
int setHWPreset(nvmpictx* ctx, int32_t preset);
int setIDRInterval(nvmpictx* ctx, uint32_t interval);
int setIFrameInterval(nvmpictx* ctx, uint32_t interval);
int setMaxPerfMode(nvmpictx* ctx, int flag);
int setRateControlMode(nvmpictx* ctx, int mode);
int enableRateControl(nvmpictx* ctx, bool enabled_rc);
int setLevel(nvmpictx* ctx, uint32_t level);
int insertVUI(nvmpictx* ctx, int flag);
int getExtControls(nvmpictx* ctx, v4l2_ext_controls* ctl);
int getMetadata(nvmpictx* ctx, uint32_t buffer_index,
                v4l2_ctrl_videoenc_outputbuf_metadata & enc_metadata);
int setConstantQP(nvmpictx* ctx, uint32_t IinitQP, uint32_t PinitQP,
               uint32_t BinitQP);
}  // namespace gxf
}  // namespace nvidia
#endif  // NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_UTILS_HPP_
