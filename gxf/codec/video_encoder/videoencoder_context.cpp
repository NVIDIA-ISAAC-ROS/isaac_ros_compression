// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <string>
#include "videoencoder_context.hpp"
#include "videoencoder_utils.hpp"

namespace nvidia {
namespace gxf {

VideoEncoderContext::VideoEncoderContext() {}

VideoEncoderContext::~VideoEncoderContext() {}

gxf_result_t VideoEncoderContext::registerInterface(gxf::Registrar* registrar) {
  if (!registrar) {
    return GXF_ARGUMENT_NULL;
  }
  gxf::Expected<void> result;

  result &= registrar->parameter(scheduling_term_, "scheduling_term",
                                 "Asynchronous Scheduling Term", "Asynchronous Scheduling Term");

  result &= registrar->parameter(device_id_, "device_id", "cuda device id",
                                 "valid device id range is 0 to (cudaGetDeviceCount() - 1)", 0);

  return gxf::ToResultCode(result);
}

// Thread function to DQ the bit stream buffer and trigger the response codelet
static void * encoder_thread_fcn(void* arg) {
  nvmpictx* ctx = reinterpret_cast<nvmpictx*>(arg);
  int32_t dqbuf_index = 0;
  int32_t bitstream_size;
  while (!ctx->eos) {
    if ((ctx->bistreamBuf_queued == 0)) {
      continue;
    }
  try_dq_again:
    if (dqbuf_on_capture_plane(ctx, &dqbuf_index, &bitstream_size) != 0) {
      if (errno == EAGAIN) {
        GXF_LOG_DEBUG("Got EAGAIN while DQBUF on Capture plane, trying again");
        goto try_dq_again;
      } else {
        GXF_LOG_ERROR("Error in dqbuf_on_capture_plane");
        return NULL;
      }
    } else {
      GXF_LOG_DEBUG("Successfully Dequeued capture plane, idx:%d", dqbuf_index);
    }

    ctx->bistreamBuf_queued = 0;
    ctx->bitstream_size = (int32_t)bitstream_size;
    ctx->dqbuf_index = dqbuf_index;
    if (!ctx->eos) {
      ctx->scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_DONE);
    }
  }
  ctx->scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_NEVER);
  return NULL;
}

gxf_result_t VideoEncoderContext::initialize() {
  ctx_ = new nvmpictx;
  if (ctx_ == nullptr) {
    GXF_LOG_ERROR("Failed to allocate memory for encoder Context");
    return GXF_FAILURE;
  }
  auto gxf_ret_code = initalizeContext();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create default encoder context");
    return gxf_ret_code;
  }
  auto ret_val = pthread_create(&(ctx_->ctx_thread), NULL, encoder_thread_fcn, ctx_);
  if (ret_val) {
    GXF_LOG_ERROR("Failed create thred:pthread_create");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderContext::initalizeContext() {
  bool isWSL = isWSLPlatform();

  ctx_->scheduling_term = scheduling_term_.get();
  ctx_->scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
  ctx_->device_id = device_id_;
  ctx_->index = 0;
  // Not using B frames
  ctx_->num_of_bframes = 0;
  ctx_->encoder_pixfmt = V4L2_PIX_FMT_H264;
  ctx_->request_count = 0;
  ctx_->response_count = 0;
  ctx_->eos = 0;
  ctx_->bistreamBuf_queued = 0;
  ctx_->output_buffer_count = 5;
  ctx_->capture_buffer_count = 5;
  ctx_->output_buffer_idx = 0;
  ctx_->dqbuf_index = 0;

  int32_t device_count;
  cudaDeviceProp prop;
  cudaError_t status;
  status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    GXF_LOG_ERROR("cudaGetDevice failed");
    return GXF_FAILURE;
  }
  if (ctx_->device_id >= device_count) {
    GXF_LOG_ERROR("invalid cuda device id set");
    return GXF_FAILURE;
  }
  status = cudaGetDeviceProperties(&prop, ctx_->device_id);
  if (status != cudaSuccess) {
    GXF_LOG_ERROR("cudaGetDeviceProperties failed");
    return GXF_FAILURE;
  }
  ctx_->is_cuvid = !prop.integrated;

  /* This call creates a new V4L2 Video Encoder object
   on the device node.
   device_ = "/dev/null" for WSL platform
   device_ = "/dev/nvidia0" for cuvid (for system with single GPU).
   device_ = "/dev/v4l2-nvenc" for tegra
  */
  if (isWSL) {
  GXF_LOG_INFO("WSL Platform, device name :%s", "/dev/null");
  ctx_->dev_fd = v4l2_open("/dev/null", 0);
  } else if (ctx_->is_cuvid) {
    /* For multi GPU systems, device = "/dev/nvidiaX",
     where X < number of GPUs in the system.
     Find the device node in the system by searching for /dev/nvidia*
    */
    char gpu_device[16];
    FILE* fd = popen(
        "ls /dev/nvidia* | grep -m 1 '/dev/nvidia[[:digit:]]' | tr -d [:space:]", "r");
    if (fd == NULL) {
      GXF_LOG_ERROR("popen() command failed on the System");
      return GXF_FAILURE;
    }
    // Read system output from the above command
    if (fgets(gpu_device, sizeof(gpu_device), fd) == NULL) {
      GXF_LOG_ERROR("Could not read device node info from System");
      return GXF_FAILURE;
    }
    pclose(fd);
    GXF_LOG_INFO("Using GPU Device, device name:%s", gpu_device);
    ctx_->dev_fd = v4l2_open(gpu_device, 0);
  } else {
    GXF_LOG_INFO("Using Tegra Device, device name:%s", "/dev/nvl2-nvenc");
    ctx_->dev_fd = v4l2_open("/dev/v4l2-nvenc", 0);
  }
  if (ctx_->dev_fd < 0) {
    GXF_LOG_ERROR("Failed to open device:v4l2_open() failed");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderContext::deinitialize() {
  int retval = 0;
  if (!ctx_->is_cuvid) {
    uint32_t idx;
    NvBufSurface* nvbuffer;
    for (idx = 0; idx < ctx_->capture_buffer_count; idx++) {
      if (ctx_->capture_buffers[idx].buf_surface != nullptr) {
        nvbuffer = reinterpret_cast<NvBufSurface *>(ctx_->capture_buffers[idx].buf_surface);
        NvBufSurfaceUnMap(nvbuffer, 0, 0);
      }
    }
  }

  ctx_->eos = 1;
  ctx_->bistreamBuf_queued = 0;
  if (ctx_->dev_fd != -1) {
    struct v4l2_encoder_cmd dcmd = {
        0,
    };
    dcmd.cmd = V4L2_ENC_CMD_STOP;
    retval = v4l2_ioctl(ctx_->dev_fd, VIDIOC_ENCODER_CMD, &dcmd);
    if (retval < 0) {
      GXF_LOG_ERROR("Error in stopping the decoder");
      return GXF_FAILURE;
    }

    retval = streamoff_plane(ctx_, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
    if (retval < 0) {
      GXF_LOG_ERROR("Error in Stream off for OUTPUT_MPLANE");
      return GXF_FAILURE;
    }

    retval = streamoff_plane(ctx_, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
    if (retval < 0) {
      GXF_LOG_ERROR("Error in Stream off for CAPTURE_MPLANE");
      return GXF_FAILURE;
    }
  }
  // unmap and unregister the nvbufsurface pointers
  if (!ctx_->is_cuvid) {
    uint32_t idx;
    NvBufSurface* nvbuffer;
    for (idx = 0; idx < ctx_->output_buffer_count; idx++) {
      if (ctx_->output_buffers[idx].buf_surface != nullptr) {
        nvbuffer = reinterpret_cast<NvBufSurface *>(ctx_->output_buffers[idx].buf_surface);
        NvBufSurfaceUnMap(nvbuffer, 0, 0);
        cudaHostUnregister(nvbuffer->surfaceList[0].mappedAddr.addr[0]);
        cudaHostUnregister(nvbuffer->surfaceList[0].mappedAddr.addr[1]);
        if (ctx_->raw_pixfmt != V4L2_PIX_FMT_NV12M) {
          cudaHostUnregister(nvbuffer->surfaceList[0].mappedAddr.addr[2]);
        }
      }
    }
  }
  if (ctx_->ctx_thread) {
    auto ret_val = pthread_join(ctx_->ctx_thread, NULL);
    if (ret_val) {
      GXF_LOG_ERROR("Failed to terminate thread:pthread_join");
      return GXF_FAILURE;
    }
  }
  v4l2_close(ctx_->dev_fd);
  delete ctx_;
  return GXF_SUCCESS;
}

bool VideoEncoderContext::isWSLPlatform() {
  GXF_LOG_DEBUG("Entering in isWSLPlatform function");

  std::ifstream file("/proc/version");
  std::string line;
  bool found = false;

  if (file.is_open()) {
    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    found = (line.find("microsoft") != std::string::npos);
    file.close();
  }

  return found;
}

}  // namespace gxf
}  // namespace nvidia
