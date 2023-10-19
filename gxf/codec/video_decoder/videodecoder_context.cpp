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

#include "videodecoder_context.hpp"
#include "videodecoder_utils.hpp"

namespace nvidia {
namespace gxf {

VideoDecoderContext::VideoDecoderContext() {}

VideoDecoderContext::~VideoDecoderContext() {}

gxf_result_t VideoDecoderContext::registerInterface(gxf::Registrar* registrar) {
  if (!registrar) {
    return GXF_ARGUMENT_NULL;
  }
  gxf::Expected<void> result;

  result &= registrar->parameter(response_scheduling_term_, "async_scheduling_term",
      "Asynchronous Scheduling Term", "Asynchronous Scheduling Term");

  return gxf::ToResultCode(result);
}

// Thread function to DQ the yuv buffer and trigger the response codelet
static void* decoder_thread_func(void* args) {
  int32_t retval = 0;
  nvmpictx* ctx = reinterpret_cast<nvmpictx *>(args);
  struct v4l2_format capture_format;
  int32_t dqbuf_index = 0;
  struct v4l2_event event;
  int32_t dqbuf_count = 0;
  struct v4l2_crop crop;
  uint32_t timer = 0;

  GXF_LOG_DEBUG("Created Application decoder thread function \n");

  /* Need to wait for the first Resolution change event, so that
  ** the decoder knows the stream resolution and can allocate
  ** appropriate buffers when REQBUFS is called.
  */
  while (!ctx->resolution_change_event) {
    if (ctx->eos) {
      GXF_LOG_DEBUG("eos is set without start of decode \n");
      return NULL;
    }
    memset(&event, 0, sizeof(struct v4l2_event));
    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_DQEVENT, &event);
    if (retval == 0) {
      GXF_LOG_DEBUG("DQed event %d ", event.type);
      if (event.type == V4L2_EVENT_RESOLUTION_CHANGE) {
        GXF_LOG_INFO("GOT Event resolution change \n");
        ctx->resolution_change_event = 1;
        break;
      }
    } else if (errno != EAGAIN) {
      GXF_LOG_ERROR("Error while DQing event, errno: %d \n", errno);
      ctx->error_in_decode_thread = 1;
      return NULL;
    }
    /* Sleep for 10 milli seconds to avoid continuous check for
    ** resolution_change_event. This will help in saving CPU
    ** cycles when no frame is fed to decoder.
    */
    usleep(10 * 1000);
  }

  while (!ctx->capture_format_set) {
    if (ctx->eos) {
      GXF_LOG_DEBUG("eos is set without start of decode \n");
      return NULL;
    }
    retval = get_fmt_capture_plane(ctx, &capture_format);
    if (retval < 0) {
      GXF_LOG_DEBUG("Capture format not set, sleeping \n");
      usleep(1 * 1000);
    } else {
      GXF_LOG_DEBUG("Capture format is set \n");
      ctx->capture_format_set = 1;
      break;
    }
  }

  // Call reqbufs
  GXF_LOG_INFO(" Capture Format Width %d Height %d \n",
               capture_format.fmt.pix_mp.width,
               capture_format.fmt.pix_mp.height);

  memset(&crop, 0, sizeof(struct v4l2_crop));
  crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_G_CROP, &crop);
  if (retval != 0) {
    GXF_LOG_ERROR("Error in getting VIDIOC_G_CROP \n");
    ctx->error_in_decode_thread = 1;
    return NULL;
  }

  GXF_LOG_INFO("Got display width %d Height %d \n", crop.c.width,
               crop.c.height);

  ctx->video_width = crop.c.width;
  ctx->video_height = crop.c.height;
  ctx->colorspace = capture_format.fmt.pix_mp.colorspace;
  ctx->quantization = capture_format.fmt.pix_mp.quantization;

  GXF_LOG_INFO(" Capture Format colorspace %d quantization %d \n",
               capture_format.fmt.pix_mp.colorspace,
               capture_format.fmt.pix_mp.quantization);

  if (!ctx->is_cuvid || ctx->planar) {
    NvBufSurfaceAllocateParams dstParams = {{0}};
    NvBufSurface* dst_nvbuf_surf = nullptr;

    // Allocate one extra buffer for semiplanar to planar conversion.
    // For Tegra, reuse same buffer for BL to PL conversion.
    dstParams.params.memType = NVBUF_MEM_DEFAULT;
    dstParams.params.width = crop.c.width;
    dstParams.params.height = crop.c.height;
    dstParams.params.layout = NVBUF_LAYOUT_PITCH;
    switch (ctx->colorspace) {
    case V4L2_COLORSPACE_DEFAULT:
    case V4L2_COLORSPACE_SMPTE170M: {
      if (ctx->quantization == V4L2_QUANTIZATION_FULL_RANGE) {
        /** For semi-planar, Specifies BT.601 colorspace - YUV420 ER multi-planar. */
        /** For planar, Specifies BT.601 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
        dstParams.params.colorFormat =
          (ctx->planar) ? NVBUF_COLOR_FORMAT_YUV420_ER : NVBUF_COLOR_FORMAT_NV12_ER;
      } else {
        /** For semi-planar, Specifies BT.601 colorspace - Y/CbCr 4:2:0 multi-planar */
        /** For planar, Specifies BT.601 colorspace - YUV420 multi-planar. */
        dstParams.params.colorFormat =
          (ctx->planar) ? NVBUF_COLOR_FORMAT_YUV420 : NVBUF_COLOR_FORMAT_NV12;
      }
    } break;
    case V4L2_COLORSPACE_REC709: {
      if (ctx->quantization == V4L2_QUANTIZATION_FULL_RANGE) {
        /** For semi-planar, Specifies BT.709 colorspace - Y/CbCr ER 4:2:0 multi-planar. */
        /** For planar, Specifies BT.709 colorspace - YUV420 ER multi-planar. */
        dstParams.params.colorFormat =
          (ctx->planar) ? NVBUF_COLOR_FORMAT_YUV420_709_ER : NVBUF_COLOR_FORMAT_NV12_709_ER;
      } else {
        /** For semi-planar, Specifies BT.709 colorspace - Y/CbCr 4:2:0 multi-planar. */
        /** For planar, Specifies BT.709 colorspace - YUV420 multi-planar. */
        dstParams.params.colorFormat =
          (ctx->planar) ? NVBUF_COLOR_FORMAT_YUV420_709 : NVBUF_COLOR_FORMAT_NV12_709;
      }
    } break;
    case  V4L2_COLORSPACE_BT2020:
    default:
      GXF_LOG_ERROR("Unsupported color format \n");
      ctx->error_in_decode_thread = 1;
      return NULL;
    }
    dstParams.memtag = NvBufSurfaceTag_VIDEO_DEC;

    retval = NvBufSurfaceAllocate(&dst_nvbuf_surf, 1, &dstParams);
    if (retval) {
      GXF_LOG_ERROR("Creation of dmabuf failed \n");
      ctx->error_in_decode_thread = 1;
      return NULL;
    }
    dst_nvbuf_surf->numFilled = 1;
    ctx->dst_dma_fd = dst_nvbuf_surf->surfaceList[0].bufferDesc;

    if (!ctx->is_cuvid) {
      cudaError_t result;
      for (uint32_t plane = 0;
          plane < dst_nvbuf_surf->surfaceList[0].planeParams.num_planes;
          plane++) {
        retval = NvBufSurfaceMap(dst_nvbuf_surf, 0, plane, NVBUF_MAP_READ_WRITE);
        if (retval) {
          GXF_LOG_ERROR("NvBufSurfaceMap failed");
          ctx->error_in_decode_thread = 1;
          return NULL;
        }
        result = cudaHostRegister(dst_nvbuf_surf->surfaceList[0].mappedAddr.addr[plane],
                                  dst_nvbuf_surf->surfaceList->planeParams.psize[plane],
                                  cudaHostAllocDefault);
        if (result != cudaSuccess) {
          GXF_LOG_ERROR("Failed in cudaHostRegister for plane %d: %s",
                        plane, cudaGetErrorString(result));
          ctx->error_in_decode_thread = 1;
          return NULL;
        }
      }
    }
  }

  retval = get_num_capture_buffers(ctx);
  if (retval != 0) {
    GXF_LOG_ERROR("Error in getting capture buffers \n");
    ctx->error_in_decode_thread = 1;
    return NULL;
  }

  GXF_LOG_INFO("Got minimum buffers for capture as %d \n",
               ctx->capture_buffer_count);

  retval = reqbufs_capture_plane(ctx);
  if (retval != 0) {
    GXF_LOG_ERROR("Error allocating the capture plane buffers \n");
    ctx->error_in_decode_thread = 1;
    return NULL;
  }

  retval = enqueue_all_capture_plane_buffers(ctx);
  if (retval != 0) {
    GXF_LOG_ERROR("Error in enqueued all capture buffers \n");
    ctx->error_in_decode_thread = 1;
    return NULL;
  }

  retval = streamon_plane(ctx, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in Stream on for CAPTURE \n");
    ctx->error_in_decode_thread = 1;
    return NULL;
  }
  GXF_LOG_DEBUG("Stream on for CAPTURE_MPLANE successful \n");

  // Now start the queue-dequeue loop on the capture plane
  while (!ctx->eos) {
    // Error handling to exit the thread if YUV output buffer is not consumed in 10 sec.
    if (ctx->cp_dqbuf_available) {
      GXF_LOG_DEBUG(
          "dqbuf'ed %d CAPTURE plane is not copied to GXF video buffer \n",
          ctx->cp_dqbuf_index);
      usleep(100);
      timer++;
      if (timer > 100000) {
        break;
      }
      continue;
    }
    // Issue DQBUF here
    GXF_LOG_DEBUG("DQBUF On Capture plane \n");
  try_dq_again:
    if (dqbuf_plane(ctx, &dqbuf_index, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) !=
        0) {
      GXF_LOG_DEBUG("Error on DQBUF On Capture plane \n");
      if (errno != EAGAIN)
        GXF_LOG_DEBUG("Error in dqbuf on capture plane, errno = %d", errno);
      if (errno == EAGAIN) {
        GXF_LOG_DEBUG("Got EAGAIN while DQBUF on Capture , trying again \n");
        goto try_dq_again;
      }
      if (errno == EPIPE) {
        GXF_LOG_INFO("received EPIPE from driver\n");
        ctx->got_eos = 1;
        break;
      }
    } else {
      GXF_LOG_DEBUG("Successfully dqbuf'ed %d CAPTURE plane Dqbuf count %d \n",
                    dqbuf_index, dqbuf_count);
    }

    dqbuf_count++;
    ctx->cp_dqbuf_index = dqbuf_index;
    ctx->cp_dqbuf_available = 1;
    timer = 0;
    if (!ctx->eos) {
      ctx->response_scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_DONE);
      GXF_LOG_DEBUG("Thread:AsynchronousEventState:EVENT_DONE");
    }
  }
  GXF_LOG_DEBUG("decoder_thread_func finished \n");
  ctx->response_scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_NEVER);
  ctx->error_in_decode_thread = 0;
  return NULL;
}

gxf_result_t VideoDecoderContext::initialize() {
  ctx_ = new nvmpictx;
  if (ctx_ == nullptr) {
    GXF_LOG_ERROR("Failed to allocate memory for decoder Context");
    return GXF_FAILURE;
  }
  GXF_LOG_INFO("Decoder context created");

  ctx_->response_scheduling_term = response_scheduling_term_.get();
  ctx_->response_scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);

  int32_t retval = 0;
  ctx_->eos = 0;
  ctx_->got_eos = 0;
  ctx_->cp_dqbuf_available = 0;
  ctx_->resolution_change_event = 0;
  ctx_->dst_dma_fd = -1;
  ctx_->error_in_decode_thread = 0;

  retval = system("lsmod | grep 'nvgpu' > /dev/null");
  if (retval == -1) {
    GXF_LOG_ERROR("Error in grep for nvgpu device");
    return GXF_FAILURE;
  } else if (retval == 0) {
      ctx_->is_cuvid = false;
  } else {
      ctx_->is_cuvid = true;
  }

  /* The call creates a new V4L2 Video Encoder object
   on the device node.
   device_ = "/dev/nvidia0" for cuvid (for single GPU system).
   device_ = "/dev/nvhost-nvdec" for tegra
  */
  if (ctx_->is_cuvid) {
    /* For multi GPU systems, device = "/dev/nvidiaX",
     where X < number of GPUs in the system.
     Find the device node(X) in the system by searching for /dev/nvidia*
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
    GXF_LOG_INFO("Using GPU Device, device name :%s", gpu_device);
    ctx_->dev_fd = v4l2_open(gpu_device, 0);
  } else {
    GXF_LOG_INFO("Using Tegra Device, device name :%s", "/dev/nvhost-nvdec");
    ctx_->dev_fd = v4l2_open("/dev/nvhost-nvdec", 0);
  }
  if (ctx_->dev_fd < 0) {
    GXF_LOG_ERROR("Failed to open decoder");
    return GXF_FAILURE;
  }

  // For the decoder, let's call S_FMT h264 pixelformat on the output plane
  // and then subscribe for events etc.. and the usual stuff.
  retval = set_output_plane_format(ctx_);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in set_fmt on OUTPUT_MPLANE");
    return GXF_FAILURE;
  }
  GXF_LOG_DEBUG("S_FMT Output plane successfull");

  retval = subscribe_events(ctx_);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in subscribe events \n");
    return GXF_FAILURE;
  }

  // Set gpu id. Needed by latest version of cuvidv4l2
  if (ctx_->is_cuvid) {
    retval = set_cuda_gpu_id(ctx_);
    if (retval < 0) {
      GXF_LOG_ERROR("Error in subscribe events \n");
      return GXF_FAILURE;
    }
  }

  pthread_create(&(ctx_->ctx_thread), NULL, decoder_thread_func, ctx_);

  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderContext::deinitialize() {
  GXF_LOG_DEBUG("Enter deinitialize function");

  int32_t retval = 0;
  struct v4l2_decoder_cmd dcmd = {
      0,
  };
  dcmd.cmd = V4L2_DEC_CMD_STOP;
  dcmd.flags = V4L2_DEC_CMD_STOP_TO_BLACK;
  v4l2_ioctl(ctx_->dev_fd, VIDIOC_DECODER_CMD, &dcmd);

  ctx_->eos = 1;
  ctx_->cp_dqbuf_available = 0;
  if (ctx_->ctx_thread)
    pthread_join(ctx_->ctx_thread, NULL);

  if (!ctx_->is_cuvid) {
    uint32_t idx;
    NvBufSurface* nvbuffer;
    for (idx = 0; idx < ctx_->output_buffer_count; idx++) {
      nvbuffer = reinterpret_cast<NvBufSurface *>(ctx_->output_buffers[idx].buf_surface);
      NvBufSurfaceUnMap(nvbuffer, 0, 0);
    }
  }

  retval = streamoff_plane(ctx_, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in Stream off for OUTPUT_MPLANE \n");
    return GXF_FAILURE;
  }

  retval = streamoff_plane(ctx_, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in Stream off for CAPTURE_MPLANE \n");
    return GXF_FAILURE;
  }

  if (ctx_->dst_dma_fd != -1) {
    NvBufSurface* nvbuf_surf = nullptr;

    retval = NvBufSurfaceFromFd(ctx_->dst_dma_fd,
                                reinterpret_cast<void**>(&nvbuf_surf));
    if (retval)
      GXF_LOG_ERROR("Failed to Get NvBufSurface from FD \n");

    for (uint32_t plane = 0;
        plane < nvbuf_surf->surfaceList[0].planeParams.num_planes; plane++) {
      NvBufSurfaceUnMap(nvbuf_surf, 0, plane);
    }
    retval = NvBufSurfaceDestroy(nvbuf_surf);
    if (retval)
      GXF_LOG_ERROR("Failed to destroy NvBufSurface \n");

    ctx_->dst_dma_fd = -1;
  }

  v4l2_close(ctx_->dev_fd);

  delete ctx_;
  return GXF_SUCCESS;
}


}  // namespace gxf
}  // namespace nvidia

