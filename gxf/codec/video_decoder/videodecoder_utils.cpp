// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "videodecoder_utils.hpp"

namespace nvidia {
namespace gxf {

int32_t get_fmt_capture_plane(nvmpictx* ctx, struct v4l2_format* fmt) {
  memset(fmt, 0, sizeof(struct v4l2_format));

  fmt->type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

  return v4l2_ioctl(ctx->dev_fd, VIDIOC_G_FMT, fmt);
}

int32_t set_capture_plane_format(nvmpictx* ctx) {
  int32_t retval = 0;
  struct v4l2_format fmt;

  memset(&fmt, 0, sizeof(fmt));

  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;

  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_FMT, &fmt);
  return retval;
}

int32_t set_output_plane_format(nvmpictx* ctx) {
  int32_t retval = 0;
  struct v4l2_format fmt;

  memset(&fmt, 0, sizeof(fmt));

  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
  fmt.fmt.pix_mp.num_planes = 1;
  fmt.fmt.pix_mp.plane_fmt[0].sizeimage = ctx->max_bitstream_size;

  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_FMT, &fmt);
  return retval;
}

int32_t reqbufs_output_plane(nvmpictx* ctx) {
  int32_t retval = 0;
  struct v4l2_requestbuffers reqbufs;
  struct v4l2_exportbuffer expbuf;
  struct v4l2_buffer query_buf;
  struct v4l2_plane planes[3];
  int32_t i = 0;
  NvBufSurface* nvbuf_surf = 0;

  memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));

  reqbufs.count = ctx->output_buffer_count;
  reqbufs.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  reqbufs.memory = V4L2_MEMORY_MMAP;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_REQBUFS, &reqbufs);

  if (retval != 0)
    return retval;

  GXF_LOG_DEBUG("Allocating the output buffers \n");

  for (i = 0; i < (int32_t)ctx->output_buffer_count; i++) {
    // CALL QUERYBUF and get the buffer sizes
    memset(&query_buf, 0, sizeof(query_buf));
    memset(planes, 0, sizeof(planes));
    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    query_buf.length = 1;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QUERYBUF, &query_buf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in QUERY_BUF \n");
      return retval;
    }

    GXF_LOG_DEBUG("QUERYBUF returned the length of buffer as %d \n",
                  query_buf.m.planes[0].length);

    // CALL EXPBUF here and retriev the buf fds
    memset(&expbuf, 0, sizeof(struct v4l2_exportbuffer));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_EXPBUF, &expbuf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in exporting the buffer \n");
      return retval;
    }
    GXF_LOG_DEBUG("Got the output buf_fd for index %d as %d \n", i, expbuf.fd);
    ctx->output_buffers[i].length = query_buf.m.planes[0].length;
    ctx->output_buffers[i].buf_fd = expbuf.fd;

    if (NvBufSurfaceFromFd(ctx->output_buffers[i].buf_fd,
                           reinterpret_cast<void **>(&nvbuf_surf)) < 0) {
      GXF_LOG_ERROR("Error in getting buffer parameters \n");
      return retval;
    }

    if (!ctx->is_cuvid) {
      if (NvBufSurfaceMap(nvbuf_surf, 0, 0, NVBUF_MAP_READ_WRITE) != 0) {
        GXF_LOG_ERROR("NvBufSurfaceMap failed in reqbufs_output_plane()");
        return -1;
      }
    }

    GXF_LOG_DEBUG("Got the nvbuf_surf pointer as %p \n", nvbuf_surf);
    ctx->output_buffers[i].buf_surface = nvbuf_surf;
    ctx->output_buffers[i].enqueued = 0;
  }
  GXF_LOG_DEBUG("Done allocating the Output buffers \n");
  return retval;
}

int32_t get_num_capture_buffers(nvmpictx* ctx) {
  struct v4l2_control ctl;
  int32_t ret;

  ctl.id = V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_G_CTRL, &ctl);
  if (ret == 0)
    ctx->capture_buffer_count = ctl.value;
  return ret;
}

int32_t reqbufs_capture_plane(nvmpictx* ctx) {
  int32_t retval = 0;
  struct v4l2_requestbuffers reqbufs;
  struct v4l2_exportbuffer expbuf;
  struct v4l2_buffer query_buf;
  struct v4l2_plane planes[3];
  int32_t i = 0;
  NvBufSurface* nvbuf_surf = 0;

  memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));

  reqbufs.count = ctx->capture_buffer_count + 5;
  GXF_LOG_INFO(" Requesting %d capture buffers \n", reqbufs.count);
  reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  reqbufs.memory = V4L2_MEMORY_MMAP;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_REQBUFS, &reqbufs);
  if (retval != 0)
    return retval;

  ctx->capture_buffer_count = reqbufs.count;
  GXF_LOG_INFO(" Allocated %d capture buffers \n", ctx->capture_buffer_count);

  for (i = 0; i < (int32_t)ctx->capture_buffer_count; i++) {
    // CALL QUERYBUF and get the buffer sizes
    memset(&query_buf, 0, sizeof(query_buf));
    memset(planes, 0, sizeof(planes));
    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    query_buf.length = 3;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QUERYBUF, &query_buf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in QUERY_BUF \n");
      return retval;
    }

    // CALL EXPBUF here and retriev the buf fds
    memset(&expbuf, 0, sizeof(struct v4l2_exportbuffer));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_EXPBUF, &expbuf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in exporting the buffer \n");
      return retval;
    }
    GXF_LOG_DEBUG("Got the buf_fd for index %d as %d \n", i, expbuf.fd);
    ctx->capture_buffers[i].buf_fd = expbuf.fd;

    if (NvBufSurfaceFromFd(ctx->capture_buffers[i].buf_fd,
                           reinterpret_cast<void **>(&nvbuf_surf)) < 0) {
      GXF_LOG_ERROR("Error in getting buffer parameters \n");
      return retval;
    }
    ctx->capture_buffers[i].buf_surface = nvbuf_surf;
    ctx->capture_buffers[i].enqueued = 0;
  }
  GXF_LOG_DEBUG("Done allocating buffers on capture plane \n");
  return retval;
}

int32_t enqueue_plane_buffer(nvmpictx* ctx, int32_t q_index, uint32_t bytes_used,
                             uint32_t buftype) {
  int32_t retval = 0;
  struct v4l2_buffer qbuf;
  struct v4l2_plane qbuf_plane;

  memset(&qbuf, 0, sizeof(qbuf));
  memset(&qbuf_plane, 0, sizeof(qbuf_plane));

  qbuf.index = q_index;
  qbuf.type = buftype;
  if (buftype == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) {
    qbuf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
    qbuf.timestamp.tv_sec = ctx->input_timestamp.tv_sec;
    qbuf.timestamp.tv_usec = ctx->input_timestamp.tv_usec;
  }
  qbuf.memory = V4L2_MEMORY_MMAP;
  qbuf.m.planes = &qbuf_plane;
  qbuf.m.planes[0].bytesused = bytes_used;
  qbuf.length = 1;
  GXF_LOG_DEBUG("flags are %d index %d \n", qbuf.flags, q_index);
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QBUF, &qbuf);
  if (retval != 0)
    return -1;
  GXF_LOG_DEBUG("enqueue plane buffer successful \n");
  return 0;
}

int32_t enqueue_all_capture_plane_buffers(nvmpictx* ctx) {
  uint32_t i;

  for (i = 0; i < ctx->capture_buffer_count; i++) {
    if (ctx->capture_buffers[i].enqueued == 0) {
      if (enqueue_plane_buffer(ctx, i, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) !=
          0) {
        GXF_LOG_ERROR("QBUF Capture plane error \n");
        return -1;
      }
      GXF_LOG_DEBUG("QBUF Capture done successfully \n");
      ctx->capture_buffers[i].enqueued = 1;
    }
  }
  return 0;
}

int32_t streamon_plane(nvmpictx* ctx, uint32_t stream_type) {
  return v4l2_ioctl(ctx->dev_fd, VIDIOC_STREAMON, &stream_type);
}

int32_t streamoff_plane(nvmpictx* ctx, uint32_t stream_type) {
  return v4l2_ioctl(ctx->dev_fd, VIDIOC_STREAMOFF, &stream_type);
}

int32_t set_disable_complete_frame_input(nvmpictx* ctx) {
  int32_t ret = 0;

  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;

  control.id = V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT;
  control.value = 1;

  GXF_LOG_DEBUG("Setting disable complete frame input");

  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_EXT_CTRLS, &ctrls);
  if (ret < 0)
    return -1;

  return 0;
}

int32_t enable_low_latency_deocde(nvmpictx* ctx) {
  int32_t ret = 0;

  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;

  control.id = ctx->is_cuvid ? V4L2_CID_MPEG_VIDEO_CUDA_LOW_LATENCY :
                               V4L2_CID_MPEG_VIDEO_DISABLE_DPB;
  control.value = 1;

  GXF_LOG_DEBUG("Enabling  low latency decode");

  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_EXT_CTRLS, &ctrls);
  if (ret < 0)
    return -1;

  return 0;
}

int32_t set_cuda_gpu_id(nvmpictx* ctx) {
  int32_t ret = 0;

  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID;
  control.value = ctx->device_id;

  GXF_LOG_DEBUG("Setting gpu id");

  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_EXT_CTRLS, &ctrls);
  if (ret < 0)
    return -1;

  return 0;
}

int32_t subscribe_events(nvmpictx* ctx) {
  struct v4l2_event_subscription sub;
  int32_t ret = 0;

  memset(&sub, 0, sizeof(struct v4l2_event_subscription));
  sub.type = V4L2_EVENT_EOS;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  if (ret != 0) {
    GXF_LOG_ERROR("Error received at VIDIOC_SUBSCRIBE_EVENT %d for EOS \n",
                  ret);
    return -1;
  }

  /* Subscribe to Resolution change event.
  ** This is required to catch whenever resolution change event
  ** is triggered to set the format on capture plane.
  */
  memset(&sub, 0, sizeof(struct v4l2_event_subscription));
  sub.type = V4L2_EVENT_RESOLUTION_CHANGE;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
  if (ret != 0) {
    GXF_LOG_ERROR(
        "Error received at VIDIOC_SUBSCRIBE_EVENT for resolution chang %d \n",
        ret);
    return -1;
  }

  GXF_LOG_DEBUG("Subscribed for events \n");
  return 0;
}

int32_t dqbuf_plane(nvmpictx* ctx, int32_t* dqed_index, uint32_t buftype) {
  int32_t ret = 0;
  struct v4l2_buffer dqbuf;
  struct v4l2_plane planes[2];
  memset(&dqbuf, 0, sizeof(dqbuf));
  memset(planes, 0, sizeof(planes));

  dqbuf.type = buftype;
  dqbuf.memory = V4L2_MEMORY_MMAP;
  dqbuf.m.planes = planes;
  dqbuf.length = 1;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_DQBUF, &dqbuf);
  if (ret != 0) {
    if (errno == EAGAIN) {
      // EAGAIN if we're just out of buffers to dequeue.
      return -1;
    }
    return -1;
  }
  *dqed_index = dqbuf.index;
  if (buftype == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    ctx->output_timestamp.tv_sec = dqbuf.timestamp.tv_sec;
    ctx->output_timestamp.tv_usec = dqbuf.timestamp.tv_usec;
  }
  GXF_LOG_DEBUG("DQBUF Plane dqed index %d \n", *dqed_index);
  return 0;
}

}  // namespace gxf
}  // namespace nvidia
