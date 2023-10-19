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
#include <string>

#include "gxf/core/component.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "videoencoder_utils.hpp"

namespace nvidia {
namespace gxf {

int set_capture_plane_format(nvmpictx* ctx) {
  int retval = 0;
  struct v4l2_format fmt;

  memset(&fmt, 0, sizeof(fmt));

  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
  fmt.fmt.pix_mp.width = ctx->width;
  fmt.fmt.pix_mp.height = ctx->height;
  fmt.fmt.pix_mp.num_planes = 1;
  fmt.fmt.pix_mp.plane_fmt[0].sizeimage = 2*ctx->width*ctx->height;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_FMT, &fmt);

  return retval;
}

int set_output_plane_format(nvmpictx* ctx) {
  int retval = 0;
  struct v4l2_format fmt;

  memset(&fmt, 0, sizeof(fmt));

  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  fmt.fmt.pix_mp.pixelformat = ctx->raw_pixfmt;
  fmt.fmt.pix_mp.width = ctx->width;
  fmt.fmt.pix_mp.height = ctx->height;

  if (ctx->raw_pixfmt == V4L2_PIX_FMT_NV12M)
    fmt.fmt.pix_mp.num_planes = 2;
  else
    fmt.fmt.pix_mp.num_planes = 3;

  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_FMT, &fmt);
  return retval;
}

int reqbufs_output_plane(nvmpictx* ctx, int32_t framerate) {
  int retval = 0;
  struct v4l2_requestbuffers reqbufs;
  struct v4l2_exportbuffer expbuf;
  struct v4l2_buffer query_buf;
  struct v4l2_plane planes[3];
  int i = 0;

  NvBufSurface* nvbuf_surf = NULL;
  struct v4l2_streamparm parm;
  memset(&parm, 0, sizeof(struct v4l2_streamparm));
  parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  parm.parm.output.timeperframe.numerator = 1;
  parm.parm.output.timeperframe.denominator = framerate;
  int ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_PARM, &parm);
  if (ret) {
    GXF_LOG_ERROR("Error in reqbufs_output_plane::VIDIOC_S_PARM");
    return ret;
  }

  memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));

  reqbufs.count = ctx->output_buffer_count;
  reqbufs.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  reqbufs.memory = V4L2_MEMORY_MMAP;

  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_REQBUFS, &reqbufs);

  if (retval != 0) {
    GXF_LOG_ERROR("Error in reqbufs_output_plane:VIDIOC_REQBUFS");
    return retval;
  }

  for (i = 0; i < static_cast<int>(ctx->output_buffer_count); i++) {
    // CALL QUERYBUF and get the buffer sizes
    memset(&query_buf, 0, sizeof(query_buf));
    memset(planes, 0, sizeof(planes));
    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    if (ctx->raw_pixfmt == V4L2_PIX_FMT_NV12M)
      query_buf.length = 2;
    else
      query_buf.length = 3;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QUERYBUF, &query_buf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in reqbufs_output_plane:VIDIOC_QUERYBUF");
      return retval;
    }

    // CALL EXPBUF here and retriev the buf fds
    memset(&expbuf, 0, sizeof(struct v4l2_exportbuffer));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_EXPBUF, &expbuf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in exporting the buffer");
      return retval;
    }
    ctx->output_buffers[i].length = query_buf.m.planes[0].length;
    ctx->output_buffers[i].buf_fd = expbuf.fd;

    if (NvBufSurfaceFromFd(ctx->output_buffers[i].buf_fd,
                           reinterpret_cast<void **>(&nvbuf_surf)) < 0) {
      GXF_LOG_ERROR("Error in reqbufs_output_plane:NvBufSurfaceFromFd");
      return retval;
    }

    if (!ctx->is_cuvid) {
      if (NvBufSurfaceMap(nvbuf_surf, 0, 0, NVBUF_MAP_READ_WRITE) != 0) {
        GXF_LOG_ERROR("NvBufSurfaceMap failed in reqbufs_output_plane()");
        return GXF_FAILURE;
      }
      cudaHostRegister(nvbuf_surf->surfaceList[0].mappedAddr.addr[0],
                       nvbuf_surf->surfaceList->planeParams.psize[0],
                       cudaHostAllocDefault);
      cudaHostRegister(nvbuf_surf->surfaceList[0].mappedAddr.addr[1],
                       nvbuf_surf->surfaceList->planeParams.psize[1],
                       cudaHostAllocDefault);
      if (ctx->raw_pixfmt != V4L2_PIX_FMT_NV12M) {
        cudaHostRegister(nvbuf_surf->surfaceList[0].mappedAddr.addr[2],
                         nvbuf_surf->surfaceList->planeParams.psize[2],
                         cudaHostAllocDefault);
      }
    }
    ctx->output_buffers[i].buf_surface = reinterpret_cast<NvBufSurface *>(nvbuf_surf);
    ctx->output_buffers[i].enqueued = 0;
  }
  GXF_LOG_DEBUG("Done allocating the Output buffers \n");
  return retval;
}

int reqbufs_capture_plane(nvmpictx* ctx) {
  int retval = 0;
  struct v4l2_requestbuffers reqbufs;
  struct v4l2_exportbuffer expbuf;
  int i = 0;
  NvBufSurface* nvbuf_surf = 0;
  struct v4l2_buffer query_buf;
  struct v4l2_plane planes[3];

  memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));

  reqbufs.count = ctx->capture_buffer_count;
  reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  reqbufs.memory = V4L2_MEMORY_MMAP;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_REQBUFS, &reqbufs);

  if (retval != 0) {
    GXF_LOG_ERROR("Error in reqbufs_capture_plane:VIDIOC_REQBUFS");
    return retval;
  }

  for (i = 0; i < static_cast<int>(ctx->capture_buffer_count); i++) {
    // CALL QUERYBUF and get the buffer sizes
    memset(&query_buf, 0, sizeof(query_buf));
    memset(planes, 0, sizeof(planes));
    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    query_buf.length = 1;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QUERYBUF, &query_buf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in reqbufs_capture_plane:VIDIOC_QUERYBUF");
      return retval;
    }

    // CALL EXPBUF here and retriev the buf fds
    memset(&expbuf, 0, sizeof(struct v4l2_exportbuffer));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_EXPBUF, &expbuf);
    if (retval != 0) {
      GXF_LOG_ERROR("Error in reqbufs_capture_plane: VIDIOC_EXPBUF");
      return retval;
    }
    GXF_LOG_DEBUG("Got the buf_fd for index %d as %d \n", i, expbuf.fd);
    ctx->capture_buffers[i].buf_fd = expbuf.fd;

    if (NvBufSurfaceFromFd(ctx->capture_buffers[i].buf_fd,
                           reinterpret_cast<void **>(&nvbuf_surf)) < 0) {
      GXF_LOG_ERROR("Error in getting buffer parameters");
      return retval;
    }

    if (!ctx->is_cuvid) {
      if (NvBufSurfaceMap(nvbuf_surf, 0, 0, NVBUF_MAP_READ_WRITE) != 0) {
        GXF_LOG_ERROR("Error in reqbufs_capture_plane: NvBufSurfaceMap");
        return GXF_FAILURE;
      }
    }
    ctx->capture_buffers[i].buf_surface = reinterpret_cast<NvBufSurface *>(nvbuf_surf);
    ctx->capture_buffers[i].enqueued = 0;
  }
  GXF_LOG_DEBUG("Done allocating buffers on capture plane");
  return retval;
}

int enqueue_output_plane_buffer(nvmpictx* ctx, int q_index, int bytesused) {
  int retval = 0;
  struct v4l2_buffer qbuf;
  struct v4l2_plane qbuf_plane;

  memset(&qbuf, 0, sizeof(qbuf));
  memset(&qbuf_plane, 0, sizeof(qbuf_plane));

  qbuf.index = q_index;
  qbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  qbuf.timestamp.tv_sec = q_index;  // Fill some timestamp
  qbuf.memory = V4L2_MEMORY_MMAP;
  qbuf.m.planes = &qbuf_plane;
  qbuf.m.planes[0].bytesused = bytesused;
  qbuf.length = 1;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QBUF, &qbuf);
  if (retval != 0)
    return -1;

  return 0;
}
int enqueue_capture_plane_buffer(nvmpictx* ctx, int q_index) {
  int retval = 0;
  struct v4l2_buffer qbuf;
  struct v4l2_plane qbuf_plane;

  memset(&qbuf, 0, sizeof(qbuf));
  memset(&qbuf_plane, 0, sizeof(qbuf_plane));

  qbuf.index = q_index;
  qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  qbuf.timestamp.tv_sec = q_index;  // Fill some timestamp
  qbuf.memory = V4L2_MEMORY_MMAP;
  qbuf.m.planes = &qbuf_plane;
  qbuf.m.planes[0].bytesused = 0;
  qbuf.length = 1;
  retval = v4l2_ioctl(ctx->dev_fd, VIDIOC_QBUF, &qbuf);

  if (retval != 0)
    return -1;

  return 0;
}

int streamon_plane(nvmpictx* ctx, unsigned int stream_type) {
  return v4l2_ioctl(ctx->dev_fd, VIDIOC_STREAMON, &stream_type);
}

int streamoff_plane(nvmpictx* ctx, unsigned int stream_type) {
  return v4l2_ioctl(ctx->dev_fd, VIDIOC_STREAMOFF, &stream_type);
}

int dqbuf_on_output_plane(nvmpictx* ctx, int* dqed_index) {
  int ret = 0;
  struct v4l2_buffer dqbuf;
  struct v4l2_plane planes[3];
  memset(&dqbuf, 0, sizeof(dqbuf));
  memset(planes, 0, sizeof(planes));

  dqbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  dqbuf.memory = V4L2_MEMORY_MMAP;
  dqbuf.m.planes = planes;
  dqbuf.length = 1;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_DQBUF, &dqbuf);
  if (ret != 0) {
    return -1;
  }
  *dqed_index = dqbuf.index;
  GXF_LOG_DEBUG("DQBUF on Output Plane done, index %d", *dqed_index);
  return 0;
}

int dqbuf_on_capture_plane(nvmpictx* ctx, int* dqed_index, int* bytes_copied) {
  int ret = 0;
  struct v4l2_buffer dqbuf;
  struct v4l2_plane planes[2];
  memset(&dqbuf, 0, sizeof(dqbuf));
  memset(planes, 0, sizeof(planes));

  dqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  dqbuf.memory = V4L2_MEMORY_MMAP;
  dqbuf.m.planes = planes;
  dqbuf.length = 1;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_DQBUF, &dqbuf);
  if (ret != 0) {
     if (!ctx->eos)
       return -1;
     else
       return 0;
  }
  *dqed_index = dqbuf.index;
  *bytes_copied = dqbuf.bytesused;
  GXF_LOG_DEBUG("DQBUF on Capture Plane done, index:%d Bytes:%d ",
                 *dqed_index, *bytes_copied);
  return 0;
}
int enqueue_all_capture_plane_buffers(nvmpictx* ctx) {
  unsigned int i;
  for (i = 0 ; i < ctx->capture_buffer_count; i++) {
    if (ctx->capture_buffers[i].enqueued == 0) {
        if (enqueue_capture_plane_buffer(ctx, i) != 0) {
          GXF_LOG_ERROR("QBUF Capture plane error");
          return -1;
        }
        ctx->capture_buffers[i].enqueued = 1;
    }
  }
  return 0;
}

int setExtControls(nvmpictx* ctx, v4l2_ext_controls* ctl) {
  int ret;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_S_EXT_CTRLS, ctl);
  return ret;
}

int setBitrate(nvmpictx* ctx, uint32_t bitrate) {
  int ret;
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_BITRATE;
  control.value = bitrate;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setProfile(nvmpictx* ctx, uint32_t profile) {
  int ret;
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
  control.value = profile;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setInitQP(nvmpictx* ctx, uint32_t IinitQP, uint32_t PinitQP,
               uint32_t BinitQP) {
  v4l2_ctrl_video_init_qp initqp;
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  initqp.IInitQP = IinitQP;
  initqp.PInitQP = PinitQP;
  initqp.BInitQP = BinitQP;

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEOENC_INIT_FRAME_QP;
  control.string = reinterpret_cast<char *>(&initqp);

  ret = setExtControls(ctx, &ctrls);
  return ret;
}
int setConstantQP(nvmpictx* ctx, uint32_t IinitQP, uint32_t PinitQP,
               uint32_t BinitQP) {
  struct v4l2_ext_control ctl;
  struct v4l2_ext_controls ctrls;
  int ret;
  memset(&ctl, 0, sizeof(ctl));
  memset(&ctrls, 0, sizeof(ctrls));
  v4l2_ctrl_video_constqp constqp;
  constqp.constQpI = IinitQP;
  constqp.constQpP = PinitQP;
  constqp.constQpB = BinitQP;
  ctrls.count = 1;
  ctrls.controls = &ctl;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
  ctl.id = V4L2_CID_MPEG_VIDEOENC_CUDA_CONSTQP;
  ctl.string = reinterpret_cast<char *>(&constqp);

  ret = setExtControls(ctx, &ctrls);
  return ret;
}
int setCABAC(nvmpictx* ctx, bool enabled) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE;
  control.value = enabled;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setInsertSpsPpsAtIdrEnabled(nvmpictx* ctx, bool enabled) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR;
  control.value = enabled;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setNumBFrames(nvmpictx* ctx, uint32_t num) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES;
  control.value = num;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setHWPresetType(nvmpictx* ctx, uint32_t type) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;


  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM;
  control.value = type;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}
int setHWPreset(nvmpictx* ctx, int32_t preset) {
    int ret;
    struct v4l2_ext_control control;
    struct v4l2_ext_controls ctrls;

    memset(&control, 0, sizeof(control));
    memset(&ctrls, 0, sizeof(ctrls));

    ctrls.count = 1;
    ctrls.controls = &control;
    ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

    control.id =  V4L2_CID_MPEG_VIDEOENC_CUDA_PRESET_ID;
    control.value = preset;

    ret = setExtControls(ctx, &ctrls);
    return ret;
}

int setIDRInterval(nvmpictx* ctx, uint32_t interval) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_IDR_INTERVAL;
  control.value = interval;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setIFrameInterval(nvmpictx* ctx, uint32_t interval) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_GOP_SIZE;
  control.value = interval;
  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setMaxPerfMode(nvmpictx* ctx, int flag) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_MAX_PERFORMANCE;
  control.value = flag;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setRateControlMode(nvmpictx* ctx, int mode) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE;
  control.value = mode;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int enableRateControl(nvmpictx* ctx, bool enabled_rc) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_FRAME_RC_ENABLE;
  control.value = enabled_rc;  // if false: disable rate control

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int setLevel(nvmpictx* ctx, uint32_t level) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEO_H264_LEVEL;
  control.value = (enum v4l2_mpeg_video_h264_level)level;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int insertVUI(nvmpictx* ctx, int flag) {
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;
  int ret;

  memset(&control, 0, sizeof(control));
  memset(&ctrls, 0, sizeof(ctrls));

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  control.id = V4L2_CID_MPEG_VIDEOENC_INSERT_VUI;
  control.value = flag;

  ret = setExtControls(ctx, &ctrls);
  return ret;
}

int getExtControls(nvmpictx* ctx, v4l2_ext_controls* ctl) {
  int ret;
  ret = v4l2_ioctl(ctx->dev_fd, VIDIOC_G_EXT_CTRLS, ctl);
  return ret;
}

int getMetadata(nvmpictx* ctx, uint32_t buffer_index,
                v4l2_ctrl_videoenc_outputbuf_metadata & enc_metadata) {
  int ret;
  v4l2_ctrl_video_metadata metadata;
  struct v4l2_ext_control control;
  struct v4l2_ext_controls ctrls;

  ctrls.count = 1;
  ctrls.controls = &control;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  metadata.buffer_index = buffer_index;
  metadata.VideoEncMetadata = &enc_metadata;

  control.id = V4L2_CID_MPEG_VIDEOENC_METADATA;
  control.string = reinterpret_cast<char *>(&metadata);

  ret = getExtControls(ctx, &ctrls);
  return ret;
}

}  // namespace gxf
}  // namespace nvidia
