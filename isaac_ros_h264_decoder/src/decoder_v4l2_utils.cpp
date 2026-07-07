// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_h264_decoder/decoder_v4l2_utils.hpp"

#include <cstring>

#include "libv4l2.h"  // NOLINT(build/include_subdir)
#include "linux/v4l2_nv_extensions.h"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_decoder
{
namespace v4l2_ioctl
{

int set_format(int fd, struct v4l2_format * fmt)
{
  return ::v4l2_ioctl(fd, VIDIOC_S_FMT, fmt);
}

int get_format(int fd, struct v4l2_format * fmt)
{
  return ::v4l2_ioctl(fd, VIDIOC_G_FMT, fmt);
}

int request_buffers(int fd, struct v4l2_requestbuffers * reqbuf)
{
  return ::v4l2_ioctl(fd, VIDIOC_REQBUFS, reqbuf);
}

int query_buffer(int fd, struct v4l2_buffer * buf)
{
  return ::v4l2_ioctl(fd, VIDIOC_QUERYBUF, buf);
}

int export_buffer(int fd, struct v4l2_exportbuffer * expbuf)
{
  return ::v4l2_ioctl(fd, VIDIOC_EXPBUF, expbuf);
}

int queue_buffer(int fd, struct v4l2_buffer * buf)
{
  return ::v4l2_ioctl(fd, VIDIOC_QBUF, buf);
}

int dequeue_buffer(int fd, struct v4l2_buffer * buf)
{
  return ::v4l2_ioctl(fd, VIDIOC_DQBUF, buf);
}

int stream_on(int fd, uint32_t buf_type)
{
  return ::v4l2_ioctl(fd, VIDIOC_STREAMON, &buf_type);
}

int stream_off(int fd, uint32_t buf_type)
{
  return ::v4l2_ioctl(fd, VIDIOC_STREAMOFF, &buf_type);
}

int set_ext_controls(int fd, struct v4l2_ext_controls * ctrls)
{
  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, ctrls);
}

int get_control(int fd, struct v4l2_control * ctrl)
{
  return ::v4l2_ioctl(fd, VIDIOC_G_CTRL, ctrl);
}

int get_crop(int fd, struct v4l2_crop * crop)
{
  return ::v4l2_ioctl(fd, VIDIOC_G_CROP, crop);
}

int subscribe_event(int fd, uint32_t event_type)
{
  struct v4l2_event_subscription sub;
  std::memset(&sub, 0, sizeof(sub));
  sub.type = event_type;
  return ::v4l2_ioctl(fd, VIDIOC_SUBSCRIBE_EVENT, &sub);
}

int dequeue_event(int fd, struct v4l2_event * event)
{
  return ::v4l2_ioctl(fd, VIDIOC_DQEVENT, event);
}

int decoder_stop(int fd)
{
  struct v4l2_decoder_cmd cmd;
  std::memset(&cmd, 0, sizeof(cmd));
  cmd.cmd = V4L2_DEC_CMD_STOP;
  cmd.flags = V4L2_DEC_CMD_STOP_TO_BLACK;
  return ::v4l2_ioctl(fd, VIDIOC_DECODER_CMD, &cmd);
}

int set_disable_complete_frame_input(int fd)
{
  struct v4l2_ext_control ctrl;
  struct v4l2_ext_controls ctrls;
  std::memset(&ctrl, 0, sizeof(ctrl));
  std::memset(&ctrls, 0, sizeof(ctrls));

  ctrl.id = V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT;
  ctrl.value = 1;
  ctrls.count = 1;
  ctrls.controls = &ctrl;

  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
}

int set_low_latency_decode(int fd, bool is_cuvid)
{
  struct v4l2_ext_control ctrl;
  struct v4l2_ext_controls ctrls;
  std::memset(&ctrl, 0, sizeof(ctrl));
  std::memset(&ctrls, 0, sizeof(ctrls));

  ctrl.id = is_cuvid ? V4L2_CID_MPEG_VIDEO_CUDA_LOW_LATENCY : V4L2_CID_MPEG_VIDEO_DISABLE_DPB;
  ctrl.value = 1;
  ctrls.count = 1;
  ctrls.controls = &ctrl;

  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
}

int set_cuda_gpu_id(int fd, int gpu_id)
{
  struct v4l2_ext_control ctrl;
  struct v4l2_ext_controls ctrls;
  std::memset(&ctrl, 0, sizeof(ctrl));
  std::memset(&ctrls, 0, sizeof(ctrls));

  ctrl.id = V4L2_CID_MPEG_VIDEO_CUDA_GPU_ID;
  ctrl.value = gpu_id;
  ctrls.count = 1;
  ctrls.controls = &ctrl;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;

  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
}

int get_min_capture_buffers(int fd, uint32_t * count)
{
  struct v4l2_control ctrl;
  ctrl.id = V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;

  int ret = ::v4l2_ioctl(fd, VIDIOC_G_CTRL, &ctrl);
  if (ret == 0) {
    *count = ctrl.value;
  }
  return ret;
}

}  // namespace v4l2_ioctl
}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia
