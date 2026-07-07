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

#include "isaac_ros_h264_encoder/encoder_v4l2_utils.hpp"

#include <cstring>

#include "libv4l2.h"  // NOLINT(build/include_subdir)
#include "linux/v4l2_nv_extensions.h"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
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

int set_stream_params(int fd, struct v4l2_streamparm * parm)
{
  return ::v4l2_ioctl(fd, VIDIOC_S_PARM, parm);
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

int encoder_stop(int fd)
{
  struct v4l2_encoder_cmd cmd;
  std::memset(&cmd, 0, sizeof(cmd));
  cmd.cmd = V4L2_ENC_CMD_STOP;
  return ::v4l2_ioctl(fd, VIDIOC_ENCODER_CMD, &cmd);
}

namespace
{
int set_single_ext_control(int fd, uint32_t id, int32_t value)
{
  struct v4l2_ext_control ctrl;
  struct v4l2_ext_controls ctrls;
  std::memset(&ctrl, 0, sizeof(ctrl));
  std::memset(&ctrls, 0, sizeof(ctrls));

  ctrl.id = id;
  ctrl.value = value;
  ctrls.count = 1;
  ctrls.controls = &ctrl;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_CODEC;

  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
}
}  // namespace

int set_h264_profile(int fd, uint32_t profile)
{
  int v4l2_profile;
  switch (profile) {
    case 0:
      v4l2_profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
      break;
    case 1:
      v4l2_profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
      break;
    case 2:
    default:
      v4l2_profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
      break;
  }
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_H264_PROFILE, v4l2_profile);
}

int set_h264_level(int fd, uint32_t level)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_H264_LEVEL, level);
}

int set_bitrate_mode(int fd, int mode)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_BITRATE_MODE, mode);
}

int set_bitrate(int fd, uint32_t bitrate)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_BITRATE, bitrate);
}

int set_gop_size(int fd, uint32_t gop_size)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_GOP_SIZE, gop_size);
}

int set_idr_interval(int fd, uint32_t interval)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_IDR_INTERVAL, interval);
}

int set_entropy_mode(int fd, bool cabac)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, cabac ? 1 : 0);
}

int set_frame_rate_control(int fd, bool enabled)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_FRAME_RC_ENABLE, enabled ? 1 : 0);
}

int set_max_performance(int fd, bool enabled)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEO_MAX_PERFORMANCE, enabled ? 1 : 0);
}

int set_h264_qp(int fd, uint32_t qp_i, uint32_t qp_p, uint32_t qp_b)
{
  struct v4l2_ext_control ctrl[3];
  struct v4l2_ext_controls ctrls;
  std::memset(ctrl, 0, sizeof(ctrl));
  std::memset(&ctrls, 0, sizeof(ctrls));

  ctrl[0].id = V4L2_CID_MPEG_VIDEO_H264_I_FRAME_QP;
  ctrl[0].value = qp_i;
  ctrl[1].id = V4L2_CID_MPEG_VIDEO_H264_P_FRAME_QP;
  ctrl[1].value = qp_p;
  ctrl[2].id = V4L2_CID_MPEG_VIDEO_H264_B_FRAME_QP;
  ctrl[2].value = qp_b;

  ctrls.count = 3;
  ctrls.controls = ctrl;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_CODEC;

  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
}

int set_cuda_preset(int fd, int32_t preset)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEOENC_CUDA_PRESET_ID, preset);
}

int set_hw_preset_type(int fd, uint32_t type)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEOENC_HW_PRESET_TYPE_PARAM, type);
}

int set_init_qp(int fd, uint32_t qp_i, uint32_t qp_p, uint32_t qp_b)
{
  v4l2_ctrl_video_init_qp initqp;
  struct v4l2_ext_control ctrl;
  struct v4l2_ext_controls ctrls;
  std::memset(&ctrl, 0, sizeof(ctrl));
  std::memset(&ctrls, 0, sizeof(ctrls));

  initqp.IInitQP = qp_i;
  initqp.PInitQP = qp_p;
  initqp.BInitQP = qp_b;

  ctrl.id = V4L2_CID_MPEG_VIDEOENC_INIT_FRAME_QP;
  ctrl.string = reinterpret_cast<char *>(&initqp);
  ctrls.count = 1;
  ctrls.controls = &ctrl;
  ctrls.ctrl_class = V4L2_CTRL_CLASS_CODEC;

  return ::v4l2_ioctl(fd, VIDIOC_S_EXT_CTRLS, &ctrls);
}

int set_num_bframes(int fd, uint32_t num)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEOENC_NUM_BFRAMES, num);
}

int set_insert_sps_pps_at_idr(int fd, bool enabled)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEOENC_INSERT_SPS_PPS_AT_IDR, enabled ? 1 : 0);
}

int set_insert_vui(int fd, bool enabled)
{
  return set_single_ext_control(fd, V4L2_CID_MPEG_VIDEOENC_INSERT_VUI, enabled ? 1 : 0);
}

}  // namespace v4l2_ioctl
}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia
