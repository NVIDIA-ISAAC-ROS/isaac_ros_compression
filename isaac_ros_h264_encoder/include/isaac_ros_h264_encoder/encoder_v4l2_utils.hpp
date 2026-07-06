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

#ifndef ISAAC_ROS_H264_ENCODER__ENCODER_V4L2_UTILS_HPP_
#define ISAAC_ROS_H264_ENCODER__ENCODER_V4L2_UTILS_HPP_

#include <linux/videodev2.h>

#include <cstdint>

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
{
namespace v4l2_ioctl
{

int set_format(int fd, struct v4l2_format * fmt);
int get_format(int fd, struct v4l2_format * fmt);
int set_stream_params(int fd, struct v4l2_streamparm * parm);
int request_buffers(int fd, struct v4l2_requestbuffers * reqbuf);
int query_buffer(int fd, struct v4l2_buffer * buf);
int export_buffer(int fd, struct v4l2_exportbuffer * expbuf);
int queue_buffer(int fd, struct v4l2_buffer * buf);
int dequeue_buffer(int fd, struct v4l2_buffer * buf);
int stream_on(int fd, uint32_t buf_type);
int stream_off(int fd, uint32_t buf_type);
int set_ext_controls(int fd, struct v4l2_ext_controls * ctrls);
int encoder_stop(int fd);

int set_h264_profile(int fd, uint32_t profile);
int set_h264_level(int fd, uint32_t level);
int set_bitrate_mode(int fd, int mode);
int set_bitrate(int fd, uint32_t bitrate);
int set_gop_size(int fd, uint32_t gop_size);
int set_idr_interval(int fd, uint32_t interval);
int set_entropy_mode(int fd, bool cabac);
int set_frame_rate_control(int fd, bool enabled);
int set_max_performance(int fd, bool enabled);
int set_h264_qp(int fd, uint32_t qp_i, uint32_t qp_p, uint32_t qp_b);
int set_cuda_preset(int fd, int32_t preset);
int set_hw_preset_type(int fd, uint32_t type);
int set_init_qp(int fd, uint32_t qp_i, uint32_t qp_p, uint32_t qp_b);
int set_num_bframes(int fd, uint32_t num);
int set_insert_sps_pps_at_idr(int fd, bool enabled);
int set_insert_vui(int fd, bool enabled);

}  // namespace v4l2_ioctl
}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_ENCODER__ENCODER_V4L2_UTILS_HPP_
