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

#ifndef ISAAC_ROS_H264_DECODER__DECODER_V4L2_UTILS_HPP_
#define ISAAC_ROS_H264_DECODER__DECODER_V4L2_UTILS_HPP_

#include <linux/videodev2.h>

#include <cstdint>

namespace nvidia
{
namespace isaac_ros
{
namespace h264_decoder
{
namespace v4l2_ioctl
{

int set_format(int fd, struct v4l2_format * fmt);
int get_format(int fd, struct v4l2_format * fmt);
int request_buffers(int fd, struct v4l2_requestbuffers * reqbuf);
int query_buffer(int fd, struct v4l2_buffer * buf);
int export_buffer(int fd, struct v4l2_exportbuffer * expbuf);
int queue_buffer(int fd, struct v4l2_buffer * buf);
int dequeue_buffer(int fd, struct v4l2_buffer * buf);
int stream_on(int fd, uint32_t buf_type);
int stream_off(int fd, uint32_t buf_type);
int set_ext_controls(int fd, struct v4l2_ext_controls * ctrls);
int get_control(int fd, struct v4l2_control * ctrl);
int get_crop(int fd, struct v4l2_crop * crop);
int subscribe_event(int fd, uint32_t event_type);
int dequeue_event(int fd, struct v4l2_event * event);
int decoder_stop(int fd);

int set_disable_complete_frame_input(int fd);
int set_low_latency_decode(int fd, bool is_cuvid);
int set_cuda_gpu_id(int fd, int gpu_id);
int get_min_capture_buffers(int fd, uint32_t * count);

}  // namespace v4l2_ioctl
}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_DECODER__DECODER_V4L2_UTILS_HPP_
