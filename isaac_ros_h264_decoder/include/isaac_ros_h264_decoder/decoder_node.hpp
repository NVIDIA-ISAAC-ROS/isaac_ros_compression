// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_H264_DECODER__DECODER_NODE_HPP_
#define ISAAC_ROS_H264_DECODER__DECODER_NODE_HPP_

#include <string>
#include <chrono>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "isaac_ros_nitros/nitros_node.hpp"

using StringList = std::vector<std::string>;

namespace nvidia
{
namespace isaac_ros
{
namespace h264_decoder
{

class DecoderNode : public nitros::NitrosNode
{
public:
  explicit DecoderNode(const rclcpp::NodeOptions &);

  ~DecoderNode();

  DecoderNode(const DecoderNode &) = delete;

  DecoderNode & operator=(const DecoderNode &) = delete;

  // The callback to be implemented by users for any required initialization
  void preLoadGraphCallback() override;
  void postLoadGraphCallback() override;

private:
  int32_t input_width_;
  int32_t input_height_;
};

}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_H264_DECODER__DECODER_NODE_HPP_
