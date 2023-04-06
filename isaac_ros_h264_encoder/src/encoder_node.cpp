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

#include "isaac_ros_h264_encoder/encoder_node.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_compressed_image_type/nitros_compressed_image.hpp"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_COMPONENT_KEY[] = "color_converter/data_receiver";
constexpr char INPUT_DEFAULT_FORMAT[] = "nitros_image_bgr8";
constexpr char INPUT_TOPIC_NAME[] = "image_raw";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_FORMAT[] = "nitros_compressed_image";
constexpr char OUTPUT_TOPIC_NAME[] = "image_compressed";

constexpr char APP_YAML_FILENAME[] = "config/nitros_encoder_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_h264_encoder";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_image_proc", "gxf/lib/image_proc/libgxf_tensorops.so"},
  {"isaac_ros_h264_encoder", "gxf/lib/codec/libgxf_codec_extension.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_h264_encoder"
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {
  "config/namespace_injector_rule.yaml"
};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEFAULT_FORMAT,
      .topic_name = INPUT_TOPIC_NAME,
    }
  },
  {OUTPUT_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_DEFAULT_FORMAT,
      .topic_name = OUTPUT_TOPIC_NAME,
      .frame_id_source_key = INPUT_COMPONENT_KEY
    }
  }
};
#pragma GCC diagnostic pop

EncoderNode::EncoderNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  input_width_(declare_parameter<int32_t>("input_width", 1920)),
  input_height_(declare_parameter<int32_t>("input_height", 1200)),
  qp_(declare_parameter<int32_t>("qp", 20)),
  hw_preset_type_(declare_parameter<int32_t>("hw_preset_type", 0)),
  profile_(declare_parameter<int32_t>("profile", 0)),
  iframe_interval_(declare_parameter<int32_t>("iframe_interval", 5)),
  config_(declare_parameter<std::string>("config", "pframe"))
{
  RCLCPP_DEBUG(get_logger(), "[EncoderNode] Constructor");

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCompressedImage>();

  startNitrosNode();
}

void EncoderNode::preLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[EncoderNode] preLoadGraphCallback().");

  NitrosNode::preLoadGraphSetParameter(
    "encoder", "nvidia::isaac::EncoderRequest", "config",
    config_);
}

void EncoderNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[EncoderNode] postLoadGraphCallback().");

  // Update encoder parameters
  getNitrosContext().setParameterUInt32(
    "encoder", "nvidia::isaac::EncoderRequest", "input_width",
    (uint32_t)input_width_);

  getNitrosContext().setParameterUInt32(
    "encoder", "nvidia::isaac::EncoderRequest", "input_height",
    (uint32_t)input_height_);

  getNitrosContext().setParameterUInt32(
    "encoder", "nvidia::isaac::EncoderRequest", "qp",
    (uint32_t)qp_);

  getNitrosContext().setParameterUInt32(
    "encoder", "nvidia::isaac::EncoderRequest", "hw_preset_type",
    (uint32_t)hw_preset_type_);

  getNitrosContext().setParameterUInt32(
    "encoder", "nvidia::isaac::EncoderRequest", "profile",
    (uint32_t)profile_);

  getNitrosContext().setParameterInt32(
    "encoder", "nvidia::isaac::EncoderRequest", "iframe_interval",
    iframe_interval_);
}

EncoderNode::~EncoderNode() {}

}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::h264_encoder::EncoderNode)
