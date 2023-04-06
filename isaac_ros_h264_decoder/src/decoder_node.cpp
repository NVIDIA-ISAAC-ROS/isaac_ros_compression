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


#include "isaac_ros_h264_decoder/decoder_node.hpp"

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
namespace h264_decoder
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

#ifdef __x86_64__
constexpr char APP_YAML_FILENAME[] = "config/nitros_decoder_node.yaml";
#else
constexpr char APP_YAML_FILENAME[] = "config/nitros_decoder_node_jetson.yaml";
#endif

constexpr char INPUT_COMPONENT_KEY[] = "decoder/input_image";
constexpr char INPUT_DEFAULT_FORMAT[] = "nitros_compressed_image";
constexpr char INPUT_TOPIC_NAME[] = "image_compressed";

constexpr char OUTPUT_COMPONENT_KEY[] = "vault/vault";
constexpr char OUTPUT_DEFAULT_FORMAT[] = "nitros_image_rgb8";
constexpr char OUTPUT_TOPIC_NAME[] = "image_uncompressed";

constexpr char PACKAGE_NAME[] = "isaac_ros_h264_decoder";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_image_proc", "gxf/lib/image_proc/libgxf_tensorops.so"},
  {"isaac_ros_h264_decoder", "gxf/lib/codec/libgxf_codec_extension.so"},
};
const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {"config/spec.yaml"};
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

DecoderNode::DecoderNode(const rclcpp::NodeOptions & options)
: nitros::NitrosNode(options,
    APP_YAML_FILENAME,
    CONFIG_MAP,
    PRESET_EXTENSION_SPEC_NAMES,
    EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES,
    EXTENSIONS,
    PACKAGE_NAME),
  input_width_(declare_parameter<int32_t>("input_width", 1920)),
  input_height_(declare_parameter<int32_t>("input_height", 1200))
{
  RCLCPP_DEBUG(get_logger(), "[DecoderNode] Constructor");

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCompressedImage>();

  startNitrosNode();
}

void DecoderNode::preLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[DecoderNode] preLoadGraphCallback().");
}

void DecoderNode::postLoadGraphCallback()
{
  RCLCPP_INFO(get_logger(), "[DecoderNode] postLoadGraphCallback().");
  // Update decoder parameters
  #ifdef __x86_64__
  getNitrosContext().setParameterUInt32(
    "decoder", "nvidia::isaac::Decoder", "input_width",
    (uint32_t)input_width_);

  getNitrosContext().setParameterUInt32(
    "decoder", "nvidia::isaac::Decoder", "input_height",
    (uint32_t)input_height_);
  #else
  getNitrosContext().setParameterUInt32(
    "decoder", "nvidia::isaac::DecoderRequest", "input_width",
    (uint32_t)input_width_);

  getNitrosContext().setParameterUInt32(
    "decoder", "nvidia::isaac::DecoderRequest", "input_height",
    (uint32_t)input_height_);
  #endif
}

DecoderNode::~DecoderNode() {}

}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::h264_decoder::DecoderNode)
