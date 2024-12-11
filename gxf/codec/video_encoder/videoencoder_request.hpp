// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_REQUEST_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_REQUEST_HPP_

#include <memory>
#include <string>

#include "gxf/core/component.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"
#include "videoencoder_context.hpp"

namespace nvidia {
namespace gxf {

// Min/Max resolution supported by the video encoder.
constexpr uint32_t  kVideoEncoderMinHeight = 128;
constexpr uint32_t  kVideoEncoderMinWidth = 128;
constexpr uint32_t  kVideoEncoderMaxHeight = 4096;
constexpr uint32_t  kVideoEncoderMaxWidth = 4096;

constexpr char kTimestampName[] = "timestamp";

enum struct EncoderInputFormat {
  kNV12 = 0,          // input format is NV12;
  kNV24 = 1,          // input format is NV24;
  kYUV420PLANAR = 3,  // input format is YUV420 planar;
  kUnsupported = 4    // Unsupported parameter
};

enum struct EncoderConfig {
  kIFrameCQP = 0,   // I frame only, CQP mode;
  kPFrameCQP = 1,   // IPP GOP, CQP
  kCustom = 2,      // Custom parmeters
  kUnsupported = 3  // Unsupported parameter
};

template <>
// Custom parameter parser for EncoderInputFormat
struct ParameterParser<EncoderInputFormat> {
  static Expected<EncoderInputFormat> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                       const char* key, const YAML::Node& node,
                                       const std::string& prefix) {
    const std::string value = node.as<std::string>();
    if (strcmp(value.c_str(), "nv12") == 0) {
      return EncoderInputFormat::kNV12;
    }
    if (strcmp(value.c_str(), "nv24") == 0) {
      return EncoderInputFormat::kNV24;
    }
    if (strcmp(value.c_str(), "yuv420planar") == 0) {
      return EncoderInputFormat::kYUV420PLANAR;
    }
     return EncoderInputFormat::kUnsupported;
  }
};

template<>
struct ParameterWrapper<EncoderInputFormat> {
  static Expected<YAML::Node> Wrap(
    gxf_context_t context,
    const EncoderInputFormat& value) {
    return Unexpected{GXF_NOT_IMPLEMENTED};
  }
};

template <>
// Custom parameter parser for EncoderConfig
struct ParameterParser<EncoderConfig> {
  static Expected<EncoderConfig> Parse(gxf_context_t context, gxf_uid_t component_uid,
                                       const char* key, const YAML::Node& node,
                                       const std::string& prefix) {
    const std::string value = node.as<std::string>();
    if (strcmp(value.c_str(), "iframe_cqp") == 0) {
      return EncoderConfig::kIFrameCQP;
    }
    if (strcmp(value.c_str(), "pframe_cqp") == 0) {
      return EncoderConfig::kPFrameCQP;
    }
    if (strcmp(value.c_str(), "custom") == 0) {
      return EncoderConfig::kCustom;
    }
    return EncoderConfig::kUnsupported;
  }
};

template<>
struct ParameterWrapper<EncoderConfig> {
  static Expected<YAML::Node> Wrap(
    gxf_context_t context,
    const EncoderConfig& value) {
    return Unexpected{GXF_NOT_IMPLEMENTED};
  }
};

class VideoEncoderRequest : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  VideoEncoderRequest();
  ~VideoEncoderRequest();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

 private:
  // Hide implementation details
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Encoder I/O related Parameters
  gxf::Parameter<gxf::Handle<gxf::Receiver>> input_frame_;
  gxf::Parameter<uint32_t> inbuf_storage_type_;
  // Async Scheduling Term required to get/set event state.
  // Encoder context
  gxf::Parameter<gxf::Handle<VideoEncoderContext>> videoencoder_context_;

  // Encoder Paramaters
  gxf::Parameter<int32_t> codec_;
  gxf::Parameter<uint32_t> input_height_;
  gxf::Parameter<uint32_t> input_width_;
  gxf::Parameter<gxf::EncoderInputFormat> input_format_;
  gxf::Parameter<int32_t> profile_;
  gxf::Parameter<int32_t> bitrate_;
  gxf::Parameter<int32_t> framerate_;
  gxf::Parameter<uint32_t> qp_;
  gxf::Parameter<int32_t> hw_preset_type_;
  gxf::Parameter<int32_t> level_;
  gxf::Parameter<int32_t> iframe_interval_;
  gxf::Parameter<int32_t> rate_control_mode_;
  gxf::Parameter<gxf::EncoderConfig> config_;

  // Queue input buffer
  gxf_result_t queueInputYUVBuf(const gxf::Handle<gxf::VideoBuffer> input_img);
  // Get input format from parameter
  gxf_result_t checkInputParams();
  // Get input format from parameter
  gxf_result_t getInputFormatFromParameter();
  // Get compression config from parameter
  gxf_result_t getEncoderSettingsFromParameters();
  // Get compression level from parameter
  gxf_result_t getProfileLevelSettingsFromParameter();
  // Initialize encoder with provide configs
  gxf_result_t setEncoderParameters();
  // Create output entity and fill meta data
  gxf_result_t prepareOutputEntity(const gxf::Entity & input_msg);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_REQUEST_HPP_
