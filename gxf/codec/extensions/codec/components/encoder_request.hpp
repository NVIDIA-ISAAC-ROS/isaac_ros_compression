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
#ifndef NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_ENCODER_REQUEST_HPP_
#define NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_ENCODER_REQUEST_HPP_

#include <cuda_runtime.h>
#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

enum struct EncoderInputFormat {
  kNV12 = 0,  // input format is nv12;
  kNV24 = 1,  // input format is nv24;
};

enum struct EncoderConfig {
  kIFrame = 0,  // i frame only mode;
  kPFrame = 1,  // group of picture size is 5 with one i frame and four p frames;
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
    return gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
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
    if (strcmp(value.c_str(), "iframe") == 0) {
      return EncoderConfig::kIFrame;
    }
    if (strcmp(value.c_str(), "pframe") == 0) {
      return EncoderConfig::kPFrame;
    }
    return gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE};
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

}
}

namespace nvidia {
namespace isaac {

class EncoderRequest : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  EncoderRequest();
  ~EncoderRequest();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  // Try to DQ the capture plane to get the compressed output
  // This function may return an empty entity when it triggered by other events
  gxf::Expected<gxf::Entity> hasDQCapturePlane();
  // Check if there is any available buffer to accept new request
  gxf::Expected<bool> isAcceptingRequest();

 private:
  // Fill the encoder context wil default values
  gxf_result_t createDefaultEncoderContext();
  // Get input format from parameter
  gxf_result_t getInputFormatFromParameter(const gxf::EncoderInputFormat & input_format);
  // Get compression config from parameter
  gxf_result_t getCompressionSettingsFromParameter(const gxf::EncoderConfig & config);
  // Get compression level from parameter
  gxf_result_t getCompressionLevelFromParameter(const int32_t & level);
  // Initialize encoder with provide configs
  gxf_result_t initializeEncoder();
  // Enqueue all the empty capture plane buffers
  gxf_result_t createBuffers();
  // Place input image into v4l2 buffer from video buffer
  gxf_result_t placeInputImage(const gxf::Entity & input_msg);
  // Create output entity and fulfill meta data
  gxf_result_t prepareOutputEntity(const gxf::Entity & input_msg);

  // Hide implementation details
  struct Impl;
  std::unique_ptr<Impl> impl_;

  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> image_receiver_;
  // Async Scheduling Term required to get/set event state.
  gxf::Parameter<gxf::Handle<gxf::AsynchronousSchedulingTerm>>
    scheduling_term_;

  gxf::Parameter<uint32_t> input_width_;
  gxf::Parameter<uint32_t> input_height_;
  gxf::Parameter<uint32_t> qp_;
  gxf::Parameter<uint32_t> hw_preset_type_;
  gxf::Parameter<uint32_t> profile_;
  gxf::Parameter<int32_t> level_;
  gxf::Parameter<int32_t> entropy_;
  gxf::Parameter<int32_t> iframe_interval_;
  gxf::Parameter<gxf::EncoderConfig> config_;
  gxf::Parameter<gxf::EncoderInputFormat> input_format_;

  cudaStream_t cuda_stream_;
  std::atomic<uint32_t> available_buffer_num_;
  std::queue<gxf::Entity> output_entity_queue_;
};

}  // namespace isaac
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_ENCODER_REQUEST_HPP_
