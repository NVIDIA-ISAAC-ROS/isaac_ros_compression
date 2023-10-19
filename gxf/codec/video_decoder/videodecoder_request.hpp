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

#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_DECODE_VIDEODECODER_REQUEST_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_DECODE_VIDEODECODER_REQUEST_HPP_

#include <memory>
#include <string>

#include "gxf/std/codelet.hpp"
#include "videodecoder_context.hpp"

namespace nvidia {
namespace gxf {

class VideoDecoderRequest : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  VideoDecoderRequest();
  ~VideoDecoderRequest();

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

  // Decoder I/O related Parameters
  gxf::Parameter<gxf::Handle<gxf::Receiver>> input_frame_;
  gxf::Parameter<uint32_t> inbuf_storage_type_;
  // Async Scheduling Term required to get/set event state.
  gxf::Parameter<gxf::Handle<gxf::AsynchronousSchedulingTerm>> scheduling_term_;
  // Decoder context
  gxf::Parameter<gxf::Handle<VideoDecoderContext>> videodecoder_context_;

  // Decoder Paramaters
  gxf::Parameter<uint32_t> codec_;
  gxf::Parameter<uint32_t> disableDPB_;
  gxf::Parameter<std::string> output_format_;
  // Create output entity and fulfill meta data
  gxf_result_t prepareOutputEntity(const gxf::Entity & input_msg);
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_DECODE_VIDEODECODER_REQUEST_HPP_
