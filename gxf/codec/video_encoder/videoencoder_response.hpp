// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_RESPONSE_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_RESPONSE_HPP_

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/handle.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"
#include "videoencoder_context.hpp"

namespace nvidia {
namespace gxf {

class VideoEncoderResponse : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  VideoEncoderResponse();
  ~VideoEncoderResponse();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  // Hide implementation details
  struct Impl;
  std::unique_ptr<Impl> impl_;

  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<VideoEncoderContext>> videoencoder_context_;
  gxf::Parameter<uint32_t> outbuf_storage_type_;
};


}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEOENCODER_RESPONSE_HPP_
