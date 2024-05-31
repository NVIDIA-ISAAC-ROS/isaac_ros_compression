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

#ifndef NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEODECODER_RESPONSE_HPP_
#define NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEODECODER_RESPONSE_HPP_

#include <memory>

#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "videodecoder_context.hpp"


namespace nvidia {
namespace gxf {

class VideoDecoderResponse : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  VideoDecoderResponse();
  ~VideoDecoderResponse();

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
  gxf::Parameter<uint32_t> outbuf_storage_type_;
  gxf::Parameter<gxf::Handle<VideoDecoderContext>> videodecoder_context_;

  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> decoded_frame_;
  nvidia::gxf::Handle<nvidia::gxf::Allocator> memory_pool_;
  // Copy YUV data from V4L2 buffer to gxf videobuffer
  gxf_result_t copyYUVFrame();
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_MULTIMEDIA_EXTENSIONS_VIDEODECODER_RESPONSE_HPP_
