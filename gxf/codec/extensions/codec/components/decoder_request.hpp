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
#ifndef NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_REQUEST_HPP_
#define NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_REQUEST_HPP_

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
#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/scheduling_terms.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {

class DecoderRequest : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  DecoderRequest();
  ~DecoderRequest();

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

  cudaStream_t cuda_stream_;
  std::atomic<uint32_t> available_buffer_num_;
  uint32_t total_buffer_num_;
  std::queue<gxf::Entity> output_entity_queue_;

  // Fill the decoder context wil default values
  gxf_result_t createDefaultDecoderContext();
  // Initialize decoder with provide configs
  gxf_result_t initializeDecoder();
  // Place input image into v4l2 buffer from video buffer
  gxf_result_t placeInputImage(const gxf::Entity & input_msg);
  // Create output entity and fulfill meta data
  gxf_result_t prepareOutputEntity(const gxf::Entity & input_msg);
};

}  // namespace isaac
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_REQUEST_HPP_
