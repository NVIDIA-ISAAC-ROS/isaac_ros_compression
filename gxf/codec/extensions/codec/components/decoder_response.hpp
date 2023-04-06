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
#ifndef NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_RESPONSE_HPP_
#define NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_RESPONSE_HPP_

#include "extensions/codec/components/decoder_request.hpp"

namespace nvidia {
namespace isaac {

class DecoderResponse : public gxf::Codelet {
 public:
  // Explicitly declare constructors and destructors
  // to get around forward declaration of Impl
  DecoderResponse();
  ~DecoderResponse();

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  gxf::Parameter<gxf::Handle<DecoderRequest>> decoder_impl_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> tx_;
};

}  // namespace isaac
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_RESPONSE_HPP_
