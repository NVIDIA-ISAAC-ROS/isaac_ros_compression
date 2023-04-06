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
#include "extensions/codec/components/encoder_response.hpp"

namespace nvidia {
namespace isaac {

EncoderResponse::EncoderResponse() {}

EncoderResponse::~EncoderResponse() {}

gxf_result_t EncoderResponse::registerInterface(nvidia::gxf::Registrar* registrar) {
  nvidia::gxf::Expected<void> result;
  result &= registrar->parameter(encoder_impl_, "encoder_impl",
    "Encoder Implementation Handle");
  result &= registrar->parameter(tx_, "tx", "TX", "Transmitter to publish output tensors");
  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t EncoderResponse::start() {
  if (!encoder_impl_.get()) {
    GXF_LOG_ERROR("Encoder unavailable");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t EncoderResponse::tick() {
  // Get input image
  if (!encoder_impl_.get()) {
      GXF_LOG_ERROR("Encoder unavailable");
      return GXF_FAILURE;
  }
  auto maybe_response = encoder_impl_.get()->hasDQCapturePlane();
  if (!maybe_response) {
    // Doesn't have output ready
    if (nvidia::gxf::ToResultCode(maybe_response) == GXF_QUERY_NOT_FOUND) {
      return GXF_SUCCESS;
    }
    return nvidia::gxf::ToResultCode(maybe_response);
  }
  auto input_img = maybe_response->get<gxf::Tensor>();
  if (!input_img) {
    return GXF_SUCCESS;
  }
  return gxf::ToResultCode(tx_->publish(maybe_response.value()));
}

gxf_result_t EncoderResponse::stop() {
  return GXF_SUCCESS;
}

}  // namespace isaac
}  // namespace nvidia
