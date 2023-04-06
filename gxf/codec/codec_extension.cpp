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
#include "extensions/codec/components/decoder_request.hpp"
#include "extensions/codec/components/decoder_response.hpp"
#include "extensions/codec/components/decoder_scheduling_term.hpp"
#include "extensions/codec/components/encoder_request.hpp"
#include "extensions/codec/components/encoder_response.hpp"
#include "extensions/codec/components/encoder_scheduling_term.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0x5b942ff659bc4502, 0xa0b000b36b53f74f, "CodecExtension",
  "Extension containing a encoder that compress input images to H.264 format", "NVIDIA", "0.0.1",
  "LICENSE");

GXF_EXT_FACTORY_ADD(
  0x88d747c54fab4a65, 0xbf22fad3e0e7f026,
  nvidia::isaac::EncoderRequest, nvidia::gxf::Codelet,
  "Get YUV image and send it to NvEnc");

GXF_EXT_FACTORY_ADD(
  0x88d747c54fab1e23, 0xbf22fad3e0e6f217,
  nvidia::isaac::EncoderResponse, nvidia::gxf::Codelet,
  "Get H.264 output from NvEnc");

GXF_EXT_FACTORY_ADD(
  0x95d747c54fab2e13, 0xbf21fad3a0e5f413,
  nvidia::isaac::EncoderRequestReceptiveSchedulingTerm, nvidia::gxf::SchedulingTerm,
  "Encoder scheduling term for request");

GXF_EXT_FACTORY_ADD(
  0x60caaa3c8f044c67, 0x8695a7d770628294,
  nvidia::isaac::DecoderRequest, nvidia::gxf::Codelet,
  "Get YUV image and send it to NvDec");

GXF_EXT_FACTORY_ADD(
  0xd875957cbc3440a9, 0x82ef543cf12ac9ae,
  nvidia::isaac::DecoderResponse, nvidia::gxf::Codelet,
  "Get H.264 output from NvDec");

GXF_EXT_FACTORY_ADD(
  0x0e161c7ee17040ee, 0x89b7fb09db770cd6 ,
  nvidia::isaac::DecoderRequestReceptiveSchedulingTerm, nvidia::gxf::SchedulingTerm,
  "Decoder scheduling term for request");

GXF_EXT_FACTORY_END()
