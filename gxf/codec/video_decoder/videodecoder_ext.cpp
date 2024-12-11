// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"
#include "videodecoder_context.hpp"
#include "videodecoder_request.hpp"
#include "videodecoder_response.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xedc9900173bd435c, 0xaf0ce013dcda3439,
                         "VideoDecoderExtension",
                         "Extension for video decode/decompression", "NVIDIA",
                         "1.3.0", "LICENSE");

GXF_EXT_FACTORY_ADD(0xc69e604d9f1d425e, 0xad5f121a7e9d3456,
                    nvidia::gxf::VideoDecoderContext, nvidia::gxf::Component,
                    "Video Video Decoder context");

GXF_EXT_FACTORY_ADD(0x39c030763a424927, 0x99600072b4e1bc69,
                    nvidia::gxf::VideoDecoderRequest, nvidia::gxf::Codelet,
                    "Starts video decoding process by queuing the Input buffer");

GXF_EXT_FACTORY_ADD(0x6cc164db5db4431e, 0x8b63a45ea1e7b8a6,
                    nvidia::gxf::VideoDecoderResponse, nvidia::gxf::Codelet,
                    "Completes the video decoding process by publishing decoded YUV buffer");

GXF_EXT_FACTORY_END()
