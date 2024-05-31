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

#include <string>

#include "gxf/core/gxf.h"
#include "gxf/std/extension_factory_helper.hpp"
#include "videoencoder_context.hpp"
#include "videoencoder_request.hpp"
#include "videoencoder_response.hpp"


GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(0xea5c44e415db4448, 0xa3a6f32004303338,
                         "VideoEncoderExtension",
                         "Extension for video encode/compression", "NVIDIA",
                         "1.2.0", "LICENSE");
GXF_EXT_FACTORY_ADD(0xc5bdaa9f4b1543c7, 0x806620195122a5b5,
                    nvidia::gxf::VideoEncoderContext, nvidia::gxf::Component,
                    "Creates Video Encoder Context");
GXF_EXT_FACTORY_ADD(0x482513543a914033, 0x9a0f8ac2230f1c9c,
                    nvidia::gxf::VideoEncoderRequest, nvidia::gxf::Codelet,
                    "Starts video encoding process by queuing the Input buffer");
GXF_EXT_FACTORY_ADD(0xc88585c4bce048d1, 0x96802309e63c1ff8,
                    nvidia::gxf::VideoEncoderResponse, nvidia::gxf::Codelet,
                    "Completes the video encoding process by publishing encoded bit stream");

GXF_EXT_FACTORY_END()
