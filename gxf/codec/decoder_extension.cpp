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
#include "extensions/codec/components/decoder.hpp"
#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0x5b942ff659bc4502, 0xa0b000b36b53f74f, "CodecExtension",
  "Extension containing a decoder that decompress H.264 to YUV",
  "NVIDIA", "0.0.1", "LICENSE");

GXF_EXT_FACTORY_ADD(
  0x6d60fe3be19342ec, 0xbab3b378d142da06,
  nvidia::isaac::Decoder, nvidia::gxf::Codelet,
  "Decode H.264 to YUV image");

GXF_EXT_FACTORY_END()
