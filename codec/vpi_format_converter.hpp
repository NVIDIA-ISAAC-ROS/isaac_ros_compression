// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// DEPRECATED: This helper exists for backward-compatible color format conversion
// inside the encoder/decoder nodes. It will be removed in the next major release
// when the encoder/decoder switch to NV12-only I/O.

#ifndef CODEC__VPI_FORMAT_CONVERTER_HPP_
#define CODEC__VPI_FORMAT_CONVERTER_HPP_

#include <vpi/CUDAInterop.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace nvidia
{
namespace isaac_ros
{
namespace codec
{

inline void FillNV12ImageData(
  VPIImageData & data,
  uint8_t * y_ptr, uint32_t y_pitch,
  uint8_t * uv_ptr, uint32_t uv_pitch,
  uint32_t width, uint32_t height)
{
  std::memset(&data, 0, sizeof(data));
  data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  data.buffer.pitch.format = VPI_IMAGE_FORMAT_NV12_ER;
  data.buffer.pitch.numPlanes = 2;
  data.buffer.pitch.planes[0].pBase = y_ptr;
  data.buffer.pitch.planes[0].width = width;
  data.buffer.pitch.planes[0].height = height;
  data.buffer.pitch.planes[0].pitchBytes = y_pitch;
  data.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_U8;
  data.buffer.pitch.planes[1].pBase = uv_ptr;
  data.buffer.pitch.planes[1].width = width / 2;
  data.buffer.pitch.planes[1].height = height / 2;
  data.buffer.pitch.planes[1].pitchBytes = uv_pitch;
  data.buffer.pitch.planes[1].pixelType = VPI_PIXEL_TYPE_2U8;
}

inline void FillInterleaved3ImageData(
  VPIImageData & data,
  uint8_t * ptr, uint32_t pitch,
  uint32_t width, uint32_t height,
  VPIImageFormat format)
{
  std::memset(&data, 0, sizeof(data));
  data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
  data.buffer.pitch.format = format;
  data.buffer.pitch.numPlanes = 1;
  data.buffer.pitch.planes[0].pBase = ptr;
  data.buffer.pitch.planes[0].width = width;
  data.buffer.pitch.planes[0].height = height;
  data.buffer.pitch.planes[0].pitchBytes = pitch;
  data.buffer.pitch.planes[0].pixelType = VPI_PIXEL_TYPE_3U8;
}

inline void FillRGB8ImageData(
  VPIImageData & data,
  uint8_t * ptr, uint32_t pitch,
  uint32_t width, uint32_t height)
{
  FillInterleaved3ImageData(data, ptr, pitch, width, height, VPI_IMAGE_FORMAT_RGB8);
}

inline void FillBGR8ImageData(
  VPIImageData & data,
  uint8_t * ptr, uint32_t pitch,
  uint32_t width, uint32_t height)
{
  FillInterleaved3ImageData(data, ptr, pitch, width, height, VPI_IMAGE_FORMAT_BGR8);
}

inline VPIStatus CreateOrSetWrapper(VPIImage & image, VPIImageData & data)
{
  if (!image) {
    return vpiImageCreateWrapper(&data, nullptr, VPI_BACKEND_CUDA, &image);
  }
  return vpiImageSetWrapper(image, &data);
}

// Stateful VPI format converter that reuses VPI images across calls.
// Caller owns the VPIStream and passes it in.
class VPIFormatConverter
{
public:
  VPIFormatConverter() = default;

  ~VPIFormatConverter()
  {
    if (input_) {
      vpiImageDestroy(input_);
    }
    if (output_) {
      vpiImageDestroy(output_);
    }
  }

  VPIFormatConverter(const VPIFormatConverter &) = delete;
  VPIFormatConverter & operator=(const VPIFormatConverter &) = delete;

  VPIStatus convert(
    VPIStream stream,
    VPIImageData & input_data,
    VPIImageData & output_data)
  {
    VPIStatus s = CreateOrSetWrapper(input_, input_data);
    if (s != VPI_SUCCESS) {return s;}
    s = CreateOrSetWrapper(output_, output_data);
    if (s != VPI_SUCCESS) {return s;}
    s = vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, input_, output_, nullptr);
    if (s != VPI_SUCCESS) {return s;}
    return vpiStreamSync(stream);
  }

private:
  VPIImage input_{nullptr};
  VPIImage output_{nullptr};
};

}  // namespace codec
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // CODEC__VPI_FORMAT_CONVERTER_HPP_
