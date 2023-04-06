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
#ifndef NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_HPP_
#define NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_HPP_

#include <cuda_runtime.h>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#include "nvcuvid.h"

namespace nvidia {
namespace isaac {

class Decoder : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override { return GXF_SUCCESS; }

  // Callback when decoding of sequence starts
  int handleVideoSequence(CUVIDEOFORMAT* video_format);
  // Callback when a decoded frame is ready to be decoded
  int handlePictureDecode(CUVIDPICPARAMS* pic_params);
  // Callback when a decoded frame is available for display
  int handlePictureDisplay(CUVIDPARSERDISPINFO* disp_info);

 private:
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> image_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  gxf::Parameter<uint32_t> input_width_;
  gxf::Parameter<uint32_t> input_height_;
  gxf::Parameter<int32_t> clock_rate_;
  gxf::Parameter<bool> low_latency_mode_;

  gxf::Handle<gxf::VideoBuffer> decoded_frame_;
  gxf::Handle<nvidia::gxf::Allocator> memory_pool_;

  // Dimension of the original image
  unsigned int luma_height_ = 0, chroma_height_ = 0, surface_height_ = 0;
  unsigned int num_chroma_planes_ = 0;

  CUcontext cu_context_ = NULL;
  CUvideoctxlock ctx_lock_;
  CUvideoparser parser_ = NULL;
  CUvideodecoder decoder_ = NULL;

  CUVIDEOFORMAT video_format_ = {};
  CUstream cuvid_stream_ = 0;

  cudaVideoChromaFormat chroma_format_;
  cudaVideoSurfaceFormat output_format_;

  // Create cuvid video parser and initialize
  gxf_result_t createVideoParser();
  // Create output entity and forward all the metadata from input message
  gxf::Expected<gxf::Entity> prepareOutputEntity(const gxf::Entity & input_msg);
  // Fulfill the metadata required by the decoder
  int fillDecoderMetadata(CUVIDEOFORMAT* video_format, int n_decode_surface);
  // Copy Luma plane from device to GXF GPU buffer
  int copyLumaPlane(CUdeviceptr src_frame, unsigned int src_pitch);
  // Copy chroma plane from device to GXF GPU buffer
  int copyChromaPlane(CUdeviceptr src_frame, unsigned int src_pitch);
};

}  // namespace isaac
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_EXTENSIONS_CODEC_COMPONENTS_DECODER_HPP_
