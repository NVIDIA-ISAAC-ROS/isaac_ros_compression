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

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "gxf/std/timestamp.hpp"

namespace {
#define CHECK_CUDA_ERROR(error_code)           \
  if (error_code != CUDA_SUCCESS) {            \
    const char* error_msg = NULL;              \
    cuGetErrorString(error_code, &error_msg);  \
    GXF_LOG_ERROR(error_msg);                  \
    return 0;                                  \
  }                                            \

constexpr char kTimestampName[] = "timestamp";

static int CUDAAPI handleVideoSequenceProc(void* user_data, CUVIDEOFORMAT* video_format) {
  return (reinterpret_cast<nvidia::isaac::Decoder* >(user_data))
    ->handleVideoSequence(video_format);
}

static int CUDAAPI handlePictureDecodeProc(void* user_data, CUVIDPICPARAMS* pic_params) {
  return (reinterpret_cast<nvidia::isaac::Decoder* >(user_data))
    ->handlePictureDecode(pic_params);
}
}  // namespace

namespace nvidia {
namespace isaac {

gxf_result_t Decoder::createVideoParser() {
  CUVIDPARSERPARAMS videoParserParameters = {};
  videoParserParameters.CodecType = cudaVideoCodec_H264;
  videoParserParameters.ulMaxNumDecodeSurfaces = 1;
  videoParserParameters.ulClockRate = clock_rate_;
  videoParserParameters.ulMaxDisplayDelay = low_latency_mode_ ? 0 : 1;
  videoParserParameters.pUserData = this;
  videoParserParameters.pfnSequenceCallback = handleVideoSequenceProc;
  videoParserParameters.pfnDecodePicture = handlePictureDecodeProc;
  videoParserParameters.pfnDisplayPicture = NULL;
  videoParserParameters.pfnGetOperatingPoint = NULL;
  auto error_code = cuvidCreateVideoParser(&parser_, &videoParserParameters);
  if (error_code != CUDA_SUCCESS) {
    const char* error_log;
    cuGetErrorString(error_code, &error_log);
    GXF_LOG_ERROR("Failed to create video parser: %s", error_log);
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

int Decoder::fillDecoderMetadata(CUVIDEOFORMAT* video_format, int n_decode_surface) {
  CUVIDDECODECREATEINFO video_decode_create_info = { 0 };
  video_decode_create_info.CodecType = video_format->codec;
  video_decode_create_info.ChromaFormat = video_format->chroma_format;
  video_decode_create_info.OutputFormat = output_format_;
  video_decode_create_info.bitDepthMinus8 = video_format->bit_depth_luma_minus8;

  if (video_format->progressive_sequence) {
    video_decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
  } else {
    video_decode_create_info.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
  }

  video_decode_create_info.ulNumOutputSurfaces = 2;
  video_decode_create_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  video_decode_create_info.ulNumDecodeSurfaces = n_decode_surface;
  video_decode_create_info.vidLock = ctx_lock_;
  video_decode_create_info.ulWidth = video_format->coded_width;
  video_decode_create_info.ulHeight = video_format->coded_height;
  video_decode_create_info.ulMaxWidth = video_format->coded_width;
  video_decode_create_info.ulMaxHeight = video_format->coded_height;
  video_decode_create_info.ulTargetWidth = video_format->coded_width;
  video_decode_create_info.ulTargetHeight = video_format->coded_height;

  // The height of UV plane is half of the luma plane
  chroma_height_ = luma_height_ * 0.5;
  surface_height_ = video_format->coded_height;

  CHECK_CUDA_ERROR(cuCtxPushCurrent(cu_context_));
  CHECK_CUDA_ERROR(cuvidCreateDecoder(&decoder_, &video_decode_create_info));
  CHECK_CUDA_ERROR(cuCtxPopCurrent(NULL));

  return 1;
}

int Decoder::copyLumaPlane(CUdeviceptr src_frame, unsigned int src_pitch) {
  CUDA_MEMCPY2D m = { 0 };
  auto decoded_frame_info = decoded_frame_->video_frame_info();

  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.srcDevice = src_frame;
  m.srcPitch = src_pitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = (CUdeviceptr)(decoded_frame_->pointer());
  m.dstPitch = decoded_frame_info.color_planes[0].stride;
  m.WidthInBytes = decoded_frame_info.color_planes[0].width *
                   decoded_frame_info.color_planes[0].bytes_per_pixel;
  m.Height = luma_height_;
  CHECK_CUDA_ERROR(cuMemcpy2DAsync(&m, cuvid_stream_));

  return 1;
}

int Decoder::copyChromaPlane(CUdeviceptr src_frame, unsigned int src_pitch) {
  CUDA_MEMCPY2D m = { 0 };
  auto decoded_frame_info = decoded_frame_->video_frame_info();

  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.srcDevice = (CUdeviceptr)(src_frame + src_pitch * surface_height_);
  m.srcPitch = src_pitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = (CUdeviceptr)(decoded_frame_->pointer() +
                 decoded_frame_info.color_planes[0].size);
  m.dstPitch = decoded_frame_info.color_planes[1].stride;
  m.WidthInBytes = decoded_frame_info.color_planes[1].width *
                   decoded_frame_info.color_planes[1].bytes_per_pixel;
  m.Height = chroma_height_;
  CHECK_CUDA_ERROR(cuMemcpy2DAsync(&m, cuvid_stream_));

  return 1;
}

gxf::Expected<gxf::Entity> Decoder::prepareOutputEntity(const gxf::Entity & input_msg) {
  auto output_entity = gxf::Entity::New(context());
  if (!output_entity) {
    GXF_LOG_ERROR("Failed to create output message");
    return gxf::ForwardError(output_entity);
  }

  auto maybe_input_intrinsics = input_msg.get<gxf::CameraModel>("intrinsics");
  if (maybe_input_intrinsics) {
    auto output_intrinsics = output_entity->add<gxf::CameraModel>(maybe_input_intrinsics->name());
    if (!output_intrinsics) {
      GXF_LOG_ERROR("Failed to add intrinsics into output");
      return gxf::ForwardError(output_intrinsics);
    }
    *output_intrinsics.value() = *maybe_input_intrinsics.value();
  }

  auto maybe_input_extrinsics = input_msg.get<gxf::Pose3D>("extrinsics");
  if (maybe_input_extrinsics) {
    auto output_extrinsics = output_entity->add<gxf::Pose3D>(maybe_input_extrinsics->name());
    if (!output_extrinsics) {
      GXF_LOG_ERROR("Failed to add extrinsics into output");
      return gxf::ForwardError(output_extrinsics);
    }
    *output_extrinsics.value() = *maybe_input_extrinsics.value();
  }

  auto maybe_input_seq_number = input_msg.get<int64_t>("sequence_number");
  if (maybe_input_seq_number) {
    auto output_seq_number = output_entity->add<int64_t>(maybe_input_seq_number->name());
    if (!output_seq_number) {
      GXF_LOG_ERROR("Failed to add sequence number into output");
      return gxf::ForwardError(output_seq_number);
    }
    *output_seq_number.value() = *maybe_input_seq_number.value();
  }

  // Follow the ISAAC convention to add unamed timestamp first
  auto maybe_sensor_timestamp = input_msg.get<gxf::Timestamp>();
  if (maybe_sensor_timestamp) {
    auto output_sensor_timestamp = output_entity->add<gxf::Timestamp>();
    if (!output_sensor_timestamp) {
      GXF_LOG_ERROR("Failed to add sensor timestamp into output");
      return gxf::ForwardError(output_sensor_timestamp);
    }
    *output_sensor_timestamp.value() = *maybe_sensor_timestamp.value();
  }

  auto maybe_gxf_timestamp = input_msg.get<gxf::Timestamp>(kTimestampName);
  if (maybe_gxf_timestamp) {
    auto output_gxf_timestamp = output_entity->add<gxf::Timestamp>(kTimestampName);
    if (!output_gxf_timestamp) {
      GXF_LOG_ERROR("Failed to add gxf timestamp into output");
      return gxf::ForwardError(output_gxf_timestamp);
    }
    *output_gxf_timestamp.value() = *maybe_gxf_timestamp.value();
  }

  return output_entity;
}

gxf_result_t Decoder::registerInterface(gxf::Registrar * registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      image_receiver_, "image_receiver",
      "Receiver to get the input image");
  result &= registrar->parameter(
      output_transmitter_, "output_transmitter",
      "Transmitter to send the compressed data");
  result &= registrar->parameter(
      pool_, "pool",
      "Memory pool for allocating output data");
  result &= registrar->parameter(
      input_height_, "input_height",
      "Input image height");
  result &= registrar->parameter(
      input_width_, "input_width",
      "Input image width");
  result &= registrar->parameter(
      clock_rate_, "clock_rate", "Time stamp units",
      "Timestamp units in Hz (0=default=10000000Hz)",
      1000);
  result &= registrar->parameter(
      low_latency_mode_, "low_latency_mode", "Max display queue delay",
      "Max display queue delay, no delay when low_latency_mode is enabled",
      false);
  return gxf::ToResultCode(result);
}

gxf_result_t Decoder::start() {
  auto error_code = cuCtxCreate(&cu_context_, 0, 0);
  if (error_code != CUDA_SUCCESS) {
    const char* error_log;
    cuGetErrorString(error_code, &error_log);
    GXF_LOG_ERROR("Failed to create cuda context: %s", error_log);
    return GXF_FAILURE;
  }

  error_code = cuvidCtxLockCreate(&ctx_lock_, cu_context_);
  if (error_code != CUDA_SUCCESS) {
    const char* error_log;
    cuGetErrorString(error_code, &error_log);
    GXF_LOG_ERROR("Failed to create cuda context lock: %s", error_log);
    return GXF_FAILURE;
  }

  auto gxf_ret_code = createVideoParser();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create video parser");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t Decoder::tick() {
  // Get input image
  auto maybe_input_message = image_receiver_->receive();
  if (!maybe_input_message) {
    GXF_LOG_ERROR("Failed to receive input message");
    return maybe_input_message.error();
  }
  auto input_image = maybe_input_message->get<gxf::Tensor>();
    if (!input_image) {
    GXF_LOG_ERROR("Failed to get image from message");
    return input_image.error();
  }

  // Create output message
  // gxf::Expected<gxf::Entity> output_message = gxf::Entity::New(context());
  gxf::Expected<gxf::Entity> output_message = prepareOutputEntity(maybe_input_message.value());
  if (!output_message) {
  GXF_LOG_ERROR("Failed to create output message");
    return output_message.error();
  }
  auto output_video_buffer = output_message->add<gxf::VideoBuffer>(input_image->name());
  if (!output_video_buffer) {
    GXF_LOG_ERROR("Failed to add output video buffer");
    return output_video_buffer.error();
  }

  luma_height_ = input_height_;
  decoded_frame_ = output_video_buffer.value();
  memory_pool_ = pool_.get();

  const int n_video_bytes = input_image.value()->size();
  uint8_t* p_video =  input_image.value()->data<uint8_t>().value();

  CUVIDSOURCEDATAPACKET packet = { 0 };
  packet.payload = p_video;
  packet.payload_size = n_video_bytes;
  packet.flags = CUVID_PKT_ENDOFPICTURE;
  if (!p_video || n_video_bytes == 0) {
    packet.flags |= CUVID_PKT_ENDOFSTREAM;
  }

  CUresult error_code = cuvidParseVideoData(parser_, &packet);
  if (error_code != CUDA_SUCCESS) {
    const char* error_log;
    cuGetErrorString(error_code, &error_log);
    GXF_LOG_ERROR("Failed to parse the video data into parser: %s", error_log);
    return GXF_FAILURE;
  }

  // Check if get valid output from decoder
  // The output could be empty if I frame is missing.
  if (!output_video_buffer.value()->size()) {
    GXF_LOG_WARNING("Decoder output is empty, current frame will be dropped");
    return GXF_SUCCESS;
  }

  return gxf::ToResultCode(output_transmitter_->publish(output_message.value()));
}

int Decoder::handleVideoSequence(CUVIDEOFORMAT* video_format) {
  int n_decode_surface = video_format->min_num_decode_surfaces;

  CUVIDDECODECAPS decodecaps;
  std::memset(&decodecaps, 0, sizeof(decodecaps));

  decodecaps.eCodecType = video_format->codec;
  decodecaps.eChromaFormat = video_format->chroma_format;
  decodecaps.nBitDepthMinus8 = video_format->bit_depth_luma_minus8;

  CHECK_CUDA_ERROR(cuCtxPushCurrent(cu_context_));
  CHECK_CUDA_ERROR(cuvidGetDecoderCaps(&decodecaps));
  CHECK_CUDA_ERROR(cuCtxPopCurrent(NULL));

  if (!decodecaps.bIsSupported) {
    GXF_LOG_ERROR("Codec is not supported on this GPU %d", decodecaps.eCodecType);
    return 0;
  }

  if ((video_format->coded_width > decodecaps.nMaxWidth) ||
    (video_format->coded_height > decodecaps.nMaxHeight)) {
    // Compare to the max width and max hieght
    GXF_LOG_ERROR("Resolution is not supported on this GPU");
    return 0;
  }

  if ((video_format->coded_width >> 4) * (video_format->coded_height >> 4)
       > decodecaps.nMaxMBCount) {
    GXF_LOG_ERROR("Image dimension exceed the GPU capacity");
    return 0;
  }

  chroma_format_ = video_format->chroma_format;
  if (chroma_format_ != cudaVideoChromaFormat_420) {
    GXF_LOG_ERROR("Only YUV420p chroma format is supported");
    return 0;
  }

  // Output NV12 when input chroma format is YUV420.
  // Could add more output format based on the chroma format.
  output_format_ = cudaVideoSurfaceFormat_NV12;
  video_format_ = *video_format;

  if (!fillDecoderMetadata(video_format, n_decode_surface)) {
    GXF_LOG_ERROR("Failed to fill the decoder metadata");
    return 0;
  }

  return n_decode_surface;
}

int Decoder::handlePictureDecode(CUVIDPICPARAMS* pic_params) {
  if (!decoder_) {
    GXF_LOG_ERROR("Decoder is not initialized");
    return 0;
  }
  CHECK_CUDA_ERROR(cuCtxPushCurrent(cu_context_));
  CHECK_CUDA_ERROR(cuvidDecodePicture(decoder_, pic_params));

  CUVIDPARSERDISPINFO disp_info;
  std::memset(&disp_info, 0, sizeof(disp_info));
  disp_info.picture_index = pic_params->CurrPicIdx;
  disp_info.progressive_frame = !pic_params->field_pic_flag;
  disp_info.top_field_first = pic_params->bottom_field_flag ^ 1;
  handlePictureDisplay(&disp_info);
  CHECK_CUDA_ERROR(cuCtxPopCurrent(NULL));

  GXF_LOG_DEBUG("Finish handle picture decode func");
  return 1;
}

int Decoder::handlePictureDisplay(CUVIDPARSERDISPINFO* disp_info) {
  CUVIDPROCPARAMS video_processing_parameters = {};
  video_processing_parameters.progressive_frame = disp_info->progressive_frame;
  video_processing_parameters.second_field = disp_info->repeat_first_field + 1;
  video_processing_parameters.top_field_first = disp_info->top_field_first;
  video_processing_parameters.unpaired_field = disp_info->repeat_first_field < 0;
  video_processing_parameters.output_stream = cuvid_stream_;

  CUdeviceptr src_frame = 0;
  unsigned int src_pitch = 0;
  CHECK_CUDA_ERROR(cuCtxPushCurrent(cu_context_));
  CHECK_CUDA_ERROR(cuvidMapVideoFrame(decoder_, disp_info->picture_index, &src_frame,
      &src_pitch, &video_processing_parameters));

  CUVIDGETDECODESTATUS decode_status;
  std::memset(&decode_status, 0, sizeof(decode_status));
  auto error_code = cuvidGetDecodeStatus(decoder_, disp_info->picture_index, &decode_status);
  if (error_code != CUDA_SUCCESS) {
    GXF_LOG_ERROR("Failed to get decode status");
    return 0;
  } else if (error_code == CUDA_SUCCESS &&
            (decode_status.decodeStatus == cuvidDecodeStatus_Error ||
             decode_status.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
    GXF_LOG_ERROR("Got decode error for current picture");
    return 0;
  }

  constexpr auto surface_layout = gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  constexpr auto storage_type = gxf::MemoryStorageType::kDevice;

  auto gxf_ret_code = decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER>(
                      input_width_, luma_height_, surface_layout, storage_type, memory_pool_);
  if (!gxf_ret_code) {
    GXF_LOG_ERROR("Failed to resize output video buffer");
    return 0;
  }

  if (!copyLumaPlane(src_frame, src_pitch)) {
    GXF_LOG_ERROR("Failed to copy luma plane");
    return 0;
  }

  // NVDEC output has luma height aligned by 2. Adjust chroma offset by aligning height
  // Need the third plane if YUV444 is considered
  if (!copyChromaPlane(src_frame, src_pitch)) {
    GXF_LOG_ERROR("Failed to copy chroma plane");
    return 0;
  }

  CHECK_CUDA_ERROR(cuStreamSynchronize(cuvid_stream_));
  CHECK_CUDA_ERROR(cuvidUnmapVideoFrame(decoder_, src_frame));
  CHECK_CUDA_ERROR(cuCtxPopCurrent(NULL));

  return 1;
}

}  // namespace isaac
}  // namespace nvidia
