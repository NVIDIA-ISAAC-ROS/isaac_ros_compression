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

#include <sys/poll.h>
#include <string>

#include "gxf/multimedia/camera.hpp"
#include "gxf/std/timestamp.hpp"
#include "videoencoder_request.hpp"
#include "videoencoder_utils.hpp"

namespace nvidia {
namespace gxf {

#define CHECK_ENCODER_ERROR(ret, error_msg)  \
  if (ret < 0) {                             \
    GXF_LOG_ERROR(error_msg);                \
    return GXF_FAILURE;                      \
  }
struct VideoEncoderRequest::Impl {
  nvmpictx* ctx;
};

VideoEncoderRequest::VideoEncoderRequest() {}

VideoEncoderRequest::~VideoEncoderRequest() {}

gxf_result_t VideoEncoderRequest::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

if (!registrar) {
    return GXF_ARGUMENT_NULL;
  }

  // I/O related parameters
  result &= registrar->parameter(input_frame_, "input_frame",
                        "Receiver to get the input frame", "");
  result &= registrar->parameter(inbuf_storage_type_, "inbuf_storage_type",
                       "Input Buffer storage(memory) type",
                       "Input Buffer storage type, 0:host mem, 1:device mem",
                       1u);
  result &= registrar->parameter(videoencoder_context_, "videoencoder_context",
    "videoencoder context", "Encoder context Handle");
  // Encoder related parameters
  result &= registrar->parameter(codec_, "codec", "Video Codec to use",
                       "Video codec,  0:H264, only H264 supported",
                       0);
  result &= registrar->parameter(input_height_, "input_height",
                       "Input frame height", "");
  result &= registrar->parameter(input_width_, "input_width",
                       "Input image width", "");
  result &= registrar->parameter(input_format_, "input_format",
                       "Input color format, nv12,nv24,yuv420planar", "nv12",
                       gxf::EncoderInputFormat::kNV12);
  result &= registrar->parameter(profile_, "profile", "Encode profile",
                       "0:Baseline Profile, 1: Main , 2: High",
                       2);
  result &= registrar->parameter(hw_preset_type_, "hw_preset_type",
                       "Encode hw preset type, select from 0 to 3", "hw preset",
                       0);
  result &= registrar->parameter(level_, "level", "Video H264 level",
                       "Maximum data rate and resolution, select from 0 to 14",
                       14);
  result &= registrar->parameter(iframe_interval_, "iframe_interval",
                       "I Frame Interval", "Interval between two I frames",
                       30);
  result &= registrar->parameter(rate_control_mode_, "rate_control_mode",
                       "Rate control mode, 0:CQP[RC off], 1:CBR, 2:VBR ",
                       "Rate control mode",
                       1);
  result &= registrar->parameter(qp_, "qp", "Encoder constant QP value",
                       "cont qp",
                       20u);
  result &= registrar->parameter(bitrate_, "bitrate", "Encoder bitrate",
                       "Bitrate of the encoded stream, in bits per second",
                       20000000);
  result &= registrar->parameter(framerate_, "framerate", "Frame Rate, FPS",
                       "Frames per second",
                       30);
  result &= registrar->parameter(config_, "config",
                       "Preset of parameters, select from pframe_cqp, iframe_cqp, custom",
                       "Preset of config",
                       gxf::EncoderConfig::kCustom);

  return gxf::ToResultCode(result);
}
gxf_result_t VideoEncoderRequest::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  if (impl_ == nullptr) {
    GXF_LOG_ERROR("Failed to get impl handle");
    return GXF_OUT_OF_MEMORY;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderRequest::deinitialize() {
  return GXF_SUCCESS;
}


gxf_result_t VideoEncoderRequest::start() {
  int32_t retval = 0;
  // Get the context pointer from encoder context component
  impl_->ctx = videoencoder_context_.get()->ctx_;
  if (impl_->ctx == nullptr) {
    GXF_LOG_ERROR("Failed to get encoder context");
    return GXF_FAILURE;
  }

  // Check input parmaters and flag error for unsupported param values
  retval = checkInputParams();
  if (retval != GXF_SUCCESS) {
    GXF_LOG_ERROR("Error in Input Parameters");
    return GXF_FAILURE;
  }
  auto gxf_ret_code = getEncoderSettingsFromParameters();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get compression settings from parameter");
    return gxf_ret_code;
  }
  // Setting the RC mode and IDR interval parmeters for cuvid case.
  if (impl_->ctx->is_cuvid) {
    if (impl_->ctx->rate_control_mode == 0) {
      CHECK_ENCODER_ERROR(setRateControlMode(impl_->ctx, V4L2_MPEG_VIDEO_BITRATE_MODE_CONSTQP),
                        "Failed to set Rate control mode")
    } else {
      CHECK_ENCODER_ERROR(setRateControlMode(impl_->ctx, V4L2_MPEG_VIDEO_BITRATE_MODE_CBR),
                        "Failed to set Rate control mode")
    }
    CHECK_ENCODER_ERROR(setIDRInterval(impl_->ctx, impl_->ctx->iframe_interval),
                        "Failed to set IFrame Interval")
  }
  // Set v4l2 parameters for capture plane(bit stream buffer for encoder)
  // only h.264 is supported.
  retval = set_capture_plane_format(impl_->ctx);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in set_capture_plane_format Error:%d", retval);
    return GXF_FAILURE;
  }

  // Set cuda device id
  retval = set_cuda_gpu_id(impl_->ctx);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in set_cuda_gpu_id:%d", retval);
    return GXF_FAILURE;
  }

  // Set v4l2 parameters for output plane(input yuv buffer for encoder)
  // YUV 420 nv12 and planar formats  pitch linear supported now
  retval = set_output_plane_format(impl_->ctx);
  if (retval != 0) {
    GXF_LOG_ERROR("Error in set_output_plane_format, Error:%d", retval);
    return GXF_FAILURE;
  }
  // Set encode paremeters
  gxf_ret_code = setEncoderParameters();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to initialize encoder");
    return gxf_ret_code;
  }
  retval = reqbufs_output_plane(impl_->ctx, framerate_);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in reqbufs_output_plane");
    return GXF_FAILURE;
  }

  // Request buffers on capture plane for bit stream buffers
  retval = reqbufs_capture_plane(impl_->ctx);
  if (retval != 0) {
    GXF_LOG_ERROR("Error in reqbufs_capture_plane");
    return GXF_FAILURE;
  }

  // Issue stream On
  retval = streamon_plane(impl_->ctx, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in Stream on for CAPTURE_MPLANE");
    return GXF_FAILURE;
  }
  retval = streamon_plane(impl_->ctx, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in Stream on for CAPTURE_MPLANE");
    return GXF_FAILURE;
  }

  retval = enqueue_all_capture_plane_buffers(impl_->ctx);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in enqueue_all_capture_plane_buffers");
    return GXF_FAILURE;
  }
  impl_->ctx->bistreamBuf_queued = 1;
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderRequest::tick() {
  // Get input frame from message
  auto maybe_input_message = input_frame_->receive();
  if (!maybe_input_message) {
    GXF_LOG_ERROR("Failed to receive input message");
    return maybe_input_message.error();
  }
  auto input_image = maybe_input_message.value().get<gxf::VideoBuffer>();
  if (!input_image) {
    GXF_LOG_ERROR("Failed to get image from message");
    return input_image.error();
  }

  auto gxf_ret_code = prepareOutputEntity(maybe_input_message.value());
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed create output entity");
    return gxf_ret_code;
  }

  // Dequeue and Enqueue v4l2 output_plane (input YUV buffer)
  int ret_val;
  ret_val = queueInputYUVBuf(input_image.value());
  if (ret_val != GXF_SUCCESS) {
    GXF_LOG_ERROR("Error in queueFrame");
  }
  impl_->ctx->request_count++;
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderRequest::stop() {
  return GXF_SUCCESS;
}
gxf_result_t VideoEncoderRequest::prepareOutputEntity(const gxf::Entity & input_msg) {
  auto output_entity = gxf::Entity::New(impl_->ctx->gxf_context);
  if (!output_entity) {
    GXF_LOG_ERROR("Failed to create output message");
    return gxf::ToResultCode(output_entity);
  }

  auto maybe_input_intrinsics = input_msg.get<gxf::CameraModel>("intrinsics");
  if (maybe_input_intrinsics) {
    auto output_intrinsics = output_entity->add<gxf::CameraModel>(maybe_input_intrinsics->name());
    if (!output_intrinsics) {
      GXF_LOG_ERROR("Failed to add intrinsics into output");
      return gxf::ToResultCode(output_intrinsics);
    }
    *output_intrinsics.value() = *maybe_input_intrinsics.value();
  }

  auto maybe_input_extrinsics = input_msg.get<gxf::Pose3D>("extrinsics");
  if (maybe_input_extrinsics) {
    auto output_extrinsics = output_entity->add<gxf::Pose3D>(maybe_input_extrinsics->name());
    if (!output_extrinsics) {
      GXF_LOG_ERROR("Failed to add extrinsics into output");
      return gxf::ToResultCode(output_extrinsics);
    }
    *output_extrinsics.value() = *maybe_input_extrinsics.value();
  }

  auto maybe_input_seq_number = input_msg.get<int64_t>("sequence_number");
  if (maybe_input_seq_number) {
    auto output_seq_number = output_entity->add<int64_t>(maybe_input_seq_number->name());
    if (!output_seq_number) {
      GXF_LOG_ERROR("Failed to add sequence number into output");
      return gxf::ToResultCode(output_seq_number);
    }
    *output_seq_number.value() = *maybe_input_seq_number.value();
  }

  // Follow the ISAAC convention to add unamed timestamp first
  auto maybe_sensor_timestamp = input_msg.get<gxf::Timestamp>();
  if (maybe_sensor_timestamp) {
    auto output_sensor_timestamp = output_entity->add<gxf::Timestamp>();
    if (!output_sensor_timestamp) {
      GXF_LOG_ERROR("Failed to add sensor timestamp into output");
      return gxf::ToResultCode(output_sensor_timestamp);
    }
    *output_sensor_timestamp.value() = *maybe_sensor_timestamp.value();
  }

  auto maybe_gxf_timestamp = input_msg.get<gxf::Timestamp>(kTimestampName);
  if (maybe_gxf_timestamp) {
    auto output_gxf_timestamp = output_entity->add<gxf::Timestamp>(kTimestampName);
    if (!output_gxf_timestamp) {
      GXF_LOG_ERROR("Failed to add gxf timestamp into output");
      return gxf::ToResultCode(output_gxf_timestamp);
    }
    *output_gxf_timestamp.value() = *maybe_gxf_timestamp.value();
  }

  auto maybe_input_image = input_msg.get<gxf::VideoBuffer>();
  if (!maybe_input_image) {
    GXF_LOG_ERROR("Failed to get input videobuffer");
    return gxf::ToResultCode(maybe_input_image);
  }
  auto output_tensor = output_entity->add<nvidia::gxf::Tensor>(maybe_input_image->name());
  if (!output_tensor) {
    GXF_LOG_ERROR("Failed to add output tensor");
    return gxf::ToResultCode(output_tensor);
  }

  impl_->ctx->output_entity_queue.push(output_entity.value());
  return GXF_SUCCESS;
}

// Function to Queue the input YUV buffer
gxf_result_t VideoEncoderRequest::queueInputYUVBuf(
                                const gxf::Handle<gxf::VideoBuffer> input_img) {
  auto input_img_info = input_img->video_frame_info();
  int retval = 0;
  int q_index = 0;

  // Get a free buffer(queue index) for yuv buffer(output plane) enqueue
  if (impl_->ctx->output_buffer_idx < impl_->ctx->output_buffer_count) {
    q_index = impl_->ctx->output_buffer_idx;
    impl_->ctx->output_buffer_idx++;
  } else {
    // Output plane dqBuffer
    while (dqbuf_on_output_plane(impl_->ctx, &q_index) != 0) {
      if (errno == EAGAIN) {
        continue;
      } else {
        GXF_LOG_ERROR("Error in dqbuf on output plane \n");
        return GXF_FAILURE;
      }
    }
  }

 /* Cuda memcpy depending on if the gxf input tensor is in host or device.
  ** For Tegra, destination type is always Host since mapped addr is used.
  */
  cudaMemcpyKind memcpytype = cudaMemcpyHostToDevice;
  switch (MemoryStorageType(inbuf_storage_type_.get())) {
  case MemoryStorageType::kHost: {
      memcpytype = cudaMemcpyHostToDevice;
  } break;
  case MemoryStorageType::kDevice: {
      memcpytype = cudaMemcpyDeviceToDevice;
  } break;
  default:
    return GXF_PARAMETER_OUT_OF_RANGE;
  }

  // Copy the input YUV buffer into v4l2 buffer.
  uint8_t* pdst_data;
  cudaError_t result;

  NvBufSurface* nvbuffer0 =
      reinterpret_cast<NvBufSurface *>(impl_->ctx->output_buffers[q_index].buf_surface);
  if (impl_->ctx->is_cuvid) {
    pdst_data = reinterpret_cast<uint8_t *>(nvbuffer0->surfaceList[0].dataPtr);
  } else {
    result = cudaHostGetDevicePointer(&pdst_data,
                             nvbuffer0->surfaceList[0].mappedAddr.addr[0], 0);
    if (result != cudaSuccess) {
      GXF_LOG_ERROR("Failed in cudaHostGetDevicePointer: %s",
                    cudaGetErrorString(result));
      return GXF_FAILURE;
    }
  }
  // Y plane copy
  uint32_t plane = 0;
  result = cudaMemcpy2D(pdst_data,
                nvbuffer0->surfaceList[plane].pitch,
                input_img->pointer(),
                input_img_info.color_planes[plane].stride,
                (input_img_info.color_planes[plane].width *
                    input_img_info.color_planes[plane].bytes_per_pixel),
                (input_img_info.color_planes[plane].height),
                memcpytype);
  if (result != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy the input buffer: %s",
                  cudaGetErrorString(result));
    return GXF_FAILURE;
  }
  // UV plane copy for nv12, U plane copy for planar
  plane = 1;
  if (impl_->ctx->is_cuvid) {
    pdst_data =
      reinterpret_cast<uint8_t *>(nvbuffer0->surfaceList[0].dataPtr) +
          (nvbuffer0->surfaceList->planeParams.pitch[0] *
           nvbuffer0->surfaceList->planeParams.height[0]);
  } else {
    result = cudaHostGetDevicePointer(&pdst_data,
                           nvbuffer0->surfaceList[0].mappedAddr.addr[plane], 0);
    if (result != cudaSuccess) {
      GXF_LOG_ERROR("Failed in cudaHostGetDevicePointer: %s",
                    cudaGetErrorString(result));
      return GXF_FAILURE;
    }
  }

  auto src_offset = input_img_info.color_planes[0].size;
  result = cudaMemcpy2D((unsigned char *)pdst_data,
                         nvbuffer0->surfaceList[0].planeParams.pitch[plane],
                         input_img->pointer()+ src_offset,
                         input_img_info.color_planes[plane].stride,
                         (input_img_info.color_planes[plane].width *
                             input_img_info.color_planes[plane].bytes_per_pixel),
                         input_img_info.color_planes[plane].height, memcpytype);

  if (result != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy the input buffer: %s",
                  cudaGetErrorString(result));
    return GXF_FAILURE;
  }

  // V plane copy for planar
  if (input_format_.get() == gxf::EncoderInputFormat::kYUV420PLANAR) {
    plane = 2;
    if (impl_->ctx->is_cuvid) {
      pdst_data =
        reinterpret_cast<uint8_t *>(nvbuffer0->surfaceList[0].dataPtr) +
            (nvbuffer0->surfaceList->planeParams.pitch[0] *
             nvbuffer0->surfaceList->planeParams.height[0]) +
            (nvbuffer0->surfaceList->planeParams.pitch[1] *
             nvbuffer0->surfaceList->planeParams.height[1]);
    } else {
      result = cudaHostGetDevicePointer(&pdst_data,
                           nvbuffer0->surfaceList[0].mappedAddr.addr[plane], 0);
      if (result != cudaSuccess) {
        GXF_LOG_ERROR("Failed in cudaHostGetDevicePointer: %s",
                      cudaGetErrorString(result));
        return GXF_FAILURE;
      }
    }

    src_offset = input_img_info.color_planes[0].size +
                  input_img_info.color_planes[1].size;
    result = cudaMemcpy2D((unsigned char *)pdst_data,
                         nvbuffer0->surfaceList[0].planeParams.pitch[plane],
                         input_img->pointer()+ src_offset,
                         input_img_info.color_planes[plane].stride,
                         (input_img_info.color_planes[plane].width *
                             input_img_info.color_planes[plane].bytes_per_pixel),
                         input_img_info.color_planes[plane].height, memcpytype);

    if (result != cudaSuccess) {
      GXF_LOG_ERROR("Failed to copy the input buffer: %s",
                    cudaGetErrorString(result));
      return GXF_FAILURE;
    }
  }

  int bytes_used = (3*input_img_info.color_planes[0].width *
                      input_img_info.color_planes[0].height)>>1;
  // Enqueue the yuv buffer(v4l2 output_plane)
  retval = enqueue_output_plane_buffer(impl_->ctx, q_index, bytes_used);
  GXF_LOG_DEBUG("videoencoder_request: enqueue of YUV done");
  if (retval < 0) {
    GXF_LOG_ERROR("Error in enqueue_output_plane_buffer");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

// Function to check validity of the input params
gxf_result_t VideoEncoderRequest::checkInputParams() {
  if (codec_ != 0) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported codec");
    return GXF_FAILURE;
  }

  if ((input_format_.get() != gxf::EncoderInputFormat::kNV12) &&
      (input_format_.get() != gxf::EncoderInputFormat::kNV24) &&
      (input_format_.get() != gxf::EncoderInputFormat::kYUV420PLANAR)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported input format");
    return GXF_FAILURE;
  }

  if ((profile_ < 0)|| (profile_ > 2)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported Profile");
    return GXF_FAILURE;
  }
  if ((level_ < 0)|| (level_ > 14)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported Level");
    return GXF_FAILURE;
  }

  int max_preset = 3;
  if (impl_->ctx->is_cuvid)
    max_preset = 7;

  if ((hw_preset_type_ < 0)|| (hw_preset_type_ > max_preset)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported HW Preset");
    return GXF_FAILURE;
  }

  if (bitrate_ <= 0) {
    GXF_LOG_ERROR("Error in input parameter: Bit rate is <=0 ");
    return GXF_FAILURE;
  }
  if (framerate_ <= 0) {
    GXF_LOG_ERROR("Error in input parameter: Frame rate is <= 0");
    return GXF_FAILURE;
  }
  if (inbuf_storage_type_ > 1) {
    GXF_LOG_ERROR("Error in input parameter: in inbuf_storage_type");
    return GXF_FAILURE;
  }
  if ((input_width_ < kVideoEncoderMinWidth) ||
      (input_width_ > kVideoEncoderMaxWidth)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported input_width");
    return GXF_FAILURE;
  }
  if (input_width_ % 2 != 0) {
    GXF_LOG_ERROR("Error in input parameter: input_width must be an even number");
    return GXF_FAILURE;
  }
  if ((input_height_ < kVideoEncoderMinHeight) ||
      (input_height_ > kVideoEncoderMaxHeight)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported input_height");
    return GXF_FAILURE;
  }
  if (input_height_ % 2 != 0) {
    GXF_LOG_ERROR("Error in input parameter: input_height must be an even number");
    return GXF_FAILURE;
  }
  if (iframe_interval_ < 0) {
    GXF_LOG_ERROR("Error in input parameter: iframe_interval_ < 0");
    return GXF_FAILURE;
  }
  if ((qp_ <= 0) || (qp_ > 51)) {
    GXF_LOG_ERROR("Error in input parameter: qp");
    return GXF_FAILURE;
  }
  if ((config_.get() != gxf::EncoderConfig::kIFrameCQP) &&
      (config_.get() != gxf::EncoderConfig::kPFrameCQP) &&
      (config_.get() != gxf::EncoderConfig::kCustom)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported config");
    return GXF_FAILURE;
  }
  if ((rate_control_mode_< 0) || (rate_control_mode_ > 3)) {
    GXF_LOG_ERROR("Error in input parameter: Unsupported rate_control_mode");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

// Map input pixel format to V4L2 pixel format
gxf_result_t
VideoEncoderRequest::getInputFormatFromParameter() {
  if (input_format_.get() == gxf::EncoderInputFormat::kNV24) {
    impl_->ctx->raw_pixfmt = V4L2_PIX_FMT_NV24M;
  } else if (input_format_.get() == gxf::EncoderInputFormat::kNV12) {
    impl_->ctx->raw_pixfmt = V4L2_PIX_FMT_NV12M;
  } else if (input_format_.get() == gxf::EncoderInputFormat::kYUV420PLANAR) {
      impl_->ctx->raw_pixfmt = V4L2_PIX_FMT_YUV420M;
  } else {
    GXF_LOG_ERROR("The input format is not supported");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

// Set encode Profile and Level parameters
gxf_result_t
VideoEncoderRequest::getProfileLevelSettingsFromParameter() {
  switch (profile_) {
  case 0:
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
    break;
  case 1:
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
    break;
  case 2:
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
    break;
  default:
    GXF_LOG_ERROR("The profile is not supported");
    return GXF_FAILURE;
  }
  switch (level_) {
  case 0:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_1_0;
    break;
  case 1:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_1_1;
    break;
  case 2:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_1_2;
    break;
  case 3:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_1_3;
    break;
  case 4:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_2_0;
    break;
  case 5:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_2_1;
    break;
  case 6:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_2_2;
    break;
  case 7:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_3_0;
    break;
  case 8:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_3_1;
    break;
  case 9:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_3_2;
    break;
  case 10:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_4_0;
    break;
  case 11:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_4_1;
    break;
  case 12:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_4_2;
    break;
  case 13:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_5_0;
    break;
  case 14:
    impl_->ctx->level = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
    break;
  default:
    GXF_LOG_ERROR("The Video H264 level is not supported");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

// Map the input parmaters to appropriate encoder settings
gxf_result_t
VideoEncoderRequest::getEncoderSettingsFromParameters() {
  auto gxf_ret_code = getInputFormatFromParameter();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get input format from parameter");
    return gxf_ret_code;
  }

  gxf_ret_code = getProfileLevelSettingsFromParameter();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get compression level from parameter");
    return gxf_ret_code;
  }

  impl_->ctx->width = input_width_;
  impl_->ctx->height = input_height_;
  impl_->ctx->outbuf_bytesused[0] =
                 (3 * impl_->ctx->width * impl_->ctx->height) >> 1;  // YUV 420

  impl_->ctx->qp = qp_;
  impl_->ctx->entropy = 1;  // CABAC
  impl_->ctx->iframe_interval = iframe_interval_;
  impl_->ctx->idr_interval = iframe_interval_;
  impl_->ctx->bitrate = bitrate_;
  impl_->ctx->rate_control_mode = rate_control_mode_;

  // Preset configs kPFrameCQP, kIFrameCQP
  if (config_.get() == gxf::EncoderConfig::kPFrameCQP) {
    impl_->ctx->qp = 20;
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
    impl_->ctx->iframe_interval = 5;
    impl_->ctx->idr_interval = 5;
    impl_->ctx->rate_control_mode = 0;
  } else if (config_.get() == gxf::EncoderConfig::kIFrameCQP) {
    impl_->ctx->qp = 20;
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
    impl_->ctx->iframe_interval = 1;
    impl_->ctx->idr_interval = 1;
    impl_->ctx->rate_control_mode = 0;
  }

  // For Tegra only config, set preset and entropy mode
  if (!impl_->ctx->is_cuvid) {
    if (config_.get() == gxf::EncoderConfig::kPFrameCQP) {
      impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
      impl_->ctx->entropy = 0;
    } else if (config_.get() == gxf::EncoderConfig::kIFrameCQP) {
      impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
      impl_->ctx->entropy = 0;
    } else {
      switch (hw_preset_type_) {
        case 0:
          impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
          break;
        case 1:
          impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_FAST;
          break;
        case 2:
          impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_MEDIUM;
          break;
        case 3:
          impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_SLOW;
          break;
        default:
          GXF_LOG_ERROR("The hw preset type is not supported");
          return GXF_FAILURE;
      }
    }
  }
  impl_->ctx->gxf_context = context();
  return GXF_SUCCESS;
}

// Function to initialize and set encoder parmaters
// The encoder settings must be called after setFormat on both planes
// and before requestBuffers on any of the planes.
gxf_result_t VideoEncoderRequest::setEncoderParameters() {
  CHECK_ENCODER_ERROR(setProfile(impl_->ctx, impl_->ctx->profile),
                      "Failed to set encoder profile")

  if (impl_->ctx->is_cuvid) {
    CHECK_ENCODER_ERROR(setHWPreset(impl_->ctx, hw_preset_type_),
                        "Failed to set hw preset type")
    if (impl_->ctx->rate_control_mode == 0) {
      CHECK_ENCODER_ERROR(setConstantQP(impl_->ctx, impl_->ctx->qp, impl_->ctx->qp,
                                    impl_->ctx->qp), "Failed to set Const QP value")
    } else {
      CHECK_ENCODER_ERROR(setBitrate(impl_->ctx, impl_->ctx->bitrate),
                          "Failed to set encoder bitrate")
    }
  } else {
    CHECK_ENCODER_ERROR(setLevel(impl_->ctx, impl_->ctx->level),
                        "Failed to set h264 video level")
    CHECK_ENCODER_ERROR(setCABAC(impl_->ctx, impl_->ctx->entropy),
                        "Failed to set entropy encoding")
    CHECK_ENCODER_ERROR(setHWPresetType(impl_->ctx, impl_->ctx->hw_preset_type),
                        "Failed to set hw preset type")
    // Set rate control mode
    if (impl_->ctx->rate_control_mode == 0) {
      CHECK_ENCODER_ERROR(enableRateControl(impl_->ctx, false),
                          "Failed to enable/Disbale rate control mode")
      CHECK_ENCODER_ERROR(setInitQP(impl_->ctx, impl_->ctx->qp, impl_->ctx->qp,
                                    impl_->ctx->qp), "Failed to set QP value")
    } else {
      CHECK_ENCODER_ERROR(enableRateControl(impl_->ctx, true),
                          "Failed to enable/Disbale rate control mode")
      if (impl_->ctx->rate_control_mode == 2) {
        CHECK_ENCODER_ERROR(setRateControlMode(impl_->ctx, V4L2_MPEG_VIDEO_BITRATE_MODE_VBR),
                            "Failed Set rate control mode")
      } else {
        CHECK_ENCODER_ERROR(setRateControlMode(impl_->ctx, V4L2_MPEG_VIDEO_BITRATE_MODE_CBR),
                            "Failed Set rate control mode")
      }
      CHECK_ENCODER_ERROR(setBitrate(impl_->ctx, impl_->ctx->bitrate),
                          "Failed to set encoder bitrate")
    }

    CHECK_ENCODER_ERROR(setIDRInterval(impl_->ctx, impl_->ctx->idr_interval),
                        "Failed to set IDR interval")

    CHECK_ENCODER_ERROR(setIFrameInterval(impl_->ctx, impl_->ctx->iframe_interval),
                        "Failed to set IFrame Interval")

    CHECK_ENCODER_ERROR(setNumBFrames(impl_->ctx, impl_->ctx->num_of_bframes),
                        "Failed to set number of B Frames")

    CHECK_ENCODER_ERROR(setInsertSpsPpsAtIdrEnabled(impl_->ctx, true),
                        "Failed to set encoder SPSPPS at IDR")

    CHECK_ENCODER_ERROR(setMaxPerfMode(impl_->ctx, true),
                        "Failed to set Max performance mode")

    CHECK_ENCODER_ERROR(insertVUI(impl_->ctx, true), "Failed to Insert VUI")
  }
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
