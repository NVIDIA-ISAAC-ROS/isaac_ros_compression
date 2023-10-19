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
#include <string>
#include "gxf/multimedia/camera.hpp"
#include "gxf/std/timestamp.hpp"
#include "videodecoder_request.hpp"
#include "videodecoder_utils.hpp"

namespace nvidia {
namespace gxf {

struct VideoDecoderRequest::Impl {
  nvmpictx* ctx;
};

VideoDecoderRequest::VideoDecoderRequest() {}

VideoDecoderRequest::~VideoDecoderRequest() {}

constexpr uint32_t kNumInputBuffer = 5;
constexpr char kDefaultOutputFormat[] = "nv12pl";

gxf_result_t VideoDecoderRequest::registerInterface(gxf::Registrar* registrar) {
  if (!registrar) {
    return GXF_ARGUMENT_NULL;
  }
  gxf::Expected<void> result;

  // I/O related parameters
  result &= registrar->parameter(input_frame_, "input_frame",
                                 "Receiver to get the input image");
  result &=
      registrar->parameter(inbuf_storage_type_, "inbuf_storage_type",
                           "Input Buffer Storage(memory) type",
                           "Input Buffer storage type, 0:kHost, 1:kDevice",
                           1u);
  result &= registrar->parameter(scheduling_term_, "async_scheduling_term",
      "Asynchronous Scheduling Term", "Asynchronous Scheduling Term");
  result &= registrar->parameter(videodecoder_context_, "videodecoder_context",
    "videodecoder context", "Decoder context Handle");

  // Decoder related parameters
  result &=
      registrar->parameter(codec_, "codec", "Video Codec to use",
                           "Video codec,  0:H264, only H264 supported", 0u);
  result &=
      registrar->parameter(disableDPB_, "disableDPB",
                           "Enable low latency decode",
                           "Works only for IPPP case", 0u);
  result &= registrar->parameter(output_format_, "output_format",
                       "Output frame video format",
                       "nv12pl and yuv420planar are supported",
                       std::string(kDefaultOutputFormat));

  return gxf::ToResultCode(result);
}

gxf_result_t VideoDecoderRequest::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  if (impl_ == nullptr) {
    GXF_LOG_ERROR("Failed to get impl handle");
    return GXF_OUT_OF_MEMORY;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderRequest::deinitialize() {
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderRequest::prepareOutputEntity(const gxf::Entity & input_msg) {
  auto output_entity = gxf::Entity::New(impl_->ctx->gxf_context);
  if (!output_entity) {
    GXF_LOG_ERROR("Failed to create output message");
    return gxf::ToResultCode(output_entity);
  }

  auto maybe_input_intrinsics = input_msg.get<gxf::CameraModel>("intrinsics");
  if (maybe_input_intrinsics) {
    auto output_intrinsics = output_entity.value().add<gxf::CameraModel>("intrinsics");
    if (!output_intrinsics) {
      GXF_LOG_ERROR("Failed add intrinsics into output");
      return gxf::ToResultCode(output_intrinsics);
    }
    *output_intrinsics.value() = *maybe_input_intrinsics.value();
  }

  auto maybe_input_extrinsics = input_msg.get<gxf::Pose3D>("extrinsics");
  if (maybe_input_extrinsics) {
    auto output_extrinsics = output_entity.value().add<gxf::Pose3D>("extrinsics");
    if (!output_extrinsics) {
      GXF_LOG_ERROR("Failed add extrinsics into output");
      return gxf::ToResultCode(output_extrinsics);
    }
    *output_extrinsics.value() = *maybe_input_extrinsics.value();
  }

  auto maybe_input_seq_number = input_msg.get<int64_t>("sequence_number");
  if (maybe_input_seq_number) {
    auto output_seq_number = output_entity.value().add<int64_t>("sequence_number");
    if (!output_seq_number) {
      GXF_LOG_ERROR("Failed add sequence number into output");
      return gxf::ToResultCode(output_seq_number);
    }
    *output_seq_number.value() = *maybe_input_seq_number.value();
  }

  // Follow the ISAAC convention to add unamed timestamp first
  auto maybe_sensor_timestamp = input_msg.get<gxf::Timestamp>();
  if (maybe_sensor_timestamp) {
    auto output_sensor_timestamp = output_entity.value().add<gxf::Timestamp>();
    if (!output_sensor_timestamp) {
      GXF_LOG_ERROR("Failed to add sensor timestamp into output");
      return gxf::ToResultCode(output_sensor_timestamp);
    }
    *output_sensor_timestamp.value() = *maybe_sensor_timestamp.value();
  }

  auto maybe_gxf_timestamp = input_msg.get<gxf::Timestamp>(kTimestampName);
  if (maybe_gxf_timestamp) {
    auto output_gxf_timestamp = output_entity.value().add<gxf::Timestamp>(kTimestampName);
    if (!output_gxf_timestamp) {
      GXF_LOG_ERROR("Failed to add gxf timestamp into output");
      return gxf::ToResultCode(output_gxf_timestamp);
    }
    *output_gxf_timestamp.value() = *maybe_gxf_timestamp.value();

    impl_->ctx->input_timestamp.tv_usec = static_cast<uint64_t>(
      maybe_gxf_timestamp.value()->acqtime % static_cast<uint64_t>(1e6));
    impl_->ctx->input_timestamp.tv_sec = static_cast<uint64_t>(
      maybe_gxf_timestamp.value()->acqtime / static_cast<uint64_t>(1e6));
  }

  auto maybe_input_image = input_msg.get<gxf::Tensor>();
  if (!maybe_input_image) {
    GXF_LOG_ERROR("Failed to get input tensor");
    return gxf::ToResultCode(maybe_input_image);
  }
  auto output_video_buffer = output_entity->add<gxf::VideoBuffer>(maybe_input_image->name());
  if (!output_video_buffer) {
    GXF_LOG_ERROR("Failed to add output videobuffer");
    return gxf::ToResultCode(output_video_buffer);
  }
  impl_->ctx->output_entity_queue.push(output_entity.value());
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderRequest::start() {
  GXF_LOG_DEBUG("Enter decoder request start function");

  impl_->ctx = videodecoder_context_.get()->ctx_;
  if (impl_->ctx == nullptr) {
    GXF_LOG_ERROR("Failed to get encoder ctx");
    return GXF_FAILURE;
  }
  impl_->ctx->gxf_context = context();

  if (codec_ != 0) {
    GXF_LOG_ERROR("Unsupported codec");
    return GXF_FAILURE;
  }

  int32_t retval = 0;
  impl_->ctx->output_buffer_count = kNumInputBuffer;
  impl_->ctx->output_buffer_idx = 0;

  // Default output format is NV12
  impl_->ctx->planar = 0;
  if (output_format_.get() == "yuv420planar") {
    impl_->ctx->planar = 1;
  }

  /* Set appropriate controls.
  ** V4L2_CID_MPEG_VIDEO_DISABLE_COMPLETE_FRAME_INPUT control is
  ** set to false so that application can send chunks of encoded
  ** data instead of forming complete frames.
  */
  retval = set_disable_complete_frame_input(impl_->ctx);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in setting control for disable frame input \n");
    return GXF_FAILURE;
  }

  /*
   Video sequences without B-frames i.e., All-Intra frames and IPPP... frames
   should not have any decoding/display latency.
   nvcuvid decoder has inherent display latency for some video contents
   which do not have num_reorder_frames=0 in the VUI.
   Strictly adhering to the standard, this display latency is expected.
   In case, the user wants to force zero display latency for such contents, we set
   V4L2_CID_MPEG_VIDEO_CUDA_LOW_LATENCY control.
  */
  if (disableDPB_ != 0) {
    retval = enable_low_latency_deocde(impl_->ctx);
    if (retval < 0) {
      GXF_LOG_ERROR("Error in enabling low latency decode \n");
      return GXF_FAILURE;
    }
  }

  retval = reqbufs_output_plane(impl_->ctx);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in reqbuf on OUTPUT_MPLANE \n");
    return GXF_FAILURE;
  }

  retval = streamon_plane(impl_->ctx, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
  if (retval < 0) {
    GXF_LOG_ERROR("Error in Stream on for OUTPUT_MPLANE \n");
    return GXF_FAILURE;
  }
  GXF_LOG_DEBUG("Stream on for OUTPUT_MPLANE successful \n");

  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderRequest::tick() {
  GXF_LOG_DEBUG("Enter tick function");
  // Get input image
  auto maybe_input_message = input_frame_->receive();
  if (!maybe_input_message) {
    GXF_LOG_ERROR("Failed to receive input message");
    return maybe_input_message.error();
  }
  auto input_image = maybe_input_message.value().get<gxf::Tensor>();
  if (!input_image) {
    GXF_LOG_ERROR("Failed to get image from message");
    return input_image.error();
  }

  auto gxf_ret_code = prepareOutputEntity(maybe_input_message.value());
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed create output entity");
    return gxf_ret_code;
  }

  if (impl_->ctx->error_in_decode_thread) {
    GXF_LOG_ERROR("Decode Failed");
    return GXF_FAILURE;
  }
  uint32_t n_video_bytes = input_image.value()->size();
  uint8_t* p_video = input_image.value()->data<uint8_t>().value();
  int32_t dqbuf_index = 0;

  GXF_LOG_DEBUG("Input bitstream address: %p and size: %d \n", p_video,
                n_video_bytes);
  if (n_video_bytes) {
    /* Cuda memcpy depending on if the gxf input tensor is in host or device.
    ** For Tegra, destination type is always Host since mapped addr is used.
    */
    cudaMemcpyKind memcpytype = cudaMemcpyHostToDevice;
    switch (MemoryStorageType(inbuf_storage_type_.get())) {
    case MemoryStorageType::kHost: {
      if (impl_->ctx->is_cuvid)
        memcpytype = cudaMemcpyHostToDevice;
      else
        memcpytype = cudaMemcpyHostToHost;
    } break;
    case MemoryStorageType::kDevice: {
      if (impl_->ctx->is_cuvid)
        memcpytype = cudaMemcpyDeviceToDevice;
      else
        memcpytype = cudaMemcpyDeviceToHost;
    } break;
    default:
      return GXF_PARAMETER_OUT_OF_RANGE;
    }

    if (impl_->ctx->output_buffer_idx < impl_->ctx->output_buffer_count) {
      dqbuf_index = impl_->ctx->output_buffer_idx;
      impl_->ctx->output_buffer_idx++;
    } else {
      while (dqbuf_plane(impl_->ctx, &dqbuf_index,
                         V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) != 0) {
        if (errno == EAGAIN) {
          GXF_LOG_DEBUG("Got EAGAIN while DQBUF on Output , trying again \n");
          continue;
        } else {
          GXF_LOG_ERROR("Error in dqbuf on output plane \n");
          return GXF_FAILURE;
        }
      }
    }

    if (n_video_bytes > max_bitstream_size) {
      GXF_LOG_ERROR("Input bitstream size is large. Size: %d max_buffer_size: %d \n",
                    n_video_bytes, max_bitstream_size);
      return GXF_FAILURE;
    }

    cudaError_t cuda_result;
    uint8_t* bitstream_buf;
    NvBufSurface* surf_ptr = reinterpret_cast<NvBufSurface *>
        (impl_->ctx->output_buffers[dqbuf_index].buf_surface);
    if (impl_->ctx->is_cuvid) {
      bitstream_buf =
          reinterpret_cast<uint8_t *>(surf_ptr->surfaceList[0].dataPtr);
    } else {
      bitstream_buf =
          reinterpret_cast<uint8_t *>(surf_ptr->surfaceList[0].mappedAddr.addr[0]);
    }

    cuda_result = cudaMemcpy(bitstream_buf, p_video, n_video_bytes, memcpytype);
    if (cuda_result != cudaSuccess) {
      GXF_LOG_ERROR("Error in bitstream cudaMemcpy() : %s",
                    cudaGetErrorString(cuda_result));
      return GXF_FAILURE;
    }

    if (enqueue_plane_buffer(impl_->ctx, dqbuf_index, n_video_bytes,
                             V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) != 0) {
      GXF_LOG_ERROR("Error in buffer enqueue for OUTPUT_MPLANE \n");
      return GXF_FAILURE;
    }
    GXF_LOG_DEBUG("Successfully enqbuf'ed %d OUTPUT plane \n", dqbuf_index);

  } else if (!impl_->ctx->got_eos) {
    GXF_LOG_INFO("Received EOS signal. Stopping the decoder");
    struct v4l2_decoder_cmd dcmd = {
        0,
    };
    dcmd.cmd = V4L2_DEC_CMD_STOP;
    dcmd.flags = V4L2_DEC_CMD_STOP_TO_BLACK;
    v4l2_ioctl(impl_->ctx->dev_fd, VIDIOC_DECODER_CMD, &dcmd);

    // Context is inactive, set event to EVENT_NEVER
    scheduling_term_.get()->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_NEVER);
  }

  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderRequest::stop() {
  GXF_LOG_DEBUG("Enter stop function");
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
