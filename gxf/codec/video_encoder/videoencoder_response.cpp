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

#include "videoencoder_response.hpp"
#include "videoencoder_utils.hpp"

#include "gxf/multimedia/camera.hpp"

namespace nvidia {
namespace gxf {

struct VideoEncoderResponse::Impl {
  nvmpictx* ctx;
};

VideoEncoderResponse::VideoEncoderResponse() {}

VideoEncoderResponse::~VideoEncoderResponse() {}

gxf_result_t VideoEncoderResponse::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  if (!registrar) {
    return GXF_ARGUMENT_NULL;
  }

  result &= registrar->parameter(output_transmitter_, "output_transmitter",
                        "Transmitter to send the compressed data", "");
  result &= registrar->parameter(pool_, "pool",
                       "Memory pool for allocating output data", "");
  result &= registrar->parameter(videoencoder_context_, "videoencoder_context",
    "videoencoder context", "Encoder context Handle");
  result &= registrar->parameter(outbuf_storage_type_, "outbuf_storage_type",
                       "Output Buffer storage(memory) type",
                       "Output uffer storage type, 0:host mem, 1:device mem",
                       1u);
  return GXF_SUCCESS;
}
gxf_result_t VideoEncoderResponse::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  if (impl_ == nullptr) {
    return GXF_OUT_OF_MEMORY;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderResponse::deinitialize() {
  return GXF_SUCCESS;
}
gxf_result_t VideoEncoderResponse::start() {
  // Get the common encoder context pointer
  impl_->ctx = videoencoder_context_.get()->ctx_;
  if (impl_->ctx == nullptr) {
    GXF_LOG_ERROR("Failed to get encoder ctx");
    return GXF_FAILURE;
  }
  if (outbuf_storage_type_ > 1) {
    GXF_LOG_ERROR("Error in input parameter:outbuf_storage_type");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoEncoderResponse::tick() {
  int ret_val;
  int dq_index = 0;
  int32_t bytes_copied;
  dq_index = impl_->ctx->dqbuf_index;

  // Copy Bit stream from v4l2 buffer to output tensor
  NvBufSurface* surf_ptr = NULL;
  uint8_t* bitstream_buf;
  surf_ptr = reinterpret_cast<NvBufSurface *>(impl_->ctx->capture_buffers[dq_index].buf_surface);

  if (impl_->ctx->is_cuvid) {
    bitstream_buf =
        reinterpret_cast<uint8_t *>(surf_ptr->surfaceList[0].dataPtr);
  } else {
    bitstream_buf =
      reinterpret_cast<uint8_t *>(surf_ptr->surfaceList[0].mappedAddr.addr[0]);
    NvBufSurfaceSyncForCpu(surf_ptr, 0, 0);
  }

  // Bit atream size
  bytes_copied = (int32_t)impl_->ctx->bitstream_size;

  // Cuda memcpy depending on if the gxf output tensor is in host or device
  gxf::MemoryStorageType storage_type = gxf::MemoryStorageType::kHost;
  if (outbuf_storage_type_.get() == 1)
    storage_type = gxf::MemoryStorageType::kDevice;
  // Cuda memcpy depending on if the gxf output tensor is in host or device
  cudaMemcpyKind memcpytype = cudaMemcpyHostToDevice;
  switch (MemoryStorageType(outbuf_storage_type_.get())) {
  case MemoryStorageType::kHost: {
    if (impl_->ctx->is_cuvid)
      memcpytype = cudaMemcpyDeviceToHost;
    else
      memcpytype = cudaMemcpyHostToHost;
  } break;
  case MemoryStorageType::kDevice: {
    if (impl_->ctx->is_cuvid)
      memcpytype = cudaMemcpyDeviceToDevice;
    else
      memcpytype = cudaMemcpyHostToDevice;
  } break;
  default:
    return GXF_PARAMETER_OUT_OF_RANGE;
  }

  // Get output message
  auto output_entity = impl_->ctx->output_entity_queue.front();
  impl_->ctx->output_entity_queue.pop();

  auto output_tensor = output_entity.get<nvidia::gxf::Tensor>();

  if (!output_tensor) {
    GXF_LOG_ERROR("Failed to get output tensor");
    return GXF_FAILURE;
  }
  output_tensor.value()->reshape<uint8_t>(gxf::Shape{bytes_copied}, storage_type, pool_);

  cudaError_t cuda_result;
  cuda_result =
      cudaMemcpy((unsigned char *)output_tensor.value()->data<uint8_t>().value(),
                 bitstream_buf, bytes_copied, memcpytype);

  if (cuda_result != cudaSuccess) {
    GXF_LOG_ERROR("Error in bitstream cudaMemcpy() : %s",
                    cudaGetErrorString(cuda_result));
     return GXF_FAILURE;
  }
  if (!impl_->ctx->is_cuvid) {
    // Add key frame indicator into output message
    v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
    if (getMetadata(impl_->ctx, dq_index, enc_metadata) != 0) {
      GXF_LOG_ERROR("Failed to get v4l2 metadata");
      return GXF_FAILURE;
    }

    auto is_key_frame = output_entity.add<bool>("is_key_frame");
    if (!is_key_frame) {
      GXF_LOG_ERROR("Failed add key frame indicator into output");
      return GXF_FAILURE;
    }
    *is_key_frame.value() = enc_metadata.KeyFrame;
  }
  ret_val = enqueue_capture_plane_buffer(impl_->ctx, dq_index);
  if (ret_val) {
    GXF_LOG_ERROR("Failed to enqueue capture plane");
    return GXF_FAILURE;
  }
  impl_->ctx->response_count++;
  if (impl_->ctx->response_count < impl_->ctx->request_count) {
      impl_->ctx->scheduling_term->setEventState(
                            nvidia::gxf::AsynchronousEventState::EVENT_WAITING);
  } else {
      impl_->ctx->scheduling_term->setEventState(
                            nvidia::gxf::AsynchronousEventState::WAIT);
  }

  impl_->ctx->bistreamBuf_queued = 1;
  return gxf::ToResultCode(output_transmitter_->publish(output_entity));
}

gxf_result_t VideoEncoderResponse::stop() {
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
