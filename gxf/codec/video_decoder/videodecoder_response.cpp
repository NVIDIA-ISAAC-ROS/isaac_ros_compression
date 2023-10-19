// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nvbufsurftransform.h"

#include "gxf/std/timestamp.hpp"
#include "videodecoder_response.hpp"
#include "videodecoder_utils.hpp"

namespace nvidia {
namespace gxf {

struct VideoDecoderResponse::Impl {
  nvmpictx* ctx;
};

VideoDecoderResponse::VideoDecoderResponse() {}

VideoDecoderResponse::~VideoDecoderResponse() {}

gxf_result_t VideoDecoderResponse::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;

  if (!registrar) {
    return GXF_ARGUMENT_NULL;
  }

  result &= registrar->parameter(output_transmitter_, "output_transmitter",
                        "Transmitter to send the yuv data", "");
  result &= registrar->parameter(pool_, "pool",
                       "Memory pool for allocating output data", "");
  result &= registrar->parameter(outbuf_storage_type_, "outbuf_storage_type",
                       "Output Buffer storage(memory) type",
                       "Output buffer storage type, 0:host mem, 1:device mem",
                       1u);
  result &= registrar->parameter(videodecoder_context_, "videodecoder_context",
    "videodecoder context", "Decoder context Handle");
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderResponse::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  if (impl_ == nullptr) {
    return GXF_OUT_OF_MEMORY;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderResponse::deinitialize() {
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderResponse::start() {
  GXF_LOG_DEBUG("Enter decoder response start function");

  impl_->ctx = videodecoder_context_.get()->ctx_;
  if (impl_->ctx == nullptr) {
    GXF_LOG_ERROR("Failed to get decoder ctx");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderResponse::tick() {
  GXF_LOG_DEBUG("VideoDecoderResponse:Tick started");

  // Get output message
  auto output_entity = impl_->ctx->output_entity_queue.front();
  impl_->ctx->output_entity_queue.pop();

  auto output_video_buffer = output_entity.get<nvidia::gxf::VideoBuffer>();
  if (!output_video_buffer) {
    GXF_LOG_ERROR("Failed to get output VideoBuffer");
    return GXF_FAILURE;
  }
  decoded_frame_ = output_video_buffer.value();
  memory_pool_ = pool_.get();

  // Copy output YUV buffer
  if (copyYUVFrame() != GXF_SUCCESS) {
    GXF_LOG_ERROR("Error in copying output YUV");
    return GXF_FAILURE;
  }

  // Add timestamp into output message
  uint64_t input_timestamp =
    impl_->ctx->output_timestamp.tv_sec * static_cast<uint64_t>(1e6) +
    impl_->ctx->output_timestamp.tv_usec;
  auto out_timestamp =
    output_entity.add<nvidia::gxf::Timestamp>(kTimestampName);
  out_timestamp.value()->acqtime = input_timestamp;

  // enqueue it back
  if (enqueue_plane_buffer(impl_->ctx, impl_->ctx->cp_dqbuf_index, 0,
                           V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) != 0) {
    GXF_LOG_ERROR("Error in QBUF on capture plane \n");
    return GXF_FAILURE;
  }
  GXF_LOG_DEBUG("Successfully qbuf'ed %d CAPTURE plane", impl_->ctx->cp_dqbuf_index);

  if (impl_->ctx->got_eos) {
    // Context is inactive, set event to EVENT_NEVER
    impl_->ctx->response_scheduling_term->setEventState(
      nvidia::gxf::AsynchronousEventState::EVENT_NEVER);
  } else {
    // Set event back to WAIT to wait for next requests
    impl_->ctx->response_scheduling_term->setEventState(
      nvidia::gxf::AsynchronousEventState::WAIT);
  }

  impl_->ctx->cp_dqbuf_available = 0;

  return gxf::ToResultCode(output_transmitter_->publish(output_entity));
}

gxf_result_t VideoDecoderResponse::stop() {
  GXF_LOG_DEBUG("Enter stop function");
  return GXF_SUCCESS;
}

gxf_result_t VideoDecoderResponse::copyYUVFrame() {
  GXF_LOG_DEBUG("Copy YUV start \n");
  constexpr auto surface_layout =
      gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  auto storage_type = gxf::MemoryStorageType::kDevice;

  /* Cuda memcpy depending on if the gxf video tensor is in host or device.
  ** For Tegra, source type is always Host since mapped addr is used.
  */
  cudaMemcpyKind memcpytype = cudaMemcpyHostToDevice;
  switch (MemoryStorageType(outbuf_storage_type_.get())) {
  case MemoryStorageType::kHost: {
    storage_type = gxf::MemoryStorageType::kHost;
    memcpytype = cudaMemcpyDeviceToHost;
  } break;
  case MemoryStorageType::kDevice: {
    storage_type = gxf::MemoryStorageType::kDevice;
    memcpytype = cudaMemcpyDeviceToDevice;
  } break;
  default:
    return GXF_PARAMETER_OUT_OF_RANGE;
  }

  nvidia::gxf::Expected<void> gxf_result;
  switch (impl_->ctx->colorspace) {
  case V4L2_COLORSPACE_DEFAULT:
  case V4L2_COLORSPACE_SMPTE170M: {
    if (impl_->ctx->quantization == V4L2_QUANTIZATION_FULL_RANGE) {
      if (impl_->ctx->planar) {
        // BT.601 multi planar 4:2:0 YUV ER
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_ER>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      } else {
        // BT.601 multi planar 4:2:0 YUV ER with interleaved UV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      }
    } else {
      if (impl_->ctx->planar) {
        // BT.601 multi planar 4:2:0 YUV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      } else {
        // BT.601 multi planar 4:2:0 YUV with interleaved UV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      }
    }
  } break;
  case V4L2_COLORSPACE_REC709: {
    if (impl_->ctx->quantization == V4L2_QUANTIZATION_FULL_RANGE) {
      if (impl_->ctx->planar) {
        // BT.709 multi planar 4:2:0 YUV ER with interleaved UV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709_ER>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      } else {
        // BT.709 multi planar 4:2:0 YUV ER with interleaved UV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709_ER>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      }
    } else {
      if (impl_->ctx->planar) {
        // BT.709 multi planar 4:2:0 YUV with interleaved UV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420_709>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      } else {
        // BT.709 multi planar 4:2:0 YUV with interleaved UV
        gxf_result =
          decoded_frame_->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_709>(
            impl_->ctx->video_width, impl_->ctx->video_height, surface_layout,
            storage_type, memory_pool_);
      }
    }
  } break;
  case  V4L2_COLORSPACE_BT2020:
  default:
    return GXF_PARAMETER_OUT_OF_RANGE;
  }

  if (ToResultCode(gxf_result) != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to resize video buffer");
    return GXF_FAILURE;
  }

  auto decoded_frame__info = decoded_frame_->video_frame_info();

  cudaError_t result;
  int32_t retval = 0;
  NvBufSurface* dst_nvbuf_surf = nullptr;
  NvBufSurface* buf_surface = reinterpret_cast<NvBufSurface *>
      (impl_->ctx->capture_buffers[impl_->ctx->cp_dqbuf_index].buf_surface);
  if (!impl_->ctx->is_cuvid || impl_->ctx->planar) {
    /* Transformation parameters are defined
    ** which are passed to the NvBufSurfTransform
    ** for required conversion.
    */
    NvBufSurfTransformRect src_rect, dest_rect;
    src_rect.top = 0;
    src_rect.left = 0;
    src_rect.width = impl_->ctx->video_width;
    src_rect.height = impl_->ctx->video_height;
    dest_rect.top = 0;
    dest_rect.left = 0;
    dest_rect.width = impl_->ctx->video_width;
    dest_rect.height = impl_->ctx->video_height;

    NvBufSurfTransformParams transform_params;
    memset(&transform_params, 0, sizeof(transform_params));

    /* @transform_flag defines the flags for enabling the
    ** valid transforms. All the valid parameters are
    **  present in the nvbufsurface header.
    */
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transform_params.transform_flip = NvBufSurfTransform_None;
    transform_params.transform_filter = NvBufSurfTransformInter_Nearest;
    transform_params.src_rect = &src_rect;
    transform_params.dst_rect = &dest_rect;

    // Written for NV12.
    retval = NvBufSurfaceFromFd(impl_->ctx->dst_dma_fd,
                                reinterpret_cast<void**>(&dst_nvbuf_surf));
    if (retval) {
      GXF_LOG_ERROR("NvBufSurfaceFromFd failed");
      return GXF_FAILURE;
    }

    /* Blocklinear to Pitchlinear transformation AND/OR
    ** SemiPlanar to Planar transformation.
    */
    retval = NvBufSurfTransform(buf_surface, dst_nvbuf_surf,
                                &transform_params);
    if (retval) {
      GXF_LOG_ERROR("NvBufSurfaceFromFd failed");
      return GXF_FAILURE;
    }
  }
  uint32_t plane = 0;
  uint8_t* psrc_data;
  if (impl_->ctx->planar) {
    buf_surface = dst_nvbuf_surf;
  }

  if (impl_->ctx->is_cuvid) {
    psrc_data =
      reinterpret_cast<uint8_t *>(buf_surface->surfaceList[0].dataPtr);
  } else {
    result = cudaHostGetDevicePointer(&psrc_data,
                 dst_nvbuf_surf->surfaceList[0].mappedAddr.addr[plane], 0);
    if (result != cudaSuccess) {
      GXF_LOG_ERROR("Failed in cudaHostGetDevicePointer for plane %d: %s",
                    plane, cudaGetErrorString(result));
      return GXF_FAILURE;
    }
    buf_surface = dst_nvbuf_surf;
  }

  result = cudaMemcpy2D(
      decoded_frame_->pointer(),
      decoded_frame__info.color_planes[plane].stride,
      psrc_data,
      buf_surface->surfaceList[0].planeParams.pitch[plane],
      decoded_frame__info.color_planes[plane].width *
          decoded_frame__info.color_planes[plane].bytes_per_pixel,
      decoded_frame__info.color_planes[plane].height,
      memcpytype);

  if (result != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy yuv output for plane %d: %s",
                  plane, cudaGetErrorString(result));
    return GXF_FAILURE;
  }

  plane = 1;
  if (impl_->ctx->is_cuvid) {
    psrc_data =
      reinterpret_cast<uint8_t *>(buf_surface->surfaceList[0].dataPtr) +
          (buf_surface->surfaceList->planeParams.pitch[0] *
           buf_surface->surfaceList->planeParams.height[0]);
  } else {
    result = cudaHostGetDevicePointer(&psrc_data,
                 dst_nvbuf_surf->surfaceList[0].mappedAddr.addr[plane], 0);
    if (result != cudaSuccess) {
      GXF_LOG_ERROR("Failed in cudaHostGetDevicePointer for plane %d: %s",
                    plane, cudaGetErrorString(result));
      return GXF_FAILURE;
    }
    buf_surface = dst_nvbuf_surf;
  }

  result = cudaMemcpy2D(
      decoded_frame_->pointer() + decoded_frame__info.color_planes[0].size,
      decoded_frame__info.color_planes[plane].stride,
      psrc_data,
      buf_surface->surfaceList[0].planeParams.pitch[plane],
      decoded_frame__info.color_planes[plane].width *
          decoded_frame__info.color_planes[plane].bytes_per_pixel,
      decoded_frame__info.color_planes[plane].height,
      memcpytype);

  if (result != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy yuv output for plane %d: %s",
                  plane, cudaGetErrorString(result));
    return GXF_FAILURE;
  }

  if (impl_->ctx->planar) {
    plane = 2;
    if (impl_->ctx->is_cuvid) {
      psrc_data =
        reinterpret_cast<uint8_t *>(buf_surface->surfaceList[0].dataPtr) +
            (buf_surface->surfaceList->planeParams.pitch[0] *
             buf_surface->surfaceList->planeParams.height[0]) +
            (buf_surface->surfaceList->planeParams.pitch[1] *
             buf_surface->surfaceList->planeParams.height[1]);
    } else {
      result = cudaHostGetDevicePointer(&psrc_data,
                   dst_nvbuf_surf->surfaceList[0].mappedAddr.addr[plane], 0);
      if (result != cudaSuccess) {
        GXF_LOG_ERROR("Failed in cudaHostGetDevicePointer for plane %d: %s",
                      plane, cudaGetErrorString(result));
        return GXF_FAILURE;
      }
      buf_surface = dst_nvbuf_surf;
    }

    result = cudaMemcpy2D(
        decoded_frame_->pointer() + decoded_frame__info.color_planes[0].size +
        decoded_frame__info.color_planes[1].size,
        decoded_frame__info.color_planes[plane].stride,
        psrc_data,
        buf_surface->surfaceList[0].planeParams.pitch[plane],
        decoded_frame__info.color_planes[plane].width *
            decoded_frame__info.color_planes[plane].bytes_per_pixel,
        decoded_frame__info.color_planes[plane].height,
        memcpytype);

    if (result != cudaSuccess) {
      GXF_LOG_ERROR("Failed to copy yuv output for plane %d: %s",
                    plane, cudaGetErrorString(result));
      return GXF_FAILURE;
    }
  }
  GXF_LOG_DEBUG("Copy YUV done \n");
  return GXF_SUCCESS;
}

}  // namespace gxf
}  // namespace nvidia
