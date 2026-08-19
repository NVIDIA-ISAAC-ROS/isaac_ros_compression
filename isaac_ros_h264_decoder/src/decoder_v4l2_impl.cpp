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

#include "isaac_ros_h264_decoder/decoder_v4l2_impl.hpp"

#include <linux/videodev2.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <queue>
#include <string>

#include "isaac_ros_h264_decoder/decoder_v4l2_utils.hpp"
#include "libv4l2.h"  // NOLINT(build/include_subdir)
#include "linux/v4l2_nv_extensions.h"
#include "nvbufsurface.h"  // NOLINT(build/include_subdir)
#include "nvbufsurftransform.h"  // NOLINT(build/include_subdir)
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_decoder
{

namespace
{
constexpr int kMaxBuffers = 32;
constexpr int kMaxPlanes = 4;
constexpr int kCudaDevPropMajorThor = 11;

struct BufferInfo
{
  void * buf_surface{nullptr};
  int buf_fd{-1};
  int enqueued{0};
  int length{0};
};

bool is_wsl_platform()
{
  std::ifstream file("/proc/version");
  std::string line;
  if (file.is_open()) {
    std::getline(file, line);
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
    return line.find("microsoft") != std::string::npos;
  }
  return false;
}
}  // namespace

struct V4L2DecoderImpl
{
  int dev_fd{-1};
  uint32_t video_width{0};
  uint32_t video_height{0};
  uint32_t colorspace{0};
  uint32_t quantization{0};
  uint32_t max_bitstream_size{2097152};

  uint32_t output_buffer_count{5};
  uint32_t capture_buffer_count{10};
  uint32_t output_buffer_idx{0};

  BufferInfo output_buffers[kMaxBuffers];
  BufferInfo capture_buffers[kMaxBuffers];

  int dst_dma_fd{-1};

  bool is_cuvid{false};
  bool low_latency{false};
  std::atomic<bool> eos{false};
  std::atomic<bool> got_eos{false};
  std::atomic<bool> resolution_change_event{false};
  std::atomic<bool> capture_format_set{false};
  std::atomic<bool> cp_dqbuf_available{false};
  std::atomic<int32_t> cp_dqbuf_index{0};
  std::atomic<bool> error_in_decode_thread{false};

  std::thread decoder_thread;
  std::mutex queue_mutex;
  std::mutex decode_mutex;  // Serializes decode_frame calls to ensure FIFO order
  // Metadata for frame correlation using FIFO queue.
  // V4L2 decoder outputs frames in the same order as input (no B-frames in stream),
  // so a simple queue maintains correct correlation.
  struct FrameMetadata
  {
    uint64_t timestamp_ns;
    std::string frame_id;
  };
  std::queue<FrameMetadata> metadata_queue;  // FIFO - V4L2 decoder preserves order

  // Frame tracking counters for monitoring
  // Invariant: frames_in == frames_out + pending (metadata_queue.size())
  // If violated, frames were dropped: dropped = frames_in - frames_out - pending
  std::atomic<uint64_t> frames_in{0};
  std::atomic<uint64_t> frames_out{0};

  V4L2Decoder * parent{nullptr};

  int set_output_plane_format();
  int get_capture_plane_format(struct v4l2_format * fmt);
  int reqbufs_output_plane();
  int reqbufs_capture_plane();
  int enqueue_plane_buffer(int q_index, uint32_t bytes_used, uint32_t buftype);
  int enqueue_all_capture_plane_buffers();
  int dqbuf_plane(int * dqed_index, uint32_t buftype);

  void invoke_callback(DecodedFrame && frame);
};

int V4L2DecoderImpl::set_output_plane_format()
{
  struct v4l2_format fmt;
  std::memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
  fmt.fmt.pix_mp.num_planes = 1;
  fmt.fmt.pix_mp.plane_fmt[0].sizeimage = max_bitstream_size;

  return v4l2_ioctl::set_format(dev_fd, &fmt);
}

int V4L2DecoderImpl::get_capture_plane_format(struct v4l2_format * fmt)
{
  std::memset(fmt, 0, sizeof(struct v4l2_format));
  fmt->type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  return v4l2_ioctl::get_format(dev_fd, fmt);
}

int V4L2DecoderImpl::reqbufs_output_plane()
{
  struct v4l2_requestbuffers reqbuf;
  std::memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.count = output_buffer_count;
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  reqbuf.memory = V4L2_MEMORY_MMAP;

  int ret = v4l2_ioctl::request_buffers(dev_fd, &reqbuf);
  if (ret != 0) {
    return ret;
  }

  for (uint32_t i = 0; i < output_buffer_count; i++) {
    struct v4l2_buffer query_buf;
    struct v4l2_plane planes[kMaxPlanes];
    std::memset(&query_buf, 0, sizeof(query_buf));
    std::memset(planes, 0, sizeof(planes));

    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    query_buf.length = 1;

    ret = v4l2_ioctl::query_buffer(dev_fd, &query_buf);
    if (ret != 0) {
      return ret;
    }

    struct v4l2_exportbuffer expbuf;
    std::memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    ret = v4l2_ioctl::export_buffer(dev_fd, &expbuf);
    if (ret != 0) {
      return ret;
    }

    output_buffers[i].length = query_buf.m.planes[0].length;
    output_buffers[i].buf_fd = expbuf.fd;

    NvBufSurface * nvbuf = nullptr;
    if (NvBufSurfaceFromFd(output_buffers[i].buf_fd,
        reinterpret_cast<void **>(&nvbuf)) < 0)
    {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] NvBufSurfaceFromFd output buffer %u failed", i);
      return -1;
    }

    if (!is_cuvid) {
      if (NvBufSurfaceMap(nvbuf, 0, 0, NVBUF_MAP_READ_WRITE) != 0) {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] NvBufSurfaceMap output buffer %u failed", i);
        return -1;
      }
    }

    output_buffers[i].buf_surface = nvbuf;
    output_buffers[i].enqueued = 0;
  }

  return 0;
}

int V4L2DecoderImpl::reqbufs_capture_plane()
{
  struct v4l2_requestbuffers reqbuf;
  std::memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.count = capture_buffer_count + 5;
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  reqbuf.memory = V4L2_MEMORY_MMAP;

  int ret = v4l2_ioctl::request_buffers(dev_fd, &reqbuf);
  if (ret != 0) {
    return ret;
  }

  capture_buffer_count = reqbuf.count;

  for (uint32_t i = 0; i < capture_buffer_count; i++) {
    struct v4l2_buffer query_buf;
    struct v4l2_plane planes[kMaxPlanes];
    std::memset(&query_buf, 0, sizeof(query_buf));
    std::memset(planes, 0, sizeof(planes));

    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    query_buf.length = 3;

    ret = v4l2_ioctl::query_buffer(dev_fd, &query_buf);
    if (ret != 0) {
      return ret;
    }

    struct v4l2_exportbuffer expbuf;
    std::memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    ret = v4l2_ioctl::export_buffer(dev_fd, &expbuf);
    if (ret != 0) {
      return ret;
    }

    capture_buffers[i].buf_fd = expbuf.fd;

    NvBufSurface * nvbuf = nullptr;
    if (NvBufSurfaceFromFd(capture_buffers[i].buf_fd,
        reinterpret_cast<void **>(&nvbuf)) < 0)
    {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] NvBufSurfaceFromFd capture buffer %u failed", i);
      return -1;
    }

    capture_buffers[i].buf_surface = nvbuf;
    capture_buffers[i].enqueued = 0;
  }

  return 0;
}

int V4L2DecoderImpl::enqueue_plane_buffer(int q_index, uint32_t bytes_used, uint32_t buftype)
{
  struct v4l2_buffer qbuf;
  struct v4l2_plane plane;
  std::memset(&qbuf, 0, sizeof(qbuf));
  std::memset(&plane, 0, sizeof(plane));

  qbuf.index = q_index;
  qbuf.type = buftype;
  qbuf.memory = V4L2_MEMORY_MMAP;
  qbuf.m.planes = &plane;
  qbuf.m.planes[0].bytesused = bytes_used;
  qbuf.length = 1;

  return v4l2_ioctl::queue_buffer(dev_fd, &qbuf);
}

int V4L2DecoderImpl::enqueue_all_capture_plane_buffers()
{
  for (uint32_t i = 0; i < capture_buffer_count; i++) {
    if (capture_buffers[i].enqueued == 0) {
      if (enqueue_plane_buffer(i, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) != 0) {
        return -1;
      }
      capture_buffers[i].enqueued = 1;
    }
  }
  return 0;
}

int V4L2DecoderImpl::dqbuf_plane(int * dqed_index, uint32_t buftype)
{
  struct v4l2_buffer dqbuf;
  struct v4l2_plane planes[kMaxPlanes];
  std::memset(&dqbuf, 0, sizeof(dqbuf));
  std::memset(planes, 0, sizeof(planes));

  dqbuf.type = buftype;
  dqbuf.memory = V4L2_MEMORY_MMAP;
  dqbuf.m.planes = planes;
  dqbuf.length = 1;

  int ret = v4l2_ioctl::dequeue_buffer(dev_fd, &dqbuf);
  if (ret != 0) {
    return -1;
  }

  *dqed_index = dqbuf.index;
  return 0;
}

void V4L2DecoderImpl::invoke_callback(DecodedFrame && frame)
{
  if (parent) {
    parent->invoke_callback(std::move(frame));
  }
}

namespace
{
void decoder_thread_func(V4L2DecoderImpl * ctx)
{
  int32_t retval = 0;
  struct v4l2_format capture_format;
  int32_t dqbuf_index = 0;
  struct v4l2_event event;
  struct v4l2_crop crop;

  uint32_t wait_count = 0;
  while (!ctx->resolution_change_event) {
    if (ctx->eos) {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] EOS received while waiting for resolution change");
      ctx->error_in_decode_thread = true;
      return;
    }

    std::memset(&event, 0, sizeof(event));
    retval = v4l2_ioctl::dequeue_event(ctx->dev_fd, &event);
    if (retval == 0) {
      RCLCPP_INFO(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] Got event type=%u (RESOLUTION_CHANGE=%u)",
        event.type, V4L2_EVENT_RESOLUTION_CHANGE);
      if (event.type == V4L2_EVENT_RESOLUTION_CHANGE) {
        RCLCPP_INFO(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] Resolution change event received after %u iterations", wait_count);
        ctx->resolution_change_event = true;
        break;
      }
    } else if (errno != EAGAIN) {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] dequeue_event failed: %s", strerror(errno));
      ctx->error_in_decode_thread = true;
      return;
    }

    wait_count++;
    if (wait_count % 10000 == 0) {
      RCLCPP_WARN(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] Still waiting for resolution_change_event, wait_count=%u, "
        "output_buffer_idx=%u",
        wait_count, ctx->output_buffer_idx);
    }
    usleep(100);
  }

  while (!ctx->capture_format_set) {
    if (ctx->eos) {
      ctx->error_in_decode_thread = true;
      return;
    }
    RCLCPP_INFO(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Waiting for capture format set event...");
    retval = ctx->get_capture_plane_format(&capture_format);
    if (retval < 0) {
      usleep(100);
    } else {
      ctx->capture_format_set = true;
      break;
    }
  }

  std::memset(&crop, 0, sizeof(crop));
  crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  retval = v4l2_ioctl::get_crop(ctx->dev_fd, &crop);
  if (retval != 0) {
    ctx->error_in_decode_thread = true;
    return;
  }

  ctx->video_width = crop.c.width;
  ctx->video_height = crop.c.height;
  ctx->colorspace = capture_format.fmt.pix_mp.colorspace;
  ctx->quantization = capture_format.fmt.pix_mp.quantization;

  RCLCPP_INFO(rclcpp::get_logger("V4L2Decoder"),
    "Decoded video: %ux%u", ctx->video_width, ctx->video_height);

  if (!ctx->is_cuvid) {
    NvBufSurfaceAllocateParams dst_params = {{0}};
    NvBufSurface * dst_nvbuf = nullptr;

    dst_params.params.memType = NVBUF_MEM_DEFAULT;
    dst_params.params.width = crop.c.width;
    dst_params.params.height = crop.c.height;
    dst_params.params.layout = NVBUF_LAYOUT_PITCH;
    dst_params.params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    dst_params.memtag = NvBufSurfaceTag_VIDEO_DEC;

    retval = NvBufSurfaceAllocate(&dst_nvbuf, 1, &dst_params);
    if (retval) {
      ctx->error_in_decode_thread = true;
      return;
    }
    dst_nvbuf->numFilled = 1;
    ctx->dst_dma_fd = dst_nvbuf->surfaceList[0].bufferDesc;

    // For Tegra (nvgpu): map each plane to CPU memory and register with CUDA so the
    // GPU can access the decoded NV12 buffer via cudaHostGetDevicePointer.
    for (uint32_t plane = 0;
      plane < dst_nvbuf->surfaceList[0].planeParams.num_planes; plane++)
    {
      retval = NvBufSurfaceMap(dst_nvbuf, 0, plane, NVBUF_MAP_READ_WRITE);
      if (retval) {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] NvBufSurfaceMap for plane %u failed", plane);
        ctx->error_in_decode_thread = true;
        return;
      }
      cudaError_t cuda_err = cudaHostRegister(
        dst_nvbuf->surfaceList[0].mappedAddr.addr[plane],
        dst_nvbuf->surfaceList[0].planeParams.psize[plane],
        cudaHostRegisterDefault);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] cudaHostRegister for plane %u failed: %s",
          plane, cudaGetErrorString(cuda_err));
        ctx->error_in_decode_thread = true;
        return;
      }
    }
  }

  retval = v4l2_ioctl::get_min_capture_buffers(ctx->dev_fd, &ctx->capture_buffer_count);
  if (retval != 0) {
    ctx->error_in_decode_thread = true;
    return;
  }

  retval = ctx->reqbufs_capture_plane();
  if (retval != 0) {
    ctx->error_in_decode_thread = true;
    return;
  }

  retval = ctx->enqueue_all_capture_plane_buffers();
  if (retval != 0) {
    ctx->error_in_decode_thread = true;
    return;
  }

  retval = v4l2_ioctl::stream_on(ctx->dev_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  if (retval < 0) {
    ctx->error_in_decode_thread = true;
    return;
  }

  uint32_t timer = 0;

  while (!ctx->eos) {
    if (ctx->cp_dqbuf_available) {
      usleep(100);
      timer++;
      if (timer > 100000) {
        break;
      }
      continue;
    }

    if (ctx->dqbuf_plane(&dqbuf_index, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) != 0) {
      if (errno == EAGAIN) {
        continue;
      }
      if (errno == EPIPE) {
        ctx->got_eos = true;
        break;
      }
      break;
    }

    ctx->cp_dqbuf_index = dqbuf_index;
    ctx->cp_dqbuf_available = true;
    timer = 0;

    // Get metadata using FIFO - V4L2 decoder outputs in order (no B-frames)
    uint64_t timestamp_ns = 0;
    std::string frame_id;
    bool metadata_valid = false;
    {
      std::lock_guard<std::mutex> lock(ctx->queue_mutex);
      if (!ctx->metadata_queue.empty()) {
        timestamp_ns = ctx->metadata_queue.front().timestamp_ns;
        frame_id = std::move(ctx->metadata_queue.front().frame_id);
        ctx->metadata_queue.pop();
        metadata_valid = true;
      } else {
        // This should never happen - it means V4L2 produced output we didn't expect.
        // The invariant (frames_in == frames_out + pending) will be violated.
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] BUG: metadata queue empty but V4L2 produced output! "
          "(frames_in=%lu, frames_out=%lu, pending=0)",
          ctx->frames_in.load(), ctx->frames_out.load());
      }
    }

    // Skip this frame if no metadata available - return buffer to V4L2
    if (!metadata_valid) {
      ctx->capture_buffers[dqbuf_index].enqueued = 0;
      ctx->enqueue_plane_buffer(dqbuf_index, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
      ctx->capture_buffers[dqbuf_index].enqueued = 1;
      ctx->cp_dqbuf_available = false;
      continue;
    }

    NvBufSurface * cap_buf = reinterpret_cast<NvBufSurface *>(
      ctx->capture_buffers[dqbuf_index].buf_surface);

    NvBufSurface * src_buf = cap_buf;

    if (!ctx->is_cuvid && ctx->dst_dma_fd >= 0) {
      NvBufSurface * dst_buf = nullptr;
      NvBufSurfaceFromFd(ctx->dst_dma_fd, reinterpret_cast<void **>(&dst_buf));

      NvBufSurfTransformRect src_rect = {0, 0, ctx->video_width, ctx->video_height};
      NvBufSurfTransformRect dst_rect = {0, 0, ctx->video_width, ctx->video_height};
      NvBufSurfTransformParams params;
      std::memset(&params, 0, sizeof(params));
      params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
      params.transform_flip = NvBufSurfTransform_None;
      params.transform_filter = NvBufSurfTransformInter_Nearest;
      params.src_rect = &src_rect;
      params.dst_rect = &dst_rect;

      NvBufSurfTransform(cap_buf, dst_buf, &params);
      src_buf = dst_buf;
    }

    // Build frame with GPU pointer - no CPU copy
    DecodedFrame frame;
    frame.width = ctx->video_width;
    frame.height = ctx->video_height;
    frame.timestamp_ns = timestamp_ns;
    frame.frame_id = std::move(frame_id);
    frame.buffer_index = dqbuf_index;

    // Get plane parameters from NvBufSurface
    auto & plane_params = src_buf->surfaceList[0].planeParams;

    if (ctx->is_cuvid) {
      frame.device_ptr = reinterpret_cast<uint8_t *>(src_buf->surfaceList[0].dataPtr);
      frame.y_pitch = plane_params.pitch[0];
      // For CUVID NV12, UV pitch is typically same as Y pitch (semi-planar in single buffer)
      // Use pitch[1] if available and non-zero, otherwise use Y pitch
      frame.uv_pitch = (plane_params.pitch[1] > 0) ? plane_params.pitch[1] : frame.y_pitch;
      // Use offset[1] if available, otherwise calculate from psize[0] or pitch*height
      if (plane_params.offset[1] > 0) {
        frame.uv_offset = plane_params.offset[1];
      } else if (plane_params.psize[0] > 0) {
        frame.uv_offset = plane_params.psize[0];
      } else {
        frame.uv_offset = frame.y_pitch * ctx->video_height;
      }
    } else {
      // For Tegra (nvgpu): each plane was mapped via NvBufSurfaceMap + cudaHostRegister.
      // Get the device pointer for each plane separately — the plane mappings
      // are not guaranteed to be contiguous in virtual memory, so the consumer
      // must use uv_device_ptr rather than device_ptr + uv_offset.
      uint8_t * y_dev_ptr = nullptr;
      uint8_t * uv_dev_ptr = nullptr;
      cudaError_t cuda_err = cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&y_dev_ptr),
        src_buf->surfaceList[0].mappedAddr.addr[0], 0);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] cudaHostGetDevicePointer Y failed: %s",
          cudaGetErrorString(cuda_err));
        ctx->cp_dqbuf_available = false;
        continue;
      }
      if (plane_params.num_planes > 1) {
        cuda_err = cudaHostGetDevicePointer(
          reinterpret_cast<void **>(&uv_dev_ptr),
          src_buf->surfaceList[0].mappedAddr.addr[1], 0);
        if (cuda_err != cudaSuccess) {
          RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
            "[V4L2Decoder] cudaHostGetDevicePointer UV failed: %s",
            cudaGetErrorString(cuda_err));
          ctx->cp_dqbuf_available = false;
          continue;
        }
      }
      frame.device_ptr = y_dev_ptr;
      frame.uv_device_ptr = uv_dev_ptr;
      frame.y_pitch = plane_params.pitch[0];
      frame.uv_pitch = (plane_params.pitch[1] > 0) ? plane_params.pitch[1] : frame.y_pitch;
      if (plane_params.offset[1] > 0) {
        frame.uv_offset = plane_params.offset[1];
      } else if (plane_params.psize[0] > 0) {
        frame.uv_offset = plane_params.psize[0];
      } else {
        frame.uv_offset = frame.y_pitch * ctx->video_height;
      }
    }

    // Callback must copy data and call return_buffer() when done
    ctx->frames_out++;
    ctx->invoke_callback(std::move(frame));

    // Don't re-enqueue buffer here - the callback will call return_buffer()
    ctx->cp_dqbuf_available = false;
  }

  ctx->error_in_decode_thread = false;
}
}  // namespace

struct V4L2Decoder::Impl : public V4L2DecoderImpl {};

V4L2Decoder::V4L2Decoder()
: impl_(std::make_unique<Impl>())
{
  impl_->parent = this;
}

V4L2Decoder::~V4L2Decoder()
{
  shutdown();
}

void V4L2Decoder::invoke_callback(DecodedFrame && frame)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (output_callback_) {
    output_callback_(std::move(frame));
  }
}

bool V4L2Decoder::initialize(const DecoderConfig & config)
{
  if (initialized_) {
    shutdown();
  }

  config_ = config;
  impl_->max_bitstream_size = config.max_bitstream_size;
  impl_->low_latency = config.low_latency;

  cudaDeviceProp prop;
  cudaError_t status = cudaGetDeviceProperties(&prop, 0);
  if (status != cudaSuccess) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"), "[V4L2Decoder] cudaGetDeviceProperties failed");
    return false;
  }

  impl_->is_cuvid = !prop.integrated || (prop.major >= kCudaDevPropMajorThor);

  bool is_wsl = is_wsl_platform();
  if (is_wsl) {
    impl_->dev_fd = v4l2_open("/dev/null", 0);
  } else if (impl_->is_cuvid) {
    char gpu_device[16];
    FILE * fd = popen(
      "ls /dev/nvidia* | grep -m 1 '/dev/nvidia[[:digit:]]' | tr -d [:space:]", "r");
    if (!fd || !fgets(gpu_device, sizeof(gpu_device), fd)) {
      if (fd) {pclose(fd);}
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] Could not find GPU device node");
      return false;
    }
    pclose(fd);
    impl_->dev_fd = v4l2_open(gpu_device, 0);
  } else {
    impl_->dev_fd = v4l2_open("/dev/v4l2-nvdec", 0);
  }

  if (impl_->dev_fd < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"), "[V4L2Decoder] Failed to open decoder device");
    return false;
  }

  if (impl_->set_output_plane_format() < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Failed to set output plane format");
    return false;
  }

  if (v4l2_ioctl::subscribe_event(impl_->dev_fd, V4L2_EVENT_EOS) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"), "[V4L2Decoder] Failed to subscribe EOS event");
    return false;
  }

  if (v4l2_ioctl::subscribe_event(impl_->dev_fd, V4L2_EVENT_RESOLUTION_CHANGE) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Failed to subscribe resolution change event");
    return false;
  }

  if (impl_->is_cuvid) {
    if (v4l2_ioctl::set_cuda_gpu_id(impl_->dev_fd, 0) < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"), "[V4L2Decoder] Failed to set CUDA GPU ID");
      return false;
    }
  }

  if (v4l2_ioctl::set_disable_complete_frame_input(impl_->dev_fd) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Failed to set disable complete frame input");
    return false;
  }

  if (config.low_latency) {
    if (v4l2_ioctl::set_low_latency_decode(impl_->dev_fd, impl_->is_cuvid) < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
        "[V4L2Decoder] Failed to enable low latency decode");
      return false;
    }
  }

  if (impl_->reqbufs_output_plane() < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Failed to request output plane buffers");
    return false;
  }

  if (v4l2_ioctl::stream_on(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Failed to stream on output plane");
    return false;
  }

  impl_->eos = false;
  impl_->decoder_thread = std::thread(decoder_thread_func, impl_.get());

  initialized_ = true;
  RCLCPP_INFO(rclcpp::get_logger("V4L2Decoder"),
    "V4L2 Decoder initialized, cuvid=%d", impl_->is_cuvid);

  return true;
}

void V4L2Decoder::shutdown()
{
  if (!initialized_) {
    return;
  }

  // Log final frame statistics
  uint64_t in_count = impl_->frames_in.load();
  uint64_t out_count = impl_->frames_out.load();
  uint64_t pending = 0;
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    pending = impl_->metadata_queue.size();
  }
  // Derive dropped from invariant: frames_in == frames_out + pending + dropped
  uint64_t dropped_count = (in_count > out_count + pending) ?
    (in_count - out_count - pending) : 0;

  RCLCPP_INFO(
    rclcpp::get_logger("V4L2Decoder"),
    "[V4L2Decoder] Shutdown stats: frames_in=%lu, frames_out=%lu, "
    "pending=%lu, dropped=%lu",
    in_count, out_count, pending, dropped_count);

  if (dropped_count > 0) {
    RCLCPP_WARN(
      rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] WARNING: %lu frames were dropped during decoding",
      dropped_count);
  }

  impl_->eos = true;
  impl_->cp_dqbuf_available = false;

  if (impl_->dev_fd >= 0) {
    v4l2_ioctl::decoder_stop(impl_->dev_fd);
    v4l2_ioctl::stream_off(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
    v4l2_ioctl::stream_off(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  }

  if (impl_->decoder_thread.joinable()) {
    impl_->decoder_thread.join();
  }

  if (!impl_->is_cuvid) {
    for (uint32_t i = 0; i < impl_->output_buffer_count; i++) {
      if (impl_->output_buffers[i].buf_surface) {
        NvBufSurface * nvbuf = reinterpret_cast<NvBufSurface *>(
          impl_->output_buffers[i].buf_surface);
        NvBufSurfaceUnMap(nvbuf, 0, 0);
        impl_->output_buffers[i].buf_surface = nullptr;
      }
    }
  }

  if (impl_->dst_dma_fd >= 0) {
    NvBufSurface * dst_buf = nullptr;
    NvBufSurfaceFromFd(impl_->dst_dma_fd, reinterpret_cast<void **>(&dst_buf));
    if (dst_buf && dst_buf->surfaceList) {
      for (uint32_t plane = 0;
        plane < dst_buf->surfaceList[0].planeParams.num_planes; plane++)
      {
        NvBufSurfaceUnMap(dst_buf, 0, plane);
      }
    }
    NvBufSurfaceDestroy(dst_buf);
    impl_->dst_dma_fd = -1;
  }

  if (impl_->dev_fd >= 0 && !impl_->error_in_decode_thread) {
    v4l2_close(impl_->dev_fd);
    impl_->dev_fd = -1;
  }

  initialized_ = false;
}

bool V4L2Decoder::decode_frame(
  const uint8_t * data,
  size_t size,
  uint64_t timestamp_ns,
  const std::string & frame_id,
  cudaStream_t stream)
{
  if (!initialized_) {
    return false;
  }

  if (impl_->error_in_decode_thread) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"), "[V4L2Decoder] Decode thread error");
    return false;
  }

  if (size == 0) {
    if (!impl_->got_eos) {
      v4l2_ioctl::decoder_stop(impl_->dev_fd);
    }
    return true;
  }

  // Serialize decode_frame calls to ensure metadata queue order matches V4L2 queue order.
  // Without this lock, concurrent calls could push metadata in wrong order.
  std::lock_guard<std::mutex> decode_lock(impl_->decode_mutex);

  int q_index = 0;
  if (impl_->output_buffer_idx < impl_->output_buffer_count) {
    q_index = impl_->output_buffer_idx;
    impl_->output_buffer_idx++;
  } else {
    while (impl_->dqbuf_plane(&q_index, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) != 0) {
      if (errno == EAGAIN) {
        continue;
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
          "[V4L2Decoder] Error in dqbuf on output plane");
        return false;
      }
    }
  }

  if (size > impl_->max_bitstream_size) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "Input size %zu exceeds max %u", size, impl_->max_bitstream_size);
    return false;
  }

  NvBufSurface * nvbuf = reinterpret_cast<NvBufSurface *>(
    impl_->output_buffers[q_index].buf_surface);

  uint8_t * dst = nullptr;
  if (impl_->is_cuvid) {
    dst = reinterpret_cast<uint8_t *>(nvbuf->surfaceList[0].dataPtr);
  } else {
    dst = reinterpret_cast<uint8_t *>(nvbuf->surfaceList[0].mappedAddr.addr[0]);
  }

  // Async copy on provided stream, then sync before V4L2 QBUF.
  // For CUVID (dGPU): D2D copy to device memory in NvBufSurface
  // For Tegra: D2H copy to mapped host memory in NvBufSurface
  cudaMemcpyKind memcpy_kind = impl_->is_cuvid ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
  cudaError_t cuda_err = cudaMemcpyAsync(dst, data, size, memcpy_kind, stream);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] cudaMemcpyAsync failed: %s", cudaGetErrorString(cuda_err));
    return false;
  }
  // Sync required because V4L2 QBUF immediately hands the buffer to the driver
  cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] cudaStreamSynchronize failed: %s", cudaGetErrorString(cuda_err));
    return false;
  }

  // Store metadata in FIFO queue for correlation
  // V4L2 decoder outputs frames in same order as input (no B-frames in stream)
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    impl_->metadata_queue.push({timestamp_ns, frame_id});
  }

  if (impl_->enqueue_plane_buffer(q_index, size, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Failed to enqueue output plane buffer");
    return false;
  }

  impl_->frames_in++;
  return true;
}

DecoderFrameStats V4L2Decoder::get_frame_stats() const
{
  DecoderFrameStats stats;
  stats.frames_in = impl_->frames_in.load();
  stats.frames_out = impl_->frames_out.load();
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    stats.pending = impl_->metadata_queue.size();
  }
  // Derive dropped from invariant: frames_in == frames_out + pending + dropped
  stats.frames_dropped = (stats.frames_in > stats.frames_out + stats.pending) ?
    (stats.frames_in - stats.frames_out - stats.pending) : 0;
  return stats;
}

void V4L2Decoder::set_output_callback(DecodedFrameCallback callback)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  output_callback_ = std::move(callback);
}

void V4L2Decoder::return_buffer(int buffer_index)
{
  if (buffer_index < 0 || buffer_index >= static_cast<int>(impl_->capture_buffer_count)) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Decoder"),
      "[V4L2Decoder] Invalid buffer index %d for return_buffer", buffer_index);
    return;
  }
  impl_->capture_buffers[buffer_index].enqueued = 0;
  impl_->enqueue_plane_buffer(buffer_index, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  impl_->capture_buffers[buffer_index].enqueued = 1;
}

uint32_t V4L2Decoder::video_width() const
{
  return impl_->video_width;
}

uint32_t V4L2Decoder::video_height() const
{
  return impl_->video_height;
}

}  // namespace h264_decoder
}  // namespace isaac_ros
}  // namespace nvidia
