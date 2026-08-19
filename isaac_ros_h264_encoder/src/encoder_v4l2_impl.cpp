// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaac_ros_h264_encoder/encoder_v4l2_impl.hpp"

#include <linux/videodev2.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <queue>
#include <string>

#include "isaac_ros_h264_encoder/encoder_v4l2_utils.hpp"
#include "libv4l2.h"  // NOLINT(build/include_subdir)
#include "linux/v4l2_nv_extensions.h"
#include "nvbufsurface.h"  // NOLINT(build/include_subdir)
#include "rclcpp/rclcpp.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace h264_encoder
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

// Metadata for frame correlation using FIFO queue.
// V4L2 encoder outputs frames in the same order as input (no B-frames),
// so a simple queue maintains correct correlation.
struct FrameMetadata
{
  uint64_t timestamp_ns;
  std::string frame_id;
};

struct V4L2EncoderImpl
{
  int dev_fd{-1};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t profile{0};
  uint32_t qp{20};
  uint32_t raw_pixfmt{0};
  uint32_t encoder_pixfmt{V4L2_PIX_FMT_H264};
  uint32_t level{14};
  uint32_t hw_preset_type{1};
  uint32_t iframe_interval{5};
  uint32_t idr_interval{5};
  uint32_t entropy{1};  // 0: CAVLC, 1: CABAC
  int32_t rate_control_mode{0};
  int32_t bitrate{20000000};
  int32_t framerate{30};

  uint32_t output_buffer_count{5};
  uint32_t capture_buffer_count{5};
  uint32_t output_buffer_idx{0};

  BufferInfo output_buffers[kMaxBuffers];
  BufferInfo capture_buffers[kMaxBuffers];
  uint32_t outbuf_bytesused[kMaxPlanes];

  bool is_cuvid{false};
  std::atomic<bool> eos{false};
  std::atomic<bool> bitstream_buf_queued{false};
  std::atomic<uint32_t> dqbuf_index{0};
  std::atomic<uint32_t> bitstream_size{0};

  std::thread encoder_thread;
  std::mutex queue_mutex;
  std::mutex encode_mutex;  // Serializes encode_frame calls to ensure FIFO order
  std::queue<FrameMetadata> metadata_queue;  // FIFO - V4L2 encoder preserves order

  // Frame tracking counters for monitoring
  // Invariant: frames_in == frames_out + pending (metadata_queue.size())
  // If violated, frames were dropped: dropped = frames_in - frames_out - pending
  std::atomic<uint64_t> frames_in{0};
  std::atomic<uint64_t> frames_out{0};

  V4L2Encoder * parent{nullptr};

  int set_capture_plane_format();
  int set_output_plane_format();
  int reqbufs_output_plane(int32_t fps);
  int reqbufs_capture_plane();
  int enqueue_capture_plane_buffer(int q_index);
  int enqueue_all_capture_plane_buffers();
  int enqueue_output_plane_buffer(int q_index, int bytes_used);
  int dqbuf_on_output_plane(int * dqed_index);
  int dqbuf_on_capture_plane(int * dqed_index, int * bytes_copied);

  void invoke_callback(EncodedFrame && frame);
};

int V4L2EncoderImpl::set_capture_plane_format()
{
  struct v4l2_format fmt;
  std::memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.pixelformat = encoder_pixfmt;
  fmt.fmt.pix_mp.width = width;
  fmt.fmt.pix_mp.height = height;
  fmt.fmt.pix_mp.num_planes = 1;
  fmt.fmt.pix_mp.plane_fmt[0].sizeimage = 2 * width * height;

  return v4l2_ioctl::set_format(dev_fd, &fmt);
}

int V4L2EncoderImpl::set_output_plane_format()
{
  struct v4l2_format fmt;
  std::memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  fmt.fmt.pix_mp.pixelformat = raw_pixfmt;
  fmt.fmt.pix_mp.width = width;
  fmt.fmt.pix_mp.height = height;

  if (raw_pixfmt == V4L2_PIX_FMT_NV12M) {
    fmt.fmt.pix_mp.num_planes = 2;
  } else {
    fmt.fmt.pix_mp.num_planes = 3;
  }

  return v4l2_ioctl::set_format(dev_fd, &fmt);
}

int V4L2EncoderImpl::reqbufs_output_plane(int32_t fps)
{
  struct v4l2_streamparm parm;
  std::memset(&parm, 0, sizeof(parm));
  parm.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  parm.parm.output.timeperframe.numerator = 1;
  parm.parm.output.timeperframe.denominator = fps;

  int ret = v4l2_ioctl::set_stream_params(dev_fd, &parm);
  if (ret != 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] VIDIOC_S_PARM failed");
    return ret;
  }

  struct v4l2_requestbuffers reqbuf;
  std::memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.count = output_buffer_count;
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  reqbuf.memory = V4L2_MEMORY_MMAP;

  ret = v4l2_ioctl::request_buffers(dev_fd, &reqbuf);
  if (ret != 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] VIDIOC_REQBUFS output failed");
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
    query_buf.length = (raw_pixfmt == V4L2_PIX_FMT_NV12M) ? 2 : 3;

    ret = v4l2_ioctl::query_buffer(dev_fd, &query_buf);
    if (ret != 0) {
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] VIDIOC_QUERYBUF output failed");
      return ret;
    }

    struct v4l2_exportbuffer expbuf;
    std::memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    ret = v4l2_ioctl::export_buffer(dev_fd, &expbuf);
    if (ret != 0) {
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] VIDIOC_EXPBUF output failed");
      return ret;
    }

    output_buffers[i].length = query_buf.m.planes[0].length;
    output_buffers[i].buf_fd = expbuf.fd;

    NvBufSurface * nvbuf = nullptr;
    if (NvBufSurfaceFromFd(output_buffers[i].buf_fd,
        reinterpret_cast<void **>(&nvbuf)) < 0)
    {
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] NvBufSurfaceFromFd output failed");
      return -1;
    }

    if (!is_cuvid) {
      if (NvBufSurfaceMap(nvbuf, 0, 0, NVBUF_MAP_READ_WRITE) != 0) {
        RCLCPP_ERROR(
          rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] NvBufSurfaceMap output failed");
        return -1;
      }
      cudaError_t cuda_err = cudaHostRegister(
        nvbuf->surfaceList[0].mappedAddr.addr[0],
        nvbuf->surfaceList[0].planeParams.psize[0], cudaHostRegisterDefault);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(
          rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] cudaHostRegister Y plane failed: %s", cudaGetErrorString(cuda_err));
        NvBufSurfaceUnMap(nvbuf, 0, 0);
        return -1;
      }
      cuda_err = cudaHostRegister(
        nvbuf->surfaceList[0].mappedAddr.addr[1],
        nvbuf->surfaceList[0].planeParams.psize[1], cudaHostRegisterDefault);
      if (cuda_err != cudaSuccess) {
        RCLCPP_ERROR(
          rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] cudaHostRegister UV plane failed: %s", cudaGetErrorString(cuda_err));
        cudaHostUnregister(nvbuf->surfaceList[0].mappedAddr.addr[0]);
        NvBufSurfaceUnMap(nvbuf, 0, 0);
        return -1;
      }
    }
    output_buffers[i].buf_surface = nvbuf;
    output_buffers[i].enqueued = 0;
  }

  return 0;
}

int V4L2EncoderImpl::reqbufs_capture_plane()
{
  struct v4l2_requestbuffers reqbuf;
  std::memset(&reqbuf, 0, sizeof(reqbuf));
  reqbuf.count = capture_buffer_count;
  reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  reqbuf.memory = V4L2_MEMORY_MMAP;

  int ret = v4l2_ioctl::request_buffers(dev_fd, &reqbuf);
  if (ret != 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] VIDIOC_REQBUFS capture failed: %d", ret);
    return ret;
  }

  for (uint32_t i = 0; i < capture_buffer_count; i++) {
    struct v4l2_buffer query_buf;
    struct v4l2_plane planes[kMaxPlanes];
    std::memset(&query_buf, 0, sizeof(query_buf));
    std::memset(planes, 0, sizeof(planes));

    query_buf.index = i;
    query_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    query_buf.memory = V4L2_MEMORY_MMAP;
    query_buf.m.planes = planes;
    query_buf.length = 1;

    ret = v4l2_ioctl::query_buffer(dev_fd, &query_buf);
    if (ret != 0) {
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] VIDIOC_QUERYBUF capture failed");
      return ret;
    }

    struct v4l2_exportbuffer expbuf;
    std::memset(&expbuf, 0, sizeof(expbuf));
    expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    expbuf.index = i;
    expbuf.fd = -1;

    ret = v4l2_ioctl::export_buffer(dev_fd, &expbuf);
    if (ret != 0) {
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] VIDIOC_EXPBUF capture failed");
      return ret;
    }

    capture_buffers[i].buf_fd = expbuf.fd;

    NvBufSurface * nvbuf = nullptr;
    if (NvBufSurfaceFromFd(capture_buffers[i].buf_fd,
        reinterpret_cast<void **>(&nvbuf)) < 0)
    {
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] NvBufSurfaceFromFd capture failed");
      return -1;
    }

    if (!is_cuvid) {
      // For Tegra: map the compressed-bitstream capture buffer to CPU memory
      // so the encoder thread can copy the bitstream into EncodedFrame::data.
      // cudaHostRegister is not used here because the NVENC bitstream buffer
      // is not a CUDA-registerable dmabuf on this platform; the frame.data
      // host-vector fallback in EncoderNode already handles the H2D copy.
      if (NvBufSurfaceMap(nvbuf, 0, 0, NVBUF_MAP_READ_WRITE) != 0) {
        RCLCPP_ERROR(
          rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] NvBufSurfaceMap capture failed");
        return -1;
      }
    }
    capture_buffers[i].buf_surface = nvbuf;
    capture_buffers[i].enqueued = 0;
  }

  return 0;
}

int V4L2EncoderImpl::enqueue_capture_plane_buffer(int q_index)
{
  struct v4l2_buffer buf;
  struct v4l2_plane plane;
  std::memset(&buf, 0, sizeof(buf));
  std::memset(&plane, 0, sizeof(plane));

  buf.index = q_index;
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.timestamp.tv_sec = q_index;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.m.planes = &plane;
  buf.m.planes[0].bytesused = 0;
  buf.length = 1;

  return v4l2_ioctl::queue_buffer(dev_fd, &buf);
}

int V4L2EncoderImpl::enqueue_all_capture_plane_buffers()
{
  for (uint32_t i = 0; i < capture_buffer_count; i++) {
    if (capture_buffers[i].enqueued == 0) {
      if (enqueue_capture_plane_buffer(i) != 0) {
        RCLCPP_ERROR(

          rclcpp::get_logger("V4L2Encoder"),

          "[V4L2Encoder] QBUF capture plane failed");
        return -1;
      }
      capture_buffers[i].enqueued = 1;
    }
  }
  return 0;
}

int V4L2EncoderImpl::enqueue_output_plane_buffer(int q_index, int bytes_used)
{
  struct v4l2_buffer buf;
  struct v4l2_plane plane;
  std::memset(&buf, 0, sizeof(buf));
  std::memset(&plane, 0, sizeof(plane));

  buf.index = q_index;
  buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.m.planes = &plane;
  buf.m.planes[0].bytesused = bytes_used;
  buf.length = 1;

  return v4l2_ioctl::queue_buffer(dev_fd, &buf);
}

int V4L2EncoderImpl::dqbuf_on_output_plane(int * dqed_index)
{
  struct v4l2_buffer buf;
  struct v4l2_plane planes[kMaxPlanes];
  std::memset(&buf, 0, sizeof(buf));
  std::memset(planes, 0, sizeof(planes));

  buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.m.planes = planes;
  buf.length = 1;

  int ret = v4l2_ioctl::dequeue_buffer(dev_fd, &buf);
  if (ret == 0) {
    *dqed_index = buf.index;
  }
  return ret;
}

int V4L2EncoderImpl::dqbuf_on_capture_plane(int * dqed_index, int * bytes_copied)
{
  struct v4l2_buffer buf;
  struct v4l2_plane planes[kMaxPlanes];
  std::memset(&buf, 0, sizeof(buf));
  std::memset(planes, 0, sizeof(planes));

  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.m.planes = planes;
  buf.length = 1;

  int ret = v4l2_ioctl::dequeue_buffer(dev_fd, &buf);
  if (ret == 0) {
    *dqed_index = buf.index;
    *bytes_copied = buf.m.planes[0].bytesused;
  }
  return ret;
}

void V4L2EncoderImpl::invoke_callback(EncodedFrame && frame)
{
  if (parent) {
    parent->invoke_callback(std::move(frame));
  }
}

namespace
{
void encoder_thread_func(V4L2EncoderImpl * ctx)
{
  int32_t dq_index = 0;
  int32_t bs_size = 0;

  while (!ctx->eos) {
    if (!ctx->bitstream_buf_queued) {
      continue;
    }

    int ret = ctx->dqbuf_on_capture_plane(&dq_index, &bs_size);
    if (ret != 0) {
      if (ctx->eos) {break;}
      if (errno == EAGAIN) {
        continue;
      }
      break;
    }

    if (ctx->eos || bs_size == 0) {break;}

    ctx->bitstream_buf_queued = false;
    ctx->bitstream_size = bs_size;
    ctx->dqbuf_index = dq_index;

    NvBufSurface * nvbuf = reinterpret_cast<NvBufSurface *>(
      ctx->capture_buffers[dq_index].buf_surface);

    // Get metadata using FIFO - V4L2 encoder outputs in order (no B-frames)
    FrameMetadata metadata{};
    bool metadata_valid = false;
    {
      std::lock_guard<std::mutex> lock(ctx->queue_mutex);
      if (!ctx->metadata_queue.empty()) {
        metadata = std::move(ctx->metadata_queue.front());
        ctx->metadata_queue.pop();
        metadata_valid = true;
        RCLCPP_DEBUG(rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] POP metadata: ts=%lu, queue_size=%zu",
          metadata.timestamp_ns, ctx->metadata_queue.size());
      } else {
        // This should never happen - it means V4L2 produced output we didn't expect.
        // The invariant (frames_in == frames_out + pending) will be violated.
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] BUG: metadata queue empty but V4L2 produced output! "
          "(frames_in=%lu, frames_out=%lu, pending=0)",
          ctx->frames_in.load(), ctx->frames_out.load());
      }
    }

    // Skip this frame if no metadata available
    if (!metadata_valid) {
      ctx->capture_buffers[dq_index].enqueued = 0;
      ctx->enqueue_capture_plane_buffer(dq_index);
      ctx->capture_buffers[dq_index].enqueued = 1;
      ctx->bitstream_buf_queued = true;
      continue;
    }

    EncodedFrame frame;
    frame.timestamp_ns = metadata.timestamp_ns;
    frame.frame_id = metadata.frame_id;
    frame.is_keyframe = false;

    if (ctx->is_cuvid) {
      // For CUVID (dGPU), dataPtr is in CUDA device memory
      frame.device_ptr = nvbuf->surfaceList[0].dataPtr;
      frame.size = bs_size;
    } else {
      // For Tegra (nvgpu): expose the CPU-mapped NVENC capture buffer pointer directly.
      // EncoderNode::on_encoded_frame issues a synchronous cudaMemcpy from this
      // pointer to the output tensor before the callback returns, after which
      // the V4L2 buffer is re-enqueued. NvBufSurfaceSyncForCpu flushes the
      // DMA-writer cache so the read sees the bytes NVENC just wrote; see the
      // sibling GXF video_encoder response at videoencoder_response.cpp:94-96.
      NvBufSurfaceSyncForCpu(nvbuf, 0, 0);
      void * host_ptr = nvbuf->surfaceList[0].mappedAddr.addr[0];
      if (host_ptr == nullptr) {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] capture buffer not mapped");
        continue;
      }
      frame.host_ptr = host_ptr;
      frame.size = bs_size;
    }

    ctx->frames_out++;
    ctx->invoke_callback(std::move(frame));

    ctx->capture_buffers[dq_index].enqueued = 0;
    ctx->enqueue_capture_plane_buffer(dq_index);
    ctx->capture_buffers[dq_index].enqueued = 1;
    ctx->bitstream_buf_queued = true;
  }
}
}  // namespace

struct V4L2Encoder::Impl : public V4L2EncoderImpl {};

V4L2Encoder::V4L2Encoder()
: impl_(std::make_unique<Impl>())
{
  impl_->parent = this;
}

V4L2Encoder::~V4L2Encoder()
{
  shutdown();
}

void V4L2Encoder::invoke_callback(EncodedFrame && frame)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (output_callback_) {
    output_callback_(std::move(frame));
  }
}

bool V4L2Encoder::initialize(const EncoderConfig & config)
{
  if (initialized_) {
    shutdown();
  }

  config_ = config;

  impl_->width = config.width;
  impl_->height = config.height;
  impl_->profile = config.profile;
  impl_->qp = config.qp;
  impl_->hw_preset_type = config.hw_preset_type;
  impl_->iframe_interval = config.iframe_interval;
  impl_->idr_interval = config.idr_interval;
  impl_->entropy = config.entropy;
  impl_->rate_control_mode = config.rate_control_mode;
  impl_->bitrate = config.bitrate;
  impl_->framerate = config.framerate;
  impl_->level = config.level;
  impl_->raw_pixfmt = V4L2_PIX_FMT_NV12M;
  impl_->encoder_pixfmt = V4L2_PIX_FMT_H264;

  cudaDeviceProp prop;
  cudaError_t status = cudaGetDeviceProperties(&prop, 0);
  if (status != cudaSuccess) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] cudaGetDeviceProperties failed");
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
      RCLCPP_ERROR(

        rclcpp::get_logger("V4L2Encoder"),

        "[V4L2Encoder] Could not find GPU device node");
      return false;
    }
    pclose(fd);
    impl_->dev_fd = v4l2_open(gpu_device, 0);
  } else {
    impl_->dev_fd = v4l2_open("/dev/v4l2-nvenc", 0);
  }

  if (impl_->dev_fd < 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] Failed to open encoder device");
    return false;
  }

  if (impl_->is_cuvid) {
    if (impl_->rate_control_mode == 0) {
      v4l2_ioctl::set_bitrate_mode(impl_->dev_fd, V4L2_MPEG_VIDEO_BITRATE_MODE_CONSTQP);
    } else {
      v4l2_ioctl::set_bitrate_mode(impl_->dev_fd, V4L2_MPEG_VIDEO_BITRATE_MODE_CBR);
    }
    v4l2_ioctl::set_idr_interval(impl_->dev_fd, impl_->iframe_interval);
  }

  if (impl_->set_capture_plane_format() < 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] Failed to set capture plane format");
    return false;
  }

  if (impl_->set_output_plane_format() < 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] Failed to set output plane format");
    return false;
  }

  v4l2_ioctl::set_h264_profile(impl_->dev_fd, impl_->profile);

  if (impl_->is_cuvid) {
    if (v4l2_ioctl::set_cuda_preset(impl_->dev_fd, impl_->hw_preset_type) < 0) {
      RCLCPP_ERROR(
        rclcpp::get_logger("V4L2Encoder"),
        "[V4L2Encoder] Failed to set CUVID hardware preset");
      return false;
    }
    // Insert SPS/PPS before each IDR frame
    v4l2_ioctl::set_insert_sps_pps_at_idr(impl_->dev_fd, true);
    if (impl_->rate_control_mode == 0) {
      v4l2_ioctl::set_h264_qp(impl_->dev_fd, impl_->qp, impl_->qp, impl_->qp);
    }
  } else {
    if (v4l2_ioctl::set_hw_preset_type(impl_->dev_fd, impl_->hw_preset_type) < 0) {
      RCLCPP_ERROR(
        rclcpp::get_logger("V4L2Encoder"),
        "[V4L2Encoder] Failed to set Tegra hardware preset");
      return false;
    }
    v4l2_ioctl::set_entropy_mode(impl_->dev_fd, impl_->entropy == 1);
    v4l2_ioctl::set_bitrate(impl_->dev_fd, impl_->bitrate);
    v4l2_ioctl::set_h264_level(impl_->dev_fd, impl_->level);
    v4l2_ioctl::set_bitrate_mode(impl_->dev_fd, impl_->rate_control_mode);
    v4l2_ioctl::set_idr_interval(impl_->dev_fd, impl_->idr_interval);
    v4l2_ioctl::set_gop_size(impl_->dev_fd, impl_->iframe_interval);
    v4l2_ioctl::set_num_bframes(impl_->dev_fd, config.num_bframes);
    v4l2_ioctl::set_insert_sps_pps_at_idr(impl_->dev_fd, true);

    if (impl_->rate_control_mode == 0) {
      v4l2_ioctl::set_frame_rate_control(impl_->dev_fd, false);
      v4l2_ioctl::set_init_qp(impl_->dev_fd, impl_->qp, impl_->qp, impl_->qp);
    }
  }

  if (impl_->reqbufs_output_plane(impl_->framerate) < 0) {
    RCLCPP_ERROR(

      rclcpp::get_logger("V4L2Encoder"),

      "[V4L2Encoder] Failed to request output plane buffers");
    return false;
  }

  if (impl_->reqbufs_capture_plane() < 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] Failed to request capture plane buffers");
    return false;
  }

  if (v4l2_ioctl::stream_on(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) < 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] Failed to stream on capture plane");
    return false;
  }

  if (v4l2_ioctl::stream_on(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) < 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] Failed to stream on output plane");
    return false;
  }

  if (impl_->enqueue_all_capture_plane_buffers() < 0) {
    RCLCPP_ERROR(
      rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] Failed to enqueue capture plane buffers");
    return false;
  }

  impl_->bitstream_buf_queued = true;
  impl_->eos = false;
  impl_->encoder_thread = std::thread(encoder_thread_func, impl_.get());

  initialized_ = true;
  RCLCPP_INFO(
    rclcpp::get_logger("V4L2Encoder"),
    "[V4L2Encoder] V4L2 Encoder initialized: %ux%u, cuvid=%d",
    config.width, config.height, impl_->is_cuvid);

  return true;
}

void V4L2Encoder::shutdown()
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
    rclcpp::get_logger("V4L2Encoder"),
    "[V4L2Encoder] Shutdown stats: frames_in=%lu, frames_out=%lu, "
    "pending=%lu, dropped=%lu",
    in_count, out_count, pending, dropped_count);

  if (dropped_count > 0) {
    RCLCPP_WARN(
      rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] WARNING: %lu frames were dropped due to metadata queue desync",
      dropped_count);
  }

  impl_->eos = true;
  impl_->bitstream_buf_queued = false;

  if (impl_->dev_fd >= 0) {
    v4l2_ioctl::encoder_stop(impl_->dev_fd);
    v4l2_ioctl::stream_off(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
    v4l2_ioctl::stream_off(impl_->dev_fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  }

  if (impl_->encoder_thread.joinable()) {
    impl_->encoder_thread.join();
  }

  if (impl_->dev_fd >= 0) {
    for (uint32_t i = 0; i < impl_->output_buffer_count; i++) {
      if (impl_->output_buffers[i].buf_surface) {
        NvBufSurface * nvbuf = reinterpret_cast<NvBufSurface *>(
          impl_->output_buffers[i].buf_surface);
        if (!impl_->is_cuvid) {
          NvBufSurfaceUnMap(nvbuf, 0, 0);
          cudaHostUnregister(nvbuf->surfaceList[0].mappedAddr.addr[0]);
          cudaHostUnregister(nvbuf->surfaceList[0].mappedAddr.addr[1]);
        }
        impl_->output_buffers[i].buf_surface = nullptr;
      }
      // Close the dma-buf FD exported via VIDIOC_EXPBUF. Without this each
      // initialize()/shutdown() cycle leaks one descriptor per buffer.
      if (impl_->output_buffers[i].buf_fd >= 0) {
        ::close(impl_->output_buffers[i].buf_fd);
        impl_->output_buffers[i].buf_fd = -1;
      }
    }

    for (uint32_t i = 0; i < impl_->capture_buffer_count; i++) {
      if (impl_->capture_buffers[i].buf_surface) {
        NvBufSurface * nvbuf = reinterpret_cast<NvBufSurface *>(
          impl_->capture_buffers[i].buf_surface);
        if (!impl_->is_cuvid) {
          NvBufSurfaceUnMap(nvbuf, 0, 0);
        }
        impl_->capture_buffers[i].buf_surface = nullptr;
      }
      if (impl_->capture_buffers[i].buf_fd >= 0) {
        ::close(impl_->capture_buffers[i].buf_fd);
        impl_->capture_buffers[i].buf_fd = -1;
      }
    }

    // Free the driver-side buffer allocations before closing the device.
    // VIDIOC_REQBUFS(count=0) releases the V4L2/NVENC encode-session buffers;
    // without it the cuvid driver leaks the session (worker threads + device
    // memory) on every teardown, eventually wedging or aborting the host
    // process when the encoder is loaded/unloaded repeatedly in one container.
    struct v4l2_requestbuffers reqbuf;
    std::memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.count = 0;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    v4l2_ioctl::request_buffers(impl_->dev_fd, &reqbuf);
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    v4l2_ioctl::request_buffers(impl_->dev_fd, &reqbuf);

    v4l2_close(impl_->dev_fd);
    impl_->dev_fd = -1;
  }

  initialized_ = false;
}

bool V4L2Encoder::encode_frame(
  const uint8_t * y_device_ptr,
  const uint8_t * uv_device_ptr,
  uint32_t y_stride,
  uint32_t uv_stride,
  uint64_t timestamp_ns,
  const std::string & frame_id,
  cudaStream_t stream)
{
  if (!initialized_) {
    return false;
  }

  // Serialize encode_frame calls to ensure metadata queue order matches V4L2 queue order.
  std::lock_guard<std::mutex> encode_lock(impl_->encode_mutex);

  static int frame_num = 0;
  frame_num++;

  int q_index = 0;
  if (impl_->output_buffer_idx < impl_->output_buffer_count) {
    q_index = impl_->output_buffer_idx;
    impl_->output_buffer_idx++;
  } else {
    while (impl_->dqbuf_on_output_plane(&q_index) != 0) {
      if (errno == EAGAIN) {
        continue;
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
          "[V4L2Encoder] Error in dqbuf on output plane");
        return false;
      }
    }
  }

  NvBufSurface * nvbuf = reinterpret_cast<NvBufSurface *>(
    impl_->output_buffers[q_index].buf_surface);
  uint8_t * dst_y = nullptr;
  uint8_t * dst_uv = nullptr;

  cudaMemcpyKind memcpy_kind = cudaMemcpyDeviceToDevice;

  uint32_t dst_y_pitch = nvbuf->surfaceList[0].planeParams.pitch[0];
  uint32_t dst_uv_pitch = nvbuf->surfaceList[0].planeParams.pitch[1];

  if (impl_->is_cuvid) {
    dst_y = reinterpret_cast<uint8_t *>(nvbuf->surfaceList[0].dataPtr);
    uint32_t dst_uv_offset = nvbuf->surfaceList[0].planeParams.offset[1];
    dst_uv = dst_y + dst_uv_offset;
  } else {
    cudaError_t cuda_err = cudaHostGetDevicePointer(
      reinterpret_cast<void **>(&dst_y), nvbuf->surfaceList[0].mappedAddr.addr[0], 0);
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
        "[V4L2Encoder] cudaHostGetDevicePointer Y failed: %s", cudaGetErrorString(cuda_err));
      return false;
    }
    cuda_err = cudaHostGetDevicePointer(
      reinterpret_cast<void **>(&dst_uv), nvbuf->surfaceList[0].mappedAddr.addr[1], 0);
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
        "[V4L2Encoder] cudaHostGetDevicePointer UV failed: %s", cudaGetErrorString(cuda_err));
      return false;
    }
  }

  // Copy Y plane
  cudaError_t err = cudaMemcpy2DAsync(
    dst_y, dst_y_pitch,
    y_device_ptr, y_stride,
    impl_->width, impl_->height,
    memcpy_kind, stream);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] cudaMemcpy2DAsync Y failed: %s", cudaGetErrorString(err));
    return false;
  }

  // Copy UV plane - use actual UV stride from input, not Y stride
  err = cudaMemcpy2DAsync(
    dst_uv, dst_uv_pitch,
    uv_device_ptr, uv_stride,
    impl_->width, impl_->height / 2,
    memcpy_kind, stream);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] cudaMemcpy2DAsync UV failed: %s", cudaGetErrorString(err));
    return false;
  }

  // Synchronize before V4L2 QBUF
  // V4L2 driver may access the buffer immediately after QBUF,
  // so we must ensure all copies are complete.
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
    return false;
  }

  // Store metadata in FIFO queue for correlation
  {
    std::lock_guard<std::mutex> lock(impl_->queue_mutex);
    impl_->metadata_queue.push({timestamp_ns, frame_id});
    RCLCPP_DEBUG(rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] PUSH metadata: ts=%lu, queue_size=%zu",
      timestamp_ns, impl_->metadata_queue.size());
  }

  int bytes_used = (3 * impl_->width * impl_->height) >> 1;
  if (impl_->enqueue_output_plane_buffer(q_index, bytes_used) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("V4L2Encoder"),
      "[V4L2Encoder] Failed to enqueue output plane buffer");
    return false;
  }

  impl_->frames_in++;
  return true;
}

EncoderFrameStats V4L2Encoder::get_frame_stats() const
{
  EncoderFrameStats stats;
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

void V4L2Encoder::set_output_callback(EncodedFrameCallback callback)
{
  std::lock_guard<std::mutex> lock(callback_mutex_);
  output_callback_ = std::move(callback);
}

}  // namespace h264_encoder
}  // namespace isaac_ros
}  // namespace nvidia
