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
#include "extensions/codec/components/decoder_request.hpp"

#include <fcntl.h>
#include <poll.h>
#include <semaphore.h>
#include <algorithm>
#include <string>
#include <thread>

#include "gxf/multimedia/camera.hpp"
#include "gxf/std/timestamp.hpp"

#include "NvVideoDecoder.h"
#include "nvbuf_utils.h"
#include "cudaEGL.h"

namespace {
#define CHECK_DECODER_ERROR(ret, error_msg)  \
  if (ret < 0) {                             \
    GXF_LOG_ERROR(error_msg);                \
    return GXF_FAILURE;                      \
  }

#define CHECK_CUDA_ERROR(error_code)           \
  if (error_code != CUDA_SUCCESS) {            \
    const char* error_msg = NULL;              \
    cuGetErrorString(error_code, &error_msg);  \
    GXF_LOG_ERROR(error_msg);                  \
    return -1;                                 \
  }                                            \

constexpr uint32_t kMaxBuffers = 24;
constexpr uint32_t kMaxWaitMilliseconds = 50;
constexpr char kTimestampName[] = "timestamp";

// Struct to store the meta data passed to the decoder
struct nvmpictx {
  // Instance of the video decoder
  NvVideoDecoder* dec;
  // Index of the output plane buffer
  uint32_t index;
  // Input image width
  uint32_t width;
  // Input image height
  uint32_t height;
  // Input iamge format, support nv24 and nv12
  uint32_t output_pixfmt;
  // Output format, only support V4L2_PIX_FMT_H264 yet
  uint32_t decoder_pixfmt{0};
  // Max number of buffers used to hold the output plane data
  uint32_t num_buffers;
  // File descriptor of destination DMA buffer
  int dst_dma_fd;
  // EOS of resolution change event
  bool got_res_event;
  // Number of completed requestes pending for DQ
  // Use atomic to guarantee the thread saftey
  std::atomic<int> request_complete_num;
  // Polling thread waits on this to be signalled to issue Poll
  sem_t pollthread_sema;
  // Polling thread, running in non-blocking mode.
  pthread_t  dec_pollthread;
  // Counter mutex
  std::mutex mtx;

  nvidia::gxf::Handle<nvidia::gxf::Allocator> pool;
  nvidia::gxf::Handle<nvidia::gxf::AsynchronousSchedulingTerm> scheduling_term;
  gxf_context_t gxf_context;
};

int copyDMAbufToCUDA(int dmabuf_fd, nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> decoded_frame) {
  auto egl_frame = NvEGLImageFromFd(NULL, dmabuf_fd);

  CUresult status;
  CUeglFrame egl_mapped_frame;
  CUgraphicsResource p_resource = NULL;

  CHECK_CUDA_ERROR(cuGraphicsEGLRegisterImage(&p_resource, egl_frame,
              CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE))

  CHECK_CUDA_ERROR(cuGraphicsResourceGetMappedEglFrame(&egl_mapped_frame, p_resource, 0, 0))

  CHECK_CUDA_ERROR(cuCtxSynchronize())

  if (egl_mapped_frame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
    GXF_LOG_ERROR("The egl frame type is not pitch linear");
    return -1;
  }

  // copy data from egl to output entity
  CUDA_MEMCPY2D m = { 0 };
  auto decoded_frame_info = decoded_frame->video_frame_info();

  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.srcDevice = (CUdeviceptr) egl_mapped_frame.frame.pArray[0];
  m.srcPitch = egl_mapped_frame.pitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = (CUdeviceptr)(decoded_frame->pointer());
  m.dstPitch = decoded_frame_info.color_planes[0].stride;
  m.WidthInBytes = decoded_frame_info.color_planes[0].width *
                   decoded_frame_info.color_planes[0].bytes_per_pixel;
  m.Height = decoded_frame_info.color_planes[0].height;
  CHECK_CUDA_ERROR(cuMemcpy2D(&m))

  m.srcDevice = (CUdeviceptr) egl_mapped_frame.frame.pArray[1];
  m.dstDevice = (CUdeviceptr)(decoded_frame->pointer() +
                 decoded_frame_info.color_planes[0].size);
  m.WidthInBytes = decoded_frame_info.color_planes[1].width *
                   decoded_frame_info.color_planes[1].bytes_per_pixel;
  m.Height = decoded_frame_info.color_planes[1].height;
  CHECK_CUDA_ERROR(cuMemcpy2D(&m))

  CHECK_CUDA_ERROR(cuGraphicsUnregisterResource(p_resource))

  NvDestroyEGLImage(NULL, egl_frame);
  return 1;
}

static int setCapturePlane(nvmpictx* ctx) {
  NvVideoDecoder* dec = ctx->dec;
  NvBufferCreateParams dst_params = {0};
  struct v4l2_format format;
  int32_t min_dec_capture_buffers;
  int ret = 0;

  ret = dec->capture_plane.getFormat(format);
  dec->capture_plane.deinitPlane();

  dst_params.width = ctx->width;;
  dst_params.height = ctx->height;
  dst_params.layout = NvBufferLayout_Pitch;
  dst_params.payloadType = NvBufferPayload_SurfArray;
  dst_params.nvbuf_tag = NvBufferTag_VIDEO_DEC;
  dst_params.colorFormat = NvBufferColorFormat_NV12;
  ret = NvBufferCreateEx(&ctx->dst_dma_fd, &dst_params);

  ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                   format.fmt.pix_mp.width,
                                   format.fmt.pix_mp.height);

  ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);

  ret = dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                      min_dec_capture_buffers + 5, false,
                                      false);

  // Capture plane STREAMON
  ret = dec->capture_plane.setStreamStatus(true);
  // Enqueue all the empty capture plane buffers
  for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_buf.memory = V4L2_MEMORY_MMAP;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
  }
  ctx->got_res_event = true;
  return ret;
}

// Callback function to pool device to check if there is any data ready
static void * decoder_pollthread_fcn(void* arg) {
  nvmpictx* ctx = reinterpret_cast<nvmpictx*>(arg);
  v4l2_ctrl_video_device_poll devicepoll;
  memset(&devicepoll, 0, sizeof(v4l2_ctrl_video_device_poll));

  // Wait here until you are signalled to issue the Poll call.
  // Check if the abort status is set , if so exit
  // Else issue the Poll on the decoder and block.
  // When the Poll returns, signal the decoder thread to continue.
  while (!ctx->dec->isInError()) {
      sem_wait(&ctx->pollthread_sema);
      devicepoll.req_events = POLLIN | POLLPRI;
      // This call shall wait in the v4l2 decoder library
      ctx->dec->DevicePoll(&devicepoll);

      std::lock_guard<std::mutex> lock(ctx->mtx);
      if (ctx->scheduling_term->getEventState() ==
          nvidia::gxf::AsynchronousEventState::EVENT_DONE) {
         GXF_LOG_DEBUG("Decoder Async Event is unexpectedly already marked DONE");
      }
      ctx->scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_DONE);
      ctx->request_complete_num++;
  }
  return NULL;
}
}  // namespace

namespace nvidia {
namespace isaac {

struct DecoderRequest::Impl {
  nvmpictx* ctx;
};

DecoderRequest::DecoderRequest() {}

DecoderRequest::~DecoderRequest() {}

gxf_result_t DecoderRequest::registerInterface(gxf::Registrar * registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      image_receiver_, "image_receiver",
      "Receiver to get the input image");
  result &= registrar->parameter(
      pool_, "pool",
      "Memory pool for allocating output data");
  result &= registrar->parameter(
      input_height_, "input_height",
      "Input image height");
  result &= registrar->parameter(
      input_width_, "input_width",
      "Input image width");
  result &= registrar->parameter(scheduling_term_, "async_scheduling_term",
      "Asynchronous Scheduling Term", "Asynchronous Scheduling Term");

  return gxf::ToResultCode(result);
}

gxf_result_t DecoderRequest::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  if (impl_ == nullptr) {
    return GXF_OUT_OF_MEMORY;
  }
  scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
  available_buffer_num_ = kMaxBuffers;
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::deinitialize() {
  impl_.reset();
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::createDefaultDecoderContext() {
  impl_->ctx = new nvmpictx;
  impl_->ctx->index = 0;
  impl_->ctx->width = input_width_;
  impl_->ctx->height = input_height_;
  impl_->ctx->decoder_pixfmt = V4L2_PIX_FMT_H264;
  impl_->ctx->output_pixfmt = V4L2_PIX_FMT_NV12M;
  impl_->ctx->num_buffers = kMaxBuffers;
  impl_->ctx->gxf_context = context();
  impl_->ctx->request_complete_num = 0;
  impl_->ctx->got_res_event = false;
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::initializeDecoder() {
  // Specify the maximum buffer size of the capture plane
  int chunk_size = input_width_ * input_height_ * 3;

  CHECK_DECODER_ERROR(impl_->ctx->dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0),
                      "Failed to subscribe resolution change event")

  CHECK_DECODER_ERROR(impl_->ctx->dec->setOutputPlaneFormat(impl_->ctx->decoder_pixfmt,
                                                            chunk_size),
                      "Failed to set output plane format")

  CHECK_DECODER_ERROR(impl_->ctx->dec->setFrameInputMode(0),
                      "Failed to set frame input mode")

  CHECK_DECODER_ERROR(impl_->ctx->dec->disableDPB(),
                      "Failed to disable DPB")
  
  CHECK_DECODER_ERROR(impl_->ctx->dec->setMaxPerfMode(true),
                      "Failed to set Max performance mode")

  CHECK_DECODER_ERROR(impl_->ctx->dec->output_plane.setupPlane(V4L2_MEMORY_USERPTR,
                                               impl_->ctx->num_buffers,
                                               false,
                                               true),
                      "Failed to setup output plane")

  CHECK_DECODER_ERROR(impl_->ctx->dec->output_plane.setStreamStatus(true),
                      "Failed to set output plane stream status")


  sem_init(&impl_->ctx->pollthread_sema, 0, 0);
  pthread_create(&impl_->ctx->dec_pollthread, NULL, decoder_pollthread_fcn, impl_->ctx);
  // Capture plane and output plane should have the same number of buffers
  // The number of allocated buffer could be differ from the provided parameter
  available_buffer_num_ = impl_->ctx->dec->output_plane.getNumBuffers();
  total_buffer_num_ = available_buffer_num_;
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::prepareOutputEntity(const gxf::Entity & input_msg) {
  auto output_entity = gxf::Entity::New(impl_->ctx->gxf_context);
  if (!output_entity) {
    GXF_LOG_ERROR("Failed to create output message");
    return output_entity.error();
  }

  auto maybe_input_intrinsics = input_msg.get<gxf::CameraModel>("intrinsics");
  if (maybe_input_intrinsics) {
    auto output_intrinsics = output_entity.value().add<gxf::CameraModel>("intrinsics");
    if (!output_intrinsics) {
      GXF_LOG_ERROR("Failed add intrinsics into output");
      return output_intrinsics.error();
    }
    *output_intrinsics.value() = *maybe_input_intrinsics.value();
  }

  auto maybe_input_extrinsics = input_msg.get<gxf::Pose3D>("extrinsics");
  if (maybe_input_extrinsics) {
    auto output_extrinsics = output_entity.value().add<gxf::Pose3D>("extrinsics");
    if (!output_extrinsics) {
      GXF_LOG_ERROR("Failed add extrinsics into output");
      return output_extrinsics.error();
    }
    *output_extrinsics.value() = *maybe_input_extrinsics.value();
  }

  auto maybe_input_seq_number = input_msg.get<int64_t>("sequence_number");
  if (maybe_input_seq_number) {
    auto output_seq_number = output_entity.value().add<int64_t>("sequence_number");
    if (!output_seq_number) {
      GXF_LOG_ERROR("Failed add sequence number into output");
      return output_seq_number.error();
    }
    *output_seq_number.value() = *maybe_input_seq_number.value();
  }

  auto maybe_input_image = input_msg.get<gxf::Tensor>();
  if (!maybe_input_image) {
    GXF_LOG_ERROR("Failed to get input tensor");
    return gxf::ToResultCode(maybe_input_image);
  }
  auto maybe_output_image = output_entity->add<gxf::VideoBuffer>(maybe_input_image->name());
  if (!maybe_output_image) {
    GXF_LOG_ERROR("Failed to add output videobuffer");
    return gxf::ToResultCode(maybe_output_image);
  }
  output_entity_queue_.push(output_entity.value());
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::start() {
  auto gxf_ret_code = createDefaultDecoderContext();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create default decoder context");
    return gxf_ret_code;
  }

  if (!(impl_->ctx->dec = NvVideoDecoder::createVideoDecoder("dec0", O_NONBLOCK))) {
    GXF_LOG_ERROR("Failed to create decoder");
    return GXF_FAILURE;
  }


  gxf_ret_code = initializeDecoder();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to initialize decoder");
    return gxf_ret_code;
  }

  cudaStreamCreate(&cuda_stream_);
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::tick() {
  // Get input image
  auto maybe_input_message = image_receiver_->receive();
  if (!maybe_input_message) {
    GXF_LOG_ERROR("Failed to receive input message");
    return maybe_input_message.error();
  }

  impl_->ctx->pool = pool_.get();
  impl_->ctx->scheduling_term = scheduling_term_.get();

  auto gxf_ret_code = prepareOutputEntity(maybe_input_message.value());
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed create output entity");
    return gxf_ret_code;
  }

  gxf_ret_code = placeInputImage(maybe_input_message.value());
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to place input image into v4l2 buffer");
    return gxf_ret_code;
  }

  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::stop() {
  sem_destroy(&impl_->ctx->pollthread_sema);
  cudaStreamDestroy(cuda_stream_);
  impl_->ctx->dec->capture_plane.stopDQThread();
  impl_->ctx->dec->capture_plane.waitForDQThread(1000);
  impl_.reset();
  return GXF_SUCCESS;
}

gxf_result_t DecoderRequest::placeInputImage(const gxf::Entity & input_msg) {
  impl_->ctx->dec->SetPollInterrupt();
  auto input_img = input_msg.get<gxf::Tensor>();
  if (!input_img) {
    GXF_LOG_ERROR("Failed to get image from message");
    return input_img.error();
  }

  // Get timestamp from input image
  auto maybe_timestamp = input_msg.get<gxf::Timestamp>(kTimestampName);
  if (!maybe_timestamp) {
    maybe_timestamp = input_msg.get<gxf::Timestamp>("");
  }
  if (!maybe_timestamp) {return maybe_timestamp.error();}

  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer *nvBuffer;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));
  v4l2_buf.m.planes = planes;

  if (impl_->ctx->dec->isInError()) {
    GXF_LOG_ERROR("ctx DEC is in error");
    return GXF_FAILURE;
  }

  if (impl_->ctx->index < total_buffer_num_) {
    nvBuffer = impl_->ctx->dec->output_plane.getNthBuffer(impl_->ctx->index);
    v4l2_buf.index = impl_->ctx->index;
    impl_->ctx->index++;
  } else {
    // Output plane dqBuffer should always succeed
    // as we have a sheduling term to ensure there are avaible buffers
    CHECK_DECODER_ERROR(impl_->ctx->dec->output_plane.dqBuffer(v4l2_buf, &nvBuffer, NULL, -1),
                        "Failed to dequeue output plane buffer")
  }
  memcpy(nvBuffer->planes[0].data, input_img.value()->pointer(), input_img.value()->size());
  nvBuffer->planes[0].bytesused = input_img.value()->size();

  // Add timestamp into v4l2 buffer
  v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
  v4l2_buf.timestamp.tv_usec = static_cast<uint64_t>(
      maybe_timestamp.value()->acqtime % static_cast<uint64_t>(1e6));
  v4l2_buf.timestamp.tv_sec = static_cast<uint64_t>(
      maybe_timestamp.value()->acqtime / static_cast<uint64_t>(1e6));

  impl_->ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);

  std::lock_guard<std::mutex> lock(impl_->ctx->mtx);
  available_buffer_num_ -= 1;
  if (scheduling_term_->getEventState() == nvidia::gxf::AsynchronousEventState::WAIT) {
    scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_WAITING);
  }
  sem_post(&impl_->ctx->pollthread_sema);
  return GXF_SUCCESS;
}

nvidia::gxf::Expected<nvidia::gxf::Entity> DecoderRequest::hasDQCapturePlane() {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane capture_planes[MAX_PLANES];
  struct v4l2_event ev;
  NvBuffer* buffer = NULL;
  int ret = 0;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(capture_planes, 0, sizeof(capture_planes));
  v4l2_buf.m.planes = capture_planes;

  impl_->ctx->dec->dqEvent(ev, impl_->ctx->got_res_event ? 0 : 500);
  if (ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
    ret = setCapturePlane(impl_->ctx);
    if (ret < 0) {
      GXF_LOG_ERROR("Failed to set capture plane");
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }
    // Ensure the first frame is decoded by the hardware
    ret = impl_->ctx->dec->capture_plane.dqBuffer(v4l2_buf, &buffer, NULL, 0);
    if (ret < 0) {
      // Record start time for timeout checking
      const auto start = std::chrono::system_clock::now();
      while (errno == EAGAIN && 
        (std::chrono::system_clock::now() - start) < std::chrono::milliseconds(kMaxWaitMilliseconds )) {
        impl_->ctx->dec->capture_plane.dqBuffer(v4l2_buf, &buffer, NULL, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (errno == EAGAIN) {
        GXF_LOG_ERROR("Failed to receive the first decoded frame");
        return nvidia::gxf::Unexpected{GXF_FAILURE};
      }
    }
  } else {
    ret = impl_->ctx->dec->capture_plane.dqBuffer(v4l2_buf, &buffer, NULL, 0);
    if (ret < 0) {
      if (errno == EAGAIN) {
        // Device poll is returned due to interupt or other events
        // No buffer to DQ from capture plane
        std::lock_guard<std::mutex> lock(impl_->ctx->mtx);
        impl_->ctx->request_complete_num -= 1;
        if (impl_->ctx->request_complete_num <= 0) {
          if (available_buffer_num_ < total_buffer_num_) {
            // Have requests on the fly, set event back to waiting
            scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_WAITING);
          } else {
            // No pending results to publish, set event back to wait
            scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
          }
        }
        sem_post(&impl_->ctx->pollthread_sema);
        GXF_LOG_DEBUG("No capture plane is ready to dequeue");
        return nvidia::gxf::Unexpected(GXF_QUERY_NOT_FOUND);
      }
      GXF_LOG_ERROR("Failed to dequeue buffer from capture plane");
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }
  }

  if (buffer->planes[0].bytesused == 0) {
    GXF_LOG_ERROR("Buffer from capture plane is empty");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  NvBufferRect src_rect, dest_rect;
  src_rect.top = 0;
  src_rect.left = 0;
  src_rect.width = impl_->ctx->width;
  src_rect.height = impl_->ctx->height;
  dest_rect.top = 0;
  dest_rect.left = 0;
  dest_rect.width = impl_->ctx->width;
  dest_rect.height = impl_->ctx->height;

  NvBufferTransformParams transform_params;
  memset(&transform_params, 0, sizeof(transform_params));
  transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
  transform_params.transform_flip = NvBufferTransform_None;
  transform_params.transform_filter = NvBufferTransform_Filter_Smart;
  transform_params.src_rect = src_rect;
  transform_params.dst_rect = dest_rect;

  // Transform block linear to pitch linear
  ret = NvBufferTransform(buffer->planes[0].fd, impl_->ctx->dst_dma_fd, &transform_params);
  if (ret < 0) {
    GXF_LOG_ERROR("Failed to transform block linear to pitch linear");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  // Get output message
  auto output_entity = output_entity_queue_.front();
  output_entity_queue_.pop();

  auto output_image = output_entity.get<nvidia::gxf::VideoBuffer>();
  if (!output_image) {
    GXF_LOG_ERROR("Failed to add output tensor");
    return nvidia::gxf::Unexpected{output_image.error()};
  }

  constexpr auto surface_layout = gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;
  constexpr auto storage_type = gxf::MemoryStorageType::kDevice;
  auto gxf_ret_code = output_image.value()->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12_ER>(
                      input_width_, input_height_, surface_layout, storage_type, impl_->ctx->pool);
  if (!gxf_ret_code) {
    GXF_LOG_ERROR("Failed to resize output video buffer");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  // Copy result from DMA buffer to CUDA buffer
  ret = copyDMAbufToCUDA(impl_->ctx->dst_dma_fd, output_image.value());
  if (ret < 0) {
    GXF_LOG_ERROR("Failed to copy data from DMA buffer to CUDA buffer");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  // Add timestamp into output message
  uint64_t input_timestamp = v4l2_buf.timestamp.tv_sec * static_cast<uint64_t>(1e6) +
                             v4l2_buf.timestamp.tv_usec;
  auto out_timestamp = output_entity.add<nvidia::gxf::Timestamp>(kTimestampName);
  out_timestamp.value()->acqtime = input_timestamp;

  // Enqueue empty buffer to capture plane
  if (impl_->ctx->dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
    GXF_LOG_ERROR("Failed to enqueue capture plane");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  std::lock_guard<std::mutex> lock(impl_->ctx->mtx);
  available_buffer_num_+=1;
  impl_->ctx->request_complete_num -= 1;
  if (impl_->ctx->request_complete_num <= 0) {
    if (available_buffer_num_ < total_buffer_num_) {
      // Have requests on the fly, set event back to waiting
      scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_WAITING);
    } else {
      // No pending results to publish, set event back to wait
      scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
    }
  }
  return output_entity;
}

nvidia::gxf::Expected<bool> DecoderRequest::isAcceptingRequest() {
  return available_buffer_num_ > 0;
}

}  // namespace isaac
}  // namespace nvidia
