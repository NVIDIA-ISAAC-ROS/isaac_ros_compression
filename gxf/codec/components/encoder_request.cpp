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
#include "extensions/codec/components/encoder_request.hpp"

#include <fcntl.h>
#include <poll.h>
#include <semaphore.h>
#include <algorithm>
#include <string>
#include <thread>

#include "gxf/multimedia/camera.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/timestamp.hpp"

#include "NvVideoEncoder.h"

namespace {
#define CHECK_ENCODER_ERROR(ret, error_msg)  \
  if (ret < 0) {                             \
    GXF_LOG_ERROR(error_msg);                \
    return GXF_FAILURE;                      \
  }
constexpr uint32_t kMaxBuffers = 24;
constexpr char kTimestampName[] = "timestamp";

// Struct to store the meta data passed to the encoder
struct nvmpictx {
  // Instance of the video encoder
  NvVideoEncoder* enc;
  // Index of the output plane buffer
  uint32_t index;
  // Input image width
  uint32_t width;
  // Input image height
  uint32_t height;
  // Profile to be used for encoding
  uint32_t profile;
  // Encoder constant QP value
  uint32_t qp;
  // Input iamge format, support nv24 and nv12
  uint32_t raw_pixfmt;
  // Output format, only support V4L2_PIX_FMT_H264 yet
  uint32_t encoder_pixfmt;
  // Maximum data rate and resolution, select from 0 to 14
  enum v4l2_mpeg_video_h264_level level;
  // Encode hw preset type, select from 0 to 3
  enum v4l2_enc_hw_preset_type hw_preset_type;
  // Entropy encoding, 0 represents CAVLC, 1 represents CABAC
  uint32_t entropy;
  // Interval between two I frames, in number of frames
  // E.g., interval=5 means IPPPPI, interval=1 means I frame only
  uint32_t iframe_interval;
  // Interval between two IDR frames, in number of frames
  uint32_t idr_interval;
  // Number of B frames in a GOP
  uint32_t num_of_bframes;
  // Max number of buffers used to hold the output plane data
  uint32_t num_buffers;
  // Encoded data size
  uint32_t packet_buf_size;
  // Number of completed requestes pending for DQ
  // Use atomic to guarantee the thread saftey
  std::atomic<int> request_complete_num;
  // Polling thread waits on this to be signaled to issue Poll
  sem_t pollthread_sema;
  // Polling thread, running in non-blocking mode.
  std::thread enc_pollthread;
  // Counter mutex
  std::mutex mtx;

  nvidia::gxf::Handle<nvidia::gxf::Allocator> pool;
  nvidia::gxf::Handle<nvidia::gxf::AsynchronousSchedulingTerm> scheduling_term;
  gxf_context_t gxf_context;
};

// Callback function to pool device to check if there is any data ready
static void * encoder_pollthread_fcn(void* arg) {
  nvmpictx* ctx = reinterpret_cast<nvmpictx*>(arg);
  v4l2_ctrl_video_device_poll devicepoll;
  memset(&devicepoll, 0, sizeof(v4l2_ctrl_video_device_poll));

  // Wait here until you are signalled to issue the Poll call.
  // Check if the abort status is set , if so exit
  // Else issue the Poll on the encoder and block.
  // When the Poll returns, signal the encoder thread to continue.
  while (!ctx->enc->isInError()) {
      sem_wait(&ctx->pollthread_sema);
      devicepoll.req_events = POLLIN | POLLPRI;
      // This call shall wait in the v4l2 encoder library
      ctx->enc->DevicePoll(&devicepoll);

      std::lock_guard<std::mutex> lock(ctx->mtx);
      if (ctx->scheduling_term->getEventState() ==
          nvidia::gxf::AsynchronousEventState::EVENT_DONE) {
         GXF_LOG_DEBUG("Encoder Async Event is unexpectedly already marked DONE");
      }
      ctx->scheduling_term->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_DONE);
      ctx->request_complete_num++;
  }
  return NULL;
}
}  // namespace

namespace nvidia {
namespace isaac {

struct EncoderRequest::Impl {
  nvmpictx* ctx;
};

EncoderRequest::EncoderRequest() {}

EncoderRequest::~EncoderRequest() {}

gxf_result_t EncoderRequest::registerInterface(gxf::Registrar * registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(
      image_receiver_, "image_receiver",
      "Receiver to get the input image");
  result &= registrar->parameter(
      pool_, "pool",
      "Memory pool for allocating output data");
  result &= registrar->parameter(
      profile_, "profile",
      "Encode profile");
  result &= registrar->parameter(
      hw_preset_type_, "hw_preset_type",
      "Encode hw preset type, select from 0 to 3");
  result &= registrar->parameter(
      level_, "level", "Video H264 level",
      "Maximum data rate and resolution, select from 0 to 14",
      14);
  result &= registrar->parameter(
      entropy_, "entropy", "Entropy Encoding",
      "Entropy Encoding, select from 0(CAVLC)), 1(CABAC)",
      0);
    result &= registrar->parameter(
      iframe_interval_, "iframe_interval", "I Frame Interval",
      "Interval between two I frames",
      5);
  result &= registrar->parameter(
      input_height_, "input_height",
      "Input image height");
  result &= registrar->parameter(
      input_width_, "input_width",
      "Input image width");
  result &= registrar->parameter(
      qp_, "qp",
      "Encoder constant QP value");
  result &= registrar->parameter(
      input_format_, "input_format",
      "Input iamge format, select from nv24, nv12",
      "nv24");
  result &= registrar->parameter(
      config_, "config",
      "Preset of parameters, select from pframe, iframe, custom",
      "pframe");
  result &= registrar->parameter(scheduling_term_, "async_scheduling_term",
      "Asynchronous Scheduling Term", "Asynchronous Scheduling Term");

  return gxf::ToResultCode(result);
}

gxf_result_t EncoderRequest::initialize() {
  impl_ = MakeUniqueNoThrow<Impl>();
  if (impl_ == nullptr) {
    return GXF_OUT_OF_MEMORY;
  }
  scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
  available_buffer_num_ = kMaxBuffers;
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::deinitialize() {
  impl_.reset();
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::createDefaultEncoderContext() {
  impl_->ctx = new nvmpictx;
  impl_->ctx->index = 0;
  impl_->ctx->width = input_width_;
  impl_->ctx->height = input_height_;
  // Not using B frames
  impl_->ctx->num_of_bframes = 0;
  impl_->ctx->encoder_pixfmt = V4L2_PIX_FMT_H264;
  impl_->ctx->num_buffers = kMaxBuffers;
  impl_->ctx->gxf_context = context();
  impl_->ctx->request_complete_num = 0;
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::getInputFormatFromParameter(const gxf::EncoderInputFormat& input_format) {
  if (input_format == gxf::EncoderInputFormat::kNV24) {
    impl_->ctx->raw_pixfmt = V4L2_PIX_FMT_NV24M;
  } else if (input_format == gxf::EncoderInputFormat::kNV12) {
    impl_->ctx->raw_pixfmt = V4L2_PIX_FMT_NV12M;
  } else {
    GXF_LOG_ERROR("The input format is not supported");
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::getCompressionSettingsFromParameter(const gxf::EncoderConfig& config) {
  if (config == gxf::EncoderConfig::kPFrame) {
    impl_->ctx->qp = 20;
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
    impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
    impl_->ctx->entropy = 0;
    impl_->ctx->iframe_interval = 5;
    impl_->ctx->idr_interval = 5;
  } else if (config == gxf::EncoderConfig::kIFrame) {
    impl_->ctx->qp = 20;
    impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
    impl_->ctx->hw_preset_type = V4L2_ENC_HW_PRESET_ULTRAFAST;
    impl_->ctx->entropy = 0;
    impl_->ctx->iframe_interval = 1;
    impl_->ctx->idr_interval = 1;
  } else {
    CHECK_ENCODER_ERROR(qp_, "QP value is invalid, please use non-negative integers")
    CHECK_ENCODER_ERROR(entropy_, "Entropy value is invalide, please select from 0 and 1")
    CHECK_ENCODER_ERROR(iframe_interval_,
                        "I frame interval is invalid, please use non-negative integers")
    impl_->ctx->qp = qp_;
    impl_->ctx->entropy = entropy_;
    impl_->ctx->iframe_interval = iframe_interval_;
    impl_->ctx->idr_interval = iframe_interval_;
    switch (profile_) {
      case 0:
        impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_MAIN;
        break;
      case 1:
        impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
        break;
      case 2:
        impl_->ctx->profile = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH;
        break;
      default:
        GXF_LOG_ERROR("The profile is not supported");
        return GXF_FAILURE;
      }
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
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::getCompressionLevelFromParameter(const int& level) {
    switch (level) {
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

gxf_result_t EncoderRequest::initializeEncoder() {
  // Specify the maximum buffer size of the capture plane
  int chunk_size = input_width_ * input_height_ * 3;

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setCapturePlaneFormat(impl_->ctx->encoder_pixfmt,
                                                             impl_->ctx->width,
                                                             impl_->ctx->height,
                                                             chunk_size),
                      "Failed to set capture plane format")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setOutputPlaneFormat(impl_->ctx->raw_pixfmt,
                                                            impl_->ctx->width,
                                                            impl_->ctx->height),
                      "Failed to set output plane format")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setHWPresetType(impl_->ctx->hw_preset_type),
                      "Failed to set hw preset type")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setProfile(impl_->ctx->profile),
                      "Failed to set encoder profile")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setLevel(impl_->ctx->level),
                      "Failed to set h264 video level")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setCABAC(impl_->ctx->entropy),
                      "Failed to set entropy encoding")

  // Disable rate control mode
  CHECK_ENCODER_ERROR(impl_->ctx->enc->setConstantQp(false),
                      "Failed to set disable rate control mode")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setInitQP(impl_->ctx->qp, impl_->ctx->qp, impl_->ctx->qp),
                      "Failed to set QP value")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setIDRInterval(impl_->ctx->idr_interval),
                      "Failed to set IDR interval")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setIFrameInterval(impl_->ctx->iframe_interval),
                      "Failed to set IFrame Interval")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setNumBFrames(impl_->ctx->num_of_bframes),
                      "Failed to set number of B Frames")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setInsertSpsPpsAtIdrEnabled(true),
                      "Failed to set encoder SPSPPS at IDR")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->setMaxPerfMode(true),
                      "Failed to set Max performance mode")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->output_plane.setupPlane(V4L2_MEMORY_USERPTR,
                                               impl_->ctx->num_buffers,
                                               false,
                                               true),
                      "Failed to setup output plane")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                                impl_->ctx->num_buffers,
                                                true,
                                                false),
                      "Failed to setup capture plane")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->subscribeEvent(V4L2_EVENT_EOS, 0, 0),
                      "Failed to create encoder")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->output_plane.setStreamStatus(true),
                      "Failed to set output plane stream status")

  CHECK_ENCODER_ERROR(impl_->ctx->enc->capture_plane.setStreamStatus(true),
                      "Failed to set capture plane stream status")

  sem_init(&impl_->ctx->pollthread_sema, 0, 0);
  impl_->ctx->enc_pollthread = std::thread(encoder_pollthread_fcn, impl_->ctx);
  impl_->ctx->enc_pollthread.detach();
  // The number of allocated buffer could be differ from the provided parameter
  available_buffer_num_ = impl_->ctx->enc->output_plane.getNumBuffers();
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::createBuffers() {
  for (uint32_t i = 0; i < impl_->ctx->enc->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    NvBuffer* nvBuffer = impl_->ctx->enc->capture_plane.getNthBuffer(i);
    cudaHostRegister(nvBuffer->planes[0].data, nvBuffer->planes[0].length, cudaHostAllocDefault);
    CHECK_ENCODER_ERROR(impl_->ctx->enc->capture_plane.qBuffer(v4l2_buf, NULL),
                        "Failed to enqueue capture plane buffer")
  }

  for (uint32_t i = 0; i < impl_->ctx->enc->output_plane.getNumBuffers(); i++) {
    NvBuffer* nvBuffer = impl_->ctx->enc->output_plane.getNthBuffer(i);
    cudaHostRegister(nvBuffer->planes[0].data, nvBuffer->planes[0].length, cudaHostAllocDefault);
    cudaHostRegister(nvBuffer->planes[1].data, nvBuffer->planes[1].length, cudaHostAllocDefault);
  }
  return GXF_SUCCESS;
}


gxf_result_t EncoderRequest::prepareOutputEntity(const gxf::Entity & input_msg) {
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

  output_entity_queue_.push(output_entity.value());
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::start() {
  auto gxf_ret_code = createDefaultEncoderContext();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create default encoder context");
    return gxf_ret_code;
  }

  gxf_ret_code = getInputFormatFromParameter(input_format_.get());
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get input format from parameter");
    return gxf_ret_code;
  }

  gxf_ret_code = getCompressionSettingsFromParameter(config_.get());
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get compression settings from parameter");
    return gxf_ret_code;
  }

  gxf_ret_code = getCompressionLevelFromParameter(level_);
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to get compression level from parameter");
    return gxf_ret_code;
  }

  if (!(impl_->ctx->enc = NvVideoEncoder::createVideoEncoder("enc0", O_NONBLOCK))) {
    GXF_LOG_ERROR("Failed to create encoder");
    return GXF_FAILURE;
  }

  gxf_ret_code = initializeEncoder();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to initialize encoder");
    return gxf_ret_code;
  }

  gxf_ret_code = createBuffers();
  if (gxf_ret_code != GXF_SUCCESS) {
    GXF_LOG_ERROR("Failed to create v4l2 buffers");
    return gxf_ret_code;
  }

  cudaStreamCreate(&cuda_stream_);
  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::tick() {
  // Get input image
  auto maybe_input_message = image_receiver_->receive();
  if (!maybe_input_message) {
    GXF_LOG_ERROR("Failed to receive input message");
    return gxf::ToResultCode(maybe_input_message);
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

gxf_result_t EncoderRequest::stop() {
  // Unregister the page-locked memory
  for (uint32_t i = 0; i < impl_->ctx->enc->output_plane.getNumBuffers(); i++) {
    NvBuffer* nvBuffer = impl_->ctx->enc->output_plane.getNthBuffer(i);
    cudaHostUnregister(nvBuffer->planes[0].data);
    cudaHostUnregister(nvBuffer->planes[1].data);
  }
  for (uint32_t i = 0; i < impl_->ctx->enc->capture_plane.getNumBuffers(); i++) {
    NvBuffer* nvBuffer = impl_->ctx->enc->capture_plane.getNthBuffer(i);
    cudaHostUnregister(nvBuffer->planes[0].data);
    cudaHostUnregister(nvBuffer->planes[1].data);
  }

  sem_destroy(&impl_->ctx->pollthread_sema);
  impl_->ctx->enc->capture_plane.stopDQThread();
  impl_->ctx->enc->capture_plane.waitForDQThread(1000);
  impl_.reset();

  return GXF_SUCCESS;
}

gxf_result_t EncoderRequest::placeInputImage(const gxf::Entity & input_msg) {
  impl_->ctx->enc->SetPollInterrupt();
  auto input_img = input_msg.get<gxf::VideoBuffer>();
  if (!input_img) {
    GXF_LOG_ERROR("Failed to get image from message");
    return gxf::ToResultCode(input_img);
  }

  // Get timestamp from input image
  auto maybe_timestamp = input_msg.get<gxf::Timestamp>(kTimestampName);
  if (!maybe_timestamp) {
    maybe_timestamp = input_msg.get<gxf::Timestamp>("");
  }
  if (!maybe_timestamp) {return gxf::ToResultCode(maybe_timestamp);}

  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer *nvBuffer;
  cudaError_t result;
  auto input_img_info = input_img.value()->video_frame_info();

  if (input_img_info.color_planes[0].width != input_width_ ||
     input_img_info.color_planes[0].height != input_height_) {
    GXF_LOG_ERROR("The preset dimension and input image does not match." \
      "input width: %d, input height: %d, preset width: %u, preset height: %u",
      input_img_info.color_planes[0].width, input_img_info.color_planes[0].height,
      input_width_.get(), input_height_.get());
    return GXF_FAILURE;
  }

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));

  v4l2_buf.m.planes = planes;

  if (impl_->ctx->enc->isInError()) {
    GXF_LOG_ERROR("ctx ENC is in error");
    return GXF_FAILURE;
  }

  if (impl_->ctx->index < impl_->ctx->enc->output_plane.getNumBuffers()) {
    nvBuffer = impl_->ctx->enc->output_plane.getNthBuffer(impl_->ctx->index);
    v4l2_buf.index = impl_->ctx->index;
    impl_->ctx->index++;
  } else {
    // Output plane dqBuffer should always succeed
    // as we have a sheduling term to ensure there are avaible buffers
    CHECK_ENCODER_ERROR(impl_->ctx->enc->output_plane.dqBuffer(v4l2_buf, &nvBuffer, NULL, -1),
                        "Failed to dequeue output plane buffer")
  }

  unsigned char *nvbuffer_plane_0;
  unsigned char *nvbuffer_plane_1;
  cudaHostGetDevicePointer(&nvbuffer_plane_0, nvBuffer->planes[0].data, 0);
  cudaHostGetDevicePointer(&nvbuffer_plane_1, nvBuffer->planes[1].data, 0);

  result = cudaMemcpy2DAsync(
      nvbuffer_plane_0,
      input_img_info.color_planes[0].width * input_img_info.color_planes[0].bytes_per_pixel,
      input_img.value()->pointer(),
      input_img_info.color_planes[0].stride,
      input_img_info.color_planes[0].width * input_img_info.color_planes[0].bytes_per_pixel,
      input_img_info.color_planes[0].height,
      cudaMemcpyDeviceToDevice,
      cuda_stream_);
  if (cudaSuccess != result) {
    GXF_LOG_ERROR("Failed to copy input from device to host: %s", cudaGetErrorString(result));
    return GXF_FAILURE;
  }

  result = cudaMemcpy2DAsync(
      nvbuffer_plane_1,
      input_img_info.color_planes[1].width * input_img_info.color_planes[1].bytes_per_pixel,
      input_img.value()->pointer() + input_img_info.color_planes[0].size,
      input_img_info.color_planes[1].stride,
      input_img_info.color_planes[1].width * input_img_info.color_planes[1].bytes_per_pixel,
      input_img_info.color_planes[1].height,
      cudaMemcpyDeviceToDevice,
      cuda_stream_);
  if (cudaSuccess != result) {
    GXF_LOG_ERROR("Failed to copy input from device to host: %s", cudaGetErrorString(result));
    return GXF_FAILURE;
  }
  cudaStreamSynchronize(cuda_stream_);

  nvBuffer->planes[0].bytesused = input_img_info.color_planes[0].width *
                                  input_img_info.color_planes[0].bytes_per_pixel *
                                  input_img_info.color_planes[0].height;
  nvBuffer->planes[1].bytesused = input_img_info.color_planes[1].width *
                                  input_img_info.color_planes[1].bytes_per_pixel *
                                  input_img_info.color_planes[1].height;

  // Add timestamp into v4l2 buffer
  v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
  v4l2_buf.timestamp.tv_usec = static_cast<uint64_t>(
      maybe_timestamp.value()->acqtime % static_cast<uint64_t>(1e6));
  v4l2_buf.timestamp.tv_sec = static_cast<uint64_t>(
      maybe_timestamp.value()->acqtime / static_cast<uint64_t>(1e6));;

  impl_->ctx->enc->output_plane.qBuffer(v4l2_buf, NULL);

  std::lock_guard<std::mutex> lock(impl_->ctx->mtx);
  available_buffer_num_ -= 1;
  if (scheduling_term_->getEventState() == nvidia::gxf::AsynchronousEventState::WAIT) {
    scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_WAITING);
  }
  sem_post(&impl_->ctx->pollthread_sema);

  return GXF_SUCCESS;
}

nvidia::gxf::Expected<nvidia::gxf::Entity> EncoderRequest::hasDQCapturePlane() {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane capture_planes[MAX_PLANES];
  NvBuffer* buffer = NULL;
  int ret = 0;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(capture_planes, 0, sizeof(capture_planes));
  v4l2_buf.m.planes = capture_planes;
  v4l2_buf.length = 1;

  ret = impl_->ctx->enc->capture_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10);
  if (ret < 0) {
    if (errno == EAGAIN) {
      // Device poll is returned due to interupt or other events
      // No buffer to DQ from capture plane
      std::lock_guard<std::mutex> lock(impl_->ctx->mtx);
      impl_->ctx->request_complete_num -= 1;
      if (impl_->ctx->request_complete_num <= 0) {
        if (available_buffer_num_ < impl_->ctx->enc->output_plane.getNumBuffers()) {
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

  if (buffer->planes[0].bytesused == 0) {
    GXF_LOG_ERROR("Buffer from capture plane is empty");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  // Get output message
  auto output_entity = output_entity_queue_.front();
  output_entity_queue_.pop();

  auto output_tensor = output_entity.get<nvidia::gxf::Tensor>();
  if (!output_tensor) {
    GXF_LOG_ERROR("Failed to get output tensor");
    return gxf::ForwardError(output_tensor);
  }

  impl_->ctx->packet_buf_size = buffer->planes[0].bytesused;
  auto gxf_tensor_result = output_tensor.value()->reshape<uint8_t>(
                             nvidia::gxf::Shape{static_cast<int>(impl_->ctx->packet_buf_size)},
                             nvidia::gxf::MemoryStorageType::kHost, impl_->ctx->pool);
  if (!gxf_tensor_result) {
    GXF_LOG_ERROR("Failed to reshape tensor");
    return gxf::ForwardError(gxf_tensor_result);
  }

  // Copy data back
  memcpy(output_tensor.value()->data<uint8_t>().value(),
         buffer->planes[0].data, buffer->planes[0].bytesused);

  // Add key frame indicator into output message
  v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
  if (impl_->ctx->enc->getMetadata(v4l2_buf.index, enc_metadata) != 0) {
    GXF_LOG_ERROR("Failed add key frame indicator into output");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  auto is_key_frame = output_entity.add<bool>("is_key_frame");
  if (!is_key_frame) {
    GXF_LOG_ERROR("Failed add key frame indicator into output");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }
  *is_key_frame.value() = enc_metadata.KeyFrame;

  // Enqueue empty buffer to capture plane
  if (impl_->ctx->enc->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
    GXF_LOG_ERROR("Failed to enqueue capture plane");
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }

  std::lock_guard<std::mutex> lock(impl_->ctx->mtx);
  available_buffer_num_+=1;
  impl_->ctx->request_complete_num -= 1;
  if (impl_->ctx->request_complete_num <= 0) {
    if (available_buffer_num_ < impl_->ctx->enc->output_plane.getNumBuffers()) {
      // Have requests on the fly, set event back to waiting
      scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::EVENT_WAITING);
    } else {
      // No pending results to publish, set event back to wait
      scheduling_term_->setEventState(nvidia::gxf::AsynchronousEventState::WAIT);
    }
  }
  return output_entity;
}

nvidia::gxf::Expected<bool> EncoderRequest::isAcceptingRequest() {
  return available_buffer_num_ > 0;
}

}  // namespace isaac
}  // namespace nvidia
