# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

project(gxf_codec_extension LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -w)
endif()
find_package(yaml-cpp)

ament_auto_add_library(gxf_video_encoder_extension SHARED
  video_encoder/videoencoder_ext.cpp
  video_encoder/videoencoder_request.cpp
  video_encoder/videoencoder_request.hpp
  video_encoder/videoencoder_response.cpp
  video_encoder/videoencoder_response.hpp
  video_encoder/videoencoder_context.cpp
  video_encoder/videoencoder_context.hpp
  video_encoder/videoencoder_utils.cpp
  video_encoder/videoencoder_utils.hpp
)

ament_auto_add_library(gxf_video_decoder_extension SHARED
  video_decoder/videodecoder_ext.cpp
  video_decoder/videodecoder_request.cpp
  video_decoder/videodecoder_request.hpp
  video_decoder/videodecoder_response.cpp
  video_decoder/videodecoder_response.hpp
  video_decoder/videodecoder_context.cpp
  video_decoder/videodecoder_context.hpp
  video_decoder/videodecoder_utils.cpp
  video_decoder/videodecoder_utils.hpp
)

find_package(CUDAToolkit)

# https://developer.nvidia.com/video_codec_sdk/downloads/v11-1-5
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

target_include_directories(gxf_video_encoder_extension PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/nvbuf_utils
  ${CMAKE_CURRENT_SOURCE_DIR}/libv4l2/include
)

target_include_directories(gxf_video_decoder_extension PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/nvbuf_utils
  ${CMAKE_CURRENT_SOURCE_DIR}/libv4l2/include
)

add_library(nvbuf_fdmap SHARED IMPORTED)
add_library(nvbufsurface SHARED IMPORTED)
add_library(nvbufsurftransform SHARED IMPORTED)
add_library(nvv4l2 SHARED IMPORTED)
add_library(cuvidv4l2 SHARED IMPORTED)

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET nvbuf_fdmap PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvbuf_fdmap.so)
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvbufsurftransform.so)
  set_property(TARGET nvv4l2 PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvv4l2.so)
  set_property(TARGET cuvidv4l2 PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libcuvidv4l2.so)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET nvbuf_fdmap PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvbuf_fdmap.so.1.0.0)
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurftransform.so)
  set_property(TARGET nvv4l2 PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvv4l2.so)
  set_property(TARGET cuvidv4l2 PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvcuvidv4l2.so)
endif()

target_link_libraries(gxf_video_encoder_extension
  CUDA::cuda_driver
  CUDA::cudart
  yaml-cpp
  cuvidv4l2
  nvv4l2
  nvbufsurface
  nvbuf_fdmap
  nvbufsurftransform
  isaac_ros_gxf::Core
  isaac_ros_gxf::Std
  isaac_ros_gxf::Multimedia
)

target_link_libraries(gxf_video_decoder_extension
  CUDA::cuda_driver
  CUDA::cudart
  yaml-cpp
  cuvidv4l2
  nvv4l2
  nvbufsurface
  nvbuf_fdmap
  nvbufsurftransform
  isaac_ros_gxf::Core
  isaac_ros_gxf::Std
  isaac_ros_gxf::Multimedia
)
