# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Dependencies
find_package(CUDAToolkit)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    core
    cuda
    multimedia
    serialization
    std
)
include(YamlCpp)

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

if( ${ARCHITECTURE} STREQUAL "x86_64" )
# x86_64
add_library(gxf_codec_extension SHARED
  decoder_extension.cpp
  components/decoder.cpp
  extensions/codec/components/decoder.hpp
)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
# aarch64
add_library(gxf_codec_extension SHARED
  codec_extension.cpp
  components/decoder_request.cpp
  extensions/codec/components/decoder_request.hpp
  components/decoder_response.cpp
  extensions/codec/components/decoder_response.hpp
  components/decoder_scheduling_term.cpp
  extensions/codec/components/decoder_scheduling_term.hpp
  components/encoder_request.cpp
  extensions/codec/components/encoder_request.hpp
  components/encoder_response.cpp
  extensions/codec/components/encoder_response.hpp
  components/encoder_scheduling_term.cpp
  extensions/codec/components/encoder_scheduling_term.hpp
)

add_library(jetson_multimedia STATIC
  /usr/src/jetson_multimedia_api/samples/common/classes/NvVideoEncoder.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvVideoDecoder.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2Element.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2ElementPlane.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
  /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp
)

target_include_directories(
  jetson_multimedia
    PUBLIC /usr/lib/aarch64-linux-gnu/tegra
           /usr/src/jetson_multimedia_api/argus/samples/utils
           /usr/src/jetson_multimedia_api/include)

target_link_libraries(
  jetson_multimedia PUBLIC
  /usr/lib/aarch64-linux-gnu/tegra/libnvbuf_utils.so
  /usr/lib/aarch64-linux-gnu/tegra/libnvv4l2.so
)

target_link_libraries(gxf_codec_extension PRIVATE jetson_multimedia)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
# https://developer.nvidia.com/video_codec_sdk/downloads/v11-1-5
target_include_directories(gxf_codec_extension PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/nvcuvid/Interface
)

# nvcuvid
add_library(nvcuvid SHARED IMPORTED)
set_property(TARGET nvcuvid PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvcuvid/linux/stubs/${ARCHITECTURE}/libnvcuvid.so)

target_link_libraries(gxf_codec_extension
  PUBLIC
    CUDA::cuda_driver
    GXF::multimedia
    GXF::cuda
    GXF::serialization
    GXF::std
    nvcuvid
    yaml-cpp
)
