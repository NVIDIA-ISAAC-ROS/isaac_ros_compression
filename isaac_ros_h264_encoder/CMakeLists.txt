# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_h264_encoder LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(CUDAToolkit REQUIRED)
find_package(vpi REQUIRED)

# V4L2 and NvBuf libraries for encoder
add_library(nvbuf_fdmap SHARED IMPORTED)
add_library(nvbufsurface SHARED IMPORTED)
add_library(nvbufsurftransform SHARED IMPORTED)
add_library(nvv4l2 SHARED IMPORTED)

if(${ARCHITECTURE} STREQUAL "x86_64")
  set_property(TARGET nvbuf_fdmap PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvbuf_fdmap.so)
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvbufsurftransform.so)
  set_property(TARGET nvv4l2 PROPERTY IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libnvv4l2.so)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  set_property(TARGET nvbuf_fdmap PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvbuf_fdmap.so.1.0.0)
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurftransform.so)
  set_property(TARGET nvv4l2 PROPERTY IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/nvidia/libnvv4l2.so)
endif()

# EncoderNode - accepts RGB8 or NV12 input (converts RGB8 to NV12 via VPI)
ament_auto_add_library(encoder_node SHARED
  src/encoder_node.cpp
  src/encoder_v4l2_impl.cpp
  src/encoder_v4l2_utils.cpp
)

target_include_directories(encoder_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../codec
  ${CMAKE_CURRENT_SOURCE_DIR}/../codec/nvbuf_utils
  ${CMAKE_CURRENT_SOURCE_DIR}/../codec/libv4l2/include
)

target_link_libraries(encoder_node
  CUDA::cuda_driver
  CUDA::cudart
  vpi
  nvv4l2
  nvbufsurface
  nvbuf_fdmap
  nvbufsurftransform
)

rclcpp_components_register_nodes(encoder_node "nvidia::isaac_ros::h264_encoder::EncoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::h264_encoder::EncoderNode;$<TARGET_FILE:encoder_node>\n")
set_target_properties(encoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Ensure NVIDIA multimedia libraries can be resolved in non-login runtimes on FastOS (DGX Spark).
if(${ARCHITECTURE} STREQUAL "aarch64")
  set(NV_CODEC_RPATH "/usr/lib/aarch64-linux-gnu/nvidia:/opt/deepstream_tegra/lib")
  set_target_properties(encoder_node PROPERTIES INSTALL_RPATH "${NV_CODEC_RPATH}")
  target_link_options(encoder_node PRIVATE "-Wl,--disable-new-dtags")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_h264_encoder_pol.py)
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
