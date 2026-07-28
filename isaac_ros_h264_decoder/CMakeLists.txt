# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_ros_h264_decoder LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(CUDAToolkit REQUIRED)
find_package(vpi REQUIRED)

# V4L2 and NvBuf libraries for decoder
add_library(nvbuf_fdmap SHARED IMPORTED)
add_library(nvbufsurface SHARED IMPORTED)
add_library(nvbufsurftransform SHARED IMPORTED)
add_library(nvv4l2 SHARED IMPORTED)
add_library(cuvidv4l2 SHARED IMPORTED)

# libnvbuf_fdmap, libnvbufsurface and libnvbufsurftransform ship without a
# DT_SONAME. When a shared library has no soname the linker records whatever path
# it was handed, so linking them by absolute IMPORTED_LOCATION bakes absolute
# paths into decoder_node's DT_NEEDED entries:
#
#   $ readelf -d libdecoder_node.so | grep NEEDED
#     /usr/lib/x86_64-linux-gnu/libnvbufsurface.so
#     /usr/lib/x86_64-linux-gnu/libnvbuf_fdmap.so
#     /usr/lib/x86_64-linux-gnu/libnvbufsurftransform.so
#
# decoder_node then loads only when those libraries sit at exactly that path, and
# an absolute DT_NEEDED cannot be redirected by RPATH, RUNPATH or LD_LIBRARY_PATH,
# which makes the node unusable from any relocated prefix.
#
# Linking them by name against a search directory records a plain DT_NEEDED
# instead, so the normal search order applies. NVBUF_LINK_LIBS below selects that
# form on x86_64. libnvv4l2.so is unaffected either way: it carries a soname of
# its own (libv4l2.so.0) and already resolves correctly.
#
# The aarch64 branch is deliberately left on the original absolute-path form. The
# Jetson nvbuf_fdmap is versioned (libnvbuf_fdmap.so.1.0.0) and whether an
# unversioned symlink exists alongside it could not be verified here, so `-l:` is
# not assumed to resolve. Applying the same treatment there would be worthwhile if
# you can confirm the layout.
# NVBUF_LIB_DIR can be overridden to build against an NVDEC installation outside
# the distribution default, which packagers on non-FHS layouts need.
if(${ARCHITECTURE} STREQUAL "x86_64")
  if(NOT NVBUF_LIB_DIR)
    set(NVBUF_LIB_DIR /usr/lib/x86_64-linux-gnu)
  endif()
  set(NVBUF_LINK_LIBS
    -l:libnvbufsurface.so
    -l:libnvbuf_fdmap.so
    -l:libnvbufsurftransform.so)
  set_property(TARGET nvv4l2 PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libnvv4l2.so)
  set_property(TARGET cuvidv4l2 PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libcuvidv4l2.so)
elseif(${ARCHITECTURE} STREQUAL "aarch64")
  if(NOT NVBUF_LIB_DIR)
    set(NVBUF_LIB_DIR /usr/lib/aarch64-linux-gnu/nvidia)
  endif()
  set(NVBUF_LINK_LIBS nvbufsurface nvbuf_fdmap nvbufsurftransform)
  set_property(TARGET nvbuf_fdmap PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libnvbuf_fdmap.so.1.0.0)
  set_property(TARGET nvbufsurface PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libnvbufsurface.so)
  set_property(TARGET nvbufsurftransform PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libnvbufsurftransform.so)
  set_property(TARGET nvv4l2 PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libnvv4l2.so)
  set_property(TARGET cuvidv4l2 PROPERTY IMPORTED_LOCATION ${NVBUF_LIB_DIR}/libnvcuvidv4l2.so)
endif()

# DecoderNode (GXF-free implementation)
ament_auto_add_library(decoder_node SHARED
  src/decoder_node.cpp
  src/decoder_v4l2_impl.cpp
  src/decoder_v4l2_utils.cpp
)

target_include_directories(decoder_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../codec
  ${CMAKE_CURRENT_SOURCE_DIR}/../codec/nvbuf_utils
  ${CMAKE_CURRENT_SOURCE_DIR}/../codec/libv4l2/include
)

if(DEFINED NVBUF_LIB_DIR)
  target_link_directories(decoder_node PRIVATE ${NVBUF_LIB_DIR})
endif()
target_link_libraries(decoder_node
  CUDA::cuda_driver
  CUDA::cudart
  vpi
  cuvidv4l2
  nvv4l2
  ${NVBUF_LINK_LIBS}
)

rclcpp_components_register_nodes(decoder_node "nvidia::isaac_ros::h264_decoder::DecoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::h264_decoder::DecoderNode;$<TARGET_FILE:decoder_node>\n")
set_target_properties(decoder_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Ensure NVIDIA multimedia libraries can be resolved in non-login runtimes on FastOS (DGX Spark).
if(${ARCHITECTURE} STREQUAL "aarch64")
  set(NV_CODEC_RPATH "/usr/lib/aarch64-linux-gnu/nvidia:/opt/deepstream_tegra/lib")
  set_target_properties(decoder_node PROPERTIES INSTALL_RPATH "${NV_CODEC_RPATH}")
  target_link_options(decoder_node PRIVATE "-Wl,--disable-new-dtags")
endif()


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_decoder_pol.py)
endif()

# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
