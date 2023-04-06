# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.23.2)
project(isaac_ros_h264_encoder LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# EncoderNode
ament_auto_add_library(encoder_node SHARED src/encoder_node.cpp)
rclcpp_components_register_nodes(encoder_node "nvidia::isaac_ros::h264_encoder::EncoderNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::h264_encoder::EncoderNode;$<TARGET_FILE:encoder_node>\n")

### Install extensions built from source

# Codec
add_subdirectory(gxf/codec)
install(TARGETS gxf_codec_extension DESTINATION share/${PROJECT_NAME}/gxf/lib/codec)

### End extensions

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  if(${ARCHITECTURE} STREQUAL "aarch64")
    find_package(launch_testing_ament_cmake REQUIRED)
    add_launch_test(test/isaac_ros_h264_encoder_pol.py)
  endif()
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)
