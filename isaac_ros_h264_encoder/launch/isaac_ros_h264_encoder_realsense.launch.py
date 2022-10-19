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
import os

from ament_index_python.packages import get_package_share_directory

import launch
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # RealSense
    realsense_config_file_path = os.path.join(
        get_package_share_directory('isaac_ros_h264_encoder'),
        'config', 'realsense.yaml'
    )

    realsense_node = ComposableNode(
        package='realsense2_camera',
        plugin='realsense2_camera::RealSenseNodeFactory',
        parameters=[realsense_config_file_path],
        remappings=[
            ('infra1/image_rect_raw', 'left/image_rect_raw_mono'),
            ('infra2/image_rect_raw', 'right/image_rect_raw_mono'),
            ('infra1/camera_info', 'left/camerainfo'),
            ('infra2/camera_info', 'right/camerainfo')
        ]
    )

    left_format_converter_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_left',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'left/image_rect_raw_mono'),
            ('image', 'left/image_rect_raw')]
    )

    right_format_converter_node = ComposableNode(
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        name='image_format_node_right',
        parameters=[{
                'encoding_desired': 'rgb8',
        }],
        remappings=[
            ('image_raw', 'right/image_rect_raw_mono'),
            ('image', 'right/image_rect_raw')]
    )

    left_encoder_node = ComposableNode(
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        name='left_encoder_node',
        parameters=[{
            'input_width': 640,
            'input_height': 480,
        }],
        remappings=[
            ('image_raw', 'left/image_rect_raw'),
            ('image_compressed', 'left/image_compressed')]
    )

    right_encoder_node = ComposableNode(
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        name='right_encoder_node',
        parameters=[{
            'input_width': 640,
            'input_height': 480,
        }],
        remappings=[
            ('image_raw', 'right/image_rect_raw'),
            ('image_compressed', 'right/image_compressed')]
    )

    container = ComposableNodeContainer(
        name='encoder_container',
        namespace='encoder',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[realsense_node, left_format_converter_node,
                                      right_format_converter_node,
                                      left_encoder_node, right_encoder_node],
        output='screen'
    )

    rosbag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '/left/camerainfo', '/right/camerainfo',
             '/left/image_compressed', '/right/image_compressed'],
        output='screen')

    return (launch.LaunchDescription([rosbag_record, container]))
