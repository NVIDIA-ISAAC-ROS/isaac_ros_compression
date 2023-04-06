
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

import launch
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch encoder to encode single image topic from r2b."""
    launch_args = [
        DeclareLaunchArgument(
            'rosbag_path',
            description='Path of the r2b rosbag'),
    ]

    rosbag_path = LaunchConfiguration('rosbag_path')

    left_encoder_node = ComposableNode(
        name='left_encoder_node',
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
            'config': 'iframe',
        }],
        remappings=[
            ('image_raw', 'hawk_0_left_rgb_image'),
            ('image_compressed', 'hawk_0_left_h264_image')
        ]
    )

    right_encoder_node = ComposableNode(
        name='right_encoder_node',
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
            'config': 'iframe',
        }],
        remappings=[
            ('image_raw', 'hawk_0_right_rgb_image'),
            ('image_compressed', 'hawk_0_right_h264_image')
        ]
    )

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path],
        output='screen')

    rosbag_record = ExecuteProcess(
        cmd=['ros2', 'bag',
             'record', '/hawk_0_left_h264_image', '/hawk_0_right_h264_image',
             '-o', 'r2b_compressed_image'],
        output='screen')

    container = ComposableNodeContainer(
        name='encoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[left_encoder_node, right_encoder_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )

    return (launch.LaunchDescription(launch_args + [rosbag_record, rosbag_play, container]))
