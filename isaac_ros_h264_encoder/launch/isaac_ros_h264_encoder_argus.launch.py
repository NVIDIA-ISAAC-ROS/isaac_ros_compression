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
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Launch the H.264 Encoder Node."""
    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
    )

    left_encoder_node = ComposableNode(
        name='left_encoder_node',
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
        }],
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('image_compressed', 'left/image_compressed')
        ]
    )

    right_encoder_node = ComposableNode(
        name='right_encoder_node',
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        parameters=[{
            'input_width': 1920,
            'input_height': 1200,
        }],
        remappings=[
            ('image_raw', 'right/image_raw'),
            ('image_compressed', 'right/image_compressed')
        ]
    )

    rosbag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '/left/camerainfo', '/right/camerainfo',
             '/left/image_compressed', '/right/image_compressed'],
        output='screen')

    container = ComposableNodeContainer(
        name='encoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[argus_stereo_node, left_encoder_node, right_encoder_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )

    return (launch.LaunchDescription([rosbag_record, container]))
