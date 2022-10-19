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
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the software compatible H.264 decoder Node."""
    launch_args = [
        DeclareLaunchArgument(
            'rosbag_path',
            description='Path of the rosbag'),
    ]

    rosbag_path = LaunchConfiguration('rosbag_path')

    h264_msgs_packet_node = Node(
        name='h264_converter_node',
        package='isaac_ros_to_h264_msgs_packet',
        executable='ToH264MsgsPacket',
        remappings=[
            ('image_compressed', 'left/image_compressed'),
            ('image_raw/compressed', 'image_raw/h264')
        ],
        output='screen'
    )
            
    republish_node = Node(
        name='republish_node',
        package='image_transport', 
        executable='republish',  
        arguments=['h264',  'raw'], 
        remappings=[                
            ('in/h264', 'image_raw/h264'),
        ],
        output='screen'
    )
    
    viewer_node = Node(
        name='image_viewer',
        package='image_view',
        executable='image_view',
        remappings=[
            ('image', '/out'),
        ],
        output='screen'
    )

    rosbag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', '--loop', rosbag_path],
        output='screen')

    return (launch.LaunchDescription(launch_args + 
                    [rosbag_play, h264_msgs_packet_node, republish_node, viewer_node]))
