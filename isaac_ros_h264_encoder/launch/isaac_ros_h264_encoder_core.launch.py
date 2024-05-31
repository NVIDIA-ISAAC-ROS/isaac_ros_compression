# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


class IsaacROSStereoH264EncoderLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        return {
            'left_encoder_node': ComposableNode(
                package='isaac_ros_h264_encoder',
                plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
                name='left_encoder_node',
                parameters=[{
                    'input_width': interface_specs['camera_resolution']['width'],
                    'input_height': interface_specs['camera_resolution']['height'],
                }],
                remappings=[
                    ('image_raw', 'left/image_rect'),
                    ('image_compressed', 'left/image_compressed')]
            ),

            'right_encoder_node': ComposableNode(
                package='isaac_ros_h264_encoder',
                plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
                name='right_encoder_node',
                parameters=[{
                    'input_width': interface_specs['camera_resolution']['width'],
                    'input_height': interface_specs['camera_resolution']['height'],
                }],
                remappings=[
                    ('image_raw', 'right/image_rect'),
                    ('image_compressed', 'right/image_compressed')]
            )
        }


def generate_launch_description():
    encoder_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='encoder_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSStereoH264EncoderLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription(
        [encoder_container] +
        IsaacROSStereoH264EncoderLaunchFragment.get_launch_actions().values())
