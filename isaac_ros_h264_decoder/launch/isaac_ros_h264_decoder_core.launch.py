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


class IsaacROSStereoH264DecoderLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        return {
            'left_decoder_node': ComposableNode(
                package='isaac_ros_h264_decoder',
                plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
                name='left_decoder_node',
                remappings=[
                    ('image_compressed', 'left/image_compressed'),
                    ('image_uncompressed', 'left/image_uncompressed')
                ]
            ),

            'right_decoder_node': ComposableNode(
                package='isaac_ros_h264_decoder',
                plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
                name='right_decoder_node',
                remappings=[
                    ('image_compressed', 'right/image_compressed'),
                    ('image_uncompressed', 'right/image_uncompressed')
                ]
            )
        }


def generate_launch_description():
    decoder_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='decoder_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=IsaacROSStereoH264DecoderLaunchFragment
        .get_composable_nodes().values(),
        output='screen'
    )

    return launch.LaunchDescription(
        [decoder_container] +
        IsaacROSStereoH264DecoderLaunchFragment.get_launch_actions().values())
