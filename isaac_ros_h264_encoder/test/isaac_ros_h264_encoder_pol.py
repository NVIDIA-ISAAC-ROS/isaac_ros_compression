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
import pathlib
import time

from isaac_ros_test import IsaacROSBaseTest, JSONConversion

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CompressedImage, Image

HEIGHT = 460
WIDTH = 460
SAVE_H264 = False


@pytest.mark.rostest
def generate_test_description():
    encoder_node = ComposableNode(
        name='encoder',
        package='isaac_ros_h264_encoder',
        plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
        namespace=IsaacROSEncoderTest.generate_namespace(),
        parameters=[{
                'input_height': HEIGHT,
                'input_width': WIDTH,
        }])

    container = ComposableNodeContainer(
        name='encoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[encoder_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSEncoderTest.generate_test_description([container])


class IsaacROSEncoderTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_encoder(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['image_raw',
                                        'image_compressed'])

        subs = self.create_logging_subscribers(
            [('image_compressed', CompressedImage)], received_messages)

        input_image_pub = self.node.create_publisher(
            Image, self.namespaces['image_raw'], self.DEFAULT_QOS
        )

        try:
            input_image = JSONConversion.load_image_from_json(test_folder / 'image.json')
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                input_image_pub.publish(input_image)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'image_compressed' in received_messages:
                    done = True
                    break
            if(SAVE_H264):
                compressed_img_msg = received_messages['image_compressed']
                compressed_img_file = open('compressed.h264', 'wb')
                compressed_img_file.write(compressed_img_msg.data)
                compressed_img_file.close()
            self.assertTrue(done, 'Didnt recieve output on encoded topic')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(input_image_pub)
