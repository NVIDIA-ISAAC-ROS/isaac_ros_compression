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

import cv2
import cv_bridge

from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import pytest
import rclpy

from sensor_msgs.msg import CompressedImage, Image

HEIGHT = 460
WIDTH = 460
VISUALIZE = False


@pytest.mark.rostest
def generate_test_description():
    decoder_node = ComposableNode(
        name='decoder',
        package='isaac_ros_h264_decoder',
        plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
        namespace=IsaacROSDecoderTest.generate_namespace(),
        parameters=[{
                'input_height': HEIGHT,
                'input_width': WIDTH,
        }])

    container = ComposableNodeContainer(
        name='decoder_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[decoder_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    return IsaacROSDecoderTest.generate_test_description([container])


class IsaacROSDecoderTest(IsaacROSBaseTest):
    filepath = pathlib.Path(os.path.dirname(__file__))

    @IsaacROSBaseTest.for_each_test_case()
    def test_image_decoder(self, test_folder):
        TIMEOUT = 10
        received_messages = {}

        self.generate_namespace_lookup(['image_compressed',
                                        'image_uncompressed'])

        subs = self.create_logging_subscribers(
            [('image_uncompressed', Image)], received_messages)

        input_image_pub = self.node.create_publisher(
            CompressedImage, self.namespaces['image_compressed'], self.DEFAULT_QOS
        )

        try:
            compressed_img_msg = CompressedImage()
            compressed_img_msg.format = 'h264'
            compressed_img_msg.data = open(test_folder / 'compressed.h264', 'rb').read()
            end_time = time.time() + TIMEOUT
            done = False

            while time.time() < end_time:
                input_image_pub.publish(compressed_img_msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)

                if 'image_uncompressed' in received_messages:
                    done = True
                    break
            if(VISUALIZE):
                image_uncompressed_msg = received_messages['image_uncompressed']
                cv_img = cv_bridge.CvBridge().imgmsg_to_cv2(image_uncompressed_msg,
                                                            desired_encoding='bgr8')
                cv2.imwrite('uncompressed.png', cv_img)
            self.assertTrue(done, 'Didnt recieve output on decoded topic')

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            self.node.destroy_publisher(input_image_pub)
