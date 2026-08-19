# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Round-trip test: encode -> decode for every supported image format.

For each format the encoder consumes (rgb8, nv12, mono8) a matching decoder is
configured (via the `output_encoding` parameter) to emit the same format, so a
single launch graph exercises encode -> decode for all formats. The test
publishes a synthetic input image and asserts that a decoded image of the
expected encoding is received.

The encoder is configured with the `iframe_cqp` preset so every frame is an
independently decodable IDR frame, letting the decoder produce output from any
single received frame.
"""

import platform
import subprocess
import time

from isaac_ros_test import IsaacROSBaseTest

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import launch_testing

import numpy as np

import pytest
import rclpy

from sensor_msgs.msg import Image

# Even dimensions are required for NV12 (4:2:0 chroma subsampling).
HEIGHT = 256
WIDTH = 256

# Encoder input formats and the matching decoder `output_encoding` values.
FORMATS = ['rgb8', 'nv12', 'mono8']

# Encoding observed by a plain sensor_msgs/Image subscriber. The decoder emits a
# NitrosImage in its configured `output_encoding`, but the NitrosImage ->
# sensor_msgs/Image type adapter converts nv12 (and nv24) to rgb8 on the way out
# to non-NITROS subscribers, since standard ROS consumers cannot handle NV12
# device memory. rgb8/mono8 pass through unchanged.
EXPECTED_ROS_ENCODING = {'rgb8': 'rgb8', 'nv12': 'rgb8', 'mono8': 'mono8'}

# GPUs whose NVENC/NVDEC engines are unavailable or incompatible.
UNSUPPORTED_COMPUTE_CAPS = ['8.0', '9.0', '10.0', '10.3']


def _codec_available() -> bool:
    """Return True if the local GPU exposes usable NVENC/NVDEC engines."""
    if platform.machine() != 'x86_64':
        # Jetson integrated codecs are always available.
        return True
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv'],
            capture_output=True, text=True)
    except FileNotFoundError:
        # nvidia-smi not installed: assume no usable codec engine.
        return False
    compute_caps = result.stdout.strip().split('\n')
    if len(compute_caps) <= 1:
        return False
    return all(cap not in UNSUPPORTED_COMPUTE_CAPS for cap in compute_caps[1:])


def _make_image(encoding: str) -> Image:
    """Build a deterministic synthetic input image in the given encoding."""
    msg = Image()
    msg.height = HEIGHT
    msg.width = WIDTH
    msg.encoding = encoding
    msg.is_bigendian = 0

    # A simple gradient so the luma plane is non-trivial.
    gradient = ((np.indices((HEIGHT, WIDTH)).sum(axis=0)) % 256).astype(np.uint8)

    if encoding == 'rgb8':
        msg.step = WIDTH * 3
        rgb = np.dstack([gradient, gradient, gradient])
        msg.data = rgb.tobytes()
    elif encoding == 'mono8':
        msg.step = WIDTH
        msg.data = gradient.tobytes()
    elif encoding == 'nv12':
        msg.step = WIDTH
        # Y plane (HxW) followed by an interleaved UV plane (H/2 x W) set to
        # neutral chroma (128).
        uv = np.full((HEIGHT // 2, WIDTH), 128, dtype=np.uint8)
        msg.data = gradient.tobytes() + uv.tobytes()
    else:
        raise ValueError(f'Unsupported test encoding: {encoding}')
    return msg


@pytest.mark.rostest
def generate_test_description():
    if not _codec_available():
        IsaacROSH264RoundTripTest.skip_test = True
        return IsaacROSH264RoundTripTest.generate_test_description(
            [launch_testing.actions.ReadyToTest()])

    IsaacROSH264RoundTripTest.skip_test = False

    nodes = []
    for fmt in FORMATS:
        nodes.append(ComposableNode(
            name=f'encoder_{fmt}',
            package='isaac_ros_h264_encoder',
            plugin='nvidia::isaac_ros::h264_encoder::EncoderNode',
            namespace=IsaacROSH264RoundTripTest.generate_namespace(),
            parameters=[{
                'input_height': HEIGHT,
                'input_width': WIDTH,
                'config': 'iframe_cqp',
            }],
            remappings=[
                ('image_raw', f'{fmt}/image_raw'),
                ('image_compressed', f'{fmt}/image_compressed'),
            ]))
        nodes.append(ComposableNode(
            name=f'decoder_{fmt}',
            package='isaac_ros_h264_decoder',
            plugin='nvidia::isaac_ros::h264_decoder::DecoderNode',
            namespace=IsaacROSH264RoundTripTest.generate_namespace(),
            parameters=[{
                'output_encoding': fmt,
            }],
            remappings=[
                ('image_compressed', f'{fmt}/image_compressed'),
                ('image_uncompressed', f'{fmt}/image_uncompressed'),
            ]))

    container = ComposableNodeContainer(
        name='h264_round_trip_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=nodes,
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )
    return IsaacROSH264RoundTripTest.generate_test_description([container])


class IsaacROSH264RoundTripTest(IsaacROSBaseTest):
    skip_test = False

    def test_round_trip_all_formats(self):
        if self.skip_test:
            self.skipTest('No NVENC/NVDEC engine available for existing GPUs.')

        TIMEOUT = 20

        raw_topics = [f'{fmt}/image_raw' for fmt in FORMATS]
        uncompressed_topics = [f'{fmt}/image_uncompressed' for fmt in FORMATS]
        self.generate_namespace_lookup(raw_topics + uncompressed_topics)

        # The encoder/decoder pipeline is continuously streaming: every frame
        # published below is an independently decodable IDR frame (iframe_cqp),
        # so the decoder emits one output per input. Since we keep publishing
        # until every format has been received, a format whose output arrives
        # early will produce several messages on its topic before the slowest
        # format catches up. Accept multiple messages per topic and require at
        # least one, rather than asserting exactly one.
        received_messages = {}
        subs = self.create_logging_subscribers(
            [(f'{fmt}/image_uncompressed', Image) for fmt in FORMATS],
            received_messages,
            accept_multiple_messages=True)

        publishers = {
            fmt: self.node.create_publisher(
                Image, self.namespaces[f'{fmt}/image_raw'], self.DEFAULT_QOS)
            for fmt in FORMATS
        }

        try:
            inputs = {fmt: _make_image(fmt) for fmt in FORMATS}

            end_time = time.time() + TIMEOUT
            pending = set(FORMATS)
            while time.time() < end_time and pending:
                for fmt in FORMATS:
                    publishers[fmt].publish(inputs[fmt])
                rclpy.spin_once(self.node, timeout_sec=0.1)
                pending = {
                    fmt for fmt in FORMATS
                    if not received_messages[f'{fmt}/image_uncompressed']
                }

            self.assertFalse(
                pending,
                f'Did not receive decoded output for formats: {sorted(pending)}')

            for fmt in FORMATS:
                msg = received_messages[f'{fmt}/image_uncompressed'][0]
                expected_encoding = EXPECTED_ROS_ENCODING[fmt]
                self.assertEqual(
                    msg.encoding, expected_encoding,
                    f'For decoder output_encoding {fmt}, expected ROS encoding '
                    f'{expected_encoding}, got {msg.encoding}')
                self.assertEqual(msg.width, WIDTH)
                self.assertEqual(msg.height, HEIGHT)

        finally:
            [self.node.destroy_subscription(sub) for sub in subs]
            [self.node.destroy_publisher(pub) for pub in publishers.values()]
