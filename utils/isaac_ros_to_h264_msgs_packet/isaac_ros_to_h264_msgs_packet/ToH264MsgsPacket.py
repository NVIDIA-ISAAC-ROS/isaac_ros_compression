# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from h264_msgs.msg import Packet
from sensor_msgs.msg import CompressedImage

import rclpy
from rclpy.node import Node


class IsaacROSToH264MsgPacketNode(Node):

    def __init__(self, name='to_h264_msgs_packet_node'):
        super().__init__(name)        
        self.sequence_number_ = 0
                
        self.subscription_ = self.create_subscription(CompressedImage, 'image_compressed',
                                                      self.listener_callback, 10)
                                                              
        self.publisher_ = self.create_publisher(Packet, 'image_raw/compressed', 10)

    def listener_callback(self, msg):        
        packet = Packet()
        packet.header = msg.header
        packet.seq = self.sequence_number_
        packet.data = msg.data
        
        self.sequence_number_ += 1        
        self.publisher_.publish(packet)


def main(args=None):
    try:
        rclpy.init(args=args)
        node = IsaacROSToH264MsgPacketNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # only shut down if context is active
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
