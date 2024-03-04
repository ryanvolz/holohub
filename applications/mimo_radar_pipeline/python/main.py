# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import logging
import sys

import cupy as cp
from holoscan.core import Application, Operator, OperatorSpec

from holohub.advanced_network_connector_rx import AdvConnectorOpRx
from holohub.advanced_network_rx import AdvNetworkOpRx

logger = logging.getLogger("MIMORadarPipeline")
logging.basicConfig(level=logging.INFO)


class PingRFOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("rf_in")

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("rf_in")
        for key, val in in_message.items():
            cp_array = cp.asarray(val)
            print(f"key: {key}, val: {cp_array.shape}")

class MIMORadarPipeline(Application):
    def compose(self):
        adv_net_rx = AdvNetworkOpRx(self, name="adv_network_rx")
        adv_rx_pkt = AdvConnectorOpRx(self, name="adv_network_connector_rx")
        ping_rf = PingRFOp(self, name="ping_rf")
        self.add_flow(adv_net_rx, adv_rx_pkt, {("bench_rx_out", "burst_in")})
        self.add_flow(adv_rx_pkt, ping_rf, {("rf_out", "rf_in")})


if __name__ == "__main__":
    config_path = sys.argv[1]
    app = MIMORadarPipeline()
    app.config(config_path)
    app.run()
