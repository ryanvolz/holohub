/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Technology
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace BasicNetwork {

PYDOC(NetworkOpBurstParams_python, R"doc(
Object containt a burst of packets.

Attributes
----------
data : memoryview
    Packet data for the burst, with packets concatenated.
num_pkts : int
    The number of packets contained in the burst.
)doc")

// PyBasicNetworkOpRx Constructor
PYDOC(BasicNetworkOpRx_python, R"doc(
Operator that outputs packet bursts from a basic network interface.

**==Named Outputs==**

    burst_out : NetworkOpBurstParams
        Object containing a burst of packets with the packet payload only.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
ip_addr : str
    IP address of interface to bind to
dst_port : int
    UDP or TCP port to listen on
l4_proto : str
    Layer 4 protocol (udp or tcp)
batch_size : int
    Number of packets in batch
max_payload_size : int
    Maximum payload size expected from sender.
)doc")

// PyBasicNetworkOpTx Constructor
PYDOC(BasicNetworkOpTx_python, R"doc(
Operator that sends packet bursts to a basic network interface.

**==Named Inputs==**

    burst_in : NetworkOpBurstParams
        Object containing a burst of packets with the packet payload only.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
ip_addr : str
    IP address of interface to bind to
dst_port : int
    UDP or TCP port to listen on
l4_proto : str
    Layer 4 protocol (udp or tcp)
max_payload_size : int
    Maximum payload size expected from sender.
min_ipg_ns : int
    Smallest gap between packets in nanoseconds
retry_connect : int
    Interval to retry connecting to server in seconds
)doc")
}  // namespace BasicNetwork

}  // namespace holoscan::doc
