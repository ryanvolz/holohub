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

namespace NetConnectorBasic {

// PyNetConnectorBasic Constructor
PYDOC(NetConnectorBasic_python, R"doc(
Operator taking RF data in UDP packets and outputting RFArray chunks.

**==Named Inputs==**

    burst_in : NetworkOpBurstParams
        Burst of packets coming from the basic network operator.

**==Named Outputs==**

    rf_out : RFArray_sc16
        Complex integer RFArray with data shape (num_samples, num_subchannels).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
buffer_size : int
    Max number of num_samples batches that can be held at once.
num_samples : int
    Number of samples per output chunk.
num_subchannels : int
    Number of IQ subchannels per sample time instance.
batch_size : int, optional
    Batch size in packets for each processing epoch.
max_packet_size : int, optional
    Maximum packet size (not including network protocol headers) expected from sender.
)doc")
}  // namespace NetConnectorBasic

}  // namespace holoscan::doc
