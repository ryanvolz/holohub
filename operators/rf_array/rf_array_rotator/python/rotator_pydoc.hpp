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

namespace Rotator {

// PyRotatorScheduled Constructor
PYDOC(RotatorScheduled_python, R"doc(
Operator that tunes given frequencies to baseband according to a cyclic time schedule.

**==Named Inputs==**

    rf_in : RFArray_fc32
        Complex RFArray with data shape (chunk_size, num_subchannels).

**==Named Outputs==**

    rf_out : RFArray_fc32
        Complex RFArray with data shape (chunk_size, num_subchannels).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
cycle_duration_secs : int
    Duration of the cycle of frequencies before it repeats.
cycle_start_timestamp : int
    Cycle start timestamp (seconds since Unix epoch).
schedule : list of {"start": start_time, "freq": frequency}
    Schedule of frequencies and their activation times in the cycle. The frequencies
    given are the RF frequency to tune, and this operator will take into account
    the known input frequency of the RF samples according to the chunk's RFMetadata.
)doc")
}  // namespace Rotator

}  // namespace holoscan::doc
