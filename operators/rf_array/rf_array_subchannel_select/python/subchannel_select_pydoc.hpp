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

namespace SubchannelSelect {

// PySubchannelSelect Constructor
PYDOC(SubchannelSelect_python, R"doc(
Operator that selects given subchannels to keep in an RFArray.

**==Named Inputs==**

    rf_in : RFArray
        RFArray with data shape (chunk_size, num_subchannels).

**==Named Outputs==**

    rf_out : RFArray
        RFArray with data shape (chunk_size, num_subchannels).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
subchannel_idx : list of int
    List of subchannel indices to keep in the array.
)doc")
}  // namespace SubchannelSelect

}  // namespace holoscan::doc
