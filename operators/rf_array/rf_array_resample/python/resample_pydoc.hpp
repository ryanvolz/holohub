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

namespace Resample {

// PyResamplePoly Constructor
PYDOC(ResamplePoly_python, R"doc(
Operator that does polyphase filtering and resampling of an RFArray chunk.

**==Named Inputs==**

    rf_in : RFArray_fc32
        Complex RFArray with data shape (chunk_size, num_subchannels).

**==Named Outputs==**

    rf_out : RFArray_fc32
        Complex RFArray with data shape (chunk_size * up // down, num_subchannels).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
chunk_size : int
    Number of samples to operate on in one chunk.
num_subchannels : int
    Number of IQ subchannels per sample time instance.
filter_coefs : array
    Filter coefficients provided to MatX's resample_poly
up : int
    Upsampling factor to pass to MatX's resample_poly
down : int
    Downsampling factor to pass to MatX's resample_poly
)doc")
}  // namespace Resample

}  // namespace holoscan::doc
