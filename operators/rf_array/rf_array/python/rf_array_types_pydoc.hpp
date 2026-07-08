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

namespace RFMetadata {

PYDOC(RFMetadata, R"doc(
Dataclass for storing metadata corresponding to a chunk of RF samples.

Attributes
----------
sample_idx : int
    Sample index of the first sample in the chunk of RF data.
sample_rate_numerator : int
    Numerator of the sample rate at which the RF data stream is sampled.
sample_rate_denominator : int
    Denominator of the sample rate at which the RF data stream is sampled.
center_freq : int
    Center frequency of the RF data samples.
)doc")

}  // namespace RFMetadata

namespace RFArray {

PYDOC(RFArray, R"doc(
Dataclass containing a data array and metadata class for a chunk of RF samples.

Attributes
----------
data : holoscan.tensor, shape (num_samples, num_subchannels)
    Array of RF samples.
metadata : RFMetadata
    Metadata corresponding to the RF data.
)doc")

PYDOC(set_deallocation_stream, R"doc(
Set the CUDA stream for stream-aware memory deallocation.

For operators that use an RFArray's data tensor on a stream separate from
the one it was created on (i.e. it was passed from another operator), this
method should be called with the newly used stream in order to defer memory
reuse util GPU operations on the stream complete. This prevents race
conditions where memory is returned to the pool while GPU kernels are still
reading from it.

This method should be used instead of using the data tensor's own
`set_deallocation_stream` method (if available), because that tensor does not
own the memory and it is instead owned by an underlying matx::tensor.

Parameters
----------
stream : int
    The memory address of the CUDA stream that last accessed this tensor's data.

)doc")
}  // namespace RFArray

}  // namespace holoscan::doc
