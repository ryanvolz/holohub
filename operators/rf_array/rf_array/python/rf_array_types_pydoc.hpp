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

namespace RFArrayTypes {

PYDOC(RFMetadata_python, R"doc(
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

PYDOC(RFArray_python, R"doc(
Dataclass containing a data array and metadata class for a chunk of RF samples.

Attributes
----------
data : matx::tensor_t<sampleType, 2>, shape (num_samples, num_subchannels)
    Array of RF samples.
metadata : RFMetadata
    Metadata corresponding to the RF data.
)doc")
}  // namespace RFArrayTypes

}  // namespace holoscan::doc
