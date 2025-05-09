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
up : int
    Upsampling factor
down : int
    Downsampling factor
outrate_cutoff : float
    Normalized low-pass filter cutoff frequency (half-amplitude point,
    where the attenuation will be -6 dB) where a value of 1.0 indicates
    half the *output* sampling rate. The value in Hertz is therefore
    ``(outrate_cutoff * out_sample_rate / 2.0)``.
outrate_transition_width : float
    Normalized width of the transition region from pass band to stop band,
    where a value of 1.0 indicates half the *output* sampling rate.
    The value in Hertz is therefore
    ``(outrate_transition_width * out_sample_rate / 2.0)``.
attenuation_db : float
    Minimum attenuation of the low-pass filter stop band in dB.
numtaps: int, optional
    The length of the filter (number of taps), overriding the value
    that would be used based on `outrate_transition_width` and
    `attenuation_db`.
kaiser_beta: float, optional
    The beta parameter for the Kaiser window (pi * alpha, controlling
    main lobe width versus side lobe level), overriding the value that
    would be used based on `outrate_transition_width` and
    `attenuation_db`.
filter_coefs : array, optional
    Filter coefficients. If provided, these will be used instead
    of ones that would be designed based on the above parameters.
)doc")
}  // namespace Resample

}  // namespace holoscan::doc
