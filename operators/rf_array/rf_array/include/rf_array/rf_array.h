/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Techonology
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

#include <matx.h>
#include <cuda/std/complex>

using float_t = float;
using complex_t = cuda::std::complex<float_t>;

using real_t = int16_t;
struct complex_int_type {
  real_t r;
  real_t i;
} __attribute__((__packed__));
using sample_t = complex_int_type;

// Metadata for RF signal
struct RFMetadata {
  uint64_t sample_idx;
  uint64_t sample_rate_numerator;
  uint64_t sample_rate_denominator;
  double center_freq;
};

// Represents a single RF transmission
template <typename sampleType>
struct RFArray {
  matx::tensor_t<sampleType, 2> data;
  RFMetadata metadata;

  RFArray(matx::tensor_t<sampleType, 2> _data, RFMetadata _metadata)
      : data{_data}, metadata{_metadata} {}
};
