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

  RFMetadata(uint64_t _sample_idx, uint64_t _sample_rate_numerator,
             uint64_t _sample_rate_denominator, double _center_freq)
      : sample_idx{_sample_idx},
        sample_rate_numerator{_sample_rate_numerator},
        sample_rate_denominator{_sample_rate_denominator},
        center_freq{_center_freq} {}
};

// Represents a single RF transmission
template <typename sampleType>
struct RFArray {
  matx::tensor_t<sampleType, 2> data;
  RFMetadata metadata;
  std::shared_ptr<holoscan::Tensor> dlpack;

  RFArray(matx::tensor_t<sampleType, 2> _data, RFMetadata _metadata)
      : data{_data}, metadata{_metadata} {
    dlpack = std::make_shared<holoscan::Tensor>(data.ToDlPack());
  }

  // Need custom assignment operators because matx::tensor_t's assignment operators are overridden
  // and can't be used like normal by the default assignment operators
  ~RFArray() = default;
  RFArray(const RFArray& other) = default;
  RFArray(RFArray&& other) = default;
  RFArray& operator=(const RFArray& other) {
    data.Shallow(other.data);
    metadata = other.metadata;
    dlpack = other.dlpack;
    return *this;
  }
  RFArray& operator=(RFArray&& other) {
    // shallow copy is the best we can do for matx::tensor_t data
    data.Shallow(other.data);
    metadata = std::move(other.metadata);
    dlpack = std::move(other.dlpack);
    return *this;
  }
};
