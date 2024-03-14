/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda/std/complex>
#include "holoscan/holoscan.hpp"
#include "matx.h"

using namespace matx;
using float_t = float;
using complex_t = cuda::std::complex<float_t>;

using real_t = int32_t;
struct complex_int_type {
  real_t r;
  real_t i;
} __attribute__((__packed__));
using sample_t = complex_int_type;

// Meta data for received signal
struct RfMetaData {
  uint64_t sample_idx;
  uint64_t sample_rate_numerator;
  uint64_t sample_rate_denominator;
  uint32_t channel_idx;
  uint32_t num_subchannels;
  uint32_t pkt_samples;
  uint16_t bits_per_int;
  unsigned is_complex : 1;
  unsigned reserved0 : 7;
  uint8_t reserved1;
  uint64_t reserved2;
  uint64_t reserved3;
  uint64_t reserved4;
} __attribute__((__packed__));

// Represents a single RF transmission
struct RFArray {
  tensor_t<sample_t, 3> data;
  uint64_t sample_idx;
  uint16_t channel_idx;
  cudaStream_t stream;

  RFArray(tensor_t<sample_t, 3> _data, uint64_t _sample_idx, uint16_t _channel_idx,
          cudaStream_t _stream)
      : data{_data}, sample_idx{_sample_idx}, channel_idx{channel_idx}, stream{_stream} {}
};
