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

#include <memory>
#include <vector>

#include <matx.h>

#include "holoscan/holoscan.hpp"
#include "rf_array/rf_array.h"

namespace holoscan::ops {

class ResamplePoly : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ResamplePoly)

  ResamplePoly() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Polyphase resampling by up/down rate
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<uint32_t> chunk_size;
  Parameter<uint16_t> num_subchannels;
  Parameter<uint16_t> up;
  Parameter<uint16_t> down;
  Parameter<std::vector<float, std::allocator<float>>> filter_coefs;

  std::shared_ptr<RFArray<complex_t>> prior_input;
  uint32_t pad_size;
  uint32_t out_pad_size;
  uint32_t out_chunk_size;
  matx::tensor_t<float, 1> filter;
  matx::tensor_t<complex_t, 2> padded_out_data;
  matx::tensor_t<complex_t, 2> padded_data;
};  // ResamplePoly

}  // namespace holoscan::ops
