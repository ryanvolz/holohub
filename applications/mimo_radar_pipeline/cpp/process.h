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

#include "common.h"

// ---------- Structures ----------
struct ComplexRFArray {
  tensor_t<complex_t, 3> data;
  tensor_t<RfMetaData, 0> metadata;
  cudaStream_t stream;

  ComplexRFArray(tensor_t<complex_t, 3> _data, tensor_t<RfMetaData, 0> _metadata,
                 cudaStream_t _stream)
      : data{_data}, metadata{_metadata}, stream{_stream} {}
};

// ---------- Operators ----------
namespace holoscan::ops {

class ComplexIntToFloatOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ComplexIntToFloatOp)

  ComplexIntToFloatOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Convert complex integer representation to floating point
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<uint16_t> num_cycles;
  Parameter<uint16_t> num_samples;
  Parameter<uint16_t> num_subchannels;

  tensor_t<complex_t, 3> complex_data;
};  // ComplexIntToFloatOp

}  // namespace holoscan::ops
