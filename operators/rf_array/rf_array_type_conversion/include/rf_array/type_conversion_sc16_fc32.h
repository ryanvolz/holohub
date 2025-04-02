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

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class TypeConversionComplexIntToFloat : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TypeConversionComplexIntToFloat)

  TypeConversionComplexIntToFloat() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Convert complex integer representation to floating point
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;
};  // TypeConversionComplexIntToFloat

}  // namespace holoscan::ops
