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

class TypeConversionComplexFloatToInt : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TypeConversionComplexFloatToInt)

  TypeConversionComplexFloatToInt() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Convert complex floating point representation to integer
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;
};  // TypeConversionComplexFloatToInt

}  // namespace holoscan::ops
