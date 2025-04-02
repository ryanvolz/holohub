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

#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class RotatorScheduled : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RotatorScheduled)

  RotatorScheduled() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Rotator with frequency shift controlled by a fixed schedule
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<double> cycle_duration_secs;
  Parameter<double> cycle_start_timestamp;
  Parameter<YAML::Node> schedule_yaml;

  std::vector<std::pair<double, double>> schedule;
  size_t schedule_idx = 0;
};  // RotatorScheduled

}  // namespace holoscan::ops
