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
#include <cstdint>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../../../operator_util.hpp"
#include "./rotator_pydoc.hpp"
#include "rf_array/rotator_scheduled.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyRotatorScheduled : public RotatorScheduled {
 public:
  // Inherit the constructors
  using RotatorScheduled::RotatorScheduled;

  // Define a constructor that fully initializes the object.
  PyRotatorScheduled(Fragment* fragment, const py::args& args, double cycle_duration_secs,
                     double cycle_start_timestamp,
                     std::list<std::map<std::string, double>>& schedule,
                     const std::string& name = "rotator_scheduled")
      : RotatorScheduled(ArgList{
            Arg{"cycle_duration_secs", cycle_duration_secs},
            Arg{"cycle_start_timestamp", cycle_start_timestamp},
            Arg{"schedule", schedule},
        }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

void bind_rf_array_rotator(py::module& m) {
  py::class_<RotatorScheduled, PyRotatorScheduled, Operator, std::shared_ptr<RotatorScheduled>>(
      m, "RotatorScheduled", doc::Rotator::doc_RotatorScheduled_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    double,
                    double,
                    std::list<std::map<std::string, double>>&,
                    const std::string&>(),
           "fragment"_a,
           "cycle_duration_secs"_a,
           "cycle_start_timestamp"_a,
           "schedule"_a,
           "name"_a = "rotator_scheduled"s,
           doc::Rotator::doc_RotatorScheduled_python);
}

}  // namespace holoscan::ops
