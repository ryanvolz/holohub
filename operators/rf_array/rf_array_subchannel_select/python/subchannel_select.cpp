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
#include "./subchannel_select_pydoc.hpp"
#include "rf_array/subchannel_select.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

template <typename sampleType>
class PySubchannelSelect : public SubchannelSelect<sampleType> {
 public:
  // Inherit the constructors
  using SubchannelSelect<sampleType>::SubchannelSelect;

  // Define a constructor that fully initializes the object.
  PySubchannelSelect(Fragment* fragment, const py::args& args,
                     std::vector<int, std::allocator<int>> subchannel_idx,
                     const std::string& name = "subchannel_select")
      : SubchannelSelect<sampleType>(ArgList{
            Arg{"subchannel_idx", subchannel_idx},
        }) {
    add_positional_condition_and_resource_args(this, args);
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    SubchannelSelect<sampleType>::setup(*this->spec_.get());
  }
};

void bind_rf_array_subchannel_select(py::module& m) {
  py::class_<SubchannelSelect<complex_int_type>,
             PySubchannelSelect<complex_int_type>,
             Operator,
             std::shared_ptr<SubchannelSelect<complex_int_type>>>(
      m, "SubchannelSelect_sc16", doc::SubchannelSelect::doc_SubchannelSelect_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::vector<int, std::allocator<int>>,
                    const std::string&>(),
           "fragment"_a,
           "subchannel_idx"_a,
           "name"_a = "subchannel_select"s,
           doc::SubchannelSelect::doc_SubchannelSelect_python);

  py::class_<SubchannelSelect<complex_t>,
             PySubchannelSelect<complex_t>,
             Operator,
             std::shared_ptr<SubchannelSelect<complex_t>>>(
      m, "SubchannelSelect_fc32", doc::SubchannelSelect::doc_SubchannelSelect_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::vector<int, std::allocator<int>>,
                    const std::string&>(),
           "fragment"_a,
           "subchannel_idx"_a,
           "name"_a = "subchannel_select"s,
           doc::SubchannelSelect::doc_SubchannelSelect_python);
}

}  // namespace holoscan::ops
