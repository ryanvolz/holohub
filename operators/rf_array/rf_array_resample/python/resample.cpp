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
#include "./resample_pydoc.hpp"
#include "rf_array/resample_poly.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyResamplePoly : public ResamplePoly {
 public:
  // Inherit the constructors
  using ResamplePoly::ResamplePoly;

  // Define a constructor that fully initializes the object.
  PyResamplePoly(Fragment* fragment, const py::args& args, uint32_t chunk_size,
                 uint16_t num_subchannels, std::vector<float, std::allocator<float>>& filter_coefs,
                 uint16_t up = 1, uint16_t down = 1, const std::string& name = "resample_poly")
      : ResamplePoly(ArgList{
            Arg{"chunk_size", chunk_size},
            Arg{"num_subchannels", num_subchannels},
            Arg{"filter_coefs", filter_coefs},
            Arg{"up", up},
            Arg{"down", down},
        }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

void bind_rf_array_resample(py::module& m) {
  py::class_<ResamplePoly, PyResamplePoly, Operator, std::shared_ptr<ResamplePoly>>(
      m, "ResamplePoly", doc::Resample::doc_ResamplePoly_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    uint16_t,
                    std::vector<float, std::allocator<float>>&,
                    uint16_t,
                    uint16_t,
                    const std::string&>(),
           "fragment"_a,
           "chunk_size"_a,
           "num_subchannels"_a,
           "filter_coefs"_a,
           "up"_a = 1,
           "down"_a = 1,
           "name"_a = "resample_poly"s,
           doc::Resample::doc_ResamplePoly_python);
}

}  // namespace holoscan::ops
