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

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../../../operator_util.hpp"
#include "./type_conversion_pydoc.hpp"
#include "rf_array/type_conversion_sc16_fc32.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyTypeConversionComplexIntToFloat : public TypeConversionComplexIntToFloat {
 public:
  // Inherit the constructors
  using TypeConversionComplexIntToFloat::TypeConversionComplexIntToFloat;

  // Define a constructor that fully initializes the object.
  PyTypeConversionComplexIntToFloat(Fragment* fragment, const py::args& args,
                                    const std::string& name = "type_conversion_sc16_fc32")
      : TypeConversionComplexIntToFloat(ArgList{}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

void bind_rf_array_type_conversion(py::module& m) {
  py::class_<TypeConversionComplexIntToFloat,
             PyTypeConversionComplexIntToFloat,
             Operator,
             std::shared_ptr<TypeConversionComplexIntToFloat>>(
      m,
      "TypeConversionComplexIntToFloat",
      doc::TypeConversion::doc_TypeConversionComplexIntToFloat_python)
      .def(py::init<Fragment*, const py::args&, const std::string&>(),
           "fragment"_a,
           "name"_a = "type_conversion_sc16_fc32"s,
           doc::TypeConversion::doc_TypeConversionComplexIntToFloat_python);
}

}  // namespace holoscan::ops
