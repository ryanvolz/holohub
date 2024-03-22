/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "../adv_networking_rx.h"
#include "./adv_networking_rx_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyAdvConnectorOpRx : public AdvConnectorOpRx {
 public:
  /* Inherit the constructors */
  using AdvConnectorOpRx::AdvConnectorOpRx;

  // Define a constructor that fully initializes the object.
  PyAdvConnectorOpRx(Fragment* fragment,
                     const std::string& name = "advanced_network_connector_rx") {
    this->add_arg(fragment->from_config("rx_params"));
    this->add_arg(fragment->from_config("radar_pipeline"));
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_advanced_network_connector_rx, m) {
  m.doc() = R"pbdoc(
Advanced Networking Connector RX Python Bindings
------------------------------------------------
.. currentmodule:: _advanced_network_connector_rx
.. autosummary::
    :toctree: _generate
    add
    subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<AdvConnectorOpRx, PyAdvConnectorOpRx, Operator, std::shared_ptr<AdvConnectorOpRx>>(
      m, "AdvConnectorOpRx", doc::AdvConnectorOpRx::doc_AdvConnectorOpRx)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "advanced_network_connector_rx"s,
           doc::AdvConnectorOpRx::doc_AdvConnectorOpRx_python)
      .def("initialize", &AdvConnectorOpRx::initialize, doc::AdvConnectorOpRx::doc_initialize)
      .def("setup", &AdvConnectorOpRx::setup, "spec"_a, doc::AdvConnectorOpRx::doc_setup);

  py::class_<RfMetaData>(m, "RfMetaData")
      .def_readwrite("sample_idx", &RfMetaData::sample_idx)
      .def_readwrite("sample_rate_numerator", &RfMetaData::sample_rate_numerator)
      .def_readwrite("sample_rate_denominator", &RfMetaData::sample_rate_denominator)
      .def_readwrite("channel_idx", &RfMetaData::channel_idx);

  py::class_<RFArray<complex_int_type>>(m, "RFArray_sc16")
      .def_readwrite("data", &RFArray<complex_int_type>::data)
      .def_readwrite("metadata", &RFArray<complex_int_type>::metadata);

  py::class_<RFArray<complex_t>>(m, "RFArray_fc32")
      .def_readwrite("data", &RFArray<complex_t>::data)
      .def_readwrite("metadata", &RFArray<complex_t>::metadata);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
