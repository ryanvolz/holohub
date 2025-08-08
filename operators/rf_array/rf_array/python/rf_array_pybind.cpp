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
#include <pybind11/pybind11.h>

#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "./rf_array_types_pydoc.hpp"
#include "rf_array/rf_array.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

#ifdef RF_ARRAY_DIGITAL_RF
void bind_rf_array_digital_rf(py::module&);
#endif
#ifdef RF_ARRAY_NET_CONNECTOR_ADVANCED
void bind_rf_array_net_connector_advanced(py::module&);
#endif
#ifdef RF_ARRAY_NET_CONNECTOR_BASIC
void bind_rf_array_net_connector_basic(py::module&);
#endif
#ifdef RF_ARRAY_RESAMPLE
void bind_rf_array_resample(py::module&);
#endif
#ifdef RF_ARRAY_ROTATOR
void bind_rf_array_rotator(py::module&);
#endif
#ifdef RF_ARRAY_SUBCHANNEL_SELECT
void bind_rf_array_subchannel_select(py::module&);
#endif
#ifdef RF_ARRAY_TYPE_CONVERSION
void bind_rf_array_type_conversion(py::module&);
#endif

PYBIND11_MODULE(_rf_array, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _rf_array
        .. autosummary::
           :toctree: _generate
           RFArray_sc16
           RFArray_fc32
           RFMetadata
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<RFMetadata>(m, "RFMetadata", doc::RFArrayTypes::doc_RFMetadata_python)
      .def(py::init<uint64_t, uint64_t, uint64_t, double>(),
           "sample_idx"_a,
           "sample_rate_numerator"_a,
           "sample_rate_denominator"_a,
           "center_freq"_a,
           doc::RFArrayTypes::doc_RFMetadata_python)
      .def_readwrite("sample_idx", &RFMetadata::sample_idx)
      .def_readwrite("sample_rate_numerator", &RFMetadata::sample_rate_numerator)
      .def_readwrite("sample_rate_denominator", &RFMetadata::sample_rate_denominator)
      .def_readwrite("center_freq", &RFMetadata::center_freq);

  py::class_<RFArray<complex_int_type>>(m, "RFArray_sc16", doc::RFArrayTypes::doc_RFArray_python)
      .def(py::init([](holoscan::Tensor data, RFMetadata metadata) {
             matx::tensor_t<complex_int_type, 2> data_matx;
             matx::make_tensor(data_matx, *data.to_dlpack());
             return RFArray(data_matx, metadata);
           }),
           "data"_a,
           "metadata"_a,
           doc::RFArrayTypes::doc_RFArray_python)
      .def_property_readonly(
          "data",
          [](const RFArray<complex_int_type>& a) { return holoscan::Tensor(a.data.ToDlPack()); },
          py::return_value_policy::take_ownership)
      .def_readonly("metadata", &RFArray<complex_int_type>::metadata);

  py::class_<RFArray<complex_t>>(m, "RFArray_fc32", doc::RFArrayTypes::doc_RFArray_python)
      .def(py::init([](holoscan::Tensor data, RFMetadata metadata) {
             matx::tensor_t<complex_t, 2> data_matx;
             matx::make_tensor(data_matx, *data.to_dlpack());
             return RFArray(data_matx, metadata);
           }),
           "data"_a,
           "metadata"_a,
           doc::RFArrayTypes::doc_RFArray_python)
      .def_property_readonly(
          "data",
          [](const RFArray<complex_t>& a) { return holoscan::Tensor(a.data.ToDlPack()); },
          py::return_value_policy::take_ownership)
      .def_readonly("metadata", &RFArray<complex_t>::metadata);

  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<std::shared_ptr<RFArray<complex_t>>>(
        "std::shared_ptr<RFArray<complex_t>>"s);
    registry.add_emitter_receiver<std::shared_ptr<RFArray<complex_int_type>>>(
        "std::shared_ptr<RFArray<complex_int_type>>"s);
    registry.add_emitter_receiver<std::shared_ptr<RFArray<sample_t>>>(
        "std::shared_ptr<RFArray<sample_t>>"s);
  });

#ifdef RF_ARRAY_DIGITAL_RF
  bind_rf_array_digital_rf(m);
#endif
#ifdef RF_ARRAY_NET_CONNECTOR_ADVANCED
  bind_rf_array_net_connector_advanced(m);
#endif
#ifdef RF_ARRAY_NET_CONNECTOR_BASIC
  bind_rf_array_net_connector_basic(m);
#endif
#ifdef RF_ARRAY_RESAMPLE
  bind_rf_array_resample(m);
#endif
#ifdef RF_ARRAY_ROTATOR
  bind_rf_array_rotator(m);
#endif
#ifdef RF_ARRAY_SUBCHANNEL_SELECT
  bind_rf_array_subchannel_select(m);
#endif
#ifdef RF_ARRAY_TYPE_CONVERSION
  bind_rf_array_type_conversion(m);
#endif
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan::ops
