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
#include <map>
#include <memory>
#include <optional>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../../../operator_util.hpp"
#include "./net_connector_basic_pydoc.hpp"
#include "rf_array/net_connector_basic.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyNetConnectorBasic : public NetConnectorBasic {
 public:
  // Inherit the constructors
  using NetConnectorBasic::NetConnectorBasic;

  // Define a constructor that fully initializes the object.
  PyNetConnectorBasic(Fragment* fragment, const py::args& args, uint16_t buffer_size,
                      uint32_t num_samples, uint16_t num_subchannels, double freq_idx_scaling = 1,
                      double freq_idx_offset = 0, bool apply_conjugate = false,
                      bool spoof_header = false, uint16_t packet_skip_bytes = 0,
                      std::optional<std::map<std::string, uint64_t>> header_metadata = std::nullopt,
                      uint32_t batch_size = 1000, uint16_t max_packet_size = 9000,
                      uint16_t batch_capacity = 4, uint32_t no_output_warn_interval = 30,
                      bool debug_print = false, int16_t packet_stream_priority = -1,
                      const std::string& name = "net_connector_basic")
      : NetConnectorBasic(ArgList{
            Arg{"buffer_size", buffer_size},
            Arg{"num_samples", num_samples},
            Arg{"num_subchannels", num_subchannels},
            Arg{"freq_idx_scaling", freq_idx_scaling},
            Arg{"freq_idx_offset", freq_idx_offset},
            Arg{"apply_conjugate", apply_conjugate},
            Arg{"spoof_header", spoof_header},
            Arg{"packet_skip_bytes", packet_skip_bytes},
            Arg{"batch_size", batch_size},
            Arg{"max_packet_size", max_packet_size},
            Arg{"batch_capacity", batch_capacity},
            Arg{"no_output_warn_interval", no_output_warn_interval},
            Arg{"debug_print", debug_print},
            Arg{"packet_stream_priority", packet_stream_priority},
        }) {
    if (header_metadata.has_value()) {
      this->add_arg(Arg{"header_metadata", header_metadata.value()});
    }
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

void bind_rf_array_net_connector_basic(py::module& m) {
  py::class_<NetConnectorBasic, PyNetConnectorBasic, Operator, std::shared_ptr<NetConnectorBasic>>(
      m, "NetConnectorBasic", doc::NetConnectorBasic::doc_NetConnectorBasic_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint16_t,
                    uint32_t,
                    uint16_t,
                    double,
                    double,
                    bool,
                    bool,
                    uint16_t,
                    std::optional<std::map<std::string, uint64_t>>,
                    uint32_t,
                    uint16_t,
                    uint16_t,
                    uint32_t,
                    bool,
                    int16_t,
                    const std::string&>(),
           "fragment"_a,
           "buffer_size"_a,
           "num_samples"_a,
           "num_subchannels"_a,
           "freq_idx_scaling"_a = 1,
           "freq_idx_offset"_a = 0,
           "apply_conjugate"_a = false,
           "spoof_header"_a = false,
           "packet_skip_bytes"_a = 0,
           "header_metadata"_a = py::none(),
           "batch_size"_a = 1000,
           "max_packet_size"_a = 9000,
           "batch_capacity"_a = 4,
           "no_output_warn_interval"_a = 30,
           "debug_print"_a = false,
           "packet_stream_priority"_a = -1,
           "name"_a = "net_connector_basic"s,
           doc::NetConnectorBasic::doc_NetConnectorBasic_python);
}

}  // namespace holoscan::ops
