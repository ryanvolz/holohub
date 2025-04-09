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
#include "./net_connector_advanced_pydoc.hpp"
#include "rf_array/net_connector_advanced.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyNetConnectorAdvanced : public NetConnectorAdvanced {
 public:
  // Inherit the constructors
  using NetConnectorAdvanced::NetConnectorAdvanced;

  // Define a constructor that fully initializes the object.
  PyNetConnectorAdvanced(Fragment* fragment, const py::args& args, uint16_t buffer_size,
                         uint32_t num_samples, uint16_t num_subchannels, uint32_t batch_size = 1000,
                         uint16_t max_packet_size = 9000, bool gpu_direct = true,
                         bool use_header_data_split = true,
                         const std::string& name = "net_connector_advanced")
      : NetConnectorAdvanced(ArgList{
            Arg{"buffer_size", buffer_size},
            Arg{"num_samples", num_samples},
            Arg{"num_subchannels", num_subchannels},
            Arg{"batch_size", batch_size},
            Arg{"max_packet_size", max_packet_size},
            Arg{"gpu_direct", gpu_direct},
            Arg{"use_header_data_split", use_header_data_split},
        }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

void bind_rf_array_net_connector_advanced(py::module& m) {
  py::class_<NetConnectorAdvanced,
             PyNetConnectorAdvanced,
             Operator,
             std::shared_ptr<NetConnectorAdvanced>>(
      m, "NetConnectorAdvanced", doc::NetConnectorAdvanced::doc_NetConnectorAdvanced_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint16_t,
                    uint32_t,
                    uint16_t,
                    uint32_t,
                    uint16_t,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "buffer_size"_a,
           "num_samples"_a,
           "num_subchannels"_a,
           "batch_size"_a = 1000,
           "max_packet_size"_a = 9000,
           "gpu_direct"_a = true,
           "use_header_data_split"_a = true,
           "name"_a = "net_connector_advanced"s,
           doc::NetConnectorAdvanced::doc_NetConnectorAdvanced_python);
}

}  // namespace holoscan::ops
