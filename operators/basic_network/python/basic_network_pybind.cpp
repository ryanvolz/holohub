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
#include <holoscan/python/core/emitter_receiver_registry.hpp>

#include "../../operator_util.hpp"
#include "./basic_network_pydoc.hpp"
#include "basic_network_operator_common.h"
#include "basic_network_operator_rx.h"
#include "basic_network_operator_tx.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

class PyBasicNetworkOpRx : public BasicNetworkOpRx {
 public:
  // Inherit the constructors
  using BasicNetworkOpRx::BasicNetworkOpRx;

  // Define a constructor that fully initializes the object.
  PyBasicNetworkOpRx(Fragment* fragment, const py::args& args, std::string& ip_addr,
                     uint16_t dst_port, std::string& l4_proto, uint32_t batch_size,
                     uint16_t max_payload_size, const std::string& name = "basic_network_rx")
      : BasicNetworkOpRx(ArgList{
            Arg{"ip_addr", ip_addr},
            Arg{"dst_port", dst_port},
            Arg{"l4_proto", l4_proto},
            Arg{"batch_size", batch_size},
            Arg{"max_payload_size", max_payload_size},
        }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyBasicNetworkOpTx : public BasicNetworkOpTx {
 public:
  // Inherit the constructors
  using BasicNetworkOpTx::BasicNetworkOpTx;

  // Define a constructor that fully initializes the object.
  PyBasicNetworkOpTx(Fragment* fragment, const py::args& args, std::string& ip_addr,
                     uint16_t dst_port, std::string& l4_proto, uint16_t max_payload_size,
                     uint32_t min_ipg_ns, int32_t retry_connect = 1,
                     const std::string& name = "basic_network_tx")
      : BasicNetworkOpTx(ArgList{
            Arg{"ip_addr", ip_addr},
            Arg{"dst_port", dst_port},
            Arg{"l4_proto", l4_proto},
            Arg{"max_payload_size", max_payload_size},
            Arg{"min_ipg_ns", min_ipg_ns},
            Arg{"retry_connect", retry_connect},
        }) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_basic_network, m) {
  m.doc() = R"pbdoc(
          Holoscan SDK Python Bindings
          ---------------------------------------
          .. currentmodule:: _basic_network
          .. autosummary::
             :toctree: _generate
             NetworkOpBurstParams
             BasicNetworkOpRx
             BasicNetworkOpTx
      )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<NetworkOpBurstParams>(
      m, "NetworkOpBurstParams", doc::BasicNetwork::doc_NetworkOpBurstParams_python)
      .def(py::init([](py::buffer data, uint32_t num_pkts) {
             // copy into our own heap memory because BasicNetworkOpTx expects to delete[]
             // the packet data when it is done sending
             auto buffer_info = data.request();
             auto buf_len = buffer_info.itemsize * buffer_info.size;
             auto buf_ptr = reinterpret_cast<uint8_t*>(buffer_info.ptr);
             auto mem = new uint8_t[buf_len];
             std::memcpy(mem, buf_ptr, buf_len);
             return NetworkOpBurstParams(mem, buf_len, num_pkts);
           }),
           "data"_a,
           "num_pkts"_a,
           doc::BasicNetwork::doc_NetworkOpBurstParams_python)
      .def_property_readonly(
          "data",
          [](const NetworkOpBurstParams& b) { return py::memoryview::from_memory(b.data, b.len); })
      .def_readonly("num_pkts", &NetworkOpBurstParams::num_pkts);

  py::class_<BasicNetworkOpRx, PyBasicNetworkOpRx, Operator, std::shared_ptr<BasicNetworkOpRx>>(
      m, "BasicNetworkOpRx", doc::BasicNetwork::doc_BasicNetworkOpRx_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::string&,
                    uint16_t,
                    std::string&,
                    uint32_t,
                    uint16_t,
                    const std::string&>(),
           "fragment"_a,
           "ip_addr"_a,
           "dst_port"_a,
           "l4_proto"_a,
           "batch_size"_a,
           "max_payload_size"_a,
           "name"_a = "basic_network_rx"s,
           doc::BasicNetwork::doc_BasicNetworkOpRx_python);

  py::class_<BasicNetworkOpTx, PyBasicNetworkOpTx, Operator, std::shared_ptr<BasicNetworkOpTx>>(
      m, "BasicNetworkOpTx", doc::BasicNetwork::doc_BasicNetworkOpTx_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::string&,
                    uint16_t,
                    std::string&,
                    uint16_t,
                    uint32_t,
                    int32_t,
                    const std::string&>(),
           "fragment"_a,
           "ip_addr"_a,
           "dst_port"_a,
           "l4_proto"_a,
           "max_payload_size"_a,
           "min_ipg_ns"_a,
           "retry_connect"_a = 1,
           "name"_a = "basic_network_tx"s,
           doc::BasicNetwork::doc_BasicNetworkOpTx_python);

  // Import the emitter/receiver registry from holoscan.core and pass it to this function to
  // register this new C++ type with the SDK.
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<std::shared_ptr<NetworkOpBurstParams>>(
        "std::shared_ptr<NetworkOpBurstParams>"s);
  });
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan::ops
