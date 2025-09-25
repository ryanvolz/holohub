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
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include "../../../operator_util.hpp"
#include "./digital_rf_pydoc.hpp"
#include "rf_array/digital_rf_sink.h"
#include "rf_array/rf_array.h"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan::ops {

template <typename sampleType>
class PyDigitalRFSink : public DigitalRFSink<sampleType> {
 public:
  // Inherit the constructors
  using DigitalRFSink<sampleType>::DigitalRFSink;

  // Define a constructor that fully initializes the object.
  PyDigitalRFSink(Fragment* fragment, const py::args& args, uint32_t chunk_size,
                  uint16_t num_subchannels, const std::filesystem::path& output_path,
                  const std::string& channel_dir, uint64_t subdir_cadence_secs = 3600,
                  uint64_t file_cadence_millisecs = 1000,
                  std::optional<const std::string> uuid = std::nullopt, int compression_level = 0,
                  bool checksum = false, bool is_continuous = true, bool marching_dots = false,
                  const std::string& name = "digital_rf_sink")
      : DigitalRFSink<sampleType>(ArgList{
            Arg{"chunk_size", chunk_size},
            Arg{"num_subchannels", num_subchannels},
            Arg{"output_path", output_path.string()},
            Arg{"channel_dir", channel_dir},
            Arg{"subdir_cadence_secs", subdir_cadence_secs},
            Arg{"file_cadence_millisecs", file_cadence_millisecs},
            Arg{"compression_level", compression_level},
            Arg{"checksum", checksum},
            Arg{"is_continuous", is_continuous},
            Arg{"marching_dots", marching_dots},
        }) {
    if (uuid.has_value()) {
      this->add_arg(Arg{"uuid", uuid.value()});
    } else {
      py::object uuid = py::module_::import("uuid");
      const std::string uuid_str = uuid.attr("uuid4")().attr("hex").cast<const std::string>();
      this->add_arg(Arg{"uuid", uuid_str});
    }
    add_positional_condition_and_resource_args(this, args);
    this->name_ = name;
    this->fragment_ = fragment;
    this->spec_ = std::make_shared<OperatorSpec>(fragment);
    DigitalRFSink<sampleType>::setup(*this->spec_.get());
  }
};

void bind_rf_array_digital_rf(py::module& m) {
  py::class_<DigitalRFSink<complex_int_type>,
             PyDigitalRFSink<complex_int_type>,
             Operator,
             std::shared_ptr<DigitalRFSink<complex_int_type>>>(
      m, "DigitalRFSink_sc16", doc::DigitalRF::doc_DigitalRFSink_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    uint16_t,
                    const std::filesystem::path&,
                    const std::string&,
                    uint64_t,
                    uint64_t,
                    std::optional<const std::string>,
                    int,
                    bool,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "chunk_size"_a,
           "num_subchannels"_a,
           "output_path"_a,
           "channel_dir"_a,
           "subdir_cadence_secs"_a = 3600,
           "file_cadence_millisecs"_a = 1000,
           "uuid"_a = py::none(),
           "compression_level"_a = 0,
           "checksum"_a = false,
           "is_continuous"_a = true,
           "marching_dots"_a = false,
           "name"_a = "digital_rf_sink"s,
           doc::DigitalRF::doc_DigitalRFSink_python);

  py::class_<DigitalRFSink<complex_t>,
             PyDigitalRFSink<complex_t>,
             Operator,
             std::shared_ptr<DigitalRFSink<complex_t>>>(
      m, "DigitalRFSink_fc32", doc::DigitalRF::doc_DigitalRFSink_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    uint16_t,
                    const std::filesystem::path&,
                    const std::string&,
                    uint64_t,
                    uint64_t,
                    std::optional<const std::string>,
                    int,
                    bool,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "chunk_size"_a,
           "num_subchannels"_a,
           "output_path"_a,
           "channel_dir"_a,
           "subdir_cadence_secs"_a = 3600,
           "file_cadence_millisecs"_a = 1000,
           "uuid"_a = py::none(),
           "compression_level"_a = 0,
           "checksum"_a = false,
           "is_continuous"_a = true,
           "marching_dots"_a = false,
           "name"_a = "digital_rf_sink"s,
           doc::DigitalRF::doc_DigitalRFSink_python);
}

}  // namespace holoscan::ops
