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
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
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
                 uint16_t num_subchannels, uint16_t up = 1, uint16_t down = 1,
                 double outrate_cutoff = 1.0, double outrate_transition_width = 0.2,
                 double attenuation_db = 100, std::optional<uint16_t> numtaps = std::nullopt,
                 std::optional<double> kaiser_beta = std::nullopt,
                 std::optional<py::array_t<float>> filter_coefs = std::nullopt,
                 const std::string& name = "resample_poly")
      : ResamplePoly(ArgList{
            Arg{"chunk_size", chunk_size},
            Arg{"num_subchannels", num_subchannels},
            Arg{"up", up},
            Arg{"down", down},
        }) {
    add_positional_condition_and_resource_args(this, args);
    py::array_t<float> taps;
    if (filter_coefs.has_value()) {
      taps = filter_coefs.value();
    } else {
      py::object np = py::module_::import("numpy");
      py::object ss = py::module_::import("scipy.signal");
      auto cutoff = outrate_cutoff / down;
      auto transition_width = outrate_transition_width / down;
      py::tuple numtaps_beta = ss.attr("kaiserord")(attenuation_db, transition_width);
      uint16_t numtaps_val;
      if (numtaps.has_value()) {
        numtaps_val = numtaps.value();
      } else {
        numtaps_val = numtaps_beta[0].cast<uint16_t>();
        // round up to nearest even-order (Type I) filter (odd number of taps)
        numtaps_val = static_cast<uint16_t>(std::ceil((numtaps_val - 1) / 2.0)) * 2 + 1;
      }
      double kaiser_beta_val;
      if (kaiser_beta.has_value()) {
        kaiser_beta_val = kaiser_beta.value();
      } else {
        kaiser_beta_val = numtaps_beta[1].cast<double>();
      }
      py::object firwin_taps = ss.attr("firwin")(
          numtaps_val, cutoff, "window"_a = py::make_tuple("kaiser", kaiser_beta_val));
      taps = np.attr("multiply")(up, firwin_taps).attr("astype")("float32");
    }
    this->add_arg(Arg{"filter_coefs", taps.cast<std::vector<float, std::allocator<float>>>()});
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_);
  }
};

void bind_rf_array_resample(py::module& m) {
  py::class_<ResamplePoly, PyResamplePoly, Operator, std::shared_ptr<ResamplePoly>>(
      m, "ResamplePoly", doc::Resample::doc_ResamplePoly_python)
      .def(py::init<Fragment*,
                    const py::args&,
                    uint32_t,
                    uint16_t,
                    uint16_t,
                    uint16_t,
                    double,
                    double,
                    double,
                    std::optional<uint16_t>,
                    std::optional<double>,
                    std::optional<py::array_t<float>>,
                    const std::string&>(),
           "fragment"_a,
           "chunk_size"_a,
           "num_subchannels"_a,
           "up"_a = 1,
           "down"_a = 1,
           "outrate_cutoff"_a = 1.0,
           "outrate_transition_width"_a = 0.2,
           "attenuation_db"_a = 100,
           "numtaps"_a = py::none(),
           "kaiser_beta"_a = py::none(),
           "filter_coefs"_a = py::none(),
           "name"_a = "resample_poly"s,
           doc::Resample::doc_ResamplePoly_python);
}

}  // namespace holoscan::ops
