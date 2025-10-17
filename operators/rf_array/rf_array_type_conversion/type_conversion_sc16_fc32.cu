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
#include <matx.h>
#include <cmath>

#include "holoscan/holoscan.hpp"
#include "rf_array/rf_array.h"
#include "rf_array/type_conversion_sc16_fc32.h"

namespace holoscan::ops {

// ----- TypeConversionComplexIntToFloat ---------------------------------------------------
void TypeConversionComplexIntToFloat::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray<sample_t>>>("rf_in");
  spec.output<std::shared_ptr<RFArray<complex_t>>>("rf_out");
}

void TypeConversionComplexIntToFloat::initialize() {
  HOLOSCAN_LOG_INFO("TypeConversionComplexIntToFloat::initialize()");
  holoscan::Operator::initialize();

  HOLOSCAN_LOG_INFO("TypeConversionComplexIntToFloat::initialize() done");
}

/**
 * @brief Convert complex integer representation to floating point
 */
void TypeConversionComplexIntToFloat::compute(InputContext& op_input, OutputContext& op_output,
                                              ExecutionContext&) {
  HOLOSCAN_LOG_TRACE("TypeConversionComplexIntToFloat::compute() called");
  auto in_ptr_maybe = op_input.receive<std::shared_ptr<RFArray<sample_t>>>("rf_in");
  cudaStream_t stream = op_input.receive_cuda_stream("rf_in", true, false);

  int num_emitted = 0;
  while (in_ptr_maybe) {
    auto in_ptr = in_ptr_maybe.value();
    HOLOSCAN_LOG_TRACE("Dim: {}, {}", in_ptr->data.Size(0), in_ptr->data.Size(1));

    // convert the data from complex int to complex float
    auto new_shp = in_ptr->data.Shape();
    new_shp[1] = 2 * new_shp[1];
    auto in_data_float_view = in_ptr->data.View<real_t, 2, typeof(new_shp)>(std::move(new_shp));
    auto in_data_float =
        matx::as_float(in_data_float_view) / (std::numeric_limits<real_t>::max() - 1);

    auto complex_data =
        matx::make_tensor<complex_t>(in_ptr->data.Shape(), matx::MATX_ASYNC_DEVICE_MEMORY, stream);
    auto out_real = complex_data.RealView();
    auto out_imag = complex_data.ImagView();
    (out_real = matx::slice(in_data_float, {0, 0}, {matx::matxEnd, matx::matxEnd}, {1, 2}))
        .run(stream);
    (out_imag = matx::slice(in_data_float, {0, 1}, {matx::matxEnd, matx::matxEnd}, {1, 2}))
        .run(stream);

    auto out_ptr = std::make_shared<RFArray<complex_t>>(complex_data, in_ptr->metadata);
    op_output.emit(out_ptr, "rf_out");
    num_emitted++;
    if (num_emitted >= op_output.outputs()["rf_out"]->queue_size()) {
      break;
    }

    // see if we have another array on the receive buffer
    in_ptr_maybe = op_input.receive<std::shared_ptr<RFArray<sample_t>>>("rf_in");
  }
}

}  // namespace holoscan::ops
