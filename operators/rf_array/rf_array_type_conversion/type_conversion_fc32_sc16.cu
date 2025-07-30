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
#include "rf_array/type_conversion_fc32_sc16.h"

namespace holoscan::ops {

// ----- TypeConversionComplexFloatToInt ---------------------------------------------------
void TypeConversionComplexFloatToInt::setup(OperatorSpec& spec) {
  spec.input<RFMessage<complex_t>>("rf_in");
  spec.output<RFMessage<sample_t>>("rf_out");
}

void TypeConversionComplexFloatToInt::initialize() {
  HOLOSCAN_LOG_INFO("TypeConversionComplexFloatToInt::initialize()");
  holoscan::Operator::initialize();

  HOLOSCAN_LOG_INFO("TypeConversionComplexFloatToInt::initialize() done");
}

/**
 * @brief Convert complex integer representation to floating point
 */
void TypeConversionComplexFloatToInt::compute(InputContext& op_input, OutputContext& op_output,
                                              ExecutionContext&) {
  HOLOSCAN_LOG_TRACE("TypeConversionComplexFloatToInt::compute() called");
  auto in_vector = op_input.receive<RFMessage<complex_t>>("rf_in").value();
  cudaStream_t stream = op_input.receive_cuda_stream("rf_in", true, false);

  RFMessage<sample_t> out_msg;

  for (auto in : in_vector) {
    HOLOSCAN_LOG_TRACE("Dim: {}, {}", in->data.Size(0), in->data.Size(1));

    // convert the data from complex float to complex int
    auto float_shp = in->data.Shape();
    float_shp[1] = 2 * float_shp[1];
    auto in_data_float_view = in->data.View<float_t, 2, typeof(float_shp)>(std::move(float_shp));

    auto complex_int_data =
        matx::make_tensor<sample_t>(in->data.Shape(), matx::MATX_ASYNC_DEVICE_MEMORY, stream);
    auto real_shp = in->data.Shape();
    real_shp[1] = 2 * real_shp[1];
    auto out_data_int_view =
        complex_int_data.View<real_t, 2, typeof(real_shp)>(std::move(real_shp));

    (out_data_int_view =
         matx::as_int16(in_data_float_view * (std::numeric_limits<real_t>::max() - 1)))
        .run(stream);

    auto params = std::make_shared<RFArray<sample_t>>(complex_int_data, in->metadata);
    out_msg.push_back(params);
  }
  op_output.emit(out_msg, "rf_out");
}

}  // namespace holoscan::ops
