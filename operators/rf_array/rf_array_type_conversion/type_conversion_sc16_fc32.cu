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
  spec.input<RFArray<sample_t>>("rf_in");
  spec.output<RFArray<complex_t>>("rf_out");
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
  auto in_maybe = op_input.receive<RFArray<sample_t>>("rf_in");
  cudaStream_t stream = op_input.receive_cuda_stream("rf_in", true, false);

  int num_emitted = 0;
  while (in_maybe) {
    auto in = in_maybe.value();
    HOLOSCAN_LOG_TRACE("Dim: {}, {}", in.data.Size(0), in.data.Size(1));

    // convert the data from complex int to complex float
    auto new_shp = in.data.Shape();
    new_shp[1] = 2 * new_shp[1];
    auto in_data_float_view = in.data.View<real_t, 2, typeof(new_shp)>(std::move(new_shp));
    // View doesn't increment the memory tracker counter to prevent deallocation of in.data's memory
    // when it is destroyed at the end of this compute(), nor does it make the allocator aware that
    // the memory is in use on this operator's stream. So we have to manually set this stream as
    // the active stream for in.data so that deallocation will happen on this stream after all of
    // the work we queue up using this memory, to prevent use after free errors.
    matx::update_stream(in.data.GetStorage().data(), stream);
    auto in_data_float =
        matx::as_float(in_data_float_view) / (std::numeric_limits<real_t>::max() - 1);

    auto complex_data =
        matx::make_tensor<complex_t>(in.data.Shape(), matx::MATX_ASYNC_DEVICE_MEMORY, stream);
    auto out_real = complex_data.RealView();
    auto out_imag = complex_data.ImagView();
    (out_real = matx::slice(in_data_float, {0, 0}, {matx::matxEnd, matx::matxEnd}, {1, 2}))
        .run(stream);
    (out_imag = matx::slice(in_data_float, {0, 1}, {matx::matxEnd, matx::matxEnd}, {1, 2}))
        .run(stream);

    auto out = RFArray<complex_t>(complex_data, in.metadata);
    op_output.emit(out, "rf_out");
    num_emitted++;
    if (num_emitted >= op_output.outputs()["rf_out"]->queue_size()) {
      break;
    }

    // see if we have another array on the receive buffer
    in_maybe = op_input.receive<RFArray<sample_t>>("rf_in");
  }
}

}  // namespace holoscan::ops
