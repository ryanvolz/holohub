/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "process.h"

namespace holoscan::ops {

// ----- ComplexIntToFloatOp ---------------------------------------------------
void ComplexIntToFloatOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray>>("rf_in");
  spec.output<std::shared_ptr<ComplexRFArray>>("rf_out");
  spec.param<uint16_t>(num_cycles,
                       "num_cycles",
                       "Number of cycles",
                       "Number of cycles of num_samples to group in processing",
                       {});
  spec.param<uint16_t>(num_samples,
                       "num_samples",
                       "Number of samples",
                       "Number of samples per cycle to group in processing",
                       {});
  spec.param<uint16_t>(num_subchannels,
                       "num_subchannels",
                       "Number of subchannels",
                       "Number of IQ subchannels per sample time instance",
                       {});
}

void ComplexIntToFloatOp::initialize() {
  HOLOSCAN_LOG_INFO("ComplexIntToFloatOp::initialize()");
  holoscan::Operator::initialize();

  make_tensor(complex_data, {num_cycles, num_samples, num_subchannels});

  HOLOSCAN_LOG_INFO("ComplexIntToFloatOp::initialize() done");
}

/**
 * @brief Convert complex integer representation to floating point
 */
void ComplexIntToFloatOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext&) {
  HOLOSCAN_LOG_INFO("ComplexIntToFloatOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray>>("rf_in").value();
  cudaStream_t stream = in->stream;

  HOLOSCAN_LOG_INFO("Dim: {}, {}, {}", in->data.Size(0), in->data.Size(1), in->data.Size(2));

  // convert the data from complex int to complex float
  auto new_shp = in->data.Shape();
  new_shp[1] = 2 * new_shp[1];
  auto in_data_float_view = in->data.View<real_t, 3, typeof(new_shp)>(std::move(new_shp));
  auto in_data_float =
      matx::as_float(in_data_float_view) / (std::numeric_limits<real_t>::max() - 1);
  (complex_data.RealView() =
       matx::slice(in_data_float, {0, 0, 0}, {matxEnd, matxEnd, matxEnd}, {1, 1, 2}))
      .run(stream);
  (complex_data.ImagView() =
       matx::slice(in_data_float, {0, 0, 1}, {matxEnd, matxEnd, matxEnd}, {1, 1, 2}))
      .run(stream);

  auto params =
      std::make_shared<ComplexRFArray>(complex_data, in->sample_idx, in->channel_idx, stream);
  op_output.emit(params, "pc_out");
}

}  // namespace holoscan::ops
