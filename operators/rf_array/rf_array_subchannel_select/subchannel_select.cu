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

#include "holoscan/holoscan.hpp"
#include "rf_array/rf_array.h"
#include "rf_array/subchannel_select.h"

namespace holoscan::ops {

// ----- SubchannelSelect ---------------------------------------------------
template <typename sampleType>
void SubchannelSelect<sampleType>::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray<sampleType>>>("rf_in");
  spec.output<std::shared_ptr<RFArray<sampleType>>>("rf_out");

  spec.param<std::vector<int, std::allocator<int>>>(
      subchannel_idx,
      "subchannel_idx",
      "Subchannel selection index",
      "Vector of subchannel indices to keep in the RFArray",
      {});
}

template <typename sampleType>
void SubchannelSelect<sampleType>::initialize() {
  HOLOSCAN_LOG_INFO("SubchannelSelect::initialize()");
  holoscan::Operator::initialize();

  idx_len = subchannel_idx.get().size();
  make_tensor(subchannel_idx_tensor, {idx_len});
  cudaMemcpy(subchannel_idx_tensor.Data(),
             subchannel_idx.get().data(),
             idx_len * sizeof(int),
             cudaMemcpyDefault);

  HOLOSCAN_LOG_INFO("SubchannelSelect::initialize() done");
}

/**
 * @brief Select RFArray subchannels to keep
 */
template <typename sampleType>
void SubchannelSelect<sampleType>::compute(InputContext& op_input, OutputContext& op_output,
                                           ExecutionContext&) {
  HOLOSCAN_LOG_TRACE("SubchannelSelect::compute() called");
  auto in_ptr_maybe = op_input.receive<std::shared_ptr<RFArray<sampleType>>>("rf_in");
  cudaStream_t stream = op_input.receive_cuda_stream("rf_in", true, false);

  int num_emitted = 0;
  while (in_ptr_maybe) {
    auto in_ptr = in_ptr_maybe.value();
    auto out_tensor = matx::make_tensor<sampleType>(
        {in_ptr->data.Size(0), idx_len}, matx::MATX_ASYNC_DEVICE_MEMORY, stream);
    (out_tensor = matx::remap<1>(in_ptr->data, subchannel_idx_tensor)).run(stream);

    auto out_ptr = std::make_shared<RFArray<sampleType>>(out_tensor, in_ptr->metadata);
    op_output.emit(out_ptr, "rf_out");
    num_emitted++;
    if (num_emitted >= op_output.outputs()["rf_out"]->queue_size()) {
      break;
    }

    // see if we have another array on the receive buffer
    in_ptr_maybe = op_input.receive<std::shared_ptr<RFArray<sampleType>>>("rf_in");
  }
}

template class SubchannelSelect<complex_int_type>;
template class SubchannelSelect<complex_t>;

}  // namespace holoscan::ops
