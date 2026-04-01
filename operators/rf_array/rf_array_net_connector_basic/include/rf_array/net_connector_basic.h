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
#pragma once

#include <matx.h>

#include "basic_network_operator_rx.h"
#include "holoscan/holoscan.hpp"
#include "rf_array/net_connector_common.h"
#include "rf_array/rf_array.h"

namespace holoscan::ops {

class NetConnectorBasic : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NetConnectorBasic)

  NetConnectorBasic() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void freeResources();
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  // Array settings
  Parameter<uint16_t> buffer_size_;
  Parameter<uint32_t> num_samples_;
  Parameter<uint16_t> num_subchannels_;

  // Packet header settings
  Parameter<double> freq_idx_scaling_;
  Parameter<double> freq_idx_offset_;
  Parameter<bool> apply_conjugate_;
  Parameter<bool> spoof_header_;
  Parameter<uint16_t> packet_skip_bytes_;
  Parameter<std::map<std::string, uint64_t>> header_metadata_;

  // Networking settings
  Parameter<uint32_t> batch_size_;       // Batch size for one processing block
  Parameter<uint16_t> batch_capacity_;
  Parameter<uint16_t> max_packet_size_;  // Maximum size of a single packet

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    int num_batches;
    cudaStream_t stream;
    cudaEvent_t evt;
  };
  std::vector<RxMsg> check_completed();
  void check_completed_and_queue_arrays(OutputContext& op_output, cudaStream_t& op_stream);

  RxMsg cur_msg_{};
  std::queue<RxMsg> out_q;

  // Buffer memory and tracking
  std::vector<void**> h_dev_ptrs_;         // Host-pinned list of device pointers
  std::vector<void*> full_batch_data_h_;   // Host aggregated batch
  std::vector<uint64_t**> ttl_pkts_drop_;  // Total packets dropped by kernel

  // Concurrent batch structures
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;
  int cur_idx = 0;

  // Holds burst buffers that cannot be freed yet
  int64_t ttl_bytes_recv_ = 0;  // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;   // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;  // Aggregate packets received in processing batch

  uint32_t max_samples_per_packet;
  size_t samples_per_arr;
  BufferTracking buffer_track;
  matx::tensor_t<sample_t, 3> rf_data;
  // Host/Device pointers to same memory storing buffer of RFMetadata
  RFMetadata* rf_metadata_h = nullptr;
  RFMetadata* rf_metadata_d = nullptr;

  // Spoofed packet header device memory structure
  RFPacketHeader* spoof_header_d = nullptr;
};  // NetConnectorBasic

}  // namespace holoscan::ops
