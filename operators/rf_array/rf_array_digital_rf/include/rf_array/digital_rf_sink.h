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
#pragma once

#include <filesystem>

#include <digital_rf.h>
#include <hdf5.h>
#include <matx.h>

#include "holoscan/holoscan.hpp"
#include "rf_array/rf_array.h"

namespace holoscan::ops {

template <typename sampleType>
class DigitalRFSink : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DigitalRFSink)

  DigitalRFSink() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Write the RF input to files in Digital RF format
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;
  void stop() override;

 private:
  static constexpr int num_concurrent = 10;  // Number of concurrent memory transfers / buffers

  void _h5type_initialize();

  Parameter<uint32_t> chunk_size;
  Parameter<uint16_t> num_subchannels;
  Parameter<std::string> channel_dir;
  Parameter<uint64_t> subdir_cadence_secs;
  Parameter<uint64_t> file_cadence_millisecs;
  Parameter<std::string> uuid;
  Parameter<int> compression_level;
  Parameter<bool> checksum;
  Parameter<bool> is_continuous;
  Parameter<bool> marching_dots;

  bool writer_initialized = false;
  hid_t hdf5_dtype;
  bool is_complex;
  uint64_t start_idx;
  uint64_t sample_rate_numerator;
  uint64_t sample_rate_denominator;
  std::filesystem::path channel_dir_path;
  Digital_rf_write_object* drf_writer;

  // Concurrent buffer structures
  std::array<cudaEvent_t, num_concurrent> events_;
  std::array<matx::tensor_t<sampleType, 2>, num_concurrent> rf_data_arrs;
  std::array<RFMetadata, num_concurrent> rf_metadatas;
  int cur_idx = 0;

  // Holds events for waiting on copy from GPU memory
  struct CopyMsg {
    int buffer_idx;
    cudaEvent_t event;
  };

  CopyMsg cur_msg_{};
  std::queue<CopyMsg> copy_q;
};  // DigitalRFSink

}  // namespace holoscan::ops
