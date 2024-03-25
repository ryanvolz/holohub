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

#include <filesystem>

#include <digital_rf.h>
#include <hdf5.h>

#include "common.h"

// ---------- Operators ----------
namespace holoscan::ops {

class ComplexIntToFloatOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ComplexIntToFloatOp)

  ComplexIntToFloatOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Convert complex integer representation to floating point
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;
};  // ComplexIntToFloatOp

template <typename sampleType>
class DigitalRFSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DigitalRFSinkOp)

  DigitalRFSinkOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Write the RF input to files in Digital RF format
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;
  void stop() override;

 private:
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
  tensor_t<sampleType, 2> rf_data;
};  // DigitalRFSinkOp

class ResamplePolyOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ResamplePolyOp)

  ResamplePolyOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;

  /**
   * @brief Polyphase resampling by up/down rate
   */
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<uint32_t> chunk_size;
  Parameter<uint16_t> num_subchannels;
  Parameter<uint16_t> up;
  Parameter<uint16_t> down;
  Parameter<std::vector<float, std::allocator<float>>> filter_coefs;

  std::shared_ptr<RFArray<complex_t>> prior_input;
  uint32_t pad_size;
  uint32_t out_pad_size;
  uint32_t out_chunk_size;
  tensor_t<float, 1> filter;
  tensor_t<complex_t, 2> padded_out_data;
  tensor_t<complex_t, 2> padded_data;
};  // ResamplePolyOp

}  // namespace holoscan::ops
