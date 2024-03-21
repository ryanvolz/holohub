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
#include <filesystem>

#include <digital_rf.h>
#include <hdf5.h>

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

  auto params = std::make_shared<ComplexRFArray>(complex_data, in->metadata, stream);
  op_output.emit(params, "rf_out");
}

// ----- DigitalRFSinkOp ---------------------------------------------------
void DigitalRFSinkOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray>>("rf_in");
  spec.param<std::string>(channel_dir,
                          "channel_dir",
                          "Channel directory",
                          "Directory for writing the Digital RF channel",
                          {});
  spec.param<uint64_t>(subdir_cadence_secs,
                       "subdir_cadence_secs",
                       "Subdirectory cadence",
                       "Subdirectory cadence in number of seconds",
                       3600);
  spec.param<uint64_t>(file_cadence_millisecs,
                       "file_cadence_millisecs",
                       "File cadence",
                       "File cadence in milliseconds",
                       1000);
  spec.param<std::string>(
      uuid, "uuid", "UUID string", "Unique identifier string for this channel", "holoscan");
  spec.param<int>(compression_level,
                  "compression_level",
                  "Compression level",
                  "HDF5 compression level (0 for none, 1-9 for gzip level)",
                  0);
  spec.param<bool>(checksum, "checksum", "Enable checksum", "Enable HDF5 checksum", false);
  spec.param<bool>(is_continuous,
                   "is_continuous",
                   "Enable continuous writing mode",
                   "Continuous writing mode (true) vs. gapped mode (false)",
                   true);
  spec.param<bool>(marching_dots,
                   "marching_dots",
                   "Enable marching dots",
                   "Enable marching dots for every file written",
                   false);

  spec.param<uint16_t>(num_cycles_,
                       "num_cycles",
                       "Number of cycles",
                       "Number of cycles of num_samples to group in processing",
                       {});
  spec.param<uint16_t>(num_samples_,
                       "num_samples",
                       "Number of samples",
                       "Number of samples per cycle to group in processing",
                       {});
  spec.param<uint16_t>(num_subchannels_,
                       "num_subchannels",
                       "Number of subchannels",
                       "Number of IQ subchannels per sample time instance",
                       {});
}

void DigitalRFSinkOp::initialize() {
  HOLOSCAN_LOG_INFO("DigitalRFSinkOp::initialize()");
  holoscan::Operator::initialize();

  // make sure the channel directory exists
  channel_dir_path = channel_dir.get();
  std::filesystem::create_directories(channel_dir_path);

  // allocate in host memory so we can access from CPU without device synchronization
  make_tensor(rf_data, {num_cycles_, num_samples_, num_subchannels_}, MATX_HOST_MEMORY);
  make_tensor(rf_metadata, MATX_HOST_MEMORY);

  HOLOSCAN_LOG_INFO("DigitalRFSinkOp::initialize() done");
}

/**
 * @brief  Write the RF input to files in Digital RF format
 */
void DigitalRFSinkOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  HOLOSCAN_LOG_INFO("DigitalRFSinkOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray>>("rf_in").value();

  // copy incoming data/metadata to host-allocated memory
  matx::copy(rf_data, in->data, in->stream);
  matx::copy(rf_metadata, in->metadata, in->stream);
  cudaStreamSynchronize(in->stream);

  auto data = rf_data;
  auto metadata = rf_metadata();

  // initialize writer using data specifications from the first array
  if (!writer_initialized) {
    start_idx = metadata.sample_idx;
    sample_rate_numerator = metadata.sample_rate_numerator;
    sample_rate_denominator = metadata.sample_rate_denominator;
    num_subchannels = data.Size(2);
    HOLOSCAN_LOG_INFO(
        "Initializing Digital RF writer with start_idx {}, sample_rate {}/{}, num_subchannels {}",
        start_idx,
        sample_rate_numerator,
        sample_rate_denominator,
        num_subchannels);
    drf_writer = digital_rf_create_write_hdf5(channel_dir_path.string().data(),
                                              hdf5_dtype,
                                              subdir_cadence_secs.get(),
                                              file_cadence_millisecs.get(),
                                              start_idx,
                                              sample_rate_numerator,
                                              sample_rate_denominator,
                                              uuid.get().data(),
                                              compression_level.get(),
                                              checksum.get(),
                                              is_complex,
                                              num_subchannels,
                                              is_continuous.get(),
                                              marching_dots.get());
    writer_initialized = true;
  }

  HOLOSCAN_LOG_INFO("Writing {} samples @ {}", data.Size(0) * data.Size(1), metadata.sample_idx);
  auto result = digital_rf_write_hdf5(
      drf_writer, metadata.sample_idx - start_idx, data.Data(), data.Size(0) * data.Size(1));
  if (result) {
    HOLOSCAN_LOG_ERROR("Digital RF write failed with error {}, sample_idx {}  write_len {}",
                       result,
                       metadata.sample_idx - start_idx,
                       data.Size(0) * data.Size(1));
    exit(result);
  }
}

void DigitalRFSinkOp::stop() {
  // clean up digital RF writer object
  auto result = digital_rf_close_write_hdf5(drf_writer);
  if (result) { HOLOSCAN_LOG_ERROR("Failed to close Digital RF writer with error {}", result); }
  writer_initialized = false;
}

}  // namespace holoscan::ops
