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
#include <filesystem>

#include <digital_rf.h>
#include <hdf5.h>
#include <matx.h>

#include "holoscan/holoscan.hpp"
#include "rf_array/digital_rf_sink.h"
#include "rf_array/rf_array.h"

namespace holoscan::ops {

// ----- DigitalRFSink ---------------------------------------------------
template <typename sampleType>
void DigitalRFSink<sampleType>::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<std::vector<RFArray<sampleType>>>>("rf_in");
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
  spec.param<uint32_t>(
      chunk_size, "chunk_size", "Chunk size", "Number of samples to operate on in one chunk", {});
  spec.param<uint16_t>(num_subchannels,
                       "num_subchannels",
                       "Number of subchannels",
                       "Number of IQ subchannels per sample time instance",
                       {});
}

template <>
void DigitalRFSink<complex_int_type>::_h5type_initialize() {
  hdf5_dtype = H5T_STD_I16LE;
  is_complex = true;
}

template <>
void DigitalRFSink<cuda::std::complex<float>>::_h5type_initialize() {
  hdf5_dtype = H5T_NATIVE_FLOAT;
  is_complex = true;
}

template <typename sampleType>
void DigitalRFSink<sampleType>::initialize() {
  HOLOSCAN_LOG_INFO("DigitalRFSink::initialize()");
  holoscan::Operator::initialize();

  _h5type_initialize();

  // make sure the channel directory exists
  channel_dir_path = channel_dir.get();
  std::filesystem::create_directories(channel_dir_path);

  HOLOSCAN_LOG_INFO("DigitalRFSink::initialize() done");
}

/**
 * @brief  Write the RF input to files in Digital RF format
 */
template <typename sampleType>
void DigitalRFSink<sampleType>::compute(InputContext& op_input, OutputContext& op_output,
                                        ExecutionContext&) {
  HOLOSCAN_LOG_TRACE("DigitalRFSink::compute() called");
  auto in_msg_ptr =
      op_input.receive<std::shared_ptr<std::vector<RFArray<sampleType>>>>("rf_in").value();
  cudaStream_t stream = op_input.receive_cuda_stream("rf_in", true, false);

  std::vector<RFArray<sampleType>> host_vector;
  std::vector<cudaEvent_t> data_ready_vector;

  for (auto in : (*in_msg_ptr)) {
    HOLOSCAN_LOG_DEBUG(
        "Copying {} samples @ {} from GPU memory", in.data.Size(0), in.metadata.sample_idx);

    // copy incoming data/metadata to host-allocated memory
    auto host_data = matx::make_tensor<sampleType>(in.data.Shape(), matx::MATX_HOST_MEMORY);
    matx::copy(host_data, in.data, stream);
    host_vector.emplace_back(host_data, in.metadata);

    cudaEvent_t event;
    cudaEventCreate(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream);
    data_ready_vector.push_back(event);
  }

  // initialize writer using data specifications from the first array
  if (!writer_initialized && !in_msg_ptr->empty()) {
    auto metadata = in_msg_ptr->front().metadata;
    start_idx = metadata.sample_idx;
    sample_rate_numerator = metadata.sample_rate_numerator;
    sample_rate_denominator = metadata.sample_rate_denominator;
    HOLOSCAN_LOG_INFO("Initializing Digital RF writer with start_idx {}, sample_rate {}/{}",
                      start_idx,
                      sample_rate_numerator,
                      sample_rate_denominator);
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
                                              num_subchannels.get(),
                                              is_continuous.get(),
                                              marching_dots.get());
    if (!drf_writer) {
      HOLOSCAN_LOG_ERROR(
          "Failed to initialize Digital RF writer with start_idx {}, sample_rate {}/{}. Exiting.",
          start_idx,
          sample_rate_numerator,
          sample_rate_denominator);
    }
    writer_initialized = true;
  }

  // wait for each copy to host memory to complete, then write
  for (size_t i = 0; i < data_ready_vector.size(); ++i) {
    cudaEventSynchronize(data_ready_vector[i]);
    auto in = host_vector[i];

    HOLOSCAN_LOG_DEBUG("Writing {} samples @ {}", in.data.Size(0), in.metadata.sample_idx);
    auto result = digital_rf_write_hdf5(
        drf_writer, in.metadata.sample_idx - start_idx, in.data.Data(), in.data.Size(0));
    if (result) {
      HOLOSCAN_LOG_ERROR("Digital RF write failed with error {}, sample_idx {}  write_len {}",
                         result,
                         in.metadata.sample_idx - start_idx,
                         in.data.Size(0));
      exit(result);
    }
  }
}

template <typename sampleType>
void DigitalRFSink<sampleType>::stop() {
  // clean up digital RF writer object
  auto result = digital_rf_close_write_hdf5(drf_writer);
  if (result) { HOLOSCAN_LOG_ERROR("Failed to close Digital RF writer with error {}", result); }
  writer_initialized = false;
}

template class DigitalRFSink<complex_int_type>;
template class DigitalRFSink<complex_t>;

}  // namespace holoscan::ops
