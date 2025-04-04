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
#include <chrono>
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
  spec.input<std::shared_ptr<RFArray<sampleType>>>("rf_in");
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

  for (int n = 0; n < num_concurrent; n++) {
    // allocate in host memory so we can access from CPU without device synchronization
    matx::make_tensor(
        rf_data_arrs[n], {chunk_size.get(), num_subchannels.get()}, matx::MATX_HOST_MEMORY);

    cudaEventCreate(&events_[n], cudaEventDisableTiming);
  }

  HOLOSCAN_LOG_INFO("DigitalRFSink::initialize() done");
}

/**
 * @brief  Write the RF input to files in Digital RF format
 */
template <typename sampleType>
void DigitalRFSink<sampleType>::compute(InputContext& op_input, OutputContext& op_output,
                                        ExecutionContext&) {
  HOLOSCAN_LOG_TRACE("DigitalRFSink::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<sampleType>>>("rf_in").value();

  if (rf_data_arrs[0].Shape() != in->data.Shape()) {
    HOLOSCAN_LOG_ERROR(
        "Incoming array shape ({}, {}) does not equal config-specified shape ({}, {})",
        in->data.Size(0),
        in->data.Size(1),
        rf_data_arrs[0].Size(0),
        rf_data_arrs[0].Size(1));
  }

  // copy incoming data/metadata to host-allocated memory
  matx::copy(rf_data_arrs[cur_idx], in->data, in->stream);
  cudaEventRecord(events_[cur_idx], in->stream);
  rf_metadatas[cur_idx] = in->metadata;
  cur_msg_.buffer_idx = cur_idx;
  cur_msg_.event = events_[cur_idx];
  copy_q.push(cur_msg_);
  HOLOSCAN_LOG_DEBUG("Buffer {}: Copying {} samples @ {} from GPU memory",
                     cur_idx,
                     rf_data_arrs[cur_idx].Size(0),
                     rf_metadatas[cur_idx].sample_idx);
  cur_idx = (++cur_idx % num_concurrent);

  // initialize writer using data specifications from the first array
  if (!writer_initialized) {
    start_idx = in->metadata.sample_idx;
    sample_rate_numerator = in->metadata.sample_rate_numerator;
    sample_rate_denominator = in->metadata.sample_rate_denominator;
    auto drf_start_idx = start_idx;
    if (drf_start_idx == 0) {
      auto now = std::chrono::system_clock::now();
      auto seconds_since_epoch =
          std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
      auto tmp_div = seconds_since_epoch / sample_rate_denominator;
      auto tmp_mod = seconds_since_epoch % sample_rate_denominator;
      tmp_div *= sample_rate_numerator;
      tmp_mod *= sample_rate_numerator;
      tmp_div += tmp_mod / sample_rate_denominator;
      tmp_mod = tmp_mod % sample_rate_denominator;
      auto remainder = tmp_mod * 1000000000;
      auto quotient = remainder / sample_rate_denominator;
      remainder = remainder % sample_rate_denominator;
      tmp_div += quotient / 1000000000;
      quotient = quotient % 1000000000;
      remainder += quotient * sample_rate_denominator;
      quotient = tmp_div;
      remainder = remainder / 1000000000 + (remainder % 1000000000 != 0);
      quotient += (remainder != 0);
      drf_start_idx = quotient;
    }
    HOLOSCAN_LOG_INFO("Initializing Digital RF writer with start_idx {}, sample_rate {}/{}",
                      drf_start_idx,
                      sample_rate_numerator,
                      sample_rate_denominator);
    drf_writer = digital_rf_create_write_hdf5(channel_dir_path.string().data(),
                                              hdf5_dtype,
                                              subdir_cadence_secs.get(),
                                              file_cadence_millisecs.get(),
                                              drf_start_idx,
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

  if (copy_q.size() == num_concurrent) {
    // copy buffers filled before we could clear any of them and write the array
    HOLOSCAN_LOG_ERROR("Fell behind in copying arrays from GPU for writing with Digital RF!");
    // wait until the oldest copy is done and we can write the next array
    cudaEventSynchronize(copy_q.front().event);
  }

  while (copy_q.size() > 0) {
    const auto next_msg = copy_q.front();
    if (cudaEventQuery(next_msg.event) == cudaSuccess) {
      HOLOSCAN_LOG_DEBUG("Buffer {}: Writing {} samples @ {}",
                         next_msg.buffer_idx,
                         rf_data_arrs[next_msg.buffer_idx].Size(0),
                         rf_metadatas[next_msg.buffer_idx].sample_idx);
      auto result = digital_rf_write_hdf5(drf_writer,
                                          rf_metadatas[next_msg.buffer_idx].sample_idx - start_idx,
                                          rf_data_arrs[next_msg.buffer_idx].Data(),
                                          rf_data_arrs[next_msg.buffer_idx].Size(0));
      if (result) {
        HOLOSCAN_LOG_ERROR("Digital RF write failed with error {}, sample_idx {}  write_len {}",
                           result,
                           rf_metadatas[next_msg.buffer_idx].sample_idx - start_idx,
                           rf_data_arrs[next_msg.buffer_idx].Size(0));
        exit(result);
      }

      copy_q.pop();
    } else {
      break;
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
