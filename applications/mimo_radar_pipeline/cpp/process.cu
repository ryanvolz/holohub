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
#include <cmath>
#include <filesystem>

#include <digital_rf.h>
#include <hdf5.h>

#include "process.h"

namespace holoscan::ops {

// ----- SubchannelSelectOp ---------------------------------------------------
template <typename sampleType>
void SubchannelSelectOp<sampleType>::setup(OperatorSpec& spec) {
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
void SubchannelSelectOp<sampleType>::initialize() {
  HOLOSCAN_LOG_INFO("SubchannelSelectOp::initialize()");
  holoscan::Operator::initialize();

  idx_len = subchannel_idx.get().size();
  make_tensor(subchannel_idx_tensor, {idx_len});
  cudaMemcpy(subchannel_idx_tensor.Data(),
             subchannel_idx.get().data(),
             idx_len * sizeof(int),
             cudaMemcpyDefault);

  HOLOSCAN_LOG_INFO("SubchannelSelectOp::initialize() done");
}

template class SubchannelSelectOp<complex_int_type>;
template class SubchannelSelectOp<complex_t>;

/**
 * @brief Select RFArray subchannels to keep
 */
template <typename sampleType>
void SubchannelSelectOp<sampleType>::compute(InputContext& op_input, OutputContext& op_output,
                                             ExecutionContext&) {
  HOLOSCAN_LOG_INFO("SubchannelSelectOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<sampleType>>>("rf_in").value();
  cudaStream_t stream = in->stream;

  auto out_tensor =
      make_tensor<sampleType>({in->data.Size(0), idx_len}, MATX_ASYNC_DEVICE_MEMORY, stream);
  (out_tensor = remap<1>(in->data, subchannel_idx_tensor)).run(stream);

  auto params = std::make_shared<RFArray<sampleType>>(out_tensor, in->metadata, stream);
  op_output.emit(params, "rf_out");
}

// ----- ComplexIntToFloatOp ---------------------------------------------------
void ComplexIntToFloatOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray<sample_t>>>("rf_in");
  spec.output<std::shared_ptr<RFArray<complex_t>>>("rf_out");
}

void ComplexIntToFloatOp::initialize() {
  HOLOSCAN_LOG_INFO("ComplexIntToFloatOp::initialize()");
  holoscan::Operator::initialize();

  HOLOSCAN_LOG_INFO("ComplexIntToFloatOp::initialize() done");
}

/**
 * @brief Convert complex integer representation to floating point
 */
void ComplexIntToFloatOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext&) {
  HOLOSCAN_LOG_INFO("ComplexIntToFloatOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<sample_t>>>("rf_in").value();
  cudaStream_t stream = in->stream;

  HOLOSCAN_LOG_INFO("Dim: {}, {}", in->data.Size(0), in->data.Size(1));

  // convert the data from complex int to complex float
  auto new_shp = in->data.Shape();
  new_shp[1] = 2 * new_shp[1];
  auto in_data_float_view = in->data.View<real_t, 2, typeof(new_shp)>(std::move(new_shp));
  auto in_data_float =
      matx::as_float(in_data_float_view) / (std::numeric_limits<real_t>::max() - 1);

  auto complex_data = make_tensor<complex_t>(in->data.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
  auto out_real = complex_data.RealView();
  auto out_imag = complex_data.ImagView();
  (out_real = matx::slice(in_data_float, {0, 0}, {matxEnd, matxEnd}, {1, 2})).run(stream);
  (out_imag = matx::slice(in_data_float, {0, 1}, {matxEnd, matxEnd}, {1, 2})).run(stream);

  auto params = std::make_shared<RFArray<complex_t>>(complex_data, in->metadata, stream);
  op_output.emit(params, "rf_out");
}

// ----- DigitalRFSinkOp ---------------------------------------------------
template <typename sampleType>
void DigitalRFSinkOp<sampleType>::setup(OperatorSpec& spec) {
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
void DigitalRFSinkOp<complex_int_type>::_h5type_initialize() {
  hdf5_dtype = H5T_STD_I16LE;
  is_complex = true;
}

template <>
void DigitalRFSinkOp<cuda::std::complex<float>>::_h5type_initialize() {
  hdf5_dtype = H5T_NATIVE_FLOAT;
  is_complex = true;
}

template <typename sampleType>
void DigitalRFSinkOp<sampleType>::initialize() {
  HOLOSCAN_LOG_INFO("DigitalRFSinkOp::initialize()");
  holoscan::Operator::initialize();

  _h5type_initialize();

  // make sure the channel directory exists
  channel_dir_path = channel_dir.get();
  std::filesystem::create_directories(channel_dir_path);

  // allocate in host memory so we can access from CPU without device synchronization
  make_tensor(rf_data, {chunk_size.get(), num_subchannels.get()}, MATX_HOST_MEMORY);

  HOLOSCAN_LOG_INFO("DigitalRFSinkOp::initialize() done");
}

/**
 * @brief  Write the RF input to files in Digital RF format
 */
template <typename sampleType>
void DigitalRFSinkOp<sampleType>::compute(InputContext& op_input, OutputContext& op_output,
                                          ExecutionContext&) {
  HOLOSCAN_LOG_INFO("DigitalRFSinkOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<sampleType>>>("rf_in").value();

  if (rf_data.Shape() != in->data.Shape()) {
    HOLOSCAN_LOG_ERROR(
        "Incoming array shape ({}, {}) does not equal config-specified shape ({}, {})",
        in->data.Size(0),
        in->data.Size(1),
        rf_data.Size(0),
        rf_data.Size(1));
  }

  // copy incoming data/metadata to host-allocated memory
  matx::copy(rf_data, in->data, in->stream);
  cudaStreamSynchronize(in->stream);

  // initialize writer using data specifications from the first array
  if (!writer_initialized) {
    start_idx = in->metadata.sample_idx;
    sample_rate_numerator = in->metadata.sample_rate_numerator;
    sample_rate_denominator = in->metadata.sample_rate_denominator;
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

  HOLOSCAN_LOG_INFO("Writing {} samples @ {}", rf_data.Size(0), in->metadata.sample_idx);
  auto result = digital_rf_write_hdf5(
      drf_writer, in->metadata.sample_idx - start_idx, rf_data.Data(), rf_data.Size(0));
  if (result) {
    HOLOSCAN_LOG_ERROR("Digital RF write failed with error {}, sample_idx {}  write_len {}",
                       result,
                       in->metadata.sample_idx - start_idx,
                       rf_data.Size(0));
    exit(result);
  }
}

template <typename sampleType>
void DigitalRFSinkOp<sampleType>::stop() {
  // clean up digital RF writer object
  auto result = digital_rf_close_write_hdf5(drf_writer);
  if (result) { HOLOSCAN_LOG_ERROR("Failed to close Digital RF writer with error {}", result); }
  writer_initialized = false;
}

template class DigitalRFSinkOp<complex_int_type>;
template class DigitalRFSinkOp<complex_t>;

// ----- ResamplePolyOp ---------------------------------------------------
void ResamplePolyOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray<complex_t>>>("rf_in");
  spec.output<std::shared_ptr<RFArray<complex_t>>>("rf_out");
  spec.param<uint32_t>(
      chunk_size, "chunk_size", "Chunk size", "Number of samples to operate on in one chunk", {});
  spec.param<uint16_t>(num_subchannels,
                       "num_subchannels",
                       "Number of subchannels",
                       "Number of IQ subchannels per sample time instance",
                       {});
  spec.param<uint16_t>(
      up, "up", "Upsampling factor", "Upsampling factor to pass to resample_poly", 1);
  spec.param<uint16_t>(
      down, "down", "Downsampling factor", "Downsampling factor to pass to resample_poly", 1);
  spec.param<std::vector<float, std::allocator<float>>>(
      filter_coefs,
      "filter_coefs",
      "Filter coefficients",
      "Filter coefficients provided to resample_poly",
      {});
}

void ResamplePolyOp::initialize() {
  HOLOSCAN_LOG_INFO("ResamplePolyOp::initialize()");
  holoscan::Operator::initialize();

  out_chunk_size = chunk_size.get() * up.get();
  if (out_chunk_size % down.get()) {
    HOLOSCAN_LOG_ERROR(
        "ResamplePolyOp up {} / down {} with chunk_size {} does not result in an integer output "
        "chunk_size",
        up.get(),
        down.get(),
        chunk_size.get());
    exit(1);
  }
  out_chunk_size = out_chunk_size / down.get();

  uint32_t filter_len = filter_coefs.get().size();
  make_tensor(filter, {filter_len});
  cudaMemcpy(
      filter.Data(), filter_coefs.get().data(), filter_len * sizeof(float), cudaMemcpyDefault);

  uint32_t filter_half_len = (filter_len - 1) / 2;
  uint16_t filter_pre_pad = down.get() - (filter_half_len % down.get());

  pad_size = (filter_half_len + filter_pre_pad);
  uint32_t padded_len = chunk_size.get() + 2 * pad_size;
  make_tensor(padded_data, {padded_len, num_subchannels.get()});
  cudaMemset(padded_data.Data(), 0, padded_data.TotalSize() * sizeof(complex_t));

  out_pad_size = pad_size / down.get();
  uint32_t padded_out_len = padded_len * up.get();
  padded_out_len =
      (padded_out_len % down.get()) ? padded_out_len / down.get() + 1 : padded_out_len / down.get();
  make_tensor(padded_out_data, {padded_out_len, num_subchannels.get()});

  HOLOSCAN_LOG_INFO("ResamplePolyOp::initialize() done");
}

/**
 * @brief Polyphase resampling by up/down rate
 */
void ResamplePolyOp::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  HOLOSCAN_LOG_INFO("ResamplePolyOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<complex_t>>>("rf_in").value();

  if (!prior_input) {
    prior_input = in;
    return;
  }

  cudaStream_t stream = prior_input->stream;

  // We need data from 3 separate chunks to compute one chunk of output and not shift the metadata
  // So for incoming chunk N to output the resampling of chunk N-1, we need:
  // [ Last pad_size samples of chunk N-2, all of chunk N-1, first pad_size samples of chunk N]

  // operating on subchannels separately, so permute them to the front
  auto padded_data_flipped = padded_data.Permute({1, 0});

  // copy last samples of chunk N-2 to the beginning of the padded tensor
  auto padded_beginning = padded_data_flipped.Slice({0, 0}, {matxEnd, pad_size});
  auto end_of_prior_prior =
      padded_data_flipped.Slice({0, chunk_size.get()}, {matxEnd, pad_size + chunk_size.get()});
  matx::copy(padded_beginning, end_of_prior_prior, stream);

  // copy prior input to middle of padded tensor
  auto padded_middle =
      padded_data_flipped.Slice({0, pad_size}, {matxEnd, pad_size + chunk_size.get()});
  auto prior_flipped = prior_input->data.Permute({1, 0});
  matx::copy(padded_middle, prior_flipped, stream);

  // copy first samples of incoming chunk to end of the padded tensor
  auto padded_end = padded_data_flipped.Slice({0, pad_size + chunk_size.get()},
                                              {matxEnd, 2 * pad_size + chunk_size.get()});
  auto incoming_flipped_beginning = in->data.Permute({1, 0}).Slice({0, 0}, {matxEnd, pad_size});
  matx::copy(padded_end, incoming_flipped_beginning, stream);

  // do the polyphase resampling
  auto padded_out_data_flipped = padded_out_data.Permute({1, 0});
  (padded_out_data_flipped = matx::resample_poly(padded_data_flipped, filter, up.get(), down.get()))
      .run(stream);

  // get output view
  auto out_data_view =
      padded_out_data.Slice({out_pad_size, 0}, {out_pad_size + out_chunk_size, matxEnd});
  // copy output to new tensor so we don't have to worry about how long the data needs
  // to live for downstream blocks to do their processing
  auto out_data = make_tensor<complex_t>(out_data_view.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
  matx::copy(out_data, out_data_view, stream);

  // create output metadata and adjust its sample index and rate according to the resampling
  auto out_metadata = prior_input->metadata;
  out_metadata.sample_idx *= up.get();
  out_metadata.sample_idx /= down.get();
  out_metadata.sample_rate_numerator *= up.get();
  out_metadata.sample_rate_denominator *= down.get();
  auto divisor = std::gcd(out_metadata.sample_rate_numerator, out_metadata.sample_rate_denominator);
  out_metadata.sample_rate_numerator /= divisor;
  out_metadata.sample_rate_denominator /= divisor;

  auto params = std::make_shared<RFArray<complex_t>>(out_data, out_metadata, stream);
  op_output.emit(params, "rf_out");

  // set incoming input to prior input for next compute
  prior_input = in;
}

// ----- ScheduledRotatorOp ---------------------------------------------------
void ScheduledRotatorOp::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFArray<complex_t>>>("rf_in");
  spec.output<std::shared_ptr<RFArray<complex_t>>>("rf_out");
  spec.param<double>(cycle_duration_secs,
                     "cycle_duration_secs",
                     "Cycle duration in seconds",
                     "Duration of the cycle of frequencies before it repeats",
                     {});
  spec.param<double>(cycle_start_timestamp,
                     "cycle_start_timestamp",
                     "Cycle start timestamp",
                     "Cycle start timestamp (seconds since Unix epoch)",
                     0);
  spec.param<YAML::Node>(schedule_yaml,
                         "schedule",
                         "Schedule",
                         "Schedule of frequencies and their activation times in the cycle",
                         {});
}

void ScheduledRotatorOp::initialize() {
  HOLOSCAN_LOG_INFO("ScheduledRotatorOp::initialize()");
  holoscan::Operator::initialize();

  for (const auto& sched_item : schedule_yaml.get()) {
    auto start = sched_item["start"].as<double>();
    auto freq = sched_item["freq"].as<double>();
    if (start >= cycle_duration_secs.get()) {
      HOLOSCAN_LOG_ERROR("Schedule step has start {} that is >= cycle_duration_secs {}",
                         start,
                         cycle_duration_secs.get());
      exit(1);
    }
    schedule.emplace_back(start, freq);
  }
  std::sort(schedule.begin(), schedule.end());
  // last entry so schedule[idx + 1].first always gives stop time
  schedule.emplace_back(cycle_duration_secs.get(), -std::numeric_limits<double>::infinity());
  // make sure all times are covered by ensuring schedule starts at time 0
  if (schedule[0].first != 0) {
    schedule.emplace(schedule.begin(), 0, -std::numeric_limits<double>::infinity());
  }

  for (const auto& sched_pair : schedule) {
    HOLOSCAN_LOG_INFO("Schedule: start {}, freq {}", sched_pair.first, sched_pair.second);
  }

  HOLOSCAN_LOG_INFO("ScheduledRotatorOp::initialize() done");
}

/**
 * @brief Rotator with frequency shift controlled by a fixed schedule
 */
void ScheduledRotatorOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext&) {
  HOLOSCAN_LOG_INFO("ScheduledRotatorOp::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<complex_t>>>("rf_in").value();
  cudaStream_t stream = in->stream;

  // calculate center frequency and timestamp of the data chunk from metadata
  double center_freq = 2e6 * in->metadata.channel_idx + 3e6;
  double sample_rate = static_cast<double>(in->metadata.sample_rate_numerator) /
                       static_cast<double>(in->metadata.sample_rate_denominator);
  uint64_t sample_sec;
  uint64_t picosecond;
  // copied from digital_rf, until function is exported
  //   digital_rf_get_timestamp_floor(in->metadata.sample_idx,
  //                                  in->metadata.sample_rate_numerator,
  //                                  in->metadata.sample_rate_denominator,
  //                                  &sample_sec,
  //                                  &picosecond);
  // calculate with divide/modulus split to avoid overflow
  // second = si * d / n = ((si / n) * d) + ((si % n) * d) / n
  uint64_t tmp_div;
  uint64_t tmp_mod;
  uint64_t tmp;
  tmp_div = in->metadata.sample_idx / in->metadata.sample_rate_numerator;
  tmp_mod = in->metadata.sample_idx % in->metadata.sample_rate_numerator;
  sample_sec = tmp_div * in->metadata.sample_rate_denominator;
  tmp = tmp_mod * in->metadata.sample_rate_denominator;
  tmp_div = tmp / in->metadata.sample_rate_numerator;
  tmp_mod = tmp % in->metadata.sample_rate_numerator;
  sample_sec += tmp_div;
  // picoseconds calculated from remainder of division to calculate seconds
  // picsecond = rem * 1e12 / n = rem * (1e12 / n) + (rem * (1e12 % n)) / n
  tmp = tmp_mod;
  tmp_div = 1000000000000 / in->metadata.sample_rate_numerator;
  tmp_mod = 1000000000000 % in->metadata.sample_rate_numerator;
  picosecond = (tmp * tmp_div) + (tmp * tmp_mod / in->metadata.sample_rate_numerator);

  double timestamp = sample_sec + picosecond / 1e12;

  // get our time elapsed within the cycle
  auto cycle_timestamp = fmod((timestamp - cycle_start_timestamp.get()), cycle_duration_secs.get());

  // use time elapsed within cycle to figure out what step we're on and get its parameters
  auto step_start = schedule[schedule_idx].first;
  auto step_stop = schedule[schedule_idx + 1].first;
  while (!(cycle_timestamp >= step_start && cycle_timestamp < step_stop)) {
    schedule_idx++;
    if (schedule_idx + 1 >= schedule.size()) { schedule_idx = 0; }
    step_start = schedule[schedule_idx].first;
    step_stop = schedule[schedule_idx + 1].first;
  }
  auto step_freq = schedule[schedule_idx].second;

  if (step_freq >= 0) {
    // set up the desired rotation
    auto freq_shift = center_freq - step_freq;
    if (std::abs(freq_shift) > sample_rate / 2) {
      HOLOSCAN_LOG_WARN(
          "Shifting center frequency {} to {} results in shift {} greater than sample rate {}. "
          "Shift "
          "will be aliased.",
          center_freq,
          step_freq,
          freq_shift,
          sample_rate);
    }
    auto aliased_freq_shift = fmod(freq_shift, sample_rate);
    HOLOSCAN_LOG_INFO("Applying frequency shift of {} (from center freq {} to desired freq {})",
                      aliased_freq_shift,
                      center_freq,
                      step_freq);
    float phase_increment = 2 * M_PI * aliased_freq_shift / sample_rate;
    float phase = 2 * M_PI * aliased_freq_shift * (cycle_timestamp - step_start);

    // do the rotation
    auto in_data_flipped = in->data.Permute({1, 0});
    auto phase_increments =
        matx::expj(matx::range<1>(in_data_flipped.Shape(), phase, phase_increment));

    auto out_data = make_tensor<complex_t>(in->data.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);
    auto out_data_flipped = out_data.Permute({1, 0});

    (out_data_flipped = in_data_flipped * phase_increments).run(stream);

    auto params = std::make_shared<RFArray<complex_t>>(out_data, in->metadata, stream);
    op_output.emit(params, "rf_out");
  } else {
    op_output.emit(in, "rf_out");
  }
}

}  // namespace holoscan::ops
