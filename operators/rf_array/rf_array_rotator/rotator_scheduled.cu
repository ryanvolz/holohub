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
#include "rf_array/rotator_scheduled.h"

namespace holoscan::ops {

// ----- RotatorScheduled ---------------------------------------------------
void RotatorScheduled::setup(OperatorSpec& spec) {
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
  spec.param<std::list<std::map<std::string, double>>>(
      schedule_list,
      "schedule",
      "Schedule",
      "Schedule of frequencies and their activation times in the cycle",
      {});
}

void RotatorScheduled::initialize() {
  HOLOSCAN_LOG_INFO("RotatorScheduled::initialize()");
  register_converter<std::list<std::map<std::string, double>>>();
  holoscan::Operator::initialize();

  for (const auto& sched_item : schedule_list.get()) {
    auto start = sched_item.at("start");
    auto freq = sched_item.at("freq");
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

  HOLOSCAN_LOG_INFO("RotatorScheduled::initialize() done");
}

/**
 * @brief Rotator with frequency shift controlled by a fixed schedule
 */
void RotatorScheduled::compute(InputContext& op_input, OutputContext& op_output,
                               ExecutionContext&) {
  HOLOSCAN_LOG_TRACE("RotatorScheduled::compute() called");
  auto in = op_input.receive<std::shared_ptr<RFArray<complex_t>>>("rf_in").value();
  cudaStream_t stream = in->stream;

  // calculate center frequency and timestamp of the data chunk from metadata
  double center_freq = 1e6 * in->metadata.channel_idx;
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

  HOLOSCAN_LOG_DEBUG(
      "Data timestamp {}: {} seconds since beginning of cycle, schedule index {}, step frequency "
      "{}",
      timestamp,
      cycle_timestamp,
      schedule_idx,
      step_freq);

  if (step_freq >= 0) {
    // set up the desired rotation
    auto freq_shift = center_freq - step_freq;
    if (std::abs(freq_shift) > sample_rate / 2) {
      HOLOSCAN_LOG_WARN(
          "Shifting frequency of {} to tuned center of {} results in a shift of {}, which is "
          "greater than the sample rate {}. Shift will be aliased.",
          step_freq,
          center_freq,
          freq_shift,
          sample_rate);
    }
    auto aliased_freq_shift = fmod(freq_shift + sample_rate / 2, sample_rate) - sample_rate / 2;
    HOLOSCAN_LOG_DEBUG(
        "Applying frequency shift of {} (moving desired frequency of {} to baseband at center freq "
        "of {})",
        aliased_freq_shift,
        step_freq,
        center_freq);
    double phase_increment = 2 * M_PI * aliased_freq_shift / sample_rate;
    double phase = 2 * M_PI * aliased_freq_shift * (cycle_timestamp - step_start);

    // do the rotation
    auto in_data_flipped = in->data.Permute({1, 0});
    auto phase_range = matx::range<0>({in_data_flipped.Size(1)}, phase, phase_increment);
    // want expj to operate on double for accuracy, but then cast to float (complex_t)
    // for compatibility with input data
    auto rotator = matx::as_type<complex_t>(matx::expj(phase_range));

    auto out_data =
        matx::make_tensor<complex_t>(in->data.Shape(), matx::MATX_ASYNC_DEVICE_MEMORY, stream);
    auto out_data_flipped = out_data.Permute({1, 0});

    (out_data_flipped = in_data_flipped * rotator).run(stream);

    auto params = std::make_shared<RFArray<complex_t>>(out_data, in->metadata, stream);
    op_output.emit(params, "rf_out");
  } else {
    op_output.emit(in, "rf_out");
  }
}

}  // namespace holoscan::ops
