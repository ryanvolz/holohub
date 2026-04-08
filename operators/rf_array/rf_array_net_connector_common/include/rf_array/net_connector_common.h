/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Techonology
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

#include <chrono>
#include <limits>
#include <map>
#include <string>

#include <linux/if_ether.h>
#include <linux/udp.h>
#include <netinet/ip.h>

#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "rf_array/rf_array.h"

// Packet header for RF signal
struct RFPacketHeader {
  uint64_t sample_idx;
  uint64_t sample_rate_numerator;
  uint64_t sample_rate_denominator;
  uint32_t freq_idx;
  uint32_t num_subchannels;
  uint32_t pkt_samples;
  uint16_t bits_per_int;
  unsigned is_complex : 1;
  unsigned reserved0 : 7;
  uint8_t reserved1;
  uint64_t reserved2;
  uint64_t reserved3;
  uint64_t reserved4;
} __attribute__((__packed__));

inline void spoofed_packet_header_from_map(RFPacketHeader* meta,
                                           std::map<std::string, uint64_t> header_vals) {
  // set default values for any that were not specified
  header_vals.try_emplace("start_sample_idx", 0);
  header_vals.try_emplace("sample_rate_numerator", 64000000);
  header_vals.try_emplace("sample_rate_denominator", 1);
  header_vals.try_emplace("freq_idx", 0);
  header_vals.try_emplace("num_subchannels", 1);
  header_vals.try_emplace("pkt_samples", 2048);
  header_vals.try_emplace("bits_per_int", 16);
  header_vals.try_emplace("is_complex", 1);

  meta->sample_idx = static_cast<uint64_t>(header_vals.at("start_sample_idx"));
  meta->sample_rate_numerator = static_cast<uint64_t>(header_vals.at("sample_rate_numerator"));
  meta->sample_rate_denominator = static_cast<uint64_t>(header_vals.at("sample_rate_denominator"));
  meta->freq_idx = static_cast<uint32_t>(header_vals.at("freq_idx"));
  meta->num_subchannels = static_cast<uint32_t>(header_vals.at("num_subchannels"));
  meta->pkt_samples = static_cast<uint32_t>(header_vals.at("pkt_samples"));
  meta->bits_per_int = static_cast<uint16_t>(header_vals.at("bits_per_int"));
  meta->is_complex = static_cast<unsigned>(header_vals.at("is_complex"));

  if (meta->sample_idx == 0) {
    // substitute sample index corresponding to the current time
    auto now = std::chrono::system_clock::now();
    auto seconds_since_epoch =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    // this time to sample index conversion is copied from digital_rf, where it not yet exposed
    // as a function (but should be in version 2.7)
    auto tmp_div = seconds_since_epoch / meta->sample_rate_denominator;
    auto tmp_mod = seconds_since_epoch % meta->sample_rate_denominator;
    tmp_div *= meta->sample_rate_numerator;
    tmp_mod *= meta->sample_rate_numerator;
    tmp_div += tmp_mod / meta->sample_rate_denominator;
    tmp_mod = tmp_mod % meta->sample_rate_denominator;
    auto remainder = tmp_mod * 1000000000;
    auto quotient = remainder / meta->sample_rate_denominator;
    remainder = remainder % meta->sample_rate_denominator;
    tmp_div += quotient / 1000000000;
    quotient = quotient % 1000000000;
    remainder += quotient * meta->sample_rate_denominator;
    quotient = tmp_div;
    remainder = remainder / 1000000000 + (remainder % 1000000000 != 0);
    quotient += (remainder != 0);
    meta->sample_idx = quotient;
  }
}

// Tracks the status of filling an RF array
struct BufferTracking {
  size_t pos;
  size_t buffer_size;
  uint32_t num_samples;
  uint16_t num_subchannels;
  uint64_t start_sample_idx;
  uint64_t total_output_samples;
  uint64_t total_dropped_samples;
  std::vector<cudaStream_t> streams_to_sync;
  std::vector<cudaEvent_t> sync_events;
  int* sample_cnt_h;
  int* sample_cnt_d;
  bool* received_end_h;
  bool* received_end_d;
  unsigned long long int* counter_h;
  unsigned long long int* counter_d;

  BufferTracking() = default;
  explicit BufferTracking(const size_t _buffer_size, const uint32_t _num_samples,
                          const uint16_t _num_subchannels,
                          std::vector<cudaStream_t> _streams_to_sync)
      : pos(0),
        buffer_size(_buffer_size),
        num_samples(_num_samples),
        num_subchannels(_num_subchannels),
        streams_to_sync(_streams_to_sync),
        start_sample_idx(0),
        total_output_samples(0),
        total_dropped_samples(0) {
    // Create sync events for each stream plus an extra one for syncing Host to Device copy
    for (auto i = 0; i < streams_to_sync.size() + 1; i++) {
      cudaEvent_t evt;
      cudaEventCreate(&evt);
      sync_events.push_back(evt);
    }

    // Reserve sample count
    cudaMallocHost((void**)&sample_cnt_h, buffer_size * sizeof(int));
    cudaMalloc((void**)&sample_cnt_d, buffer_size * sizeof(int));
    memset(sample_cnt_h, 0, buffer_size * sizeof(int));
    cudaMemset(sample_cnt_d, 0, buffer_size * sizeof(int));

    // Reserve end-of-array signal
    cudaMallocHost((void**)&received_end_h, buffer_size * sizeof(bool));
    cudaMalloc((void**)&received_end_d, buffer_size * sizeof(bool));
    memset(received_end_h, 0, buffer_size * sizeof(bool));
    cudaMemset(received_end_d, 0, buffer_size * sizeof(bool));

    // Reserve buffer counter
    cudaMallocHost((void**)&counter_h, buffer_size * sizeof(unsigned long long int));
    cudaMalloc((void**)&counter_d, buffer_size * sizeof(unsigned long long int));
    memset(counter_h, 0, buffer_size * sizeof(unsigned long long int));
    cudaMemset(counter_d, 0, buffer_size * sizeof(unsigned long long int));
  }

  void free_memory() {
    for (auto evt : sync_events) { cudaEventDestroy(evt); }
    if (sample_cnt_h) {
      cudaFreeHost(sample_cnt_h);
      sample_cnt_h = nullptr;
    }
    if (sample_cnt_d) {
      cudaFree(sample_cnt_d);
      sample_cnt_d = nullptr;
    }
    if (received_end_h) {
      cudaFreeHost(received_end_h);
      received_end_h = nullptr;
    }
    if (received_end_d) {
      cudaFree(received_end_d);
      received_end_d = nullptr;
    }
    if (counter_h) {
      cudaFreeHost(counter_h);
      counter_h = nullptr;
    }
    if (counter_d) {
      cudaFree(counter_d);
      counter_d = nullptr;
    }
  }

  cudaError_t transferSamples(const cudaMemcpyKind kind, cudaStream_t stream) {
    void* src;
    void* dst;

    if (kind == cudaMemcpyHostToDevice) {
      src = sample_cnt_h;
      dst = sample_cnt_d;
    } else {
      src = sample_cnt_d;
      dst = sample_cnt_h;
    }
    return HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(dst, src, buffer_size * sizeof(int), kind, stream));
  }

  cudaError_t transferEndArray(const cudaMemcpyKind kind, cudaStream_t stream) {
    void* src;
    void* dst;

    if (kind == cudaMemcpyHostToDevice) {
      src = received_end_h;
      dst = received_end_d;
    } else if (kind == cudaMemcpyDeviceToHost) {
      src = received_end_d;
      dst = received_end_h;
    } else {
      HOLOSCAN_LOG_ERROR("Unknown option {}", fmt::underlying(kind));
      return cudaErrorInvalidValue;
    }
    return HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(dst, src, buffer_size * sizeof(bool), kind, stream));
  }

  cudaError_t transferCounters(const cudaMemcpyKind kind, cudaStream_t stream) {
    void* src;
    void* dst;

    if (kind == cudaMemcpyHostToDevice) {
      src = counter_h;
      dst = counter_d;
    } else {
      src = counter_d;
      dst = counter_h;
    }
    return HOLOSCAN_CUDA_CALL(
        cudaMemcpyAsync(dst, src, buffer_size * sizeof(unsigned long long int), kind, stream));
  }

  // TODO: Faster way than three separate memcpy's?
  cudaError_t transfer(const cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t err;
    err = transferSamples(kind, stream);
    if (err != cudaSuccess) {
      return err;
    }
    err = transferEndArray(kind, stream);
    if (err != cudaSuccess) {
      return err;
    }
    err = transferCounters(kind, stream);
    if (err != cudaSuccess) {
      return err;
    }
    return cudaSuccess;
  }

  cudaError_t completed_at_pos(size_t completed_pos, cudaStream_t stream) {
    cudaError_t err;
    size_t buf_idx = completed_pos % buffer_size;

    auto dropped_iq_samples = (num_samples * num_subchannels) - sample_cnt_h[buf_idx];
    auto dropped_samples = dropped_iq_samples / num_subchannels;

    if (total_output_samples == 0) {
      // We haven't output any samples yet, so set start_sample_idx
      start_sample_idx = counter_h[buf_idx] * num_samples;
      // If we have samples "missing" from this first buffer, assume they are at the beginning,
      // don't count them as missing, and move start_sample_idx forward accordingly
      start_sample_idx += dropped_samples;
      dropped_samples = 0;
    }

    // Log if we are outputting with dropped samples
    if (dropped_samples > 0) {
      HOLOSCAN_LOG_WARN("Outputting sample buffer {} with {} dropped samples",
                        counter_h[buf_idx],
                        dropped_samples);
    }

    if (completed_pos != pos) {
      // We skipped some buffers entirely, increment dropped samples accordingly
      auto skipped_buffer_samples = (completed_pos - pos) * num_samples;
      HOLOSCAN_LOG_WARN("Skipped empty sample buffers {} through {}, dropping {} samples",
                        pos,
                        completed_pos - 1,
                        skipped_buffer_samples);
      dropped_samples += skipped_buffer_samples;
    }

    // Set the next buffer position expected
    pos = completed_pos + 1;

    // Update total and dropped sample count
    total_output_samples += (sample_cnt_h[buf_idx] / num_subchannels);
    total_dropped_samples += dropped_samples;

    // Reset the tracking values and copy to device memory in sync with all relevant streams
    for (auto i = 0; i < streams_to_sync.size(); i++) {
      auto sync_stream = streams_to_sync[i];
      auto sync_event = sync_events[i];
      cudaEventRecord(sync_event, sync_stream);
      cudaStreamWaitEvent(stream, sync_event);
    }
    auto buf_and_idx = std::make_tuple(this, buf_idx);
    err = HOLOSCAN_CUDA_CALL(cudaLaunchHostFunc(stream, reset_fun, &buf_and_idx));
    if (err != cudaSuccess) {
      return err;
    }
    err = transfer(cudaMemcpyHostToDevice, stream);
    cudaEventRecord(sync_events.back(), stream);
    for (auto i = 0; i < streams_to_sync.size(); i++) {
      auto sync_stream = streams_to_sync[i];
      cudaStreamWaitEvent(sync_stream, sync_events.back());
    }
    return err;
  }

  static void reset_fun(void* data) {
    auto buf_and_idx = *static_cast<std::tuple<BufferTracking*, size_t>*>(data);
    auto* self = std::get<0>(buf_and_idx);
    auto buf_idx = std::get<1>(buf_and_idx);
    self->received_end_h[buf_idx] = false;
    self->sample_cnt_h[buf_idx] = 0;
    self->counter_h[buf_idx] += self->buffer_size;
  };

  size_t find_start_idx() {
    if (pos != 0) {
      return pos % buffer_size;
    }
    // Find the starting buffer index by finding the lowest non-zero counter index
    size_t start_idx = 0;
    unsigned long long int lowest_counter = ULLONG_MAX;
    for (size_t i = 0; i < buffer_size; i++) {
      if (counter_h[i] != 0 && counter_h[i] < lowest_counter) {
        lowest_counter = counter_h[i];
        start_idx = i;
      }
    }
    // Set position now that we have a start index
    pos = counter_h[start_idx];
    return start_idx;
  }

  size_t find_ready_idx(size_t start_idx) {
    for (size_t i = 0; i < buffer_size; i++) {
      const size_t buf_idx = (start_idx + i) % buffer_size;

      // Move to next buffer in loop if this one is completely empty
      if (sample_cnt_h[buf_idx] == 0) {
        continue;
      }

      // Output the next buffer if it is either completed, the next one is completed,
      // or packets have already been placed two buffers or more beyond
      bool packets_seen_two_ahead_plus = false;
      for (size_t j = 2; j < buffer_size; j++) {
        const size_t j_buf_idx = (buf_idx + j) % buffer_size;
        if (sample_cnt_h[j_buf_idx] > 0 && (counter_h[j_buf_idx] > counter_h[buf_idx])) {
          packets_seen_two_ahead_plus = true;
          break;
        }
      }
      if (received_end_h[buf_idx] || received_end_h[(buf_idx + 1) % buffer_size] ||
          packets_seen_two_ahead_plus) {
        return buf_idx;
      }
      // Samples pending but nothing ready to output yet, return no ready index (buffer_size)
      return buffer_size;
    }
    // Sample counts are all zeros, which shouldn't happen, but anyway we have nothing to output
    return buffer_size;
  }
};

void place_packet_data(sample_t* out, RFMetadata* out_metadata, void* const* const in,
                       int* sample_cnt, bool* received_end, unsigned long long int* buffer_counter,
                       const uint32_t num_pkts, const uint16_t buffer_size,
                       const uint32_t num_samples, const uint16_t num_subchannels,
                       const uint32_t max_samples_per_packet, const double freq_idx_scaling,
                       const double freq_idx_offset, const bool apply_conjugate,
                       const RFPacketHeader* spoof_header, const uint64_t total_pkts,
                       const uint16_t packet_skip_bytes, cudaStream_t stream);
