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
#include <matx.h>
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
  unsigned long long pos;
  size_t buffer_size;
  uint32_t num_samples;
  uint64_t start_sample_idx;
  uint64_t total_output_samples;
  uint64_t total_dropped_samples;
  int* sample_cnt_h;
  int* sample_cnt_d;
  int* full_cnt_h;
  int* full_cnt_d;
  unsigned long long int* counter_h;
  unsigned long long int* counter_d;
  unsigned long long int* completed_pos_h;
  unsigned long long int* completed_pos_d;

  BufferTracking() = default;
  explicit BufferTracking(const size_t _buffer_size, const uint32_t _num_samples)
      : pos(0),
        buffer_size(_buffer_size),
        num_samples(_num_samples),
        start_sample_idx(0),
        total_output_samples(0),
        total_dropped_samples(0) {
    // Reserve sample count
    cudaMallocHost((void**)&sample_cnt_h, buffer_size * sizeof(int));
    cudaMalloc((void**)&sample_cnt_d, buffer_size * sizeof(int));
    memset(sample_cnt_h, 0, buffer_size * sizeof(int));
    cudaMemset(sample_cnt_d, 0, buffer_size * sizeof(int));

    // Reserve end-of-array signal (initialized to 1 so kernel initiates reset on start)
    cudaMallocHost((void**)&full_cnt_h, buffer_size * sizeof(int));
    cudaMalloc((void**)&full_cnt_d, buffer_size * sizeof(int));
    memset(full_cnt_h, 1, buffer_size * sizeof(int));
    cudaMemset(full_cnt_d, 1, buffer_size * sizeof(int));

    // Reserve current buffer counter
    cudaMallocHost((void**)&counter_h, buffer_size * sizeof(unsigned long long int));
    cudaMalloc((void**)&counter_d, buffer_size * sizeof(unsigned long long int));
    memset(counter_h, 0, buffer_size * sizeof(unsigned long long int));
    cudaMemset(counter_d, 0, buffer_size * sizeof(unsigned long long int));

    // Reserve filled buffer position
    cudaMallocHost((void**)&completed_pos_h, sizeof(unsigned long long int));
    cudaMalloc((void**)&completed_pos_d, sizeof(unsigned long long int));
    memset(completed_pos_h, 0, sizeof(unsigned long long int));
    cudaMemset(completed_pos_d, 0, sizeof(unsigned long long int));
  }

  void free_memory() {
    if (sample_cnt_h) {
      cudaFreeHost(sample_cnt_h);
      sample_cnt_h = nullptr;
    }
    if (sample_cnt_d) {
      cudaFree(sample_cnt_d);
      sample_cnt_d = nullptr;
    }
    if (full_cnt_h) {
      cudaFreeHost(full_cnt_h);
      full_cnt_h = nullptr;
    }
    if (full_cnt_d) {
      cudaFree(full_cnt_d);
      full_cnt_d = nullptr;
    }
    if (counter_h) {
      cudaFreeHost(counter_h);
      counter_h = nullptr;
    }
    if (counter_d) {
      cudaFree(counter_d);
      counter_d = nullptr;
    }
    if (completed_pos_h) {
      cudaFreeHost(completed_pos_h);
      completed_pos_h = nullptr;
    }
    if (completed_pos_d) {
      cudaFree(completed_pos_d);
      completed_pos_d = nullptr;
    }
  }

  // TODO: Faster way than separate memcpy's?
  void transfer(cudaStream_t stream) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpyAsync(
            sample_cnt_h, sample_cnt_d, buffer_size * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "Failed to transfer sample_cnt");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpyAsync(
            full_cnt_h, full_cnt_d, buffer_size * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "Failed to transfer full_cnt");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpyAsync(counter_h,
                                                   counter_d,
                                                   buffer_size * sizeof(unsigned long long int),
                                                   cudaMemcpyDeviceToHost,
                                                   stream),
                                   "Failed to transfer buffer_counter");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpyAsync(completed_pos_h,
                                                   completed_pos_d,
                                                   sizeof(unsigned long long int),
                                                   cudaMemcpyDeviceToHost,
                                                   stream),
                                   "Failed to transfer completed_pos");
  }

  void completed_at_pos(size_t completed_pos, cudaStream_t stream) {
    size_t buf_idx = completed_pos % buffer_size;

    // Signal to kernel that data copy is complete by copying this buffer's counter into
    // the kernel's completed_pos following the copy command in the stream
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpyAsync(completed_pos_d,
                                                   &counter_d[buf_idx],
                                                   sizeof(unsigned long long int),
                                                   cudaMemcpyDeviceToDevice,
                                                   stream),
                                   "Failed to transfter buffer_counter to completed_pos");

    auto dropped_samples = num_samples - full_cnt_h[buf_idx];

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
    total_output_samples += full_cnt_h[buf_idx];
    total_dropped_samples += dropped_samples;
  }

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
      if (full_cnt_h[buf_idx] == 0 && sample_cnt_h[buf_idx] == 0) {
        continue;
      }

      // If this non-empty buffer is back in time from position, ignore and exit
      if (counter_h[buf_idx] < pos) {
        break;
      }

      // Output the next buffer if it has been marked as full
      if (full_cnt_h[buf_idx] > 0) {
        return buf_idx;
      }
      // Samples pending but nothing ready to output yet, break to return no ready index
      break;
    }
    // Return flag for no ready index (buffer_size)
    return buffer_size;
  }
};

template <typename SampleT>
void place_packet_data(matx::tensor_t<SampleT, 3>& out, RFMetadata* out_metadata,
                       void* const* const in, int* sample_cnt, int* full_cnt,
                       unsigned long long int* buffer_counter,
                       unsigned long long int* completed_pos, const uint32_t num_pkts,
                       const uint32_t max_samples_per_packet, const double freq_idx_scaling,
                       const double freq_idx_offset, const bool apply_conjugate,
                       const RFPacketHeader* spoof_header, const uint64_t total_pkts,
                       const uint16_t packet_skip_bytes, const bool debug_print,
                       cudaStream_t stream);
