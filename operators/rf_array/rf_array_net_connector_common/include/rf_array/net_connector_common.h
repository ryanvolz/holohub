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

#include <linux/if_ether.h>
#include <linux/udp.h>
#include <netinet/ip.h>

#include "holoscan/holoscan.hpp"
#include "rf_array/rf_array.h"

// Compiler option that allows us to spoof packet metadata. This functionality
// can be useful when testing, where we have a packet generator that isn't
// transmitting data that isn't generating packets that use our data format.
#define SPOOF_PACKET_DATA true
#define SPOOF_SAMPLES_PER_PKT 2048  // byte count must be less than 'max_packet_size' config
#define SPOOF_SKIP_DATA_BYTES 0     // number of bytes to skip in actual packet to get to data

// IPV4 UDP packet using Linux headers
struct UDPIPV4Pkt {
  struct ethhdr eth;
  struct iphdr ip;
  struct udphdr udp;
  uint8_t payload[];
} __attribute__((packed));

// Packet header for RF signal
struct RfPktHeader {
  uint64_t sample_idx;
  uint64_t sample_rate_numerator;
  uint64_t sample_rate_denominator;
  uint32_t channel_idx;
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

// Tracks the status of filling an RF array
struct BufferTracking {
  size_t pos;
  size_t pos_wrap;
  size_t buffer_size;
  int* sample_cnt_h;
  int* sample_cnt_d;
  bool* received_end_h;
  bool* received_end_d;
  unsigned long long int* counter_h;
  unsigned long long int* counter_d;

  BufferTracking() = default;
  explicit BufferTracking(const size_t _buffer_size)
      : pos(0), pos_wrap(0), buffer_size(_buffer_size) {
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
    return cudaMemcpyAsync(dst, src, buffer_size * sizeof(int), kind, stream);
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
      HOLOSCAN_LOG_ERROR("Unknown option {}", kind);
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(dst, src, buffer_size * sizeof(bool), kind, stream);
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
    return cudaMemcpyAsync(dst, src, buffer_size * sizeof(unsigned long long int), kind, stream);
  }

  // TODO: Faster way than three separate memcpy's?
  cudaError_t transfer(const cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t err;
    err = transferSamples(kind, stream);
    if (err != cudaSuccess) { return err; }
    err = transferEndArray(kind, stream);
    if (err != cudaSuccess) { return err; }
    err = transferCounters(kind, stream);
    if (err != cudaSuccess) { return err; }
    return cudaSuccess;
  }

  void increment() {
    received_end_h[pos_wrap] = false;
    sample_cnt_h[pos_wrap] = 0;
    counter_h[pos_wrap] += buffer_size;
    pos++;
    pos_wrap = pos % buffer_size;
  }

  bool is_ready(const size_t samples_per_arr) {
    return received_end_h[pos_wrap] || sample_cnt_h[pos_wrap] >= samples_per_arr;
  }
};

void place_packet_data(sample_t* out, RFMetadata* out_metadata, void* const* const in,
                       int* sample_cnt, bool* received_end, unsigned long long int* buffer_counter,
                       const uint32_t num_pkts, const uint16_t buffer_size,
                       const uint32_t num_samples, const uint16_t num_subchannels,
                       const uint32_t max_samples_per_packet, const uint64_t total_pkts,
                       cudaStream_t stream);
