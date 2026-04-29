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

#include <stdio.h>

#include <matx.h>

#include "rf_array/net_connector_common.h"
#include "rf_array/rf_array.h"

template <typename SampleT>
__global__ void place_packet_data_kernel(
    typename matx::detail::base_type_t<matx::tensor_t<SampleT, 3>> out, RFMetadata* out_metadata,
    const void* const* const __restrict__ in, int* sample_cnt, int* full_cnt,
    unsigned long long int* buffer_counter, unsigned long long int* completed_pos,
    const uint32_t max_samples_per_packet, const double freq_idx_scaling,
    const double freq_idx_offset, const bool apply_conjugate, const RFPacketHeader* spoof_header,
    const uint64_t total_pkts, const uint16_t packet_skip_bytes) {
  const auto buffer_size = out.Size(0);
  const auto num_samples = out.Size(1);
  const auto num_subchannels = out.Size(2);
  const uint32_t pkt_idx = blockIdx.x;

  // Warmup
  if (in == nullptr) {
    return;
  }

  const RFPacketHeader* meta;
  const SampleT* samples;
  if (spoof_header == nullptr) {
    meta = reinterpret_cast<const RFPacketHeader*>(in[pkt_idx]);
    samples = reinterpret_cast<const SampleT*>(meta + 1);
  } else {
    // Use spoofed header and generate sample index from the packet count, assuming
    // all of the packets are arriving in order
    RFPacketHeader meta_obj = *spoof_header;
    meta_obj.sample_idx += static_cast<uint64_t>(meta_obj.pkt_samples * (total_pkts + pkt_idx));
    meta = &meta_obj;
    samples = reinterpret_cast<const SampleT*>(reinterpret_cast<const char*>(in[pkt_idx]) +
                                               packet_skip_bytes);
  }

  if (total_pkts < gridDim.x && threadIdx.x == 0 && meta->pkt_samples > max_samples_per_packet) {
    // Only warn for first batch of packets, because if the packets have this wrong once
    // chances are it is always wrong.
    if (blockIdx.x == 0) {
      // Only output full warning once per kernel call, if that
      printf("WARNING: Packet has invalid pkt_samples = %u in header\n", meta->pkt_samples);
    }
    //  I for invalid, since if this happens it can happen a lot make it very terse
    printf("I");
  }
  if (total_pkts < gridDim.x && threadIdx.x == 0 && meta->num_subchannels != num_subchannels) {
    // Only warn for first batch of packets, because if the packets have this wrong once
    // chances are it is always wrong.
    if (blockIdx.x == 0) {
      // Only output full warning once per kernel call, if that
      printf("WARNING: Packet has invalid num_subchannels = %u != %u in header\n",
             meta->num_subchannels,
             static_cast<uint32_t>(num_subchannels));
    }
    //  I for invalid, since if this happens it can happen a lot make it very terse
    printf("I");
  }

  uint64_t global_sample_idx = meta->sample_idx;
  const uint32_t pkt_samples = min(meta->pkt_samples, max_samples_per_packet);
  const uint64_t global_stop_sample_idx = global_sample_idx + pkt_samples;
  uint16_t pkt_iq_idx = 0;

  while (global_sample_idx < global_stop_sample_idx) {
    // break sample index down into (buffer, sample) index
    unsigned long long int global_buffer_idx = global_sample_idx / num_samples;
    uint32_t sample_idx = global_sample_idx % num_samples;
    uint16_t buffer_idx = global_buffer_idx % buffer_size;

    uint32_t samples_before_next_buffer = num_samples - sample_idx;
    uint32_t samples_remaining_in_packet = global_stop_sample_idx - global_sample_idx;
    uint32_t samples_to_write = min(samples_remaining_in_packet, samples_before_next_buffer);

    // Check if samples are too old to be written to the buffer
    if (global_buffer_idx < buffer_counter[buffer_idx]) {
      if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
          // Only output full warning once per kernel call, if that
          printf(
              "WARNING: Packet with sample_idx = %llu implies an old buffer_idx: %llu (current: "
              "%llu). "
              "Copying this data has been skipped.\n",
              meta->sample_idx,
              global_buffer_idx,
              buffer_counter[buffer_idx]);
        } else {
          // L for Late or oLd, since if this happens it can happen a lot make it very terse
          printf("L");
        }
      }
      // Check if packet's samples would write into a full buffer that has not been copied out yet
    } else if (full_cnt[buffer_idx] > 0 && buffer_counter[buffer_idx] > *completed_pos) {
      if (threadIdx.x == 0) {
        if (blockIdx.x == 0) {
          // Only output full warning once per kernel call, if that
          printf(
              "WARNING: Samples arrived for buffer %llu which would overwrite full buffer %llu "
              "(completed buffer position: %llu). Copying this data has been skipped.\n",
              global_buffer_idx,
              buffer_counter[buffer_idx],
              *completed_pos);
        } else {
          // F for full
          printf("F");
        }
      }
    } else {
      // Samples can be written to the buffer
      // (If it was marked full with full_cnt[buffer_idx] > 0, then since we're here
      //  buffer_counter[buffer_idx] <= *completed_pos and the data has been copied out.
      //  If it wasn't marked full, then clearly we can write to it.)

      // Copy data
      for (uint32_t i = threadIdx.x; i < samples_to_write; i += blockDim.x) {
        for (uint32_t j = 0; j < num_subchannels; j++) {
          out(buffer_idx, sample_idx + i, j) = samples[pkt_iq_idx + i * num_subchannels + j];
          if (apply_conjugate) {
            out(buffer_idx, sample_idx + i, j).i *= -1;
          }
        }
      }

      if (threadIdx.x == 0) {
        // If we're writing to this buffer, then full_cnt[buffer_idx] should be 0 and
        // buffer_counter[buffer_idx] should be global_buffer_idx.
        if (full_cnt[buffer_idx] != 0 || buffer_counter[buffer_idx] != global_buffer_idx) {
          // We have to ensure full_cnt is zeroed before the buffer_counter is updated
          // to ensure that (full_cnt[buffer_idx] > 0 && buffer_counter[buffer_idx] >
          // *completed_pos) doesn't evaluate to True for other blocks which might be racing this
          // one for this buffer_idx which would then mistakenly think the buffer is full.
          full_cnt[buffer_idx] = 0;
          buffer_counter[buffer_idx] = global_buffer_idx;
          // If either of those values was changed, then we need to reset the array metadata.
          // (sample_cnt was already set to 0 when full_cnt was set nonzero to avoid a race now)
          // POTENTIAL RACE NOTES:
          // 1) The buffer counter and metadata are all the same for a given buffer_idx so
          //    races on reading/writing those values are moot.
          // 2) full_cnt is only incremented if the buffer completely fills (only one thread
          //    can do this) or if it is old enough to be emitted, in which case we can accept a
          //    race causing a slight miscount although that would be unlikely since we shouldn't
          //    be getting packets that old.
          out_metadata[buffer_idx].sample_idx = global_buffer_idx * num_samples;
          out_metadata[buffer_idx].sample_rate_numerator = meta->sample_rate_numerator;
          out_metadata[buffer_idx].sample_rate_denominator = meta->sample_rate_denominator;
          out_metadata[buffer_idx].center_freq =
              freq_idx_scaling * meta->freq_idx + freq_idx_offset;
        }

        // Count number of samples written to buffer across all packets / blocks
        atomicAdd(&sample_cnt[buffer_idx], samples_to_write);

        // We're writing to this buffer, so consider any buffers at least two steps in the past
        // that have samples to be full
        auto full_buffer_idx = global_buffer_idx - 2;

        if (sample_cnt[buffer_idx] >= num_samples) {
          // If samples are not duplicated, then only one thread across the whole kernel
          // can get here. So we don't have to do atomic operations.
          // Signal to host that a buffer is "full" and how many valid samples it contains
          full_cnt[buffer_idx] = sample_cnt[buffer_idx];
          // Immediately reset the buffer sample count to 0 to avoid future race conditions
          sample_cnt[buffer_idx] = 0;

          // Since this buffer is full, now consider prior buffer full if it has samples
          full_buffer_idx = global_buffer_idx - 1;
        }

        // Step through prior buffers and if they are older than full_buffer_idx then
        // consider them full by moving any pending sample_cnt to full_cnt
        for (size_t i = 1; i < buffer_size; i++) {
          const size_t chk_buf_idx = (buffer_idx - i) % buffer_size;
          if (buffer_counter[chk_buf_idx] <= *completed_pos) {
            // reached a buffer that has already been completed so we can stop checking
            break;
          }
          if (buffer_counter[chk_buf_idx] <= full_buffer_idx || full_cnt[chk_buf_idx] > 0) {
            // Increment full_cnt by sample_cnt while resetting sample_cnt to 0
            // (if other blocks subsequently increment sample_cnt, they will end up here to add
            //  those additional samples to full_cnt)
            atomicAdd(&full_cnt[chk_buf_idx], atomicExch(&sample_cnt[chk_buf_idx], 0));
          }
        }
      }
    }

    // update loop counter variables regardless
    global_sample_idx += samples_to_write;
    pkt_iq_idx += samples_to_write * num_subchannels;
  }
}

template <typename SampleT>
void place_packet_data(matx::tensor_t<SampleT, 3>& out, RFMetadata* out_metadata,
                       void* const* const in, int* sample_cnt, int* full_cnt,
                       unsigned long long int* buffer_counter,
                       unsigned long long int* completed_pos, const uint32_t num_pkts,
                       const uint32_t max_samples_per_packet, const double freq_idx_scaling,
                       const double freq_idx_offset, const bool apply_conjugate,
                       const RFPacketHeader* spoof_header, const uint64_t total_pkts,
                       const uint16_t packet_skip_bytes, cudaStream_t stream) {
  // Each block processes an individual packet
  place_packet_data_kernel<SampleT><<<num_pkts, 128, 0, stream>>>(out,
                                                                  out_metadata,
                                                                  in,
                                                                  sample_cnt,
                                                                  full_cnt,
                                                                  buffer_counter,
                                                                  completed_pos,
                                                                  max_samples_per_packet,
                                                                  freq_idx_scaling,
                                                                  freq_idx_offset,
                                                                  apply_conjugate,
                                                                  spoof_header,
                                                                  total_pkts,
                                                                  packet_skip_bytes);
}

template void place_packet_data<sample_t>(
    matx::tensor_t<sample_t, 3>& out, RFMetadata* out_metadata, void* const* const in,
    int* sample_cnt, int* full_cnt, unsigned long long int* buffer_counter,
    unsigned long long int* completed_pos, const uint32_t num_pkts,
    const uint32_t max_samples_per_packet, const double freq_idx_scaling,
    const double freq_idx_offset, const bool apply_conjugate, const RFPacketHeader* spoof_header,
    const uint64_t total_pkts, const uint16_t packet_skip_bytes, cudaStream_t stream);
