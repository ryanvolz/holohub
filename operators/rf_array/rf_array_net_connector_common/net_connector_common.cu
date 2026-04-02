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
#include "rf_array/net_connector_common.h"
#include "rf_array/rf_array.h"

__global__ void place_packet_data_kernel(
    sample_t* out, RFMetadata* out_metadata, const void* const* const __restrict__ in,
    int* sample_cnt, bool* received_end, unsigned long long int* buffer_counter,
    const uint16_t buffer_size, const uint32_t num_samples, const uint16_t num_subchannels,
    const uint32_t max_samples_per_packet, const double freq_idx_scaling,
    const double freq_idx_offset, bool apply_conjugate, const RFPacketHeader* spoof_header,
    const uint64_t total_pkts, const uint16_t packet_skip_bytes) {
  const uint32_t sample_stride = static_cast<uint32_t>(num_subchannels);
  const uint32_t buffer_stride = sample_stride * num_samples;
  const uint32_t pkt_idx = blockIdx.x;

  // Warmup
  if (out == nullptr) return;

  const RFPacketHeader* meta;
  const sample_t* samples;
  if (spoof_header == nullptr) {
    meta = reinterpret_cast<const RFPacketHeader*>(in[pkt_idx]);
    samples = reinterpret_cast<const sample_t*>(meta + 1);
  } else {
    // Use spoofed header and generate sample index from the packet count, assuming
    // all of the packets are arriving in order
    RFPacketHeader meta_obj = *spoof_header;
    meta_obj.sample_idx += static_cast<uint64_t>(meta_obj.pkt_samples * (total_pkts + pkt_idx));
    meta = &meta_obj;
    samples = reinterpret_cast<const sample_t*>(reinterpret_cast<const char*>(in[pkt_idx]) +
                                                packet_skip_bytes);
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

    // Write samples only if they are not old
    if (global_buffer_idx >= buffer_counter[buffer_idx]) {
      // Compute pointer in buffer memory
      uint32_t idx_offset = sample_idx * sample_stride + buffer_idx * buffer_stride;

      // Copy data
      for (uint32_t i = threadIdx.x; i < samples_to_write * num_subchannels; i += blockDim.x) {
        out[idx_offset + i] = samples[pkt_iq_idx + i];
        if (apply_conjugate) { out[idx_offset + i].i *= -1; }
      }

      if (threadIdx.x == 0) {
        if (atomicExch(&buffer_counter[buffer_idx], global_buffer_idx) != global_buffer_idx) {
          // reset the buffer metadata for the current cycle
          sample_cnt[buffer_idx] = 0;
          received_end[buffer_idx] = false;
          // set metadata the first time we write to this buffer idx
          // (sample_idx corresponding to the start of the output array)
          out_metadata[buffer_idx].sample_idx = global_buffer_idx * num_samples;
          out_metadata[buffer_idx].sample_rate_numerator = meta->sample_rate_numerator;
          out_metadata[buffer_idx].sample_rate_denominator = meta->sample_rate_denominator;
          out_metadata[buffer_idx].center_freq =
              freq_idx_scaling * meta->freq_idx + freq_idx_offset;
        }

        // todo Smarter way than atomicAdd
        atomicAdd(&sample_cnt[buffer_idx], samples_to_write * num_subchannels);

        if (sample_cnt[buffer_idx] >= num_subchannels * num_samples) {
          received_end[buffer_idx] = true;
        }
      }
    }

    // update loop counter variables regardless
    global_sample_idx += samples_to_write;
    pkt_iq_idx += samples_to_write * num_subchannels;
  }
}

void place_packet_data(sample_t* out, RFMetadata* out_metadata, void* const* const in,
                       int* sample_cnt, bool* received_end, unsigned long long int* buffer_counter,
                       const uint32_t num_pkts, const uint16_t buffer_size,
                       const uint32_t num_samples, const uint16_t num_subchannels,
                       const uint32_t max_samples_per_packet, const double freq_idx_scaling,
                       const double freq_idx_offset, bool apply_conjugate,
                       const RFPacketHeader* spoof_header, const uint64_t total_pkts,
                       const uint16_t packet_skip_bytes, cudaStream_t stream) {
  // Each block processes an individual packet
  place_packet_data_kernel<<<num_pkts, 128, buffer_size * sizeof(int), stream>>>(
      out,
      out_metadata,
      in,
      sample_cnt,
      received_end,
      buffer_counter,
      buffer_size,
      num_samples,
      num_subchannels,
      max_samples_per_packet,
      freq_idx_scaling,
      freq_idx_offset,
      apply_conjugate,
      spoof_header,
      total_pkts,
      packet_skip_bytes);
}
