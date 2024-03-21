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
#include "adv_networking_rx.h"  // TODO: Rename networking connectors

#if SPOOF_PACKET_DATA
/**
 * This function converts the packet count to packet metadata. We just treat the
 * packet count as if all of the packets are arriving in order and write the the
 * metadata accordingly. This functionalitycan be useful when testing, where we
 * have a packet generator that isn't generating packets that use our data format.
 */
__device__ __forceinline__ void gen_meta_from_pkt_cnt(RfPktHeader* meta, const uint64_t pkt_cnt,
                                                      const uint16_t num_subchannels) {
  meta->sample_idx = static_cast<uint64_t>(SPOOF_SAMPLES_PER_PKT * pkt_cnt);
  meta->sample_rate_numerator = 1000000;
  meta->sample_rate_denominator = 1;
  meta->channel_idx = 0;
  meta->num_subchannels = num_subchannels;
  meta->pkt_samples = SPOOF_SAMPLES_PER_PKT;
  meta->bits_per_int = 128;
  meta->is_complex = 1;
}
#endif

__global__ void place_packet_data_kernel(sample_t* out, RfMetaData* out_metadata,
                                         const void* const* const __restrict__ in, int* sample_cnt,
                                         bool* received_end, unsigned long long int* buffer_counter,
                                         const uint16_t buffer_size, const uint16_t num_cycles,
                                         const uint16_t num_samples, const uint16_t num_subchannels,
                                         const uint32_t max_samples_per_packet,
                                         const uint64_t total_pkts) {
  const uint32_t sample_stride = static_cast<uint32_t>(num_subchannels);
  const uint32_t cycle_stride = sample_stride * num_samples;
  const uint32_t buffer_stride = cycle_stride * num_cycles;
  const uint32_t pkt_idx = blockIdx.x;

#if SPOOF_PACKET_DATA
  // Generate fake packet meta-data from the packet count
  RfPktHeader meta_obj;
  RfPktHeader* meta = &meta_obj;
  gen_meta_from_pkt_cnt(meta, total_pkts + pkt_idx, num_subchannels);
  const sample_t* samples = reinterpret_cast<const sample_t*>(in[pkt_idx]);
#else
  const RfPktHeader* meta = reinterpret_cast<const RfPktHeader*>(in[pkt_idx]);
  const sample_t* samples = reinterpret_cast<const sample_t*>(meta + 1);
#endif

  uint64_t global_sample_idx = meta->sample_idx;
  const uint32_t pkt_samples = min(meta->pkt_samples, max_samples_per_packet);
  const uint64_t global_stop_sample_idx = global_sample_idx + pkt_samples;
  uint16_t pkt_iq_idx = 0;

  while (global_sample_idx < global_stop_sample_idx) {
    // break sample index down into (buffer, cycle, sample) index
    uint64_t global_cycle_idx = global_sample_idx / num_samples;
    uint16_t sample_idx = global_sample_idx % num_samples;
    unsigned long long int global_buffer_idx = global_cycle_idx / num_cycles;
    uint16_t cycle_idx = global_cycle_idx % num_cycles;
    uint16_t buffer_idx = global_buffer_idx % buffer_size;

    uint32_t samples_before_next_buffer = (num_cycles - cycle_idx) * num_samples - sample_idx;
    uint32_t samples_remaining_in_packet = global_stop_sample_idx - global_sample_idx;
    uint32_t samples_to_write = min(samples_remaining_in_packet, samples_before_next_buffer);

    // update loop counter variables here before we possibly drop samples and continue loop
    global_sample_idx += samples_to_write;
    pkt_iq_idx += samples_to_write * num_subchannels;

    // Drop old samples
    if (global_buffer_idx < buffer_counter[buffer_idx]) { continue; }

    // Compute pointer in buffer memory
    uint32_t idx_offset =
        sample_idx * sample_stride + cycle_idx * cycle_stride + buffer_idx * buffer_stride;

    // Copy data
    for (uint16_t i = threadIdx.x; i < samples_to_write * num_subchannels; i += blockDim.x) {
      out[idx_offset + i] = samples[pkt_iq_idx + i];
    }

    if (threadIdx.x == 0) {
      if (atomicExch(&buffer_counter[buffer_idx], global_buffer_idx) != global_buffer_idx) {
        // reset the buffer metadata for the current cycle
        sample_cnt[buffer_idx] = 0;
        received_end[buffer_idx] = 0;
      }
      if (sample_cnt[buffer_idx] == 0) {
        // set metadata the first time we write to this buffer idx
        // (sample_idx corresponding to the start of the output array)
        out_metadata[buffer_idx].sample_idx = global_buffer_idx * (num_cycles * num_samples);
        out_metadata[buffer_idx].sample_rate_numerator = meta->sample_rate_numerator;
        out_metadata[buffer_idx].sample_rate_denominator = meta->sample_rate_denominator;
        out_metadata[buffer_idx].channel_idx = meta->channel_idx;
      }

      // todo Smarter way than atomicAdd
      atomicAdd(&sample_cnt[buffer_idx], samples_to_write * num_subchannels);

      if (sample_cnt[buffer_idx] >= num_subchannels * num_samples * num_cycles) {
        received_end[buffer_idx] = true;
      }
    }
  }
}

void place_packet_data(sample_t* out, RfMetaData* out_metadata, void* const* const in,
                       int* sample_cnt, bool* received_end, unsigned long long int* buffer_counter,
                       const uint32_t num_pkts, const uint16_t buffer_size,
                       const uint16_t num_cycles, const uint16_t num_samples,
                       const uint16_t num_subchannels, const uint32_t max_samples_per_packet,
                       const uint64_t total_pkts, cudaStream_t stream) {
  // Each block processes an individual packet
  place_packet_data_kernel<<<num_pkts, 128, buffer_size * sizeof(int), stream>>>(
      out,
      out_metadata,
      in,
      sample_cnt,
      received_end,
      buffer_counter,
      buffer_size,
      num_cycles,
      num_samples,
      num_subchannels,
      max_samples_per_packet,
      total_pkts);
}

namespace holoscan::ops {

void AdvConnectorOpRx::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<AdvNetBurstParams>>("burst_in");
  spec.output<std::shared_ptr<RFArray>>("rf_out");

  // Radar settings
  spec.param<uint16_t>(buffer_size_,
                       "buffer_size",
                       "Size of RF buffer",
                       "Max number of num_cycles batches that can be held at once",
                       {});
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

  // Networking settings
  spec.param<bool>(use_hds_,
                   "use_header_data_split",
                   "Use header-data split",
                   "Header-data split is enabled for incoming packets",
                   true);
  spec.param<bool>(gpu_direct_,
                   "gpu_direct",
                   "GPUDirect enabled",
                   "GPUDirect is enabled for incoming packets",
                   true);
  spec.param<uint32_t>(batch_size_,
                       "batch_size",
                       "Batch size",
                       "Batch size in packets for each processing epoch",
                       1000);
  spec.param<uint16_t>(max_packet_size_,
                       "max_packet_size",
                       "Max packet size",
                       "Maximum packet size expected from sender",
                       9000);
}

void AdvConnectorOpRx::initialize() {
  HOLOSCAN_LOG_INFO("AdvConnectorOpRx::initialize()");
  holoscan::Operator::initialize();

  cudaStreamCreate(&proc_stream);

  // Maximum number of RF samples (of num_subchannels I/Q samples) per packet
  max_samples_per_packet =
      (max_packet_size_.get() - sizeof(RfPktHeader)) / (num_subchannels_.get() * sizeof(sample_t));

  HOLOSCAN_LOG_INFO("Max samples per packet: {}", max_samples_per_packet);

  if (max_samples_per_packet * batch_size_.get() > (num_cycles_.get() * num_samples_.get())) {
    HOLOSCAN_LOG_ERROR(
        "Specified packet batch_size could fill more than one RF array, but at most one array can "
        "be produced per compute() call. Increase total array size to at least {} * "
        "num_subchannels samples, or decrease batch_size to at most {}",
        max_samples_per_packet * batch_size_.get(),
        (num_cycles_.get() * num_samples_.get()) / max_samples_per_packet);
    exit(1);
  }

  // Total number of I/Q samples per array
  samples_per_arr = num_cycles_.get() * num_samples_.get() * num_subchannels_.get();

  // Configuration checks
  if (!(use_hds_.get() && gpu_direct_.get())) {
    HOLOSCAN_LOG_ERROR("Only configured to run with Header-Data Split and GPUDirect");
    exit(1);
  } else if (use_hds_.get() && !gpu_direct_.get()) {
    HOLOSCAN_LOG_ERROR("If Header-Data Split mode is enabled, GPUDirect needs to be too");
    exit(1);
  }

  // Allocate memory and create CUDA streams for each concurrent batch
  for (int n = 0; n < num_concurrent; n++) {
    if (gpu_direct_.get()) {
      cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get());
    }

    cudaStreamCreate(&streams_[n]);
    cudaEventCreate(&events_[n]);
  }

  buffer_track = AdvBufferTracking(buffer_size_.get());
  make_tensor(rf_data,
              {buffer_size_.get(), num_cycles_.get(), num_samples_.get(), num_subchannels_.get()});
  make_tensor(rf_metadata, {buffer_size_.get()});

#if SPOOF_PACKET_DATA
  // Compute packets delivered per pulse and max waveform ID based on parameters
  const size_t spoof_pkt_size =
      sizeof(sample_t) * num_subchannels_.get() * SPOOF_SAMPLES_PER_PKT + sizeof(RfPktHeader);
  HOLOSCAN_LOG_WARN("Spoofing packet metadata, ignoring packet header.");
  if (spoof_pkt_size >= max_packet_size_.get()) {
    HOLOSCAN_LOG_ERROR("Max packets size ({}) can't fit the expected samples ({})",
      max_packet_size_.get(), SPOOF_SAMPLES_PER_PKT);
    exit(1);
  }
#endif

  HOLOSCAN_LOG_INFO("AdvConnectorOpRx::initialize() complete");
}

std::vector<AdvConnectorOpRx::RxMsg> AdvConnectorOpRx::free_bufs() {
  std::vector<AdvConnectorOpRx::RxMsg> completed;

  // Loop over all batches, checking if any have completed
  while (out_q.size() > 0) {
    const auto first = out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      completed.push_back(first);
      for (auto m = 0; m < first.num_batches; m++) {
        adv_net_free_all_burst_pkts_and_burst(first.msg[m]);
      }
      out_q.pop();
    } else {
      break;
    }
  }
  return completed;
}

void AdvConnectorOpRx::free_bufs_and_emit_arrays(OutputContext& op_output) {
  std::vector<AdvConnectorOpRx::RxMsg> completed_msgs = free_bufs();
  if (completed_msgs.empty()) {
    return;
  }
  cudaStream_t stream = completed_msgs[0].stream;

  buffer_track.transfer(cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (size_t i = 0; i < buffer_track.buffer_size; i++) {
    const size_t pos_wrap = (buffer_track.pos + i) % buffer_track.buffer_size;
    if (!buffer_track.received_end_h[pos_wrap]) { continue; }

    // Received End-of-Array (EOA) message, emit to downstream operators
    auto params = std::make_shared<RFArray>(
        rf_data.Slice<3>({static_cast<index_t>(pos_wrap), 0, 0, 0},
                         {matxDropDim, matxEnd, matxEnd, matxEnd}),
        rf_metadata.Slice<0>({static_cast<index_t>(pos_wrap)}, {matxDropDim}),
        proc_stream);

    op_output.emit(params, "rf_out");
    HOLOSCAN_LOG_INFO("Buffer {}: Emitting sample buffer {} with {}/{} IQ samples",
                      buffer_track.pos + i,
                      buffer_track.counter_h[pos_wrap],
                      buffer_track.sample_cnt_h[pos_wrap],
                      samples_per_arr);

    // Increment the tracker 'i' number of times. This allows us to not get hung on arrays
    // where the EOA was either dropped or missed. Ex: if the EOA for array 11 was dropped,
    // we will emit array 12 when its EOA arrives, incrementing from 10 -> 12.
    for (size_t j = 0; j <= i; j++) { buffer_track.increment(); }
    HOLOSCAN_LOG_INFO("Next sample cycle expected: {}", buffer_track.pos);

    buffer_track.transfer(cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    break;
  }
}

void AdvConnectorOpRx::compute(InputContext& op_input,
                               OutputContext& op_output,
                               ExecutionContext& context) {
  // todo Some sort of warm start for the processing stages?
  int64_t ttl_bytes_in_cur_batch_ = 0;

  auto burst_opt = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in");
  if (!burst_opt) {
    free_bufs();
    return;
  }
  auto burst = burst_opt.value();

  // Header data split saves off the GPU pointers into a host-pinned buffer to reassemble later.
  // Once enough packets are aggregated, a reorder kernel is launched. In CPU-only mode the
  // entire burst buffer pointer is saved and freed once an entire batch is received.
  if (gpu_direct_.get() && use_hds_.get()) {
    for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
      h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] = adv_net_get_gpu_pkt_ptr(burst, p);
      ttl_bytes_in_cur_batch_ += adv_net_get_gpu_pkt_len(burst, p)
                               + adv_net_get_cpu_pkt_len(burst, p);
    }
    ttl_bytes_recv_ += ttl_bytes_in_cur_batch_;
  }

  aggr_pkts_recv_ += adv_net_get_num_pkts(burst);
  cur_msg_.msg[cur_msg_.num_batches++] = burst;

  // Once we've aggregated enough packets, do some work
  if (aggr_pkts_recv_ >= batch_size_.get()) {
    if (gpu_direct_.get()) {
      do {
        free_bufs_and_emit_arrays(op_output);
        if (out_q.size() == num_concurrent) {
          HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
          cudaStreamSynchronize(streams_[cur_idx]);
        }
      } while (out_q.size() >= num_concurrent);

      // Copy packet I/Q contents to appropriate location in 'rf_data'
      place_packet_data(rf_data.Data(),
                        rf_metadata.Data(),
                        h_dev_ptrs_[cur_idx],
                        buffer_track.sample_cnt_d,
                        buffer_track.received_end_d,
                        buffer_track.counter_d,
                        aggr_pkts_recv_,
                        buffer_size_.get(),
                        num_cycles_.get(),
                        num_samples_.get(),
                        num_subchannels_.get(),
                        max_samples_per_packet,
                        ttl_pkts_recv_,  // only needed if spoofing packets
                        streams_[cur_idx]);

      cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
      cur_msg_.stream = streams_[cur_idx];
      cur_msg_.evt    = events_[cur_idx];
      out_q.push(cur_msg_);
      cur_msg_.num_batches = 0;

      ttl_pkts_recv_ += aggr_pkts_recv_;

      if (cudaGetLastError() != cudaSuccess)  {
        HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                           batch_size_.get(),
                           batch_size_.get() * max_packet_size_.get());
        exit(1);
      }
    } else {
      adv_net_free_all_burst_pkts_and_burst(burst);
    }
    aggr_pkts_recv_ = 0;
    cur_idx = (++cur_idx % num_concurrent);
  }
}

void AdvConnectorOpRx::stop() {
  HOLOSCAN_LOG_INFO(
    "\n"
    "AdvConnectorOpRx exit report:\n"
    "--------------------------------\n"
    " - Received bytes:     {}\n"
    " - Received packets:   {}\n",
    ttl_bytes_recv_,
    ttl_pkts_recv_);
}

}  // namespace holoscan::ops
