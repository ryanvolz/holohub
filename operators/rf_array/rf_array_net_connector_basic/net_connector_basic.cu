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
#include "holoscan/holoscan.hpp"
#include "rf_array/net_connector_basic.h"
#include "rf_array/net_connector_common.h"

namespace holoscan::ops {

void NetConnectorBasic::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  spec.output<std::shared_ptr<RFArray<sample_t>>>("rf_out").connector(
      holoscan::IOSpec::ConnectorType::kDoubleBuffer,
      holoscan::Arg("capacity", static_cast<uint64_t>(100)));

  // Array settings
  spec.param<uint16_t>(buffer_size_,
                       "buffer_size",
                       "Size of RF buffer",
                       "Max number of num_samples batches that can be held at once",
                       {});
  spec.param<uint32_t>(
      num_samples_, "num_samples", "Number of samples", "Number of samples per output chunk", {});
  spec.param<uint16_t>(num_subchannels_,
                       "num_subchannels",
                       "Number of subchannels",
                       "Number of IQ subchannels per sample time instance",
                       {});

  // Packet header settings
  spec.param<double>(
      freq_idx_scaling_,
      "freq_idx_scaling",
      "Frequency scaling factor",
      "Multiplier to apply to the frequency index from header metadata to calculate "
      "the center frequency: center_freq = freq_idx_scaling * freq_idx + freq_idx_offset",
      1);
  spec.param<double>(freq_idx_offset_,
                     "freq_idx_offset",
                     "Frequency offset",
                     "Additive offset to apply to the center frequency calculated from header "
                     "metadata: center_freq = freq_idx_scaling * freq_idx + freq_idx_offset",
                     0);
  spec.param<bool>(spoof_header_,
                   "spoof_header",
                   "Spoof the RFMetadata header",
                   "Whether or not to ignore the packet header and spoof its metadata",
                   false);
  spec.param<uint16_t>(packet_skip_bytes_,
                       "packet_skip_bytes",
                       "Number of bytes to skip",
                       "If spoofing packet header, number of bytes to skip at the beginning of "
                       "each packet before reading data",
                       0);
  spec.param<std::map<std::string, uint64_t>>(header_metadata_,
                                              "header_metadata",
                                              "Spoofed header values",
                                              "Metadata values to use in spoofed header",
                                              {});

  // Networking settings
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

void NetConnectorBasic::initialize() {
  HOLOSCAN_LOG_INFO("NetConnectorBasic::initialize()");
  register_converter<std::map<std::string, uint64_t>>();
  holoscan::Operator::initialize();

  cudaStreamCreateWithFlags(&proc_stream, cudaStreamNonBlocking);

  // Maximum number of RF samples (of num_subchannels I/Q samples) per packet
  max_samples_per_packet = (max_packet_size_.get() - sizeof(RFPacketHeader)) /
                           (num_subchannels_.get() * sizeof(sample_t));

  HOLOSCAN_LOG_INFO("Max samples per packet: {}", max_samples_per_packet);

  if (max_samples_per_packet * batch_size_.get() > num_samples_.get() * buffer_size_.get()) {
    HOLOSCAN_LOG_ERROR(
        "Specified packet batch_size produces more samples than can fit in the specified sample "
        "buffer (num_samples * buffer_size). Increase num_samples * buffer_size to at least {}, or "
        "decrease batch_size to at most {}",
        max_samples_per_packet * batch_size_.get(),
        num_samples_.get() * buffer_size_.get() / max_samples_per_packet);
    exit(1);
  }

  // Total number of I/Q samples per array
  samples_per_arr = num_samples_.get() * num_subchannels_.get();

  if (spoof_header_.get()) {
    uint16_t pkt_size;
    RFPacketHeader* spoof_header_h;
    cudaMallocHost((void**)&spoof_header_h, sizeof(RFPacketHeader));
    spoofed_packet_header_from_map(spoof_header_h, header_metadata_.get());
    uint32_t pkt_samples = spoof_header_h->pkt_samples;
    pkt_size = sizeof(sample_t) * num_subchannels_.get() * pkt_samples + packet_skip_bytes_.get();
    HOLOSCAN_LOG_WARN("Spoofing packet metadata, ignoring packet header.");
    if (pkt_size > max_packet_size_.get()) {
      HOLOSCAN_LOG_ERROR("Max packets size ({}) can't fit the expected samples ({})",
                         max_packet_size_.get(),
                         pkt_samples);
      exit(1);
    }
    // override max_packet_size_ since we know the fixed value
    max_packet_size_ = pkt_size;
    cudaMalloc((void**)&spoof_header_d, sizeof(RFPacketHeader));
    cudaMemcpy((void*)spoof_header_d,
               (void*)spoof_header_h,
               sizeof(RFPacketHeader),
               cudaMemcpyHostToDevice);
    cudaFreeHost(spoof_header_h);
  }

  // Allocate memory and create CUDA streams for each concurrent batch
  for (int n = 0; n < num_concurrent; n++) {
    cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get());
    // host-pinned memory for full batch data that we put the packets into in CPU mode
    cudaMallocHost(&full_batch_data_h_[n], batch_size_.get() * max_packet_size_.get());
    // populate the host-pinned device pointers since we know them ahead of time
    // (we have to assume that all packets are the same size because the basic network
    //  operator packs them all together in a burst and leaves no way to separate them
    //  except by parsing a payload header, which we don't want to do on the CPU)
    for (int p = 0; p < batch_size_.get(); p++) {
      h_dev_ptrs_[n][p] = reinterpret_cast<void*>(reinterpret_cast<char*>(full_batch_data_h_[n]) +
                                                  p * max_packet_size_.get());
    }

    cudaStreamCreateWithFlags(&streams_[n], cudaStreamNonBlocking);
    cudaEventCreate(&events_[n]);
    // Warmup
    place_packet_data(
        nullptr, nullptr, nullptr, 0, 0, 0, 16, 16, 0, 0, 0, 0, 0, nullptr, 0, 0, streams_[n]);
    cudaStreamSynchronize(streams_[n]);
  }

  buffer_track = BufferTracking(buffer_size_.get());
  matx::make_tensor(rf_data, {buffer_size_.get(), num_samples_.get(), num_subchannels_.get()});
  matx::make_tensor(rf_metadata, {buffer_size_.get()});

  HOLOSCAN_LOG_INFO("NetConnectorBasic::initialize() complete");
}

void NetConnectorBasic::freeResources() {
  HOLOSCAN_LOG_INFO("NetConnectorBasic::freeResources() start");
  for (int n = 0; n < num_concurrent; n++) {
    if (full_batch_data_h_[n]) { cudaFreeHost(full_batch_data_h_[n]); }
    if (h_dev_ptrs_[n]) { cudaFreeHost(h_dev_ptrs_[n]); }
    if (streams_[n]) { cudaStreamDestroy(streams_[n]); }
    if (events_[n]) { cudaEventDestroy(events_[n]); }
  }
  if (buffer_track.sample_cnt_h) { cudaFreeHost(buffer_track.sample_cnt_h); }
  if (buffer_track.sample_cnt_d) { cudaFree(buffer_track.sample_cnt_d); }
  if (buffer_track.received_end_h) { cudaFreeHost(buffer_track.received_end_h); }
  if (buffer_track.received_end_d) { cudaFree(buffer_track.received_end_d); }
  if (buffer_track.counter_h) { cudaFreeHost(buffer_track.counter_h); }
  if (buffer_track.counter_d) { cudaFree(buffer_track.counter_d); }
  if (proc_stream) { cudaStreamDestroy(proc_stream); }
  if (spoof_header_d) { cudaFree(spoof_header_d); }
  HOLOSCAN_LOG_INFO("NetConnectorBasic::freeResources() complete");
}

std::vector<NetConnectorBasic::RxMsg> NetConnectorBasic::check_completed() {
  std::vector<NetConnectorBasic::RxMsg> completed;

  // Loop over all batches, checking if any have completed
  while (out_q.size() > 0) {
    const auto first = out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      HOLOSCAN_LOG_DEBUG("Batch of packets successfully copied to GPU memory");
      completed.push_back(first);
      out_q.pop();
    } else {
      break;
    }
  }
  return completed;
}

void NetConnectorBasic::check_completed_and_emit_arrays(OutputContext& op_output) {
  std::vector<NetConnectorBasic::RxMsg> completed_msgs = check_completed();
  if (completed_msgs.empty()) { return; }
  cudaStream_t stream = completed_msgs[0].stream;

  buffer_track.transfer(cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (size_t i = 0; i < buffer_track.buffer_size; i++) {
    const size_t pos_wrap = (buffer_track.pos + i) % buffer_track.buffer_size;
    if (!buffer_track.received_end_h[pos_wrap]) { continue; }

    // Received End-of-Array (EOA) message, emit to downstream operators
    auto out_metadata_tensor = matx::make_tensor<RFMetadata>({}, matx::MATX_HOST_MEMORY);
    matx::copy(out_metadata_tensor,
               rf_metadata.Slice<0>({static_cast<matx::index_t>(pos_wrap)}, {matx::matxDropDim}),
               stream);
    cudaStreamSynchronize(stream);
    auto out_metadata = out_metadata_tensor();
    auto params = std::make_shared<RFArray<sample_t>>(
        rf_data.Slice<2>({static_cast<matx::index_t>(pos_wrap), 0, 0},
                         {matx::matxDropDim, matx::matxEnd, matx::matxEnd}),
        out_metadata,
        proc_stream);

    op_output.emit(params, "rf_out");
    HOLOSCAN_LOG_DEBUG("Buffer {}: Emitting sample buffer {} with {}/{} IQ samples",
                       buffer_track.pos + i,
                       buffer_track.counter_h[pos_wrap],
                       buffer_track.sample_cnt_h[pos_wrap],
                       samples_per_arr);

    // Increment the tracker 'i' number of times. This allows us to not get hung on arrays
    // where the EOA was either dropped or missed. Ex: if the EOA for array 11 was dropped,
    // we will emit array 12 when its EOA arrives, incrementing from 10 -> 12.
    for (size_t j = 0; j <= i; j++) { buffer_track.increment(); }
    HOLOSCAN_LOG_TRACE("Next sample cycle expected: {}", buffer_track.pos);

    buffer_track.transfer(cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    break;
  }
}

void NetConnectorBasic::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  HOLOSCAN_LOG_TRACE("NetConnectorBasic::compute() called");
  // todo Some sort of warm start for the processing stages?
  auto burst_opt = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  if (!burst_opt) {
    check_completed_and_emit_arrays(op_output);
    return;
  }

  auto burst = burst_opt.value();

  HOLOSCAN_LOG_DEBUG("Handling burst of {} packets and {} bytes, with {} packets already in buffer",
                     burst->num_pkts,
                     burst->len,
                     aggr_pkts_recv_);

  // Track packet payloads for the current burst
  /* CPU Mode
   * Copy each packet payload in a continuous host-pinned buffer, copy of that larger buffer to
   * the GPU will occur later (copying each packet to GPU directly would be too expensive).
   */

  auto burst_pkts_remaining = burst->num_pkts;
  // FIXME: assuming all packets are the same size, see also: filling h_dev_ptrs_ in initialize()
  auto pkt_size = burst->len / burst->num_pkts;

  while (burst_pkts_remaining > 0) {
    auto num_pkts_to_copy =
        std::min(burst_pkts_remaining, static_cast<uint32_t>(batch_size_.get() - aggr_pkts_recv_));
    auto copy_len = num_pkts_to_copy * pkt_size;
    auto batch_offset = aggr_pkts_recv_ * pkt_size;
    memcpy((char*)full_batch_data_h_[cur_idx] + batch_offset, burst->data, copy_len);

    ttl_bytes_recv_ += copy_len;
    aggr_pkts_recv_ += num_pkts_to_copy;
    burst_pkts_remaining -= num_pkts_to_copy;

    // Once we've aggregated enough packets, do some work
    if (aggr_pkts_recv_ >= batch_size_.get()) {
      HOLOSCAN_LOG_DEBUG(
          "{} packets collected exceeding batch size of {}, packing into array on GPU",
          aggr_pkts_recv_,
          batch_size_.get());
      do {
        check_completed_and_emit_arrays(op_output);
        if (out_q.size() >= num_concurrent) {
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
                        num_samples_.get(),
                        num_subchannels_.get(),
                        max_samples_per_packet,
                        freq_idx_scaling_.get(),
                        freq_idx_offset_.get(),
                        spoof_header_d,
                        ttl_pkts_recv_,            // only needed if spoofing packets
                        packet_skip_bytes_.get(),  // only needed if spoofing packets
                        streams_[cur_idx]);

      cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
      cur_msg_.stream = streams_[cur_idx];
      cur_msg_.evt = events_[cur_idx];
      out_q.push(cur_msg_);
      cur_msg_.num_batches = 0;

      ttl_pkts_recv_ += aggr_pkts_recv_;

      if (cudaGetLastError() != cudaSuccess) {
        HOLOSCAN_LOG_ERROR(
            "CUDA error dispatching batch from queue number {} after {} total packets received",
            cur_idx,
            ttl_pkts_recv_);
        exit(1);
      }
      aggr_pkts_recv_ = 0;
      cur_idx = (++cur_idx % num_concurrent);
    }
  }

  // free packets in burst
  delete[] burst->data;
}

void NetConnectorBasic::stop() {
  HOLOSCAN_LOG_INFO(
      "\n"
      "NetConnectorBasic exit report:\n"
      "--------------------------------\n"
      " - Processed bytes:     {}\n"
      " - Processed packets:   {}\n",
      ttl_bytes_recv_,
      ttl_pkts_recv_);

  freeResources();
}

}  // namespace holoscan::ops
