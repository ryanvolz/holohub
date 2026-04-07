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
#include "advanced_network/common.h"
#include "advanced_network/types.h"
#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "rf_array/net_connector_advanced.h"
#include "rf_array/net_connector_common.h"

using namespace holoscan::advanced_network;

namespace holoscan::ops {

void NetConnectorAdvanced::setup(OperatorSpec& spec) {
  // No output condition so operator will always run when input is available,
  // regardless of whether downstream operators keep up
  spec.output<std::shared_ptr<RFArray<sample_t>>>("rf_out").condition(ConditionType::kNone);

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
  spec.param<bool>(
      apply_conjugate_,
      "apply_conjugate",
      "Apply complex conjugate to the data",
      "Whether or not to take the complex conjugate of the RF data (i.e. invert spectrum)",
      false);
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
  spec.param<std::string>(interface_name_,
                          "interface_name",
                          "Name of the network interface",
                          "Name of the interface to use from the advanced_network config",
                          "rx_port");
  spec.param<uint16_t>(queue_id_,
                       "queue_id",
                       "Queue to process",
                       "ID of the queue from the advanced_network config to process",
                       0);
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
  spec.param<uint16_t>(batch_capacity_,
                       "batch_capacity",
                       "Batch capacity",
                       "Input buffer capacity in number of network packet batches",
                       4);
}

void NetConnectorAdvanced::initialize() {
  HOLOSCAN_LOG_INFO("NetConnectorAdvanced::initialize()");
  register_converter<std::map<std::string, uint64_t>>();
  holoscan::Operator::initialize();

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

  // Configuration checks
  if (use_hds_.get() && !gpu_direct_.get()) {
    HOLOSCAN_LOG_ERROR("If Header-Data Split mode is enabled, GPUDirect needs to be too");
    exit(1);
  }

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

  // Set vector sizes based on batch_capacity parameter
  h_dev_ptrs_.resize(batch_capacity_.get());
  full_batch_data_h_.resize(batch_capacity_.get());
  ttl_pkts_drop_.resize(batch_capacity_.get());
  streams_.resize(batch_capacity_.get());
  events_.resize(batch_capacity_.get());

  // Allocate memory and create CUDA streams for each concurrent batch
  for (int n = 0; n < batch_capacity_.get(); n++) {
    cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get());
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("Could not allocate cuda memory for h_dev_ptrs_");
    }
    if (!gpu_direct_.get()) {
      // host-pinned memory for full batch data that we put the packets into in CPU mode
      cudaMallocHost(&full_batch_data_h_[n], batch_size_.get() * max_packet_size_.get());
      // populate the host-pinned device pointers since we know them ahead of time
      for (int p = 0; p < batch_size_.get(); p++) {
        h_dev_ptrs_[n][p] = reinterpret_cast<void*>(reinterpret_cast<char*>(full_batch_data_h_[n]) +
                                                    p * max_packet_size_.get());
      }
    }

    cudaStreamCreateWithFlags(&streams_[n], cudaStreamNonBlocking);
    cudaEventCreate(&events_[n]);
    // Warmup
    place_packet_data(nullptr,
                      nullptr,
                      nullptr,
                      0,
                      0,
                      0,
                      16,
                      16,
                      0,
                      0,
                      0,
                      0,
                      0,
                      false,
                      nullptr,
                      0,
                      0,
                      streams_[n]);
    if (cudaStreamSynchronize(streams_[n]) != cudaSuccess) {
      HOLOSCAN_LOG_ERROR(cudaGetErrorString(cudaGetLastError()));
      exit(1);
    }
  }

  buffer_track = BufferTracking(buffer_size_.get());
  matx::make_tensor(rf_data, {buffer_size_.get(), num_samples_.get(), num_subchannels_.get()});

  HOLOSCAN_CUDA_CALL(cudaMallocHost(&rf_metadata_h, buffer_size_.get() * sizeof(RFMetadata)));
  HOLOSCAN_CUDA_CALL(cudaMalloc(&rf_metadata_d, buffer_size_.get() * sizeof(RFMetadata)));
  HOLOSCAN_CUDA_CALL(cudaMemset(rf_metadata_d, 0, buffer_size_.get() * sizeof(RFMetadata)));
  HOLOSCAN_CUDA_CALL(cudaMemcpy(rf_metadata_h,
                                rf_metadata_d,
                                buffer_size_.get() * sizeof(RFMetadata),
                                cudaMemcpyDeviceToHost));

  if (cudaGetLastError() != cudaSuccess) {
    exit(1);
  }

  HOLOSCAN_LOG_INFO("NetConnectorAdvanced::initialize() complete");
}

void NetConnectorAdvanced::freeResources() {
  HOLOSCAN_LOG_INFO("NetConnectorAdvanced::freeResources() start");
  for (int n = 0; n < batch_capacity_.get(); n++) {
    if (full_batch_data_h_[n]) { cudaFreeHost(full_batch_data_h_[n]); }
    if (h_dev_ptrs_[n]) { cudaFreeHost(h_dev_ptrs_[n]); }
    if (streams_[n]) { cudaStreamDestroy(streams_[n]); }
    if (events_[n]) { cudaEventDestroy(events_[n]); }
  }
  buffer_track.free_memory();
  if (rf_metadata_h) {
    cudaFreeHost(rf_metadata_h);
  }
  if (rf_metadata_d) {
    cudaFree(rf_metadata_d);
  }
  if (spoof_header_d) { cudaFree(spoof_header_d); }
  HOLOSCAN_LOG_INFO("NetConnectorAdvanced::freeResources() complete");
}

std::vector<NetConnectorAdvanced::RxMsg> NetConnectorAdvanced::free_bufs() {
  std::vector<NetConnectorAdvanced::RxMsg> completed;

  // Loop over all batches, checking if any have completed
  while (out_q.size() > 0) {
    const auto first = out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      HOLOSCAN_LOG_DEBUG("Batch of packets successfully copied to GPU memory");
      completed.push_back(first);
      for (auto m = 0; m < first.num_batches; m++) { free_all_packets_and_burst_rx(first.msg[m]); }
      out_q.pop();
    } else {
      break;
    }
  }
  return completed;
}

void NetConnectorAdvanced::free_bufs_and_queue_arrays(OutputContext& op_output,
                                                      cudaStream_t& op_stream) {
  // We have to wait for the packet placement to finish because we don't know if a buffer is
  // filled until we check the result of the copy
  std::vector<NetConnectorAdvanced::RxMsg> completed_msgs = free_bufs();
  if (completed_msgs.empty()) {
    return;
  }
  cudaStream_t completed_batch_stream = completed_msgs[0].stream;

  for (size_t i = 0; i < buffer_track.buffer_size; i++) {
    const size_t pos_wrap = (buffer_track.pos + i) % buffer_track.buffer_size;
    HOLOSCAN_LOG_DEBUG("Buffer {}: cnt {} (end {})",
                       buffer_track.counter_h[pos_wrap],
                       buffer_track.sample_cnt_h[pos_wrap],
                       buffer_track.received_end_h[pos_wrap]);
  }

  auto start_pos = buffer_track.pos;
  for (size_t i = 0; i < buffer_track.buffer_size; i++) {
    const size_t pos_wrap = (start_pos + i) % buffer_track.buffer_size;
    const size_t one_ahead = (start_pos + i + 1) % buffer_track.buffer_size;

    // Move to next buffer in loop if this one is completely empty (likely when starting up)
    if (buffer_track.sample_cnt_h[pos_wrap] == 0) {
      continue;
    }

    // Output the next buffer if it is either completed, the next one is completed,
    // or packets have already been placed two buffers or more beyond
    bool packets_seen_two_ahead_plus = false;
    for (size_t j = 2; j < buffer_track.buffer_size; j++) {
      const size_t j_pos_wrap = (start_pos + i + j) % buffer_track.buffer_size;
      if (buffer_track.counter_h[j_pos_wrap] > buffer_track.counter_h[pos_wrap]) {
        packets_seen_two_ahead_plus = true;
        break;
      }
    }
    if (!buffer_track.received_end_h[pos_wrap] && !buffer_track.received_end_h[one_ahead] &&
        !packets_seen_two_ahead_plus) {
      // Samples pending but nothing ready to output yet, exit loop
      break;
    }
    if (buffer_track.pos == 0 && !buffer_track.received_end_h[pos_wrap]) {
      // Ignore partially filled buffer when first starting up (pos == 0)
      continue;
    }

    // Log if we are outputting with missing samples
    if (buffer_track.sample_cnt_h[pos_wrap] < num_samples_.get() * num_subchannels_.get()) {
      HOLOSCAN_LOG_WARN(
          "Outputting sample buffer {} with {} missing IQ samples",
          buffer_track.counter_h[pos_wrap],
          num_samples_.get() * num_subchannels_.get() - buffer_track.sample_cnt_h[pos_wrap]);
    }

    // Copy buffer to output vector
    auto out_data_slice = rf_data.Slice<2>({static_cast<matx::index_t>(pos_wrap), 0, 0},
                                           {matx::matxDropDim, matx::matxEnd, matx::matxEnd});
    auto out_data = matx::make_tensor<sample_t>(
        out_data_slice.Shape(), matx::MATX_ASYNC_DEVICE_MEMORY, op_stream);
    matx::copy(out_data, out_data_slice, op_stream);
    auto out_ptr = std::make_shared<RFArray<sample_t>>(out_data, rf_metadata_h[pos_wrap]);
    // Need to manually set stream on output because it was not gotten by receive_cuda_stream
    op_output.set_cuda_stream(op_stream, "rf_out");
    op_output.emit(out_ptr, "rf_out");

    HOLOSCAN_LOG_DEBUG(
        "Emitting sample buffer {} with {} IQ samples from internal staging buffer {}",
        buffer_track.counter_h[pos_wrap],
        buffer_track.sample_cnt_h[pos_wrap],
        pos_wrap);

    // Synchronize completed_batch_stream with op_stream so we know data copying is done before
    // resetting the buffer and continuing with further packet copying on the batch streams
    cudaEvent_t op_stream_done;
    cudaEventCreate(&op_stream_done);
    cudaEventRecord(op_stream_done, op_stream);
    cudaStreamWaitEvent(completed_batch_stream, op_stream_done);

    // Reset data buffer to 0 after data is copied out
    auto real_shp = out_data_slice.Shape();
    real_shp[1] = 2 * real_shp[1];
    auto out_data_int_view = out_data_slice.View<real_t, 2, typeof(real_shp)>(std::move(real_shp));
    (out_data_int_view = matx::zeros()).run(completed_batch_stream);

    // Set buffer to next position after the one just completed
    // (place_packet_data kernel will take care of resetting counters)
    buffer_track.completed_at_pos(buffer_track.counter_h[pos_wrap], completed_batch_stream);
    HOLOSCAN_LOG_TRACE("Next sample cycle expected: {}", buffer_track.pos);
  }
}

void NetConnectorAdvanced::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context) {
  HOLOSCAN_LOG_TRACE("NetConnectorAdvanced::compute() called");
  int64_t ttl_bytes_in_cur_batch_ = 0;

  auto maybe_stream = context.allocate_cuda_stream("op_stream");
  if (!maybe_stream) {
    const auto& error = maybe_stream.error();
    throw std::runtime_error(
        fmt::format("Failed to allocate cuda stream with error: {}", error.what()));
  }
  cudaStream_t op_stream = maybe_stream.value();

  if (port_id_ == -1) {
    // initialize on first compute since we don't init the Advanced Network Operator until
    // the application starts in order to not collect packets until everything is ready
    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid network interface {} specified in the config",
                         interface_name_.get());
      exit(1);
    }
  }

  BurstParams* burst;
  auto burst_status = get_rx_burst(&burst, port_id_, queue_id_.get());
  while (burst_status == Status::SUCCESS) {
    auto burst_size = get_num_packets(burst);

    HOLOSCAN_LOG_DEBUG("Handling burst of {} packets with {} packets already in buffer",
                       burst_size,
                       aggr_pkts_recv_);

    auto burst_pkts_remaining = burst_size;

    while (burst_pkts_remaining > 0) {
      // Can't proceed until batch that we're aggregating packets into has been cleared from prior
      // processing, so wait for the corresponding event to complete
      if (cudaEventQuery(events_[cur_idx]) != cudaSuccess) {
        HOLOSCAN_LOG_DEBUG("Waiting on event to clear batch with index {}", cur_idx);
        HOLOSCAN_CUDA_CALL(cudaEventSynchronize(events_[cur_idx]));
      }

      auto num_pkts_to_copy =
          std::min(burst_pkts_remaining, static_cast<int64_t>(batch_size_.get() - aggr_pkts_recv_));

      // Track packet payloads for the current burst
      if (gpu_direct_.get()) {
        // GPUDirect mode (needs to match if the ANO queue uses one or more memory regions)
        // Save off the GPU pointers into a host-pinned buffer (h_dev_ptrs_) to reassemble later.
        if (use_hds_.get()) {
          // Header-Data-Split: header to CPU, payload to GPU
          // NOTE: current App assumes only two memory region segments, one for header (CPU),
          //       and one for payload (GPU).

          for (int p = 0; p < num_pkts_to_copy; p++) {
            // Get pointers to payload data on GPU
            // NOTE: It's (1) here since the GPU memory region is second in the list for this queue.
            //       The first region (0) is for headers on CPU, ignored here.
            // NOTE: currently ordering pointers in the order packets come in. If headers had
            // segment
            //       ID, the index in h_dev_ptrs_ should use that (instead of aggr_pkts_recv_ + p).
            h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] = get_segment_packet_ptr(burst, 1, p);
            ttl_bytes_in_cur_batch_ +=
                get_segment_packet_length(burst, 0, p) + get_segment_packet_length(burst, 1, p);
          }
        } else {
          // Batched: headers and payload to GPU (queue memory regions should be a single GPU
          // segment)
          for (int p = 0; p < num_pkts_to_copy; p++) {
            // Get pointers to payload data on GPU (shifting by IPv4 UDP header size)
            // NOTE: currently ordering pointers in the order packets come in. If headers had
            // segment
            //       ID, the index in h_dev_ptrs_ should use that (instead of aggr_pkts_recv_ + p).
            h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] =
                reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, p)) +
                sizeof(UDPIPV4Pkt);
            ttl_bytes_in_cur_batch_ += get_segment_packet_length(burst, 0, p);
          }
        }
      } else {
        /* CPU Mode (needs to match if the ANO queue uses no GPU memory regions)
         * Copy each packet payload in a continuous host-pinned buffer, copy of that larger buffer
         * to the GPU will occur later (copying each packet to GPU directly would be too expensive).
         *
         * NOTE: this assume huge pages memory regions. With host-pinned memory regions, this could
         * be skipped, though probably not faster given the higher perf to write to huge pages.
         */

        auto batch_offset = aggr_pkts_recv_ * max_packet_size_.get();

        for (int p = 0; p < num_pkts_to_copy; p++) {
          // Payload address (UDPIPV4Pkt: + 1 skips the header)
          auto payload_ptr = static_cast<UDPIPV4Pkt*>(get_segment_packet_ptr(burst, 0, p)) + 1;
          // Payload length (packet length minus header length)
          auto pkt_len = get_segment_packet_length(burst, 0, p);
          auto payload_len = pkt_len - sizeof(UDPIPV4Pkt);

          // Copy payload to aggregated CPU buffers now
          memcpy((char*)full_batch_data_h_[cur_idx] + batch_offset + p * max_packet_size_.get(),
                 payload_ptr,
                 payload_len);

          // Count bytes received
          ttl_bytes_in_cur_batch_ += pkt_len;

          // TODO: could free CPU packets now
        }
      }
      ttl_bytes_recv_ += ttl_bytes_in_cur_batch_;

      aggr_pkts_recv_ += num_pkts_to_copy;
      burst_pkts_remaining -= num_pkts_to_copy;
      // If we're finished with the packet burst, then add it to the current message to be released
      if (burst_pkts_remaining == 0) { cur_msg_.msg[cur_msg_.num_batches++] = burst; }

      // Once we've aggregated enough packets, do some work
      if (aggr_pkts_recv_ >= batch_size_.get()) {
        HOLOSCAN_LOG_DEBUG(
            "{} packets collected exceeding batch size of {}, packing into array on GPU",
            aggr_pkts_recv_,
            batch_size_.get());
        do {
          free_bufs_and_queue_arrays(op_output, op_stream);
          if (out_q.size() >= batch_capacity_.get()) {
            HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
            const auto first_msg = out_q.front();
            if (cudaStreamSynchronize(first_msg.stream) != cudaSuccess) {
              HOLOSCAN_LOG_ERROR(
                  "Failed to synchronize to next stream for placing packet data. Ending with "
                  "error:");
              HOLOSCAN_LOG_ERROR(cudaGetErrorString(cudaGetLastError()));
              exit(1);
            }
          }
        } while (out_q.size() >= batch_capacity_.get());

        // Copy packet I/Q contents to appropriate location in 'rf_data'
        place_packet_data(rf_data.Data(),
                          rf_metadata_d,
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
                          apply_conjugate_.get(),
                          spoof_header_d,
                          ttl_pkts_recv_,            // only needed if spoofing packets
                          packet_skip_bytes_.get(),  // only needed if spoofing packets
                          streams_[cur_idx]);
        // Get updated buffer tracking information back to host
        buffer_track.transfer(cudaMemcpyDeviceToHost, streams_[cur_idx]);
        // Get updated rf_metadata buffer back to host
        HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(rf_metadata_h,
                                           rf_metadata_d,
                                           buffer_size_.get() * sizeof(RFMetadata),
                                           cudaMemcpyDeviceToHost,
                                           streams_[cur_idx]));

        HOLOSCAN_CUDA_CALL(cudaEventRecord(events_[cur_idx], streams_[cur_idx]));
        cur_msg_.stream = streams_[cur_idx];
        cur_msg_.evt = events_[cur_idx];
        out_q.push(cur_msg_);
        cur_msg_.num_batches = 0;

        ttl_pkts_recv_ += aggr_pkts_recv_;

        auto cuda_err_status = cudaGetLastError();
        if (cuda_err_status != cudaSuccess) {
          HOLOSCAN_LOG_ERROR(
              "CUDA error dispatching batch from queue number {} after {} total packets received: "
              "{}",
              cur_idx,
              ttl_pkts_recv_,
              cudaGetErrorString(cuda_err_status));
          exit(1);
        }
        aggr_pkts_recv_ = 0;
        cur_idx = (++cur_idx % batch_capacity_.get());
      }
    }

    // see if we have another burst on the receive buffer
    burst_status = get_rx_burst(&burst, port_id_, queue_id_.get());
  }

  // One final check for completed arrays before exiting
  free_bufs_and_queue_arrays(op_output, op_stream);
}

void NetConnectorAdvanced::stop() {
  HOLOSCAN_LOG_INFO(
      "\n"
      "NetConnectorAdvanced exit report:\n"
      "--------------------------------\n"
      " - Processed bytes:     {}\n"
      " - Processed packets:   {}\n",
      ttl_bytes_recv_,
      ttl_pkts_recv_);

  freeResources();
}

}  // namespace holoscan::ops
