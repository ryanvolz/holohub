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
#include <chrono>

#include "holoscan/holoscan.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "rf_array/net_connector_basic.h"
#include "rf_array/net_connector_common.h"

namespace holoscan::ops {

void NetConnectorBasic::setup(OperatorSpec& spec) {
  // We'd want to set the receiver connector capacity to match the batch_capacity parameter,
  // but there's no good way to do that other than to do when creating the operator within
  // an application. So just after you create a NetConnectorBasic operator, access "burst_in"
  // in the `inputs` map and call `connector()` to add a kDoubleBuffer resource with capacity
  // set to the value of the batch_capacity parameter.
  // No input condition so operator will run regardless of whether input is available
  // (this enables compute to be called regularly so we can emit a warning when a certain
  //  time has passed since anything was output)
  spec.input<std::shared_ptr<NetworkOpBurstParams>>("burst_in").condition(ConditionType::kNone);
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

  // Miscellaneous
  spec.param<uint32_t>(no_output_warn_interval_,
                       "no_output_warn_interval",
                       "Warning interval for no output",
                       "Interval in seconds between warnings about no output being produced",
                       30);
  spec.param<bool>(debug_print_,
                   "debug_print",
                   "Debug printing enabled",
                   "Enable packet kernel debug printing",
                   false);
  spec.param<int16_t>(packet_stream_priority_,
                      "packet_stream_priority",
                      "Packet stream priority",
                      "Desired priority for the streams running the packet processing kernel",
                      -1);
}

void NetConnectorBasic::initialize() {
  HOLOSCAN_LOG_INFO("NetConnectorBasic::initialize()");
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

  // Log requested packet stream priority and whether it is within the valid range
  int least_priority = 0;
  int greatest_priority = -1;

  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
      "Failed to get stream priority range");
  HOLOSCAN_LOG_INFO(
      "Requested packet stream priority is {}, valid range: {} (greatest) to {} (least).",
      packet_stream_priority_.get(),
      greatest_priority,
      least_priority);

  // Set vector sizes based on batch_capacity parameter
  h_dev_ptrs_.resize(batch_capacity_.get());
  full_batch_data_h_.resize(batch_capacity_.get());
  ttl_pkts_drop_.resize(batch_capacity_.get());
  streams_.resize(batch_capacity_.get());
  events_.resize(batch_capacity_.get());

  // Allocate memory and create CUDA streams for each concurrent batch
  for (int n = 0; n < batch_capacity_.get(); n++) {
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

    cudaStreamCreateWithPriority(
        &streams_[n], cudaStreamNonBlocking, packet_stream_priority_.get());
    cudaEventCreate(&events_[n]);
  }

  buffer_track = BufferTracking(buffer_size_.get(), num_samples_.get());

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

  for (int n = 0; n < batch_capacity_.get(); n++) {
    // Warmup
    place_packet_data(rf_data,
                      rf_metadata_d,
                      nullptr,
                      buffer_track.sample_cnt_d,
                      buffer_track.full_cnt_d,
                      buffer_track.counter_d,
                      buffer_track.completed_pos_d,
                      16,
                      max_samples_per_packet,
                      freq_idx_scaling_.get(),
                      freq_idx_offset_.get(),
                      apply_conjugate_.get(),
                      spoof_header_d,
                      ttl_bytes_recv_,
                      packet_skip_bytes_.get(),
                      debug_print_.get(),
                      streams_[n]);
    if (cudaStreamSynchronize(streams_[n]) != cudaSuccess) {
      HOLOSCAN_LOG_ERROR(cudaGetErrorString(cudaGetLastError()));
      exit(1);
    }
  }

  HOLOSCAN_LOG_INFO("NetConnectorBasic::initialize() complete");
}

void NetConnectorBasic::freeResources() {
  HOLOSCAN_LOG_INFO("NetConnectorBasic::freeResources() start");
  for (int n = 0; n < batch_capacity_.get(); n++) {
    if (full_batch_data_h_[n]) {
      cudaFreeHost(full_batch_data_h_[n]);
    }
    if (h_dev_ptrs_[n]) {
      cudaFreeHost(h_dev_ptrs_[n]);
    }
    if (streams_[n]) {
      cudaStreamDestroy(streams_[n]);
    }
    if (events_[n]) {
      cudaEventDestroy(events_[n]);
    }
  }
  buffer_track.free_memory();
  if (rf_metadata_h) {
    cudaFreeHost(rf_metadata_h);
  }
  if (rf_metadata_d) {
    cudaFree(rf_metadata_d);
  }
  if (spoof_header_d) {
    cudaFree(spoof_header_d);
  }
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

void NetConnectorBasic::check_completed_and_queue_arrays(OutputContext& op_output,
                                                         cudaStream_t& op_stream) {
  // We have to wait for the packet placement to finish because we don't know if a buffer is
  // filled until we check the result of the copy
  std::vector<NetConnectorBasic::RxMsg> completed_msgs = check_completed();
  if (completed_msgs.empty()) {
    return;
  }

  for (size_t i = 0; i < buffer_track.buffer_size; i++) {
    const size_t pos_wrap = (buffer_track.pos + i) % buffer_track.buffer_size;
    HOLOSCAN_LOG_TRACE("Buffer {}: sample_cnt {} (full_cnt {})",
                       buffer_track.counter_h[pos_wrap],
                       buffer_track.sample_cnt_h[pos_wrap],
                       buffer_track.full_cnt_h[pos_wrap]);
  }
  HOLOSCAN_LOG_TRACE("Buffer completed_pos {}", *buffer_track.completed_pos_h);

  auto buf_idx = buffer_track.find_ready_idx();
  while (buf_idx != buffer_track.buffer_size) {
    // We have something to output!

    // Get copy of data to output, update buffer tracking, and signal to kernel
    auto out_data = buffer_track.completed_at_pos(buf_idx, rf_data, op_stream);
    auto out_ptr = std::make_shared<RFArray<sample_t>>(out_data, rf_metadata_h[buf_idx]);
    op_output.emit(out_ptr, "rf_out");
    last_emit = std::chrono::steady_clock::now();

    HOLOSCAN_LOG_DEBUG("Emitting sample buffer {} with {} samples from internal staging buffer {}",
                       buffer_track.counter_h[buf_idx],
                       buffer_track.full_cnt_h[buf_idx],
                       buf_idx);
    HOLOSCAN_LOG_TRACE("Next sample cycle expected: {}", buffer_track.pos);

    // See if we have another buffer ready
    buf_idx = buffer_track.find_ready_idx();
  }
}

void NetConnectorBasic::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  HOLOSCAN_LOG_TRACE("NetConnectorBasic::compute() called");
  auto burst_maybe = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  cudaStream_t op_stream = op_input.receive_cuda_stream("burst_in", true, false);

  if (!last_emit) {
    // on first run set the time of last emit
    last_emit = std::chrono::steady_clock::now();
  }

  while (burst_maybe) {
    auto burst = burst_maybe.value();

    HOLOSCAN_LOG_DEBUG(
        "Handling burst of {} packets and {} bytes, with {} packets already in buffer",
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
      // Can't proceed until batch that we're aggregating packets into has been cleared from prior
      // processing, so wait for the corresponding event to complete
      if (cudaEventQuery(events_[cur_idx]) != cudaSuccess) {
        HOLOSCAN_LOG_DEBUG("Waiting on event to clear batch with index {}", cur_idx);
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventSynchronize(events_[cur_idx]),
                                       "Failed to synchronize on cleared batch");
      }

      auto num_pkts_to_copy = std::min(burst_pkts_remaining,
                                       static_cast<uint32_t>(batch_size_.get() - aggr_pkts_recv_));
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
          check_completed_and_queue_arrays(op_output, op_stream);
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
        place_packet_data(rf_data,
                          rf_metadata_d,
                          h_dev_ptrs_[cur_idx],
                          buffer_track.sample_cnt_d,
                          buffer_track.full_cnt_d,
                          buffer_track.counter_d,
                          buffer_track.completed_pos_d,
                          aggr_pkts_recv_,
                          max_samples_per_packet,
                          freq_idx_scaling_.get(),
                          freq_idx_offset_.get(),
                          apply_conjugate_.get(),
                          spoof_header_d,
                          ttl_pkts_recv_,            // only needed if spoofing packets
                          packet_skip_bytes_.get(),  // only needed if spoofing packets
                          debug_print_.get(),
                          streams_[cur_idx]);
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
        // Get updated buffer tracking information back to host
        buffer_track.transfer(streams_[cur_idx]);
        // Get updated rf_metadata buffer back to host
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpyAsync(rf_metadata_h,
                                                       rf_metadata_d,
                                                       buffer_size_.get() * sizeof(RFMetadata),
                                                       cudaMemcpyDeviceToHost,
                                                       streams_[cur_idx]),
                                       "Failed to transfer rf_metadata");

        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(events_[cur_idx], streams_[cur_idx]),
                                       "Failed to record place_packet_data completed event");
        cur_msg_.stream = streams_[cur_idx];
        cur_msg_.evt = events_[cur_idx];
        out_q.push(cur_msg_);
        cur_msg_.num_batches = 0;

        ttl_pkts_recv_ += aggr_pkts_recv_;
        aggr_pkts_recv_ = 0;
        cur_idx = (++cur_idx % batch_capacity_.get());
      }
    }

    // free packets in burst
    delete[] burst->data;

    // see if we have another burst on the receive buffer
    burst_maybe = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  }

  // One final check for completed arrays before exiting
  check_completed_and_queue_arrays(op_output, op_stream);

  // Check to see if it has been a while since anything was output, and warn if it has
  auto now = std::chrono::steady_clock::now();
  auto duration_since_emit_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now - last_emit.value()).count();
  if (duration_since_emit_seconds > no_output_warn_interval_.get()) {
    HOLOSCAN_LOG_WARN("No arrays have been output in at least the last {} seconds!",
                      no_output_warn_interval_.get());
    last_emit = now;
  }
}

void NetConnectorBasic::stop() {
  HOLOSCAN_LOG_INFO(
      "\n"
      "NetConnectorBasic exit report:\n"
      "--------------------------------\n"
      " - Processed bytes:     {}\n"
      " - Processed packets:   {}\n"
      " - Output samples:      {}\n"
      " - Dropped samples:     {}\n",
      ttl_bytes_recv_,
      ttl_pkts_recv_,
      buffer_track.total_output_samples,
      buffer_track.total_dropped_samples);

  freeResources();
}

}  // namespace holoscan::ops
